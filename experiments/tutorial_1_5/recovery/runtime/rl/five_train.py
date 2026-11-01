"""튜토리얼 1~5의 전문가 학습과 균등한 PPO 통합 과정을 재현 가능하게 실행한다."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from .five_env import FiveEnv, OBS_SIZE
from .five_models import ALGORITHMS, FiveExtractor, action_probs


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sources():
    files = ['rl/five_env.py', 'rl/five_models.py', 'rl/five_train.py',
             'rl/all_tutorial_env.py', 'rl/map_input_env.py', 'pingu_ball/engine.py',
             'pingu_ball/special_lab.py', 'pingu_ball/specials.py', 'pingu_ball/playfield.py',
             'pingu_ball/config.py', 'pingu_ball/geometry.py']
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}


def vector(stages, reward_mode='contact'):
    return DummyVecEnv([lambda s=s: FiveEnv(stage=s, reward_mode=reward_mode) for s in stages])


def evaluate(model, stages, count=20, seed=10000, reward_mode='contact', epsilon=0.):
    """행동 추출 난수를 분리한다. 확률적 평가를 기본으로 하고 맵마다 최대확률 재생을 한 번 한다."""
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    rows = []
    try:
        for stage in stages:
            env = vector([stage]*4, reward_mode)
            rngs = [np.random.default_rng(seed+stage*100003+i*104729) for i in range(4)]
            samples = []
            try:
                remaining = [count//4+int(i<count%4) for i in range(4)]
                obs = env.reset()
                while sum(remaining):
                    probs = action_probs(model, obs)
                    actions = []
                    for rng, p in zip(rngs, probs):
                        p = p/p.sum()
                        action = int(rng.choice(3, p=p)) if not epsilon else int(p.argmax())
                        if epsilon and rng.random() < epsilon:
                            action = int(rng.integers(3))
                        actions.append(action)
                    obs, _, done, infos = env.step(np.asarray(actions))
                    for i in range(4):
                        if done[i] and remaining[i]:
                            samples.append({k: v for k, v in infos[i].items()
                                            if k not in ('terminal_observation', 'TimeLimit.truncated')})
                            remaining[i] -= 1
            finally:
                env.close()
            raw = FiveEnv(stage=stage, reward_mode=reward_mode)
            obs, _ = raw.reset()
            while not raw.unwrapped.done:
                action = int(action_probs(model, obs[None])[0].argmax())
                obs, _, _, _, greedy = raw.step(action)
            wins = [r['seconds'] for r in samples if r['success']]
            rows.append(dict(stage=stage, episodes=count, successes=len(wins),
                success_rate=len(wins)/count, mean_success_seconds=float(np.mean(wins)) if wins else None,
                mean_return=float(np.mean([r['episode_return'] for r in samples])),
                failures=dict(Counter(r['reason'] for r in samples if not r['success'])),
                greedy=greedy, raw=samples))
            raw.close()
    finally:
        model.policy.set_training_mode(was_training)
    return dict(protocol='sampled_policy' if not epsilon else f'greedy_plus_{epsilon}_input_noise',
                seed=seed, maps=rows, mean_success_rate=float(np.mean([r['success_rate'] for r in rows])),
                min_success_rate=min(r['success_rate'] for r in rows))


def validation_score(ev):
    return (ev['min_success_rate'], ev['mean_success_rate'],
            sum(r['greedy']['success'] for r in ev['maps']),
            -sum(r['mean_success_seconds'] or 40. for r in ev['maps']))


class Monitor(BaseCallback):
    def __init__(self, folder, stages, budget, reward_mode, eval_every=20000, eval_count=20):
        super().__init__()
        self.folder, self.stages = Path(folder), stages
        self.budget, self.reward_mode = budget, reward_mode
        self.eval_every, self.eval_count = eval_every, eval_count
        self.next_eval = eval_every
        self.best_score = (-1, -1, -1, -1e9)
        self.best_step = 0
        self.rows = []
        self.episodes = self.successes = self.ticks = 0
        self.started = time.perf_counter()
        self.per_map = {str(s): dict(episodes=0, successes=0, transitions=0) for s in stages}

    def _on_step(self):
        for i, info in enumerate(self.locals['infos']):
            self.ticks += info['tick_delta']
            row = self.per_map[str(info['stage'])]
            row['transitions'] += 1
            if self.locals['dones'][i]:
                self.episodes += 1
                self.successes += int(info['success'])
                row['episodes'] += 1
                row['successes'] += int(info['success'])
        return True

    def checkpoint(self, model):
        ev = evaluate(model, self.stages, self.eval_count, 11000, self.reward_mode)
        score = validation_score(ev)
        if score >= self.best_score:
            self.best_score, self.best_step = score, model.num_timesteps
            model.save(self.folder/'best')
            write(self.folder/'best_validation.json', dict(step=self.best_step, evaluation=ev))
        diagnostics = {k: float(v) for k, v in model.logger.name_to_value.items()
                       if k.startswith('train/') and np.isscalar(v) and np.isfinite(v)}
        row = dict(decisions=model.num_timesteps, physics_ticks=self.ticks, episodes=self.episodes,
                   training_successes=self.successes, per_map=self.per_map, validation=ev,
                   diagnostics=diagnostics, entropy=model.ent_coef,
                   elapsed_seconds=time.perf_counter()-self.started)
        self.rows.append(json.loads(json.dumps(row)))
        write(self.folder/'metrics.json', self.rows)
        model.save(self.folder/'latest')
        write(self.folder/'live.json', {k: v for k, v in row.items() if k != 'validation'} |
              dict(validation={r['stage']: r['success_rate'] for r in ev['maps']}))
        model.logger.dump(step=model.num_timesteps)
        print(json.dumps(dict(step=model.num_timesteps, success=self.successes,
              validation={r['stage']: r['success_rate'] for r in ev['maps']},
              seconds=round(row['elapsed_seconds']))), flush=True)

    def after_update(self, model):
        fraction = min(1., model.num_timesteps/self.budget)
        model.ent_coef = .02*(1-fraction)+.005*fraction
        if model.num_timesteps >= self.next_eval or model.num_timesteps >= self.budget:
            self.checkpoint(model)
            self.next_eval += self.eval_every


def make_model(algorithm, vec, seed, joint=False):
    kwargs = dict(policy='MlpPolicy', env=vec, seed=seed, device='cpu', verbose=0,
                  gamma=1., learning_rate=.0001 if joint else .0003, gae_lambda=.98,
                  ent_coef=.02, vf_coef=.5, max_grad_norm=.5,
                  policy_kwargs=dict(features_extractor_class=FiveExtractor,
                                     net_arch=dict(pi=[128, 128], vf=[128, 128])))
    if algorithm == 'a2c':
        model = ALGORITHMS[algorithm](**kwargs, n_steps=64, normalize_advantage=True)
    else:
        model = ALGORITHMS[algorithm](**kwargs, n_steps=128, batch_size=256,
                                      n_epochs=5, target_kl=.02, clip_range=.2)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    return model


def finish(model, monitor, folder, config, test_count):
    if not monitor.rows or monitor.rows[-1]['decisions'] != model.num_timesteps:
        monitor.checkpoint(model)
    model.save(folder/'latest')
    best = ALGORITHMS[config['algorithm']].load(folder/'best.zip', device='cpu')
    final = evaluate(best, monitor.stages, test_count, 910000+config['seed'], monitor.reward_mode)
    # 독립적인 강건성 진단이며 체크포인트나 교사 모델 선택에는 사용하지 않는다.
    robust = evaluate(best, monitor.stages, min(20, test_count), 1910000+config['seed'], monitor.reward_mode, epsilon=.05)
    result = dict(config=config, decisions=model.num_timesteps, physics_ticks=monitor.ticks,
                  episodes=monitor.episodes, training_successes=monitor.successes,
                  elapsed_seconds=time.perf_counter()-monitor.started,
                  best_step=monitor.best_step, best_validation=read(folder/'best_validation.json'),
                  final=final, robustness=robust, sources_after=sources(),
                  per_map=monitor.per_map, final_test_used_for_selection=False)
    assert config['sources'] == result['sources_after']
    write(folder/'summary.json', result)
    return result


def expert_job(payload):
    root, algorithm, stage, seed, budget, mode, test_count = payload
    torch.set_num_threads(1)
    folder = Path(root)/'experts'/algorithm/f'seed_{seed}'/f'stage_{stage:02d}'
    if (folder/'summary.json').exists():
        old = read(folder/'summary.json')
        assert old['config']['sources'] == sources()
        return str(folder/'summary.json')
    if (folder/'config.json').exists():
        raise RuntimeError(f'Incomplete run, do not overwrite: {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'console.log').open('w', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            vec = vector([stage]*4, mode)
            try:
                model = make_model(algorithm, vec, seed)
                model.set_logger(configure(str(folder), ['csv']))
                config = dict(algorithm=algorithm, stage=stage, seed=seed, decisions_budget=budget,
                              reward_mode=mode, from_scratch=True, teacher=False, original_spawn=True,
                              action_count=3, observation_size=OBS_SIZE, sources=sources(),
                              reward='clear +1 / death-or-40s-limit -1; first native support landing +.01, episode cap .1' if mode=='contact' else 'clear +1 / death-or-40s-limit -1',
                              test_count=test_count, validation_count=20, validation_seed=11000,
                              primary_evaluation='sampled_policy', previous_models_loaded=[])
                write(folder/'config.json', config)
                model.save(folder/'initial')
                monitor = Monitor(folder, [stage], budget, mode)
                model.after_update = monitor.after_update
                monitor.checkpoint(model)
                model.learn(total_timesteps=budget, callback=monitor, log_interval=None)
                finish(model, monitor, folder, config, test_count)
            finally:
                vec.close()
    return str(folder/'summary.json')


def select_teachers(root, seed, gate):
    selected = {}
    for stage in range(1, 6):
        candidates = []
        # 통합 실험의 초기값을 짝지으며, 가장 잘된 학습 초기값만 골라 사용하지 않는다.
        for algorithm in ('a2c', 'ppo'):
            folder = root/'experts'/algorithm/f'seed_{seed}'/f'stage_{stage:02d}'
            summary = read(folder/'summary.json')
            ev = summary['best_validation']['evaluation']
            candidates.append((validation_score(ev), algorithm, folder, ev))
        _, algorithm, folder, ev = max(candidates, key=lambda item: item[0])
        selected[str(stage)] = dict(algorithm=algorithm, checkpoint=str((folder/'best.zip').resolve()),
                                   validation=ev, passes_gate=ev['min_success_rate']>=gate)
    return selected


def collect(model, stage, count, seed, mode, controller=None):
    """목표 확률은 항상 해당 맵의 교사에게서 얻는다. 선택적으로 학생이 방문한 상태도 수집한다."""
    env = vector([stage]*4, mode)
    rng = np.random.default_rng(seed)
    observations, targets = [], []
    obs = env.reset()
    model.policy.set_training_mode(False)
    if controller is not None:
        controller.policy.set_training_mode(False)
    try:
        while len(observations)*4 < count:
            target = action_probs(model, obs)
            act_probs = target if controller is None else action_probs(controller, obs)
            observations.append(obs.copy())
            targets.append(target.copy())
            actions = [rng.choice(3, p=p/p.sum()) for p in act_probs]
            obs, _, _, _ = env.step(actions)
    finally:
        env.close()
    return np.concatenate(observations)[:count], np.concatenate(targets)[:count]


def distill(student, datasets, epochs, seed):
    """한 판의 길이와 관계없이 모든 맵의 학습 기여도를 정확히 같게 한다."""
    rng = np.random.default_rng(seed)
    student.policy.set_training_mode(True)
    optimizer = torch.optim.Adam(student.policy.parameters(), lr=.0003)
    losses = []
    size = min(len(x) for x, _ in datasets.values())
    for epoch in range(epochs):
        order = {s: rng.permutation(len(x))[:size] for s, (x, _) in datasets.items()}
        epoch_loss = []
        for start in range(0, size, 128):
            parts = []
            for s, (x, target) in datasets.items():
                ids = order[s][start:start+128]
                logits = student.policy.get_distribution(torch.from_numpy(x[ids])).distribution.logits
                q = torch.from_numpy(target[ids])
                parts.append((q*(q.clamp_min(1e-8).log()-logits)).sum(1).mean())
            loss = torch.stack(parts).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.policy.parameters(), .5)
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
        losses.append(dict(epoch=epoch+1, kl=float(np.mean(epoch_loss))))
    return losses


def consolidation_job(payload):
    root, seed, budget, gate, mode, test_count, allow_weak = payload
    torch.set_num_threads(1)
    root = Path(root)
    folder = root/'consolidated'/f'seed_{seed}'
    if (folder/'completion.json').exists():
        return str(folder/'completion.json')
    if (folder/'config.json').exists():
        raise RuntimeError(f'Incomplete consolidation: {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    selection = select_teachers(root, seed, gate)
    write(folder/'teachers.json', selection)
    failed = [int(s) for s, row in selection.items() if not row['passes_gate']]
    if failed and not allow_weak:
        write(folder/'completion.json', dict(status='blocked_by_teacher_quality', failed_stages=failed,
                                            gate=gate, training_started=False))
        return str(folder/'completion.json')
    with (folder/'console.log').open('w', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            teachers = {int(s): ALGORITHMS[r['algorithm']].load(r['checkpoint'], device='cpu') for s, r in selection.items()}
            for t in teachers.values():
                t.policy.set_training_mode(False)
            vec = vector([s for s in range(1, 6) for _ in range(2)], mode)
            try:
                student = make_model('ppo', vec, seed+100, joint=True)
                student.set_logger(configure(str(folder), ['csv']))
                config = dict(algorithm='ppo', seed=seed, stages=list(range(1, 6)),
                              decisions_budget=budget, reward_mode=mode, from_scratch=False,
                              initialization='balanced distillation from self-trained map experts',
                              teacher_gate=gate, weak_teacher_stages=failed, sources=sources(),
                              original_spawn=True, equal_map_transition_weight=.2,
                              joint_algorithm='PPO + per-map equally weighted teacher KL auxiliary gradient (.1 to .01)',
                              test_count=test_count, validation_count=20,
                              primary_evaluation='sampled_policy', human_demonstrations=False)
                write(folder/'config.json', config)
                datasets = {s: collect(t, s, 6000, seed*1000+s, mode) for s, t in teachers.items()}
                for s, (x, q) in datasets.items():
                    np.savez_compressed(folder/f'teacher_data_{s}.npz', states=x, probabilities=q)
                history = []
                # 학생이 방문한 상태를 보충하되, 항상 해당 맵의 전문가에게 목표 행동을 질의한다.
                for cycle in range(3):
                    history.append(dict(cycle=cycle, losses=distill(student, datasets, 8, seed+cycle*100)))
                    ev = evaluate(student, list(range(1, 6)), 20, 11000, mode)
                    history[-1]['validation'] = ev
                    student.save(folder/f'distill_cycle_{cycle}')
                    write(folder/'distillation.json', history)
                    if cycle < 2:
                        for s, teacher in teachers.items():
                            x, q = collect(teacher, s, 2000, seed*10000+s+cycle*100, mode, controller=student)
                            oldx, oldq = datasets[s]
                            datasets[s] = (np.concatenate([oldx, x]), np.concatenate([oldq, q]))
                best_cycle = max(range(3), key=lambda i: validation_score(history[i]['validation']))
                chosen = ALGORITHMS['ppo'].load(folder/f'distill_cycle_{best_cycle}.zip', device='cpu')
                student.policy.load_state_dict(chosen.policy.state_dict())
                student.save(folder/'distilled')
                distilled = evaluate(student, list(range(1, 6)), test_count, 910000+seed, mode)
                write(folder/'distilled_test.json', dict(chosen_cycle=best_cycle, final=distilled,
                                                       test_used_for_selection=False))
                # 새로 수집한 통합 실행 경험으로 가치 함수를 학습한다. 전문가 가치 함수를 평균 내지 않는다.
                monitor = Monitor(folder, list(range(1, 6)), budget, mode)
                student.teachers = teachers
                student.joint_budget = budget
                student.after_update = monitor.after_update
                monitor.checkpoint(student)
                student.learn(total_timesteps=budget, callback=monitor, log_interval=None)
                finish(student, monitor, folder, config, test_count)
                write(folder/'completion.json', dict(status='completed', gate_passed=not failed,
                                                     weak_teacher_stages=failed, sources=sources()))
            finally:
                vec.close()
    return str(folder/'completion.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expert-steps', type=int, default=200000)
    parser.add_argument('--joint-steps', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--reward-mode', choices=['contact', 'terminal'], default='contact')
    parser.add_argument('--test-count', type=int, default=100)
    parser.add_argument('--gate', type=float, default=.8)
    parser.add_argument('--allow-weak-teachers', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    plan = dict(stages=list(range(1, 6)), seeds=seeds, algorithms=['a2c', 'ppo'],
                expert_steps=args.expert_steps, joint_steps=args.joint_steps,
                reward_mode=args.reward_mode, test_count=args.test_count,
                gate=args.gate, allow_weak_teachers=args.allow_weak_teachers,
                validation_count=20, validation_seed=11000, sources=sources(),
                general_maps=False, git_actions=False, human_demonstrations=False,
                primary_evaluation='sampled_policy', previous_experiments_loaded=False)
    if (args.root/'plan.json').exists():
        assert read(args.root/'plan.json') == plan, 'Changed protocol requires a new root'
    else:
        write(args.root/'plan.json', plan)
        for path in plan['sources']:
            dst = args.root/'source_snapshot'/path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
    tasks = [(str(args.root), a, s, seed, args.expert_steps, args.reward_mode, args.test_count)
             for seed in seeds for s in range(1, 6) for a in ('a2c', 'ppo')]
    completed = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(expert_job, task) for task in tasks]
        for future in as_completed(pending):
            path = future.result()
            completed.append(path)
            assert sources() == plan['sources'], 'Training sources changed during run'
            write(args.root/'progress.json', dict(phase='experts', completed=completed, total=len(tasks),
                                                  elapsed_seconds=time.perf_counter()-started))
            print(f'EXPERT {len(completed)}/{len(tasks)} {path}', flush=True)
        pending = [pool.submit(consolidation_job, (str(args.root), seed, args.joint_steps,
                   args.gate, args.reward_mode, args.test_count, args.allow_weak_teachers)) for seed in seeds]
        consolidation = []
        for future in as_completed(pending):
            consolidation.append(future.result())
            write(args.root/'progress.json', dict(phase='consolidation', completed=completed,
                consolidation=consolidation, total=len(tasks), elapsed_seconds=time.perf_counter()-started))
            print('CONSOLIDATION '+consolidation[-1], flush=True)
    write(args.root/'completion.json', dict(complete=True, experts=len(completed), consolidation=consolidation,
          consolidation_status=[read(p)['status'] for p in consolidation],
          elapsed_seconds=time.perf_counter()-started, sources=sources()))


if __name__ == '__main__':
    main()
