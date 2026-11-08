"""PDF의 맵1 PPO·33항목 보상 실험. 시드 하나·50만 행동 스텝으로 처음부터 학습한다."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from pingu_ball.engine import World
from .five_models import action_probs
from .five_train import read, write
from .map_a_env import OBS_SIZE
from .map_a_models import make_model, load_policy
from .map1_spec_env import SpecEnv, preflight
from .map1_spec_rewards import RULES, audit_reward
from .map1_spec_curriculum import SnapshotBank, SpecCurriculumEnv, ROLES, DecisionLog, is_turn_decision
from .map1_spec_curriculum_stats import CurriculumStatsMixin, evaluate_local
from .stabilized_train import coefficients
from .stabilized_models import FixedRate


PPO_SETTINGS = dict(learning_rate=.0003, gamma=1., gae_lambda=.98, n_envs=4,
    n_steps=128, rollout_decisions=512, batch_size=256, n_epochs=5, clip_range=.2,
    target_kl=.02, vf_coef=.5, max_grad_norm=.5, normalize_advantage=True,
    clip_range_vf=None, device='cpu', reward_scale=.01, limit_seconds=180.)


def sources():
    """학습·물리 관련 소스를 보관해 어떤 코드로 학습했는지 재확인한다."""
    paths = [p for folder in ('rl', 'pingu_ball') for p in Path(folder).rglob('*.py')]
    paths.append(Path('assets/rl_maps/layouts.json'))
    return {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def aggregate(rows, usage, seed, greedy, native, error):
    """보상 획득률과 단순 목표 발판 도달률을 구분한다."""
    n = len(rows)
    wins = [r['seconds'] for r in rows if r['success']]
    def total(key):
        counts = Counter()
        for row in rows:
            counts.update(row[key])
        return dict(counts)
    logs = [r.get('turn_policy', DecisionLog().summary()) for r in rows]
    turns = sum(log['decisions'] for log in logs)
    turn_policy = dict(decisions=turns,
        actions=np.sum([log['actions'] for log in logs], axis=0).astype(int).tolist())
    for direction in ('left', 'stay', 'right'):
        turn_policy[direction+'_actions'] = sum(log[direction+'_actions'] for log in logs)
        turn_policy['mean_'+direction+'_probability'] = (sum(
            (log['mean_'+direction+'_probability'] or 0.)*log['decisions'] for log in logs)/turns) if turns else None
    return dict(episodes=n, successes=len(wins), success_rate=len(wins)/n,
        mean_return=float(np.mean([r['episode_return'] for r in rows])),
        mean_reward_items=float(np.mean([len(r['reward_items']) for r in rows])),
        upper_entry_rate=float(np.mean([r['upper_entered'] for r in rows])),
        mean_success_seconds=float(np.mean(wins)) if wins else None,
        mean_episode_seconds=float(np.mean([r['seconds'] for r in rows])),
        reward_item_rates={rule.id: sum(rule.id in r['reward_items'] for r in rows)/n for rule in RULES},
        target_reach_rates={rule.id: sum(bool(set(rule.target) & set(r['landed_platforms']))
                                          for r in rows)/n for rule in RULES},
        reward_payouts={rule.id: sum(rule.id in r['reward_items'] for r in rows) for rule in RULES},
        duplicate_landings=total('duplicate_landings'), unpaid_landings=total('unpaid_landings'),
        failures=dict(Counter(r['reason'] for r in rows if not r['success'])),
        failure_regions=dict(Counter(r['last_region'] for r in rows if not r['success'])),
        last_regions=dict(Counter(r['last_region'] for r in rows)), actions=usage.tolist(),
        seed=seed, protocol='greedy' if greedy else 'categorical_policy_sampling',
        native_comparison=native, max_native_error=error if native else None, turn_policy=turn_policy, raw=rows)


def evaluate(model, count=20, seed=11000, greedy=False, native=False):
    """학습과 분리된 난수로 원래 시작점에서 평가한다. 시험 결과로 모델을 고르지 않는다."""
    training = model.policy.training
    model.policy.set_training_mode(False)
    envs = [SpecEnv() for _ in range(min(4, count))]
    observations = [env.reset()[0] for env in envs]
    rngs = [np.random.default_rng(seed+i*104729) for i in range(len(envs))]
    remaining = [count//len(envs)+int(i < count % len(envs)) for i in range(len(envs))]
    worlds = [World(env.unwrapped.level, auto_reset=False) for env in envs] if native else []
    rows, error, usage = [], 0., np.zeros(9, dtype=np.int64)
    logs = [DecisionLog() for _ in envs]
    try:
        while any(remaining):
            active = [i for i, n in enumerate(remaining) if n]
            probabilities = action_probs(model, np.stack([observations[i] for i in active]))
            for i, p in zip(active, probabilities):
                action = int(p.argmax()) if greedy else int(rngs[i].choice(9, p=p/p.sum()))
                usage[action] += 1
                if is_turn_decision(envs[i].unwrapped):
                    logs[i].add(action, p)
                def tick(world, number, direction):
                    nonlocal error
                    other = worlds[i]
                    other.step(direction, min(other.physics.dt, 180.-other.time))
                    for field in ('x', 'y', 'vx', 'vy'):
                        error = max(error, abs(getattr(world.ball, field)-getattr(other.ball, field)))
                    assert error < 1e-8
                    assert (world.cleared, world.failed, world.events) == (other.cleared, other.failed, other.events)
                raw = envs[i].unwrapped
                _, _, done, _, info = raw.step(action, on_tick=tick if native else None)
                observations[i] = envs[i].observation(raw.observe())
                if done:
                    audit_reward(info)
                    info['turn_policy'] = logs[i].summary()
                    logs[i] = DecisionLog()
                    rows.append(info)
                    remaining[i] -= 1
                    if remaining[i]:
                        observations[i] = envs[i].reset()[0]
                        if native:
                            worlds[i].reset()
    finally:
        for env in envs:
            env.close()
        model.policy.set_training_mode(training)
    return aggregate(rows, usage, seed, greedy, native, error)


def selection_score(evaluation):
    """검증 완주율 우선, 동률이면 33항목 진행도·위층 진입률·완주 시간을 사용한다."""
    return (evaluation['success_rate'], evaluation['mean_reward_items'],
            evaluation['upper_entry_rate'], -(evaluation['mean_success_seconds'] or 180.))


class StepStats(CurriculumStatsMixin, BaseCallback):
    """환경별 행동 한 번을 1스텝으로 합산한다. 종료한 판 수는 별도 지표다."""
    def __init__(self, folder, budget, emit=None, bank=None):
        super().__init__()
        self.folder, self.budget, self.emit = Path(folder), budget, emit
        self.steps = self.episodes = self.ticks = self.successes = 0
        self.parts, self.payouts = Counter(), Counter()
        self.last_live, self.last_console = 0., time.monotonic()
        self.curriculum = bank is not None
        if self.curriculum:
            self.setup_curriculum(bank)

    def status(self):
        result = dict(budget_unit='steps', requested_steps=self.budget, training_steps=self.steps,
            training_episodes=self.episodes, physics_ticks=self.ticks, training_successes=self.successes,
            training_reward_parts=dict(self.parts), training_reward_payouts=dict(self.payouts))
        if self.curriculum:
            result.update(self.curriculum_status())
        return result

    def _on_step(self):
        records = []
        if self.curriculum:
            self.record_curriculum_step()
        # 네 환경을 한 번 실행하면 전이 네 개가 생기므로 합계 4스텝이다.
        self.steps += len(self.locals['infos'])
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            self.ticks += info['tick_delta']
            if done:
                audit_reward(info)
                self.episodes += 1
                self.successes += int(info['success'])
                self.parts.update(info['reward_parts'])
                self.payouts.update(info.get('new_reward_items', info['reward_items']))
                # 끝난 판의 지급·미지급·중복 기록도 저장해 후속 원인 분석에 사용한다.
                # 벡터 환경이 덧붙인 최종 관측 배열은 정책 버퍼에만 두고 JSON에는 넣지 않는다.
                record = {k: v for k, v in info.items() if k != 'terminal_observation'}
                records.append(dict(record, episode=self.episodes, physics_ticks=self.ticks,
                                    training_steps=self.steps))
        if records:
            with (self.folder/'episodes.jsonl').open('a', encoding='utf-8') as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        now = time.monotonic()
        if now-self.last_live >= 20:
            write(self.folder/'live.json', self.status())
            self.last_live = now
        if self.emit is not None and now-self.last_console >= 60:
            self.emit(f'학습 중 {self.steps:,}/{self.budget:,}스텝 · 종료 {self.episodes:,}판'
                      f' · 물리 프레임 {self.ticks:,}')
            self.last_console = now
        # 예산 도달 중에도 수집 묶음을 끝까지 채워 마지막 경험까지 PPO 갱신에 사용한다.
        # 종료 여부는 run_job()이 완성된 512스텝 묶음 뒤에 판단한다.
        return True


def run_job(root, seed, options):
    folder = Path(root)/f'seed_{seed}'
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    curriculum = options.get('curriculum', False)
    bank = SnapshotBank(seed, options.get('snapshot_capacity', 32)) if curriculum else None
    vec = DummyVecEnv([lambda role=role: SpecCurriculumEnv(bank=bank, role=role) for role in ROLES]
                      if curriculum else [SpecEnv for _ in range(4)])
    model = None
    started = time.perf_counter()
    try:
        with (folder/'console.log').open('w', encoding='utf-8', buffering=1) as log:
            def emit(message):
                text = f'[시드 {seed}] {message}'
                print(text, flush=True)
                log.write(text+'\n')
            model = make_model(vec, seed)
            model.set_logger(configure(str(folder), ['csv']))
            config = dict(seed=seed, variant='map1-pdf-33', **options, **PPO_SETTINGS,
                sources=sources(), budget_unit='steps', from_scratch=True,
                previous_models_loaded=[], observation_size=OBS_SIZE, original_spawn=not curriculum,
                validation_original_start=True, test_original_start=True,
                start_roles=list(ROLES) if curriculum else ['original']*4,
                demonstrations=False, entropy_progress_unit='training_steps',
                entropy_schedule={'until_20_percent': .02, 'at_80_percent': .005, 'at_100_percent': .001},
                reward_items=33, maximum_raw_return=133., policy_layers=[128, 128])
            write(folder/'config.json', config)
            model.save(folder/'initial')
            probabilities = action_probs(model, vec.reset())[0]
            assert model.num_timesteps == 0 and np.allclose(probabilities, np.full(9, 1/9))
            write(folder/'initialization.json', dict(steps=0, probabilities=probabilities.tolist(),
                parameter_sha256=parameter_hash(model), previous_models_loaded=[]))
            collector = StepStats(folder, options['steps'], emit, bank=bank)
            history, best_score, best_step = [], (-1., -1., -1., -1e9), 0
            next_eval, last_updated_decisions = options['eval_every'], 0

            def checkpoint():
                nonlocal best_score, best_step
                evaluation = evaluate(model, options['validation_count'], 11000)
                local = evaluate_local(model, bank, options['local_validation_count']) if curriculum else None
                candidate = selection_score(evaluation)
                if candidate >= best_score:
                    best_score, best_step = candidate, model.num_timesteps
                    model.save(folder/'best')
                    write(folder/'best_validation.json', dict(step=best_step,
                        training_steps=collector.steps, training_episodes=collector.episodes, evaluation=evaluation))
                row = dict(decisions=model.num_timesteps, **collector.status(), validation=evaluation,
                    local_validation=local,
                    entropy=model.ent_coef, learning_rate=.0003, last_updated_decisions=last_updated_decisions,
                    elapsed_seconds=time.perf_counter()-started,
                    diagnostics={k: float(v) for k, v in model.logger.name_to_value.items()
                        if k.startswith('train/') and np.isscalar(v) and np.isfinite(v)})
                history.append(row)
                write(folder/'metrics.json', history)
                model.save(folder/'latest')
                if bank is not None:
                    bank.save(folder/'states.pkl.gz')
                    write(folder/'states_manifest.json', bank.manifest())
                write(folder/'live.json', dict(collector.status(), validation_success_rate=evaluation['success_rate'],
                                               mean_reward_items=evaluation['mean_reward_items']))
                model.logger.dump(step=model.num_timesteps)
                assert config['sources'] == sources()
                emit(f'{collector.steps:,}/{collector.budget:,}스텝 · 종료 {collector.episodes:,}판'
                     f' · 검증 완주 {evaluation["success_rate"]:.0%}'
                     f' · 보상 항목 {evaluation["mean_reward_items"]:.2f}/33 · 위층 {evaluation["upper_entry_rate"]:.0%}')

            checkpoint()
            while model.num_timesteps < collector.budget:
                rate, entropy = coefficients('A', model.num_timesteps, collector.budget)
                model.learning_rate, model.lr_schedule, model.ent_coef = rate, FixedRate(rate), entropy
                before = model._n_updates
                model.learn(total_timesteps=512, reset_num_timesteps=False, callback=collector, log_interval=None)
                assert collector.steps == model.num_timesteps
                if model._n_updates > before:
                    last_updated_decisions = model.num_timesteps
                if collector.steps >= next_eval or collector.steps >= collector.budget:
                    checkpoint()
                    next_eval = (collector.steps//options['eval_every']+1)*options['eval_every']
            emit('목표 스텝 수 도달. 별도 난수로 선택본·마지막본을 시험합니다.')
            best = load_policy(folder/'best.zip')
            final = evaluate(best, options['test_count'], 910000+seed)
            same = parameter_hash(model) == parameter_hash(best)
            latest = final if same else evaluate(model, options['test_count'], 910000+seed)
            selected_greedy = evaluate(best, 1, 123, greedy=True, native=True)
            latest_greedy = selected_greedy if same else evaluate(model, 1, 123, greedy=True, native=True)
            selected_local = evaluate_local(best, bank, options['local_validation_count'], seed=921000+seed) if curriculum else None
            latest_local = selected_local
            if curriculum and not same:
                latest_local = evaluate_local(model, bank, options['local_validation_count'], seed=921000+seed)
            assert config['sources'] == sources()
            result = dict(config=config, **collector.status(), final=final, latest_test=latest,
                selected_greedy=selected_greedy, latest_greedy=latest_greedy,
                selected_local_diagnostics=selected_local, latest_local_diagnostics=latest_local,
                latest_matches_selected_weights=same, best_step=best_step,
                best_validation=read(folder/'best_validation.json'), decisions=model.num_timesteps,
                last_updated_decisions=last_updated_decisions,
                tail_decisions_not_updated=model.num_timesteps-last_updated_decisions,
                steps_over_budget=model.num_timesteps-collector.budget,
                active_episodes_at_stop=[env.unwrapped.info() for env in vec.envs if env.unwrapped.ticks > 0],
                elapsed_seconds=time.perf_counter()-started, sources_after=sources(), test_used_for_selection=False)
            assert 0 <= result['steps_over_budget'] < PPO_SETTINGS['rollout_decisions']
            write(folder/'summary.json', result)
            emit(f'완료: 실제 {collector.steps:,}스텝 · 종료 {collector.episodes:,}판'
                 f' · 선택본 시험 완주율 {final["success_rate"]:.0%}')
            return str(folder/'summary.json')
    finally:
        if model is not None and hasattr(model, '_logger'):
            model.logger.close()
        vec.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='새 결과 폴더')
    parser.add_argument('--seed', type=int, default=2026, help='학습 시드 하나. 기본값 2026')
    parser.add_argument('--steps', '--total-timesteps', dest='steps', type=int, default=500000,
                        help='환경4개 합산 행동 스텝 수. 기본 500000, 마지막512스텝 묶음까지 수집·갱신')
    parser.add_argument('--eval-every', type=int, default=25000, help='환경4개 합산 학습 스텝 기준 검증 간격')
    parser.add_argument('--validation-count', type=int, default=20)
    parser.add_argument('--test-count', type=int, default=100)
    parser.add_argument('--curriculum', action='store_true', help='실제 도달 상태를 저장해 계단·2층 연습에 사용')
    parser.add_argument('--snapshot-capacity', type=int, default=32, help='구간별 최대 저장 상태 수')
    parser.add_argument('--local-validation-count', type=int, default=10, help='저장 상태가 있는 구간별 진단 평가 판 수')
    parser.add_argument('--dry-run', action='store_true', help='학습·결과 폴더 생성 없이 설정·물리 연결 검사')
    args = parser.parse_args(argv)
    if min(args.steps, args.eval_every, args.validation_count, args.test_count,
           args.snapshot_capacity, args.local_validation_count) <= 0 or args.seed < 0:
        parser.error('스텝 수·검증 간격·평가 판 수는 양수, 시드는 0 이상이어야 합니다')
    return args


def main(argv=None):
    args = parse_args(argv)
    if Path.cwd().resolve() != Path(__file__).resolve().parents[1]:
        raise RuntimeError('프로젝트 최상위 폴더에서 실행해 주세요')
    options = {k: getattr(args, k) for k in ('steps', 'eval_every', 'validation_count', 'test_count',
                                           'curriculum', 'snapshot_capacity', 'local_validation_count')}
    root = args.root.resolve()
    check = preflight()
    if args.dry_run:
        print(json.dumps(dict(root=str(root), seed=args.seed, settings=options, ppo=PPO_SETTINGS,
            preflight=check, reward_rules=[asdict(r) for r in RULES], training_started=False,
            budget_unit='steps', expected_training_steps=((args.steps+511)//512)*512,
            original_spawn=not args.curriculum, curriculum=args.curriculum,
            start_roles=list(ROLES) if args.curriculum else ['original']*4, observation_size=OBS_SIZE,
            entropy_schedule='학습 스텝 진행률 20%까지 .02, 80%까지 .005, 마지막 .001로 선형 감소'),
            ensure_ascii=False, indent=2))
        return
    if root.exists():
        raise RuntimeError('이전 결과를 보존합니다. 새로운 --root 경로를 사용해 주세요')
    env = SpecEnv()
    try:
        plan = dict(variant='map1-pdf-33', seed=args.seed, seeds=[args.seed], **options, ppo=PPO_SETTINGS,
            sources=sources(), budget_unit='steps', level=asdict(env.unwrapped.level),
            rules=[asdict(r) for r in RULES], preflight=check, original_spawn=not args.curriculum,
            validation_original_start=True, test_original_start=True,
            start_roles=list(ROLES) if args.curriculum else ['original']*4,
            previous_weights_loaded=False, demonstrations=False, git_actions=False,
            expected_training_steps=((args.steps+511)//512)*512,
            step_definition='환경 하나의 행동 선택·전이 한 번. 네 환경 동시 실행은 합계4스텝. 물리 프레임과 다름.',
            stop_rule='요청 스텝 이상이 된 첫512스텝 수집 묶음을 모두 갱신한 뒤 종료. 최대511스텝 초과.',
            tail_rule='콜백으로 수집 도중 중단하지 않으므로 마지막 묶음도 PPO 갱신에 사용.')
    finally:
        env.close()
    root.mkdir(parents=True)
    write(root/'plan.json', plan)
    for name in plan['sources']:
        target = root/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    write(root/'versions.json', {n: importlib.metadata.version(n)
        for n in ('torch', 'stable-baselines3', 'numpy', 'gymnasium')})
    started = time.perf_counter()
    try:
        summary = run_job(root, args.seed, options)
        assert plan['sources'] == sources()
    except Exception as error:
        write(root/'failure.json', dict(error=repr(error), complete=False))
        raise
    write(root/'completion.json', dict(complete=True, models=1, summaries=[summary], sources=sources(),
        elapsed_seconds=time.perf_counter()-started))
    print('학습·최종 시험 완료. rl.map1_spec_report 명령으로 그래프를 만드세요.', flush=True)


if __name__ == '__main__':
    main()
