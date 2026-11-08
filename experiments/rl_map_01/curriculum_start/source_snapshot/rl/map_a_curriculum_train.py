"""맵1의 A 설정을 유지하고 학습 시작 상태만 커리큘럼으로 바꾸는 실행 파일.

이 모듈을 명령으로 실행해야만 학습이 시작된다. 기존 실험을 덮어쓰지 않으며,
검증·최종 시험은 저장소가 연결되지 않은 원래 A 환경에서 수행한다.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from .five_models import action_probs
from .five_train import read, write
from .map_a_env import MapAEnv, SPECS, LABELS, OBS_SIZE, CAPACITY
from .map_a_models import make_model, load_policy
from .map_a_train import evaluate, selection_score, sources as original_sources
from .map_a_curriculum_env import CurriculumEnv, SnapshotBank, StartSchedule, audit_training_reward
from .stabilized_models import FixedRate
from .stabilized_train import coefficients


def sources():
    """이번 실행에 쓰이는 기존 코드와 추가 코드의 지문을 기록한다."""
    result = original_sources()
    for name in ('rl/map_a_curriculum_env.py', 'rl/map_a_curriculum_train.py'):
        result[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    return result


def parameter_hash(model):
    """같은 시드의 기존 A와 초기 가중치가 같은지 비교할 수 있는 지문이다."""
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


class CurriculumStats(BaseCallback):
    """새로 실행한 프레임만 세고 원래 시작 성공과 구간 연습 성공을 분리한다."""
    def __init__(self, folder, bank, schedule, on_progress=None):
        super().__init__()
        self.folder, self.bank, self.schedule = Path(folder), bank, schedule
        self.ticks = self.episodes = 0
        self.parts, self.frames_by_start = Counter(), Counter()
        self.episodes_by_start, self.successes_by_start = Counter(), Counter()
        self.last_live = 0.
        self.last_console = time.monotonic()
        self.on_progress = on_progress

    def status(self):
        return dict(physics_ticks=self.ticks, training_episodes=self.episodes,
            episodes_by_start=dict(self.episodes_by_start), successes_by_start=dict(self.successes_by_start),
            frames_by_start=dict(self.frames_by_start), middle_probability=self.schedule.probability,
            training_reward_parts=dict(self.parts), snapshot_bank=self.bank.manifest())

    def _on_step(self):
        records = []
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            # 복원 이전의 시각·누적 틱을 더하면 학습량이 부풀려지므로 tick_delta만 사용한다.
            delta, kind = info['tick_delta'], info['start_kind']
            self.ticks += delta
            self.frames_by_start[kind] += delta
            if done:
                audit_training_reward(info)
                self.episodes += 1
                self.episodes_by_start[kind] += 1
                self.successes_by_start[kind] += int(info['success'])
                self.parts.update(info['reward_parts'])
                keys = ('success', 'reason', 'seconds', 'decisions', 'milestones', 'armed',
                    'episode_return', 'learning_return', 'reward_parts', 'start_kind', 'start_group',
                    'start_snapshot_id', 'start_world_time', 'start_middle_probability', 'segment_seconds',
                    'segment_ticks', 'segment_decisions', 'initial_milestones', 'initial_armed',
                    'initial_reward_parts', 'new_milestones')
                records.append(dict(episode=self.episodes, physics_ticks=self.ticks,
                                    **{key: info[key] for key in keys}))
        # 환경의 공유 시계와 집계를 대조해 프레임 중복 계산이나 누락을 찾는다.
        assert self.ticks == self.schedule.ticks
        if records:
            with (self.folder/'episodes.jsonl').open('a', encoding='utf-8') as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        if time.monotonic()-self.last_live >= 20:
            write(self.folder/'live.json', self.status())
            self.last_live = time.monotonic()
        if self.on_progress is not None and time.monotonic()-self.last_console >= 60:
            self.on_progress(f'학습 중: {self.ticks:,}/{self.schedule.budget:,} 물리 프레임, '
                f'원래 시작 종료 {self.episodes_by_start["original"]}판, '
                f'구간 연습 종료 {self.episodes_by_start["middle"]}판, '
                f'저장 상태 {self.bank.manifest()["count"]}개')
            self.last_console = time.monotonic()
        return True


def train_job(payload):
    """한 시드를 초기 가중치부터 학습하고 독립 시험까지 완료한다."""
    root, seed, options = payload
    folder = Path(root)/f'seed_{seed}'
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    bank = SnapshotBank(seed=seed, capacity=options['bank_capacity'])
    schedule = StartSchedule(options['physics_ticks'], options['middle_probability'],
                             options['final_original_fraction'])
    # 네 환경은 하나의 저장소와 시계를 쓴다. 다른 시드는 별도 저장소를 만든다.
    vec = DummyVecEnv([lambda: CurriculumEnv(bank=bank, schedule=schedule) for _ in range(4)])
    started = time.perf_counter()
    model = None
    try:
        with (folder/'console.log').open('w', encoding='utf-8', buffering=1) as log:
            def emit(message):
                text = f'[시드 {seed}] {message}'
                print(text, flush=True)
                log.write(text+'\n')

            model = make_model(vec, seed)
            model.set_logger(configure(str(folder), ['csv']))
            config = dict(seed=seed, variant='A-map1-curriculum', map_kind='reinforcement', map_number=1,
                observation_size=OBS_SIZE, from_scratch=True, previous_models_loaded=[],
                sources=sources(), **options,
                learning_rate=.0003, gamma=1., gae_lambda=.98, n_envs=4, n_steps=128,
                batch_size=256, n_epochs=5, clip_range=.2, target_kl=.02, vf_coef=.5,
                max_grad_norm=.5, reward_scale=.01, limit_seconds=180., demonstrations=False,
                validation_original_start=True, test_original_start=True)
            write(folder/'config.json', config)
            model.save(folder/'initial')
            observation = vec.reset()
            probabilities = action_probs(model, observation)[0]
            assert np.allclose(probabilities, np.full(9, 1/9)) and model.num_timesteps == 0
            write(folder/'initialization.json', dict(steps=0, probabilities=probabilities.tolist(),
                parameter_sha256=parameter_hash(model), previous_models_loaded=[]))
            collector = CurriculumStats(folder, bank, schedule, on_progress=emit)
            history, best_score, best_step = [], (-1., -1., -1., -1e9), 0
            next_eval = options['eval_every']

            def checkpoint():
                nonlocal best_score, best_step
                # 평가에는 저장소를 전달하지 않고, 저장 상태와 선택 난수의 보존도 확인한다.
                bank_before = bank.manifest()
                rng_before = json.dumps(bank.rng.bit_generator.state, sort_keys=True)
                ticks_before = schedule.ticks
                evaluation = evaluate(model, options['validation_count'], 11000)
                assert bank_before == bank.manifest() and ticks_before == schedule.ticks
                assert rng_before == json.dumps(bank.rng.bit_generator.state, sort_keys=True)
                candidate = selection_score(evaluation)
                if candidate >= best_score:
                    best_score, best_step = candidate, model.num_timesteps
                    model.save(folder/'best')
                    write(folder/'best_validation.json', dict(step=best_step,
                        physics_ticks=collector.ticks, evaluation=evaluation))
                row = dict(decisions=model.num_timesteps, **collector.status(), validation=evaluation,
                    entropy=model.ent_coef, learning_rate=.0003,
                    elapsed_seconds=time.perf_counter()-started,
                    diagnostics={key: float(value) for key, value in model.logger.name_to_value.items()
                        if key.startswith('train/') and np.isscalar(value) and np.isfinite(value)})
                history.append(row)
                write(folder/'metrics.json', history)
                model.save(folder/'latest')
                bank.save(folder/'states.pkl.gz')
                write(folder/'states_manifest.json', bank.manifest())
                write(folder/'live.json', dict(collector.status(),
                    validation_success_rate=evaluation['success_rate'],
                    mean_milestones=evaluation['mean_milestones']))
                model.logger.dump(step=model.num_timesteps)
                assert config['sources'] == sources()
                emit(f'물리 프레임 {collector.ticks:,} / {schedule.budget:,}, '
                     f'원래 시작 검증 {evaluation["success_rate"]:.0%}, '
                     f'구간 {evaluation["mean_milestones"]:.2f}/16, 저장 상태 {bank.manifest()["count"]}개')

            emit('초기 검증을 시작합니다. 검증·최종 시험은 모두 원래 출발점입니다.')
            checkpoint()
            while collector.ticks < schedule.budget:
                rate, entropy = coefficients('A', collector.ticks, schedule.budget)
                model.learning_rate = rate
                model.lr_schedule = FixedRate(rate)
                model.ent_coef = entropy
                # PPO 갱신을 중간에 자르지 않아 512행동 수집 단위만큼 예산을 넘을 수 있다.
                model.learn(total_timesteps=512, reset_num_timesteps=False,
                            callback=collector, log_interval=None)
                if collector.ticks >= next_eval or collector.ticks >= schedule.budget:
                    checkpoint()
                    next_eval += options['eval_every']
            emit('학습 종료. 선택본과 마지막본의 독립 시험을 시작합니다.')
            best = load_policy(folder/'best.zip')
            final = evaluate(best, options['test_count'], 910000+seed)
            same = all(torch.equal(value, best.policy.state_dict()[key])
                       for key, value in model.policy.state_dict().items())
            latest = final if same else evaluate(model, options['test_count'], 910000+seed)
            selected_greedy = evaluate(best, 1, 123, greedy=True, native=True)
            latest_greedy = selected_greedy if same else evaluate(model, 1, 123, greedy=True, native=True)
            assert config['sources'] == sources()
            result = dict(config=config, final=final, latest_test=latest,
                selected_greedy=selected_greedy, latest_greedy=latest_greedy,
                latest_matches_selected_weights=same, best_step=best_step,
                best_validation=read(folder/'best_validation.json'), decisions=model.num_timesteps,
                **collector.status(), elapsed_seconds=time.perf_counter()-started,
                sources_after=sources(), test_used_for_selection=False)
            write(folder/'summary.json', result)
            emit(f'완료: 원래 시작 성공률 선택본 {final["success_rate"]:.0%}, 마지막본 {latest["success_rate"]:.0%}')
        return str(folder/'summary.json')
    finally:
        # Windows에서도 결과 파일을 다른 도구가 바로 읽을 수 있도록 로그 핸들을 닫는다.
        if model is not None and hasattr(model, '_logger'):
            model.logger.close()
        vec.close()


def parse_args(argv=None):
    """실행 설정을 검사한다. --dry-run은 모델 생성이나 학습을 하지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='새 결과 폴더. 기존 폴더는 덮어쓰지 않음')
    parser.add_argument('--physics-ticks', type=int, default=6_000_000, help='모델당 네 환경 합산 물리 프레임')
    parser.add_argument('--eval-every', type=int, default=500_000)
    parser.add_argument('--validation-count', type=int, default=20)
    parser.add_argument('--test-count', type=int, default=100)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--middle-probability', type=float, default=.5)
    parser.add_argument('--final-original-fraction', type=float, default=.2)
    parser.add_argument('--bank-capacity', type=int, default=24)
    parser.add_argument('--dry-run', action='store_true', help='설정만 출력. 학습과 폴더 생성 없음')
    args = parser.parse_args(argv)
    numeric = (args.physics_ticks, args.eval_every, args.validation_count,
               args.test_count, args.workers, args.bank_capacity)
    if min(numeric) <= 0:
        parser.error('프레임·평가 횟수·작업 수·저장 개수는 양수여야 한다')
    try:
        args.seed_values = [int(value) for value in args.seeds.split(',')]
        if not args.seed_values or min(args.seed_values) < 0 or len(set(args.seed_values)) != len(args.seed_values):
            raise ValueError('시드는 중복 없는 0 이상의 정수여야 한다')
        StartSchedule(args.physics_ticks, args.middle_probability, args.final_original_fraction)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    if Path.cwd().resolve() != Path(__file__).resolve().parents[1]:
        raise RuntimeError('프로젝트 최상위 폴더에서 실행해 주세요')
    options = {key: getattr(args, key) for key in ('physics_ticks', 'eval_every', 'validation_count',
        'test_count', 'middle_probability', 'final_original_fraction', 'bank_capacity')}
    root = args.root.resolve()
    if args.dry_run:
        print(json.dumps(dict(root=str(root), seeds=args.seed_values, workers=args.workers,
                              settings=options, training_started=False), ensure_ascii=False, indent=2))
        return
    if root.exists():
        raise RuntimeError('기존 결과를 덮어쓰지 않습니다. 새 --root 경로를 지정해 주세요')
    env = MapAEnv()
    try:
        plan = dict(variant='A-map1-curriculum', map_kind='reinforcement', map_number=1,
            map_title=env.unwrapped.level.title, level=asdict(env.unwrapped.level),
            seeds=args.seed_values, physics_budget=args.physics_ticks, **options,
            sources=sources(), observation_size=OBS_SIZE, object_capacity=CAPACITY,
            milestones=[asdict(spec) for spec in SPECS], milestone_labels=list(LABELS),
            curriculum=True, previous_weights_loaded=False, demonstrations=False,
            training_start='original_and_self_collected', validation_start='original', test_start='original',
            new_episode_reward_only=True, git_actions=False)
    finally:
        env.close()
    root.mkdir(parents=True)
    write(root/'plan.json', plan)
    for name in plan['sources']:
        target = root/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    write(root/'versions.json', {name: importlib.metadata.version(name)
        for name in ('torch', 'stable-baselines3', 'numpy', 'gymnasium')})
    completed, started = [], time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.seed_values))) as pool:
            tasks = [pool.submit(train_job, (str(root), seed, options)) for seed in args.seed_values]
            for future in as_completed(tasks):
                completed.append(future.result())
                write(root/'progress.json', dict(completed=completed, total=len(args.seed_values),
                                                elapsed_seconds=time.perf_counter()-started))
                print(f'완료 {len(completed)}/{len(args.seed_values)}: {completed[-1]}', flush=True)
    except Exception as error:
        write(root/'failure.json', dict(error=repr(error), completed=completed, complete=False))
        raise
    assert plan['sources'] == sources()
    write(root/'completion.json', dict(complete=True, models=len(completed), summaries=completed,
                                       sources=sources(), elapsed_seconds=time.perf_counter()-started))
    print('커리큘럼 학습·최종 평가 완료. 보고서 명령으로 그래프를 생성할 수 있습니다.', flush=True)


if __name__ == '__main__':
    main()
