"""시드 하나를 에피소드 수 기준으로 학습하는 맵1 커리큘럼 실행 파일.

--dry-run은 설정만 출력한다. 실제 명령을 실행했을 때만 PPO 학습을 시작한다.
기존 프레임 실험과 소스를 보존하고 새 결과 폴더만 생성한다.
"""
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

from .five_models import action_probs
from .five_train import read, write
from .map_a_env import MapAEnv, SPECS, LABELS, OBS_SIZE
from .map_a_models import make_model, load_policy
from .map_a_train import evaluate, selection_score
from .map_a_curriculum_train import sources as previous_sources, parameter_hash
from .map_a_episode_env import EpisodeEnv, EpisodeBank, EpisodeSchedule, audit_training_reward
from .stabilized_train import coefficients
from .stabilized_models import FixedRate


def sources():
    """기존 코드와 이번 환경·실행 파일의 지문을 기록한다."""
    result = previous_sources()
    for name in ('rl/map_a_episode_env.py', 'rl/map_a_episode_train.py'):
        result[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    return result


class EpisodeStats(BaseCallback):
    """원래 시작과 중간 시작의 종료 판을 합산하고 목표 판 수에서 수집을 중단한다."""
    def __init__(self, folder, bank, schedule, emit=None):
        super().__init__()
        self.folder, self.bank, self.schedule = Path(folder), bank, schedule
        self.emit = emit
        self.ticks = self.episodes = 0
        self.parts, self.frames_by_start = Counter(), Counter()
        self.episodes_by_start, self.successes_by_start = Counter(), Counter()
        self.last_live, self.last_console = 0., time.monotonic()

    def status(self):
        return dict(budget_unit='episodes', requested_episodes=self.schedule.budget,
            training_episodes=self.episodes, physics_ticks=self.ticks,
            episodes_by_start=dict(self.episodes_by_start), successes_by_start=dict(self.successes_by_start),
            frames_by_start=dict(self.frames_by_start), training_reward_parts=dict(self.parts),
            middle_probability=self.schedule.probability, snapshot_bank=self.bank.manifest())

    def _on_step(self):
        records = []
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            # 프레임은 종료 예산이 아니라 실제 경험량을 설명하는 지표로만 센다.
            self.ticks += info['tick_delta']
            kind = info['start_kind']
            self.frames_by_start[kind] += info['tick_delta']
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
                    'initial_reward_parts', 'new_milestones', 'local_goal_ids', 'local_goal_reached',
                    'landed_platforms')
                records.append(dict(episode=self.episodes, physics_ticks=self.ticks,
                                    **{key: info[key] for key in keys}))
        assert self.ticks == self.schedule.ticks and self.episodes == self.schedule.completed
        if records:
            with (self.folder/'episodes.jsonl').open('a', encoding='utf-8') as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        if time.monotonic()-self.last_live >= 20:
            write(self.folder/'live.json', self.status())
            self.last_live = time.monotonic()
        if self.emit is not None and time.monotonic()-self.last_console >= 60:
            self.emit(f'학습 중 {self.episodes:,}/{self.schedule.budget:,}판, '
                f'실제 물리 프레임 {self.ticks:,}, 저장 상태 {self.bank.manifest()["count"]}개')
            self.last_console = time.monotonic()
        # SB3의 표준 중단 동작을 사용한다. 같은 벡터 스텝에서 최대 4판이 함께 끝날 수 있다.
        return self.episodes < self.schedule.budget


def run_job(root, seed, options):
    """한 시드를 처음부터 학습한다. 검증·시험은 항상 원래 출발점에서 실시한다."""
    folder = Path(root)/f'seed_{seed}'
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    bank = EpisodeBank(seed, options['bank_capacity'])
    schedule = EpisodeSchedule(options['episodes'], options['middle_probability'])
    vec = DummyVecEnv([lambda: EpisodeEnv(bank=bank, schedule=schedule) for _ in range(4)])
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
            config = dict(seed=seed, variant='A-map1-episode-v2', budget_unit='episodes', **options,
                sources=sources(), from_scratch=True, previous_models_loaded=[], observation_size=OBS_SIZE,
                learning_rate=.0003, gamma=1., gae_lambda=.98, n_envs=4, n_steps=128,
                batch_size=256, n_epochs=5, clip_range=.2, target_kl=.02, vf_coef=.5,
                max_grad_norm=.5, reward_scale=.01, limit_seconds=180., demonstrations=False,
                validation_original_start=True, test_original_start=True,
                entropy_progress_unit='completed_training_episodes')
            write(folder/'config.json', config)
            model.save(folder/'initial')
            probabilities = action_probs(model, vec.reset())[0]
            assert model.num_timesteps == 0 and np.allclose(probabilities, np.full(9, 1/9))
            write(folder/'initialization.json', dict(steps=0, probabilities=probabilities.tolist(),
                parameter_sha256=parameter_hash(model), previous_models_loaded=[]))
            collector = EpisodeStats(folder, bank, schedule, emit)
            history, best_score, best_step = [], (-1., -1., -1., -1e9), 0
            next_eval, last_updated_decisions = options['eval_every'], 0

            def checkpoint():
                nonlocal best_score, best_step
                before = (bank.manifest(), schedule.completed, schedule.ticks,
                          json.dumps(bank.rng.bit_generator.state, sort_keys=True))
                evaluation = evaluate(model, options['validation_count'], 11000)
                after = (bank.manifest(), schedule.completed, schedule.ticks,
                         json.dumps(bank.rng.bit_generator.state, sort_keys=True))
                assert before == after  # 검증을 학습 판 수나 저장소 선택 이력에 포함하지 않는다.
                schedule.observe_validation(evaluation['success_rate'])
                candidate = selection_score(evaluation)
                if candidate >= best_score:
                    best_score, best_step = candidate, model.num_timesteps
                    model.save(folder/'best')
                    write(folder/'best_validation.json', dict(step=best_step,
                        training_episodes=collector.episodes, physics_ticks=collector.ticks, evaluation=evaluation))
                row = dict(decisions=model.num_timesteps, **collector.status(), validation=evaluation,
                    entropy=model.ent_coef, learning_rate=.0003, last_updated_decisions=last_updated_decisions,
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
                emit(f'{collector.episodes:,}/{schedule.budget:,}판 · 원래 시작 검증 '
                     f'{evaluation["success_rate"]:.0%} · 구간 {evaluation["mean_milestones"]:.2f}/16 · '
                     f'다음 중간 시작 확률 {schedule.probability:.0%}')

            checkpoint()
            while collector.episodes < schedule.budget:
                # 기존 A의 계수 함수를 사용하되, 진행률의 분자·분모는 판 수로 전달한다.
                rate, entropy = coefficients('A', collector.episodes, schedule.budget)
                model.learning_rate, model.lr_schedule, model.ent_coef = rate, FixedRate(rate), entropy
                updates_before = model._n_updates
                model.learn(total_timesteps=512, reset_num_timesteps=False,
                            callback=collector, log_interval=None)
                if model._n_updates > updates_before:
                    last_updated_decisions = model.num_timesteps
                if collector.episodes >= next_eval or collector.episodes >= schedule.budget:
                    checkpoint()
                    # 검증은 완성된 PPO 수집 단위 직후에 하므로 정확히 500판보다 조금 늦을 수 있다.
                    next_eval = (collector.episodes//options['eval_every']+1)*options['eval_every']
            emit('목표 판 수 도달. 원래 출발점에서 최종 시험을 시작합니다.')
            best = load_policy(folder/'best.zip')
            final = evaluate(best, options['test_count'], 910000+seed)
            same = all(torch.equal(value, best.policy.state_dict()[key])
                       for key, value in model.policy.state_dict().items())
            latest = final if same else evaluate(model, options['test_count'], 910000+seed)
            selected_greedy = evaluate(best, 1, 123, greedy=True, native=True)
            latest_greedy = selected_greedy if same else evaluate(model, 1, 123, greedy=True, native=True)
            assert config['sources'] == sources()
            result = dict(config=config, **collector.status(), final=final, latest_test=latest,
                selected_greedy=selected_greedy, latest_greedy=latest_greedy,
                latest_matches_selected_weights=same, best_step=best_step,
                best_validation=read(folder/'best_validation.json'), decisions=model.num_timesteps,
                last_updated_decisions=last_updated_decisions,
                tail_decisions_not_updated=model.num_timesteps-last_updated_decisions,
                episodes_over_budget=collector.episodes-schedule.budget,
                elapsed_seconds=time.perf_counter()-started, sources_after=sources(), test_used_for_selection=False)
            assert 0 <= result['episodes_over_budget'] <= 3
            write(folder/'summary.json', result)
            emit(f'완료: 실제 종료 {collector.episodes:,}판, 선택본 성공률 {final["success_rate"]:.0%}')
            return str(folder/'summary.json')
    finally:
        if model is not None and hasattr(model, '_logger'):
            model.logger.close()
        vec.close()


def parse_args(argv=None):
    """기본 시드는 하나이며, 예산과 검증 간격의 단위는 에피소드다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='새 결과 폴더')
    parser.add_argument('--seed', type=int, default=1, help='학습할 시드 하나. 기본값 1')
    parser.add_argument('--episodes', type=int, default=10_000, help='원래·중간 시작을 합친 목표 학습 판 수')
    parser.add_argument('--eval-every', type=int, default=500, help='검증할 학습 에피소드 간격')
    parser.add_argument('--validation-count', type=int, default=20)
    parser.add_argument('--test-count', type=int, default=100)
    parser.add_argument('--middle-probability', type=float, default=.5)
    parser.add_argument('--bank-capacity', type=int, default=24)
    parser.add_argument('--dry-run', action='store_true', help='설정만 출력하며 학습·폴더 생성을 하지 않음')
    args = parser.parse_args(argv)
    if min(args.episodes, args.eval_every, args.validation_count, args.test_count, args.bank_capacity) <= 0:
        parser.error('에피소드·검증·시험·저장 개수는 양수여야 한다')
    if args.seed < 0:
        parser.error('시드는 0 이상의 정수여야 한다')
    try:
        EpisodeSchedule(args.episodes, args.middle_probability)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    if Path.cwd().resolve() != Path(__file__).resolve().parents[1]:
        raise RuntimeError('프로젝트 최상위 폴더에서 실행해 주세요')
    options = {key: getattr(args, key) for key in ('episodes', 'eval_every', 'validation_count',
                                                 'test_count', 'middle_probability', 'bank_capacity')}
    root = args.root.resolve()
    if args.dry_run:
        print(json.dumps(dict(root=str(root), seed=args.seed, settings=options,
                              budget_unit='episodes', training_started=False), ensure_ascii=False, indent=2))
        return
    if root.exists():
        raise RuntimeError('기존 결과를 보존합니다. 새 --root 경로를 지정해 주세요')
    env = MapAEnv()
    try:
        plan = dict(variant='A-map1-episode-v2', budget_unit='episodes', episode_budget=args.episodes,
            seed=args.seed, seeds=[args.seed], **options, sources=sources(),
            map_kind='reinforcement', map_number=1, map_title=env.unwrapped.level.title,
            level=asdict(env.unwrapped.level), milestones=[asdict(s) for s in SPECS], milestone_labels=list(LABELS),
            observation_size=OBS_SIZE, training_start='original_and_self_collected',
            validation_start='original', test_start='original', previous_weights_loaded=False,
            curriculum=True, demonstrations=False, git_actions=False,
            stop_rule='목표 판 수 이상에서 표준 SB3 콜백 중단. 네 환경 동시 종료로 최대3판 초과.',
            tail_rule='종료 시 미완성 PPO 수집 묶음은 표준 SB3 동작에 따라 갱신에서 제외.')
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
    started = time.perf_counter()
    try:
        summary = run_job(root, args.seed, options)
    except Exception as error:
        write(root/'failure.json', dict(error=repr(error), complete=False))
        raise
    assert plan['sources'] == sources()
    write(root/'completion.json', dict(complete=True, models=1, summaries=[summary], sources=sources(),
                                       elapsed_seconds=time.perf_counter()-started))
    print('에피소드 기준 학습과 최종 평가를 완료했습니다. 보고서 명령으로 그래프를 생성하세요.', flush=True)


if __name__ == '__main__':
    main()
