"""기존 행동·관측·PPO 설정을 고정하고 통합 보상으로 1~5번을 새로 학습한다."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
from pathlib import Path
import shutil
import time
import torch
from stable_baselines3.common.logger import configure
from . import five_train as old
from . import milestone_train as baseline
from .combined_env import CombinedEnv, audit_reward
from .milestone_env import OBS_SIZE

write, read = old.write, old.read
PROTOCOL = dict(baseline.PROTOCOL, reward_variant='combined-v1',
    no_distance_or_contact_or_survival_reward=False,
    stage2=dict(target_height=.5, target_horizontal=.5, auxiliary_window_seconds=3.,
                vertical_tolerance=16., target_landing='기존 핵심 구간 +10 유지; 시간 제한 추가 없음'),
    stage4=dict(lift_first=12., lift_high=12., upper_first=18.,
                approach_per_pixel=.045, approach_cap=12.96, lift_high_y=334.),
    unchanged_stages=[1,3,5], unchanged_action_and_observation=True,
    auxiliary_flags_in_policy_observation=False,
    note='보상만 변경한다. 보조 보상의 내부 추적 상태는 진단 로그에만 기록한다.')


def sources():
    result = baseline.ORIGINAL_SOURCES()
    for name in ('milestone_env.py', 'milestone_models.py', 'milestone_train.py',
                 'combined_rewards.py', 'combined_env.py', 'combined_train.py'):
        path = 'rl/' + name
        result[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


def install():
    baseline.install()
    old.FiveEnv = CombinedEnv
    old.sources = sources


class CombinedMonitor(baseline.MilestoneMonitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completed_parts = Counter()
        self.audited_episodes = 0

    def _on_step(self):
        result = super()._on_step()
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                audit_reward(info)
                self.completed_parts.update(info['reward_parts'])
                self.audited_episodes += 1
        return result

    def checkpoint(self, model):
        super().checkpoint(model)
        for row in self.rows[-1]['validation']['maps']:
            for episode in row['raw']:
                audit_reward(episode)
        self.rows[-1]['training_reward_parts'] = dict(self.completed_parts)
        self.rows[-1]['audited_training_episodes'] = self.audited_episodes
        write(self.folder / 'metrics.json', self.rows)
        write(self.folder / 'reward_audit.json', dict(passed=True,
              episodes=self.audited_episodes, reward_parts=dict(self.completed_parts)))


def train_job(payload):
    install()
    root, stage, seed, budget, tests = payload
    torch.set_num_threads(1)
    folder = Path(root) / 'experts' / 'ppo' / f'seed_{seed}' / f'stage_{stage:02d}'
    if (folder / 'summary.json').exists():
        assert read(folder / 'summary.json')['config']['sources'] == sources()
        return str(folder / 'summary.json')
    if (folder / 'config.json').exists():
        raise RuntimeError(f'이전 미완료 학습은 덮어쓰지 않는다: {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'console.log').open('w', encoding='utf-8', buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        vec = old.vector([stage] * 4, 'combined')
        try:
            model = baseline.make_model('ppo', vec, seed)
            model.set_logger(configure(str(folder), ['csv']))
            config = dict(algorithm='ppo', stage=stage, seed=seed, decisions_budget=budget,
                reward_mode='combined', from_scratch=True, original_spawn=True,
                action_count=3, observation_size=OBS_SIZE, sources=sources(), protocol=PROTOCOL,
                test_count=tests, validation_count=20, validation_seed=11000,
                primary_evaluation='sampled_policy', previous_models_loaded=[],
                reward='기존 성공 +100 / 실패 -25 / 핵심 구간 +10 + 2·4번 보조 보상; 학습 /100')
            write(folder / 'config.json', config)
            model.save(folder / 'initial')
            monitor = CombinedMonitor(folder, [stage], budget, 'combined')
            model.after_update = monitor.after_update
            monitor.checkpoint(model)
            model.learn(total_timesteps=budget, callback=monitor, log_interval=None)
            result = old.finish(model, monitor, folder, config, tests)
            for row in result['final']['maps'] + result['robustness']['maps']:
                for episode in row['raw']:
                    audit_reward(episode)
            assert sources() == config['sources']
        finally:
            vec.close()
    return str(folder / 'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=400000)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--test-count', type=int, default=100)
    args = parser.parse_args()
    if min(args.steps, args.workers, args.test_count) <= 0:
        parser.error('학습량·작업 수·시험 수는 양수여야 한다')
    seeds = [int(s) for s in args.seeds.split(',')]
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = dict(stages=[1,2,3,4,5], algorithms=['ppo'], seeds=seeds,
                expert_steps=args.steps, test_count=args.test_count, protocol=PROTOCOL,
                sources=sources(), previous_weights_loaded=False, human_actions=False,
                git_actions=False, general_maps=False, consolidation=False,
                comparison='동일한 1~5 공통 학습 구조에서 2·4번 보조 보상만 추가',
                historical_baseline='experiments/tutorial_1_5/baseline/results')
    if (root / 'plan.json').exists():
        assert read(root / 'plan.json') == plan, '조건이 달라지면 새 결과 폴더가 필요하다'
    else:
        write(root / 'plan.json', plan)
        for path in sources():
            dest = root / 'source_snapshot' / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        baseline.export_milestones(root)
    tasks = [(str(root), stage, seed, args.steps, args.test_count)
             for seed in seeds for stage in range(1,6)]
    started, completed = time.perf_counter(), []
    write(root / 'progress.json', dict(phase='training', completed=[], total=len(tasks)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(train_job, payload) for payload in tasks]
        for future in as_completed(futures):
            completed.append(future.result())
            assert sources() == plan['sources']
            write(root / 'progress.json', dict(phase='training', completed=completed,
                  total=len(tasks), elapsed_seconds=time.perf_counter()-started))
            print(f'완료 {len(completed)}/{len(tasks)} {completed[-1]}', flush=True)
    write(root / 'completion.json', dict(complete=True, models=len(completed),
          summaries=completed, sources=sources(), elapsed_seconds=time.perf_counter()-started))
    print('모든 학습과 별도 시험 완료', flush=True)


if __name__ == '__main__':
    main()
