"""팀 공통 환경에서 공통 보상과 개인 보상 추가 실험을 동일 PPO 설정으로 비교한다."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.five_models import action_probs
from rl.milestone_models import MilestoneExtractor, MilestonePPO
from tutorial_2.tutorial_test.environment_v2 import CombinedEnv

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def source_hashes():
    names = (
        'rl/milestone_env.py', 'rl/five_env.py', 'rl/map_input_env.py',
        'rl/all_tutorial_env.py', 'rl/milestone_models.py', 'rl/five_models.py',
        'pingu_ball/engine.py', 'pingu_ball/special_lab.py',
        'tutorial_2/tutorial_test/environment_v2.py', 'tutorial_2/tutorial_test/personal_rewards_v2.py',
        'tutorial_2/tutorial_test/train_ppo_v2.py',
    )
    return {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in names}


def output_dir(stage, reward, seed, root=ROOT / 'results_v2'):
    return Path(root) / f'tutorial{stage}' / reward / f'seed_{seed}'


def make_vector(stage, reward, repeat=6, count=4):
    return DummyVecEnv([
        lambda s=stage, r=reward, t=repeat: CombinedEnv(stage=s, reward=r, repeat=t)
        for _ in range(count)
    ])


def entropy(progress):
    progress = max(0.0, min(1.0, float(progress)))
    if progress <= 0.2:
        return 0.02
    if progress <= 0.8:
        return 0.02 + (0.005 - 0.02) * (progress - 0.2) / 0.6
    return 0.005 + (0.001 - 0.005) * (progress - 0.8) / 0.2


def create_ppo(env, seed):
    model = MilestonePPO(
        'MlpPolicy', env, seed=seed, device='cpu', verbose=0,
        gamma=1.0, learning_rate=0.0003, gae_lambda=0.98,
        ent_coef=0.02, vf_coef=0.5, max_grad_norm=0.5,
        n_steps=128, batch_size=256, n_epochs=5,
        clip_range=0.2, target_kl=0.02,
        policy_kwargs=dict(
            features_extractor_class=MilestoneExtractor,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
    )
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    return model


def evaluate(model, stage, reward, trials=100, seed=60000, mode='stochastic', repeat=6):
    if trials <= 0 or mode not in ('stochastic', 'greedy'):
        raise ValueError('평가 횟수는 양수, 평가 방식은 stochastic 또는 greedy여야 합니다.')
    model.policy.set_training_mode(False)
    env = make_vector(stage, reward, repeat=repeat, count=4)
    remaining = [trials // 4 + int(i < trials % 4) for i in range(4)]
    rngs = [np.random.default_rng(seed + stage * 100003 + i * 104729) for i in range(4)]
    outcomes = []
    try:
        observations = env.reset()
        while sum(remaining):
            probabilities = action_probs(model, observations)
            actions = []
            for i, (probability, rng) in enumerate(zip(probabilities, rngs)):
                if remaining[i] == 0:
                    actions.append(1)
                elif mode == 'greedy':
                    actions.append(int(probability.argmax()))
                else:
                    probability = probability / probability.sum()
                    actions.append(int(rng.choice(3, p=probability)))
            observations, _, dones, infos = env.step(np.asarray(actions, dtype=np.int64))
            for i, done in enumerate(dones):
                if done and remaining[i] > 0:
                    outcomes.append({key: value for key, value in infos[i].items()
                                     if key not in ('terminal_observation', 'TimeLimit.truncated')})
                    remaining[i] -= 1
    finally:
        env.close()
    successes = [row for row in outcomes if row['success']]
    mean = lambda values: float(np.mean(values)) if values else None
    return {
        'stage': stage, 'reward': reward, 'mode': mode, 'seed': seed,
        'trials': trials, 'wins': len(successes),
        'success_rate': round(len(successes) / trials, 4),
        'success_pct': round(100 * len(successes) / trials, 2),
        'mean_success_seconds': mean([row['seconds'] for row in successes]),
        'mean_episode_seconds': mean([row['seconds'] for row in outcomes]),
        'mean_raw_reward': mean([row['episode_return'] for row in outcomes]),
        'mean_personal_reward': mean([row.get('personal_total', 0.) for row in outcomes]),
        'mean_milestones': mean([len(row['milestones']) for row in outcomes]),
        'mean_personal_counts': {name: round(sum(row.get('personal_counts', {}).get(name, 0) for row in outcomes) / trials, 3)
                                 for name in ('target_height', 'target_horizontal', 'lift_first', 'lift_high')},
        'failures': dict(Counter(str(row['reason']) for row in outcomes if not row['success'])),
    }


class Monitor(BaseCallback):
    def __init__(self, folder, stage, reward, steps, eval_every, eval_trials, seed):
        super().__init__()
        self.folder = Path(folder)
        self.stage, self.reward, self.steps = stage, reward, steps
        self.eval_every, self.eval_trials = eval_every, eval_trials
        self.seed, self.next_eval = seed, eval_every
        self.episodes = self.training_wins = self.physics_ticks = 0
        self.best_score = None
        self.best_step = None

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            self.physics_ticks += int(info.get('tick_delta', 0))
            if done:
                self.episodes += 1
                self.training_wins += int(info['success'])
        if self.model.num_timesteps >= self.next_eval:
            self.checkpoint()
            self.next_eval = (self.model.num_timesteps // self.eval_every + 1) * self.eval_every
        return True

    def checkpoint(self):
        stats = evaluate(self.model, self.stage, self.reward, trials=self.eval_trials,
                         seed=11000, mode='stochastic')
        seconds = stats['mean_success_seconds']
        score = (stats['success_rate'], stats['mean_milestones'],
                 -(seconds if seconds is not None else 1e9))
        if self.best_score is None or score > self.best_score:
            self.best_score, self.best_step = score, self.model.num_timesteps
            self.model.save(str(self.folder / 'best'))
            write_json(self.folder / 'best_validation.json',
                       {'decisions': self.best_step, 'evaluation': stats})
        row = {
            'decisions': self.model.num_timesteps,
            'physics_ticks': self.physics_ticks,
            'episodes': self.episodes,
            'training_wins': self.training_wins,
            'entropy_coefficient': float(self.model.ent_coef),
            **{f'eval_{key}': value for key, value in stats.items()
               if isinstance(value, (int, float)) and value is not None},
        }
        with (self.folder / 'evaluations.csv').open('a', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        self.model.save(str(self.folder / 'last'))
        print(f'[평가] {self.stage}단계 {self.reward} 스텝={self.model.num_timesteps} '
              f'성공={stats["wins"]}/{stats["trials"]} ({stats["success_pct"]}%) '
              f'개인보상 평균={stats["mean_personal_reward"]:.2f}', flush=True)
        return stats


def train(args):
    if args.steps < 1 or args.eval_every < 1 or args.eval_trials < 1:
        raise ValueError('학습량과 평가 간격 및 평가 횟수는 양수여야 합니다.')
    folder = output_dir(args.stage, args.reward, args.seed, args.root)
    if folder.exists():
        raise FileExistsError(f'이미 실험 폴더가 있습니다. 기존 결과 보호를 위해 중단: {folder}')
    folder.mkdir(parents=True)
    torch.set_num_threads(1)
    config = {
        'stage': args.stage, 'reward': args.reward, 'seed': args.seed,
        'steps': args.steps, 'eval_every': args.eval_every, 'eval_trials': args.eval_trials,
        'algorithm': 'PPO', 'from_scratch': True, 'n_envs': 4,
        'n_steps': 128, 'batch_size': 256, 'epochs': 5,
        'lr': 0.0003, 'gamma': 1.0, 'gae_lambda': 0.98,
        'clip_range': 0.2, 'target_kl': 0.02,
        'entropy_schedule': '0.02(0~20%) → 0.005(80%) → 0.001(100%)',
        'action': 'left/idle/right, repeat=6 frames (0.05s)',
        'evaluation_policy': 'stochastic',
        'reward_scale': 0.01,
        'reward_version': 'v2_horizontal_requires_simultaneous_height_alignment',
        'source_hashes': source_hashes(),
    }
    write_json(folder / 'config.json', config)
    vector = make_vector(args.stage, args.reward, count=4)
    try:
        model = create_ppo(vector, args.seed)
        model.set_logger(configure(str(folder), ['csv']))
        model.save(str(folder / 'initial'))
        monitor = Monitor(folder, args.stage, args.reward, args.steps,
                          args.eval_every, args.eval_trials, args.seed)
        model.after_update = lambda m: setattr(m, 'ent_coef', entropy(m.num_timesteps / args.steps))
        print(f'[학습 시작] {args.stage}단계 보상={args.reward} '
              f'스텝={args.steps} 시드={args.seed} 결과={folder}', flush=True)
        model.learn(total_timesteps=args.steps, callback=monitor, log_interval=None)
        if monitor.best_step is None or monitor.best_step != model.num_timesteps:
            monitor.checkpoint()
        model.save(str(folder / 'last'))
        best = MilestonePPO.load(str(folder / 'best.zip'), device='cpu')
        final = evaluate(best, args.stage, args.reward, trials=args.test_trials,
                         seed=910000 + args.seed, mode='stochastic')
        write_json(folder / 'final_evaluation.json', final)
        write_json(folder / 'summary.json', {
            'config': config, 'decisions': model.num_timesteps,
            'episodes': monitor.episodes, 'training_wins': monitor.training_wins,
            'best_step': monitor.best_step, 'final_evaluation': final,
            'sources_unchanged': source_hashes() == config['source_hashes'],
        })
        print(f'[완료] {args.stage}단계 {args.reward} best={monitor.best_step} '
              f'독립평가={final["wins"]}/{final["trials"]} ({final["success_pct"]}%)', flush=True)
    finally:
        vector.close()


def load_saved(folder, reward, stage):
    config_path = Path(folder) / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'학습 설정 파일이 없습니다: {config_path}')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['stage'] != stage or config['reward'] != reward:
        raise ValueError('모델의 단계/보상 실험 조건이 명령어와 다릅니다.')
    if config['source_hashes'] != source_hashes():
        raise RuntimeError('학습 당시 게임·공통 환경·개인 보상 코드가 변경됐습니다. 결과를 섞지 않도록 중단합니다.')
    path = Path(folder) / 'best.zip'
    if not path.exists():
        raise FileNotFoundError(path)
    return MilestonePPO.load(str(path), device='cpu')


def run_evaluate(args, reward):
    folder = output_dir(args.stage, reward, args.seed, args.root)
    model = load_saved(folder, reward, args.stage)
    result = evaluate(model, args.stage, reward, trials=args.trials,
                      seed=args.eval_seed, mode=args.mode)
    write_json(folder / f'evaluation_{args.mode}_{args.trials}_{args.eval_seed}.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def smoke(args):
    env = CombinedEnv(stage=args.stage, reward=args.reward)
    try:
        observation, info = env.reset(seed=args.seed)
        print(f'[환경 확인] {args.stage}단계 {args.reward} 관측={observation.shape} '
              f'행동={env.action_space.n}')
        rng = np.random.default_rng(args.seed)
        for i in range(args.actions):
            observation, reward, done, _, info = env.step(int(rng.integers(3)))
            if done:
                print(f'[종료] 행동={i+1} 성공={info["success"]} '
                      f'점수={info["episode_return"]:.2f} 개인점수={info["personal_total"]:.2f}')
                observation, info = env.reset()
        print('[환경 확인 완료] PPO 학습은 실행하지 않았습니다.')
    finally:
        env.close()


def cli():
    parser = argparse.ArgumentParser(description='개인 보상 v2 정렬 조건 PPO 비교')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('smoke', 'train', 'evaluate', 'compare'):
        p = sub.add_parser(command)
        p.add_argument('--stage', type=int, choices=(2, 4), required=True)
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--root', type=Path, default=ROOT / 'results_v2')
        if command != 'compare':
            p.add_argument('--reward', choices=('common', 'combined'), default='combined')
        if command == 'smoke':
            p.add_argument('--actions', type=int, default=100)
        elif command == 'train':
            p.add_argument('--steps', type=int, default=1000000)
            p.add_argument('--eval-every', type=int, default=20000)
            p.add_argument('--eval-trials', type=int, default=20)
            p.add_argument('--test-trials', type=int, default=100)
        else:
            p.add_argument('--trials', type=int, default=500)
            p.add_argument('--eval-seed', type=int, default=60000)
            p.add_argument('--mode', choices=('stochastic', 'greedy'), default='stochastic')
    args = parser.parse_args()
    if args.command == 'smoke':
        smoke(args)
    elif args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        run_evaluate(args, args.reward)
    else:
        baseline = run_evaluate(args, 'common')
        personal = run_evaluate(args, 'combined')
        print(f'\n[같은 환경·정책 설정·평가 시드] 튜토리얼 {args.stage}단계')
        print(f'공통 보상만: {baseline["success_pct"]}% '
              f'({baseline["wins"]}/{baseline["trials"]})')
        print(f'공통 + 개인 보상: {personal["success_pct"]}% '
              f'({personal["wins"]}/{personal["trials"]})')


if __name__ == '__main__':
    cli()
