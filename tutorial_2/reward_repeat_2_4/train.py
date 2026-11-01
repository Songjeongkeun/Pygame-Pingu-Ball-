"""2/4 tutorial PPO comparison: same team policy, observations and PERSONAL rewards.

Only environment variable under test: 3-action hold duration, 6 vs 12 physics
frames. Training from random initialization (no cloning, DAgger, pretrained
policies, demonstrations or warm-start). Physical time budget is measured by
actual World.step ticks, not by the number of agent decisions.

Run from the PinguBall REPO ROOT: python -m tutorial_2.reward_repeat_2_4.train --help
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl import five_train as team_train
from rl.milestone_train import install
from rl.milestone_env import MilestoneEnv, OBS_SIZE
from rl.milestone_models import MilestonePPO
from rl.five_models import action_probs
from .reward_env import RewardOnlyEnv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / 'results' / 'reward_repeat_2_4'
ENTROPY_COEF = .05  # same constant used in prior successful first-spring exploration


def folder_for(args):
    return (Path(args.out) / f'stage_{args.stage:02d}' / args.reward /
            f'repeat_{args.repeat}' / f'seed_{args.seed}')


def make_env(stage: int, reward: str, repeat: int):
    if stage not in (2, 4):
        raise ValueError('Stages 1, 3 and 5 must stay untouched')
    if repeat not in (6, 12):
        raise ValueError('Use 6 or 12 physics frames per decision')
    if reward == 'personal':
        env = RewardOnlyEnv(stage, repeat=repeat)
    elif reward == 'original':
        env = MilestoneEnv(stage=stage, limit=180., repeat=repeat)
    else:
        raise ValueError('Unknown reward option')
    if env.action_space.n != 3 or env.observation_space.shape != (OBS_SIZE,):
        raise AssertionError('Common team action or observation changed')
    actual_repeat = (env.unwrapped.repeat if reward == 'original'
                     else env.team.unwrapped.repeat)
    if actual_repeat != repeat:
        raise AssertionError('Physics action repeat was not applied')
    return env


def make_vec(stage: int, reward: str, repeat: int, n=4):
    return DummyVecEnv([lambda: make_env(stage, reward, repeat) for _ in range(n)])


def safe_mean(rows, field, default=0.):
    return float(np.mean([float(r.get(field, default)) for r in rows]))


def evaluate(model, stage, reward, repeat, trials=20, seed=50000):
    """Four independent sampled-action streams (team's validation protocol)."""
    if trials < 1:
        raise ValueError('trials must be >=1')
    rngs = [np.random.default_rng(seed + stage * 100003 + i * 104729) for i in range(4)]
    remaining = [trials // 4 + int(i < trials % 4) for i in range(4)]
    records = []
    action_counts = np.zeros(3, dtype=np.int64)
    probability_sum = np.zeros(3, dtype=np.float64)
    entropy_sum = 0.
    count = 0
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    vec = make_vec(stage, reward, repeat)
    try:
        obs = vec.reset()
        while any(remaining):
            probs = np.asarray(action_probs(model, obs), dtype=np.float64)
            if probs.shape != (4, 3):
                raise AssertionError(f'Expected common PPO output (4,3), received {probs.shape}')
            actions = np.zeros(4, dtype=np.int64)
            for i in range(4):
                p = probs[i]
                if not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
                    raise ValueError('Invalid policy probabilities')
                p = p / p.sum()
                actions[i] = int(rngs[i].choice(3, p=p))
                if remaining[i] > 0:
                    count += 1
                    action_counts[actions[i]] += 1
                    probability_sum += p
                    entropy_sum -= float(np.sum(p * np.log(np.clip(p, 1e-12, 1.))))
            obs, _, dones, infos = vec.step(actions)
            for i, done in enumerate(dones):
                if done and remaining[i] > 0:
                    records.append(dict(infos[i]))
                    remaining[i] -= 1
    finally:
        vec.close()
        model.policy.set_training_mode(was_training)

    n = len(records)
    if n != trials:
        raise AssertionError(f'Expected {trials} finished episodes, observed {n}')
    parts = sorted({k for r in records for k in r.get('reward_parts', {})})
    part_means = {key: float(np.mean([float(r.get('reward_parts', {}).get(key, 0)) for r in records])) for key in parts}
    max_x = [float(r['max_x']) for r in records if r.get('max_x') is not None]
    out = dict(stage=stage, repeat=repeat, reward_mode=reward, trials=n, seed=seed,
               wins=sum(int(bool(r['success'])) for r in records),
               mean_return=safe_mean(records, 'episode_return'),
               mean_bonus=safe_mean(records, 'bonus_return'),
               mean_seconds=safe_mean(records, 'seconds'),
               mean_max_x=float(np.mean(max_x)) if max_x else None,
               mean_policy_entropy=entropy_sum / max(1, count),
               mean_action_probabilities=(probability_sum/max(1, count)).round(5).tolist(),
               actual_action_counts=action_counts.tolist(),
               mean_decisions_per_episode=float(count/n),
               reward_parts_mean=part_means,
               failure_reasons=dict(Counter(str(r.get('reason')) for r in records if not r['success'])),
               protocol='team_sampled_4_streams')
    out['success_rate'] = out['wins']/n
    if stage == 2 and reward == 'personal':
        out.update(first_spring_rate=safe_mean(records, 'first_spring'),
                   middle_rate=safe_mean(records, 'middle'),
                   spring2_rate=safe_mean(records, 'spring2'),
                   upper_rate=safe_mean(records, 'upper'),
                   spring3_rate=safe_mean(records, 'spring3'),
                   finish_rate=safe_mean(records, 'finish'),
                   mean_target_landings=safe_mean(records, 'target_landing_count'),
                   mean_springs_seen=safe_mean(records, 'springs_seen'))
    elif stage == 4 and reward == 'personal':
        out.update(lift_rate=safe_mean(records, 'lift'),
                   lift_high_rate=safe_mean(records, 'lift_high'),
                   upper_rate=safe_mean(records, 'upper'))
    else:
        out['mean_milestones'] = float(np.mean([len(r.get('milestones', [])) for r in records]))
    return out


def save_json(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def score(ev):
    if ev['stage'] == 2 and ev['reward_mode'] == 'personal':
        return (ev['success_rate'], ev['spring3_rate'], ev['upper_rate'],
                ev['spring2_rate'], ev['mean_target_landings'],
                ev['first_spring_rate'], ev['mean_max_x'] or 0)
    if ev['stage'] == 4 and ev['reward_mode'] == 'personal':
        return (ev['success_rate'], ev['upper_rate'], ev['lift_high_rate'],
                ev['lift_rate'], ev['mean_max_x'] or 0)
    return (ev['success_rate'], ev.get('mean_milestones', 0), ev['mean_max_x'] or 0)


class PhysicsMonitor(BaseCallback):
    """Track actual physical World.step calls, including short terminal actions."""
    def __init__(self, folder, args, start_frames=0, start_episodes=0, start_wins=0):
        super().__init__(verbose=0)
        self.folder = Path(folder)
        self.args = args
        self.physical_frames = int(start_frames)
        self.episodes = int(start_episodes)
        self.training_wins = int(start_wins)
        self.next_eval = (start_frames // args.eval_every_frames + 1)*args.eval_every_frames
        self.best = None
        self.collapse_checks = 0
        self.stopped_for_budget = False
        self.stopped_for_collapse = False
        self.last_saved_frames = -1
        self.started = time.perf_counter()
        best = self.folder/'best_validation.json'
        if best.exists():
            self.best = score(json.loads(best.read_text(encoding='utf-8'))['validation'])

    def _on_step(self):
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            self.physical_frames += int(info['tick_delta'])
            if done:
                self.episodes += 1
                self.training_wins += int(bool(info.get('success')))
        if self.physical_frames >= self.args.physics_budget:
            self.stopped_for_budget = True
            return False
        return not self.stopped_for_collapse

    def after_update(self, model):
        # Attach this callback to the original team's PPO train() update hook.
        if self.physical_frames >= self.next_eval:
            self.checkpoint()
            self.next_eval = (self.physical_frames // self.args.eval_every_frames + 1)*self.args.eval_every_frames

    def checkpoint(self):
        ev = evaluate(self.model, self.args.stage, self.args.reward, self.args.repeat,
                      self.args.validation_trials, self.args.validation_seed)
        rec = dict(decisions=int(self.model.num_timesteps),
                   physical_frames=self.physical_frames, episodes=self.episodes,
                   training_wins=self.training_wins,
                   entropy_coefficient=float(self.model.ent_coef),
                   validation=ev, elapsed_s=round(time.perf_counter()-self.started, 1))
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder/'validation_history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False)+'\n')
        self.model.save(str(self.folder/'last.zip'))
        save_json(self.folder/'latest.json', rec)
        self.last_saved_frames = self.physical_frames
        result_score = score(ev)
        if self.best is None or result_score > self.best:
            self.best = result_score
            self.model.save(str(self.folder/'best.zip'))
            save_json(self.folder/'best_validation.json', rec)
        if self.args.stage == 2 and self.args.reward == 'personal':
            checkpoints = (f"spring1={ev['first_spring_rate']:.0%} "
                           f"middle={ev['middle_rate']:.0%} spring2={ev['spring2_rate']:.0%} "
                           f"upper={ev['upper_rate']:.0%}")
        elif self.args.stage == 4 and self.args.reward == 'personal':
            checkpoints = (f"lift={ev['lift_rate']:.0%} "
                           f"lift_high={ev['lift_high_rate']:.0%} upper={ev['upper_rate']:.0%}")
        else:
            checkpoints = f"milestones={ev.get('mean_milestones', 0):.2f}"
        x_label = f"{ev['mean_max_x']:.1f}" if ev['mean_max_x'] is not None else 'n/a'
        print(f"[CHECK] stage={self.args.stage} repeat={self.args.repeat} "
              f"decisions={self.model.num_timesteps} physics={self.physical_frames}/{self.args.physics_budget} "
              f"win={ev['wins']}/{ev['trials']} {checkpoints} "
              f"max_x={x_label} policy_H={ev['mean_policy_entropy']:.3f}/1.099 "
              f"reward={ev['mean_return']:+.2f} ent_coef={self.model.ent_coef:.3f} "
              f"best={self.best[0]:.0%}", flush=True)
        if not self.args.no_early_stop and ev['wins'] == 0 and ev['mean_policy_entropy'] < self.args.collapse_entropy:
            self.collapse_checks += 1
        else:
            self.collapse_checks = 0
        if self.collapse_checks >= 2:
            self.stopped_for_collapse = True
            print('[EARLY STOP] 2 consecutive zero-win checks with near-zero action entropy', flush=True)
        return rec


def team_model(vec, seed):
    m = team_train.make_model('ppo', vec, seed, joint=False)
    if not (m.action_space.n == 3 and m.observation_space.shape == (OBS_SIZE,)
            and m.n_steps == 128 and m.batch_size == 256 and m.n_epochs == 5
            and m.gamma == 1. and m.gae_lambda == .98 and m.learning_rate == .0003
            and m.clip_range(1.) == .2):
        raise AssertionError('The TEAM PPO configuration changed unexpectedly')
    m.ent_coef = ENTROPY_COEF
    return m


def run_train(args):
    if args.stage not in (2,4) or args.reward not in ('personal','original') \
            or args.repeat not in (6,12) or args.physics_budget < 1 \
            or args.eval_every_frames < 1 or args.validation_trials < 1 or args.test_trials < 0 \
            or args.seed < 0:
        raise ValueError('Invalid training arguments')
    install()
    torch.set_num_threads(1)
    folder = folder_for(args)
    config_file = folder/'config.json'
    config = dict(stage=args.stage, reward=args.reward, repeat=args.repeat, seed=args.seed,
                  envs=4, action_count=3, common_observation=OBS_SIZE,
                  max_seconds=180., physics_hz=120, personal_bonus_source='tutorial_2_4',
                  first_spring_bonus=0., from_scratch=True,
                  BC=False, DAgger=False, pretrained=False, teacher=False,
                  ppo=dict(learning_rate=.0003, gamma=1., gae_lambda=.98,
                           n_steps=128, batch_size=256, n_epochs=5,
                           clip_range=.2, target_kl=.02,
                           entropy_coefficient=ENTROPY_COEF),
                  validation_trials=args.validation_trials,
                  validation_seed=args.validation_seed,
                  eval_every_frames=args.eval_every_frames)
    if args.resume:
        if not config_file.is_file() or not (folder/'last.zip').is_file() or not (folder/'latest.json').is_file():
            raise FileNotFoundError('No previous run to resume. '+str(folder))
        old = json.loads(config_file.read_text(encoding='utf-8'))
        if old != config:
            raise ValueError('Resume configuration mismatch: original saved settings differ')
        latest = json.loads((folder/'latest.json').read_text(encoding='utf-8'))
        start_frames = int(latest['physical_frames'])
        start_episodes = int(latest['episodes'])
        start_wins = int(latest['training_wins'])
        if start_frames >= args.physics_budget:
            raise ValueError(f'Saved at {start_frames} frames, target {args.physics_budget}; increase --physics-budget')
    else:
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError('Experiment already exists; use --resume, or change --seed. '+str(folder))
        folder.mkdir(parents=True, exist_ok=True)
        save_json(config_file, config)
        start_frames = start_episodes = start_wins = 0

    vec = make_vec(args.stage, args.reward, args.repeat)
    try:
        if args.resume:
            m = MilestonePPO.load(str(folder/'last.zip'), env=vec, device='cpu')
            if m.action_space.n != 3 or m.observation_space.shape != (OBS_SIZE,):
                raise ValueError('Loaded model does not match team PPO contract')
            # check architecture and optimizer settings without zeroing learned weights
            if not (m.n_steps == 128 and m.batch_size == 256 and m.n_epochs == 5
                    and m.gamma == 1. and m.gae_lambda == .98 and m.learning_rate == .0003
                    and m.clip_range(1.) == .2):
                raise ValueError('Incompatible PPO checkpoint hyperparameters')
            print(f'[RESUME] stage={args.stage} repeat={args.repeat} '
                  f'physics={start_frames}->{args.physics_budget} decisions={m.num_timesteps}',flush=True)
        else:
            m = team_model(vec, args.seed)
            print(f'[START] stage={args.stage} reward={args.reward} repeat={args.repeat} '
                  f'physics_budget={args.physics_budget} FROM SCRATCH PPO',flush=True)
        m.verbose = 0
        m.ent_coef = ENTROPY_COEF
        monitor = PhysicsMonitor(folder, args, start_frames=start_frames,
                                 start_episodes=start_episodes, start_wins=start_wins)
        m.after_update = monitor.after_update
        # Large enough maximum decision budget; stop by measured physical ticks.
        # PPO gamma/lr/entropy do NOT depend on this total_timesteps argument.
        m.learn(total_timesteps=args.physics_budget + 1024,
                callback=monitor, reset_num_timesteps=not args.resume)
        # Save even when a physical-budget stop happens midway through a PPO rollout.
        if monitor.last_saved_frames != monitor.physical_frames:
            monitor.checkpoint()
        if not (folder/'best.zip').is_file():
            raise RuntimeError('PPO did not create best.zip')
    finally:
        vec.close()
    result = dict(config=config, completed_steps=int(m.num_timesteps),
                  completed_physics_frames=monitor.physical_frames,
                  requested_physics_budget=args.physics_budget,
                  stopped_for_budget=monitor.stopped_for_budget,
                  stopped_for_collapse=monitor.stopped_for_collapse,
                  best_validation=json.loads((folder/'best_validation.json').read_text(encoding='utf-8')))
    if args.test_trials > 0:
        selected = MilestonePPO.load(str(folder/'best.zip'), device='cpu')
        result['independent_test'] = evaluate(selected, args.stage, args.reward, args.repeat,
                                             args.test_trials, args.test_seed)
        print(f"[TEST] stage={args.stage} {result['independent_test']['wins']}/"
              f"{args.test_trials} seed={args.test_seed}",flush=True)
    save_json(folder/'summary.json', result)
    print(f'[SAVED] {folder}',flush=True)


def run_smoke(args):
    """Assert reward replacement does not change physics or model observations."""
    rng = np.random.default_rng(args.seed)
    a = make_env(args.stage, 'original', args.repeat)
    b = make_env(args.stage, 'personal', args.repeat)
    episodes = 0
    frames = 0
    try:
        obs_a, _ = a.reset(seed=args.seed)
        obs_b, _ = b.reset(seed=args.seed)
        np.testing.assert_allclose(obs_a, obs_b, rtol=0, atol=0)
        for _ in range(args.decisions):
            action = int(rng.integers(0,3))
            obs_a, _, da, ta, ia = a.step(action)
            obs_b, _, db, tb, ib = b.step(action)
            np.testing.assert_allclose(obs_a, obs_b, rtol=0, atol=1e-7)
            if (da,ta,ia['tick_delta'],ia['success']) != (db,tb,ib['tick_delta'],ib['success']):
                raise AssertionError('Reward-only environment changed native physics')
            frames += ia['tick_delta']
            if da or ta:
                episodes += 1
                obs_a, _ = a.reset(seed=args.seed+episodes)
                obs_b, _ = b.reset(seed=args.seed+episodes)
                np.testing.assert_allclose(obs_a, obs_b, rtol=0, atol=0)
    finally:
        a.close();b.close()
    print(f'[SMOKE PASS] stage={args.stage} repeat={args.repeat} '
          f'decisions={args.decisions} actual_physics_frames={frames} '
          f'episodes={episodes}: SAME obs/physics/done',flush=True)


def run_evaluate(args):
    install()
    torch.set_num_threads(1)
    path = Path(args.model)
    if not path.is_file():
        raise FileNotFoundError(path)
    config = path.with_name('config.json')
    if not config.is_file():
        raise FileNotFoundError('Evaluation checkpoint must have a matching config.json')
    original = json.loads(config.read_text(encoding='utf-8'))
    for key, wanted in [('stage', args.stage), ('reward', args.reward), ('repeat', args.repeat)]:
        if original.get(key) != wanted:
            raise ValueError(f'Checkpoint {key} mismatch: saved {original.get(key)}, requested {wanted}')
    model = MilestonePPO.load(str(path), device='cpu')
    if model.action_space.n != 3 or model.observation_space.shape != (OBS_SIZE,):
        raise AssertionError('Model is not a 3-action common-observation PPO')
    result = evaluate(model, args.stage, args.reward, args.repeat, args.trials, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        save_json(args.output, result)


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest='command', required=True)
    tr = sp.add_parser('train', help='PPO scratch train with actual physical tick budget')
    tr.add_argument('--stage', required=True, choices=(2,4), type=int)
    tr.add_argument('--reward', choices=('personal','original'), default='personal')
    tr.add_argument('--repeat', choices=(6,12), type=int, default=12)
    tr.add_argument('--physics-budget', type=int, default=240000,
                    help='Total actual World.step ticks across four envs; use 240000 then 480000 --resume')
    tr.add_argument('--seed', type=int, default=42)
    tr.add_argument('--out', type=Path, default=DEFAULT_ROOT)
    tr.add_argument('--eval-every-frames', type=int, default=60000)
    tr.add_argument('--validation-trials', type=int, default=20)
    tr.add_argument('--validation-seed', type=int, default=50000)
    tr.add_argument('--test-trials', type=int, default=0)
    tr.add_argument('--test-seed', type=int, default=70000)
    tr.add_argument('--resume', action='store_true')
    tr.add_argument('--no-early-stop', action='store_true')
    tr.add_argument('--collapse-entropy', type=float, default=.1)
    sm = sp.add_parser('smoke', help='Compare personal reward environment with team physics')
    sm.add_argument('--stage', required=True, choices=(2,4), type=int)
    sm.add_argument('--repeat', choices=(6,12), type=int, default=12)
    sm.add_argument('--decisions', type=int, default=2000)
    sm.add_argument('--seed', type=int, default=42)
    ev = sp.add_parser('evaluate', help='Independent policy evaluation')
    ev.add_argument('--stage', required=True, choices=(2,4), type=int)
    ev.add_argument('--reward', choices=('personal','original'), default='personal')
    ev.add_argument('--repeat', choices=(6,12), type=int, default=12)
    ev.add_argument('--model', required=True)
    ev.add_argument('--trials', type=int, default=200)
    ev.add_argument('--seed', type=int, default=70000)
    ev.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.command == 'train':
        run_train(args)
    elif args.command == 'smoke':
        run_smoke(args)
    else:
        run_evaluate(args)


if __name__ == '__main__':
    cli()
