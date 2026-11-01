"""Stage 2 personal PPO + 1/3/4/5 unchanged team PPO, one stage-selection CLI.

This is a DISPATCHER for individual expert policies, not a single joint network.
See README.md before using existing models in a different runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .core import STAGE2_MODEL, RESULTS, PROJECT_ROOT, make_env, load_stage2_model, archive_command


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def check_physics(steps=50, seed=42):
    """Stage2 personal vs team milestone: identical actions => same obs/physics.

    The two reward totals MUST be allowed to differ, because this is the
    intended stage-specific reward replacement. The replaced personal reward
    MUST NOT include the team's old milestone reward twice.
    """
    from rl.milestone_env import MilestoneEnv
    from rl.milestone_env import OBS_SIZE
    team = MilestoneEnv(stage=2, repeat=6, limit=180.)
    personal = make_env(2)
    rng = np.random.default_rng(seed)
    tick_count = 0
    episodes = 0
    try:
        a, _ = team.reset(seed=seed)
        b, _ = personal.reset(seed=seed)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)
        for index in range(steps):
            act = int(rng.integers(3))
            a, team_reward, done_a, truncated_a, info_a = team.step(act)
            b, personal_reward, done_b, truncated_b, info_b = personal.step(act)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6,
                                       err_msg=f'Observation changed at decision {index}')
            if (done_a, truncated_a) != (done_b, truncated_b):
                raise AssertionError(f'Done changed at decision {index}')
            if int(info_a['tick_delta']) != int(info_b['tick_delta']):
                raise AssertionError(f'Physics tick count changed at decision {index}')
            tick_count += int(info_a['tick_delta'])
            if not np.isclose(float(info_a['episode_return']),
                              float(info_b['original_team_return']), atol=1e-6):
                raise AssertionError('Team original reward accounting changed')
            expected = (float(info_b['personal_terminal']) + float(info_b['personal_bonus'])) / 100.
            if not np.isclose(personal_reward, expected, atol=1e-6):
                raise AssertionError('Old team milestone reward was added to personal PPO reward')
            if not np.isfinite(team_reward):
                raise AssertionError('Non-finite original reward')
            wa, wb = team.unwrapped.world, personal.team.unwrapped.world
            xyz = ('x', 'y', 'vx', 'vy')
            np.testing.assert_allclose([getattr(wa.ball, name) for name in xyz],
                                       [getattr(wb.ball, name) for name in xyz],
                                       rtol=0, atol=2e-6)
            if done_a or truncated_a:
                episodes += 1
                a, _ = team.reset(seed=seed + episodes)
                b, _ = personal.reset(seed=seed + episodes)
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
        if personal.action_space.n != 3 or personal.observation_space.shape != (OBS_SIZE,):
            raise AssertionError('Stage2 common model contract changed')
        print(f'[PHYSICS/REWARD PASS] {steps} decisions / {tick_count} native ticks, '
              f'{episodes} resets: same observations/actions/physics, only rewards replaced', flush=True)
    finally:
        team.close()
        personal.close()
    return {'steps': steps, 'physics_ticks': tick_count, 'resets': episodes,
            'action_count': 3, 'observation_size': OBS_SIZE, 'action_repeat': 6,
            'unchanged_team_physics': True, 'stage2_personal_reward_only': True}


def evaluate_stage2(model, episodes, seed):
    """Use team 4-stream sampled-policy protocol on our integrated environment."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.five_models import action_probs
    from tutorial_2.reward_repeat_2_4.train import safe_mean
    if episodes < 1:
        raise ValueError('episodes must be positive')
    rngs = [np.random.default_rng(seed + 2 * 100003 + i * 104729) for i in range(4)]
    remaining = [episodes // 4 + int(i < episodes % 4) for i in range(4)]
    vec = DummyVecEnv([lambda: make_env(2) for _ in range(4)])
    records = []
    counts = np.zeros(3, dtype=np.int64)
    try:
        obs = vec.reset()
        while any(remaining):
            probabilities = np.asarray(action_probs(model, obs), dtype=np.float64)
            if probabilities.shape != (4, 3):
                raise ValueError(f'Expected team (4,3) policy distribution, got {probabilities.shape}')
            actions = np.zeros(4, dtype=np.int64)
            for i, p in enumerate(probabilities):
                if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
                    raise ValueError('Invalid action distribution')
                p = p / p.sum()
                actions[i] = int(rngs[i].choice(3, p=p))
                if remaining[i]:
                    counts[actions[i]] += 1
            obs, _, dones, infos = vec.step(actions)
            for i, done in enumerate(dones):
                if done and remaining[i] > 0:
                    records.append(dict(infos[i]))
                    remaining[i] -= 1
    finally:
        vec.close()
    if len(records) != episodes:
        raise AssertionError(f'Expected {episodes} episodes, got {len(records)}')
    successes = sum(bool(row['success']) for row in records)
    result = dict(stage=2, policy='personal_PPO_6frame', algorithm='ppo',
                  observation='team_MilestoneEnv_flat', actions=3, repeat=6,
                  evaluation_seed=seed, episodes=episodes, successes=successes,
                  success_rate=successes/episodes,
                  mean_return=safe_mean(records, 'episode_return'),
                  mean_bonus=safe_mean(records, 'bonus_return'),
                  mean_seconds=safe_mean(records, 'seconds'),
                  first_spring_rate=safe_mean(records, 'first_spring'),
                  middle_rate=safe_mean(records, 'middle'),
                  spring2_rate=safe_mean(records, 'spring2'),
                  upper_rate=safe_mean(records, 'upper'),
                  spring3_rate=safe_mean(records, 'spring3'),
                  finish_rate=safe_mean(records, 'finish'),
                  failures=dict(Counter(str(row.get('reason')) for row in records if not row['success'])),
                  action_counts=counts.tolist(), protocol='team_sampled_4_streams')
    return result


def play_stage2(model, seed=20261101, greedy=False):
    """Show the native pygame game using policy decisions & personal reward ticks."""
    import pygame
    from pingu_ball.app import App
    from rl.five_models import action_probs
    from tutorial_2.reward_repeat_2_4.reward_env import RewardOnlyEnv

    class VisualEnv(RewardOnlyEnv):
        tick_hook = None
        def _on_tick(self, world, tick, direction):
            super()._on_tick(world, tick, direction)
            if self.tick_hook is not None:
                self.tick_hook(world, tick, direction)

    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    pygame.init()
    env = VisualEnv(stage=2, repeat=6)
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as temporary:
        app = App(Path(temporary) / 'progress.json', headless=False)
        app.sound_on = False
        app.start_special(2)
        clock = pygame.time.Clock()
        def tick_hook(world, tick, direction):
            app.world = world
            if tick % 12 == 0 or world.cleared or world.failed:
                app.render()
            clock.tick(120)
        env.tick_hook = tick_hook
        actions = []
        try:
            obs, _ = env.reset(seed=seed)
            interrupted = False
            while not env.team.unwrapped.done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        interrupted = True
                if interrupted:
                    break
                p = action_probs(model, obs[None])[0]
                if not np.isfinite(p).all() or p.sum() <= 0:
                    raise ValueError('Invalid model action probabilities')
                p = p/p.sum()
                action = int(p.argmax()) if greedy else int(rng.choice(3, p=p))
                actions.append(action)
                obs, _, _, _, info = env.step(action)
            print(f'[PLAY STAGE 2] actions={len(actions)} interrupted={interrupted} '
                  f'success={bool(info["success"]) if actions else False} '
                  f'reward=personal repeat=6', flush=True)
        finally:
            env.close()
            pygame.quit()


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    verify = sub.add_parser('verify', help='Check stage2 common interface, physics, reward, checkpoint')
    verify.add_argument('--steps', type=int, default=100)
    verify.add_argument('--archive', action='store_true', help='Also verify all old model archives')
    evaluate = sub.add_parser('evaluate', help='Evaluate selected expert(s), no training')
    evaluate.add_argument('--stage', choices=['1','2','3','4','5','all'], default='2')
    evaluate.add_argument('--episodes', type=int, default=200)
    evaluate.add_argument('--seed', type=int, default=120000)
    evaluate.add_argument('--output', type=Path)
    play = sub.add_parser('play', help='Visual playback of selected expert')
    play.add_argument('--stage', choices=['1','2','3','4','5'], default='2')
    play.add_argument('--seed', type=int, default=20261101)
    play.add_argument('--greedy', action='store_true')
    for c in (verify, evaluate, play):
        c.add_argument('--stage2-model', type=Path, default=STAGE2_MODEL)
    return parser


def main(argv=None):
    parser = args_parser()
    args = parser.parse_args(argv)
    if args.command == 'verify':
        if args.steps < 1:
            parser.error('steps must be positive')
        physics = check_physics(steps=args.steps)
        model, vec = load_stage2_model(args.stage2_model)
        try:
            print(f'[MODEL PASS] {args.stage2_model} | 3 actions, 6-frame env, shared MilestoneExtractor', flush=True)
        finally:
            vec.close()
        if args.archive:
            archive_command('verify')
        report = {'physics': physics, 'model': str(args.stage2_model),
                  'model_loaded': True, 'archived_team_models_verified': args.archive}
        save_json(RESULTS / 'verify.json', report)
        print(f'[SAVED] {RESULTS / "verify.json"}', flush=True)
        return
    if args.command == 'play':
        stage = int(args.stage)
        if stage == 2:
            model, vec = load_stage2_model(args.stage2_model)
            try:
                play_stage2(model, args.seed, args.greedy)
            finally:
                vec.close()
        else:
            archive_command('play', stage=stage, seed=args.seed, greedy=args.greedy)
        return
    if args.episodes <= 0:
        parser.error('episodes must be positive')
    stages = list(range(1, 6)) if args.stage == 'all' else [int(args.stage)]
    results = []
    out_folder = RESULTS / f'eval_seed_{args.seed}_n_{args.episodes}'
    for stage in stages:
        if stage == 2:
            model, vec = load_stage2_model(args.stage2_model)
            try:
                row = evaluate_stage2(model, args.episodes, args.seed)
            finally:
                vec.close()
            row['model'] = str(args.stage2_model)
        else:
            output = out_folder / f'stage_{stage:02d}_original_team.json'
            archive_command('evaluate', stage=stage, episodes=args.episodes,
                            seed=args.seed, output=output)
            response = json.loads(output.read_text(encoding='utf8'))
            archived = response['results'][0]
            row = dict(stage=stage, policy='archived_team_ppo_unmodified', algorithm='ppo',
                       evaluation_seed=args.seed, episodes=args.episodes,
                       successes=int(archived['successes']),
                       success_rate=float(archived['success_rate']),
                       mean_return=archived.get('mean_return'),
                       model=archived['model'], protocol='team_archived_sampled')
        results.append(row)
        print(f'[INTEGRATED] stage={stage} {row["successes"]}/{args.episodes} '
              f'({row["success_rate"]:.1%}) mode={row["policy"]}', flush=True)
    report = dict(kind='stage_dispatch_individual_expert_PPO_NOT_single_joint_network',
                  note='Stage 4 remains original team PPO baseline; no retraining or reward change',
                  seed=args.seed, episodes_per_stage=args.episodes,
                  stage2_model=str(args.stage2_model), stages=results)
    output = args.output or out_folder / 'integrated_results.json'
    save_json(output, report)
    print(f'[SAVED] {output}', flush=True)


if __name__ == '__main__':
    main()
