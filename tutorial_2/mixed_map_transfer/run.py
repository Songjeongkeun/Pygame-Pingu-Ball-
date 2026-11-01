"""Transfer evaluation without PPO training, checkpoint rewriting, or team source changes.

    python -m tutorial_2.mixed_map_transfer.run inspect
    python -m tutorial_2.mixed_map_transfer.run evaluate --maps 1-10 --episodes 5 --limit 60
    python -m tutorial_2.mixed_map_transfer.run evaluate --maps 1-10 --episodes 20 --limit 180
    python -m tutorial_2.mixed_map_transfer.run play --map 1 --seed 120000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from collections import Counter

from .adapter_utils import parse_map_selection

TUTORIAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = TUTORIAL_ROOT / 'results' / 'mixed_map_transfer'


def ensure_root():
    """Explicitly fail if unpacked outside original game repository root."""
    root = TUTORIAL_ROOT.parent
    for rel in ('pingu_ball/rl_maps.py', 'rl/milestone_env.py',
                'assets/rl_maps/layouts.json',
                'tutorial_2/stage2_common_bridge/core.py'):
        if not (root / rel).is_file():
            raise FileNotFoundError(f'Must run from the original game repository with {rel}: {root}')
    return root


def checkpoint_hash():
    from tutorial_2.stage2_common_bridge.core import STAGE2_MODEL
    if not STAGE2_MODEL.is_file():
        raise FileNotFoundError(f'Original frozen stage2 checkpoint not found: {STAGE2_MODEL}')
    h = hashlib.sha256()
    with STAGE2_MODEL.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest(), STAGE2_MODEL


def load_policy():
    # Reuse exactly the previously integrated, validated stage2 checkpoint
    # loader. It constructs the original tutorial env for loading only; the
    # transfer rollout steps a separate MixedMapTransferEnv, never that env.
    from tutorial_2.stage2_common_bridge.core import load_stage2_model
    from rl.milestone_env import OBS_SIZE
    model, vec = load_stage2_model()
    if model.observation_space.shape != (OBS_SIZE,) or model.action_space.n != 3:
        vec.close()
        raise RuntimeError('Stage2 trained checkpoint schema mismatch')
    return model, vec


def one_episode(model, env, seed, *, greedy=False, on_step=None, on_reset=None):
    # Reset stochastic actor for reproducibility. World physics is already deterministic.
    model.set_random_seed(int(seed))
    obs, start_info = env.reset(seed=int(seed))
    if on_reset is not None:
        on_reset()
    total = 0.
    while True:
        action, _ = model.predict(obs, deterministic=greedy)
        obs, reward, done, truncated, info = env.step(int(action))
        total += float(reward)
        if on_step is not None and on_step(info):
            info = dict(info, reason='viewer_interrupted', interrupted=True)
            break
        if done or truncated:
            break
    return dict(info, seed=int(seed), sampled=not greedy,
                scaled_eval_terminal_return=round(total, 4))


def do_inspect(args):
    ensure_root()
    from pingu_ball.rl_maps import rl_map_layouts
    from rl.milestone_env import OBS_SIZE, CAPACITY, LOCAL_COUNT
    from .env import MixedMapTransferEnv
    layouts = rl_map_layouts()
    print(f'[SCHEMA] PPO stage2 obs={OBS_SIZE}, actions=3, repeat=6; full slots={CAPACITY}, local slots={LOCAL_COUNT}')
    print('[NOTICE] OOD adapter: nearest objects, ZERO tutorial milestones, onehot stage2; NOT the trained stage2 environment')
    for layout in layouts:
        env = MixedMapTransferEnv(layout['number'])
        try:
            obs, meta = env.reset(seed=42)
            if obs.shape != (OBS_SIZE,) or not env.observation_space.contains(obs):
                raise AssertionError(f'map {layout["number"]} invalid observation')
            for _ in range(3):
                obs, _, done, truncated, _ = env.step(1)
                if obs.shape != (OBS_SIZE,) or not env.observation_space.contains(obs):
                    raise AssertionError(f'map {layout["number"]} observation failed after physics step')
                if done or truncated:
                    break
            print(f'[MAP {layout["number"]:02d}] {layout["title"]} '
                  f'entities={meta["full_entities"]} (full visible={min(CAPACITY,meta["full_entities"])}) '
                  f'obs={obs.shape}, game={env.level.width}x{env.level.height}')
        finally:
            env.close()
    checkpoint_digest, checkpoint = checkpoint_hash()
    print(f'[FROZEN MODEL] {checkpoint} sha256={checkpoint_digest}')
    print('[READY] Map loading and observation dimensions checked. No PPO evaluation performed yet.')


def do_evaluate(args):
    ensure_root()
    if args.episodes < 1:
        raise ValueError('--episodes must be >= 1')
    maps = parse_map_selection(args.maps)
    from .env import MixedMapTransferEnv
    from rl.milestone_env import OBS_SIZE
    digest, ckpt = checkpoint_hash()
    model, vec = load_policy()
    out = Path(args.output).resolve() if args.output else DEFAULT_OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    trials = []
    summaries = []
    try:
        for map_no in maps:
            env = MixedMapTransferEnv(map_no, limit=args.limit)
            try:
                initial_obs, initial_info = env.reset(seed=args.seed)
                if initial_obs.shape != (OBS_SIZE,) or model.observation_space.shape != initial_obs.shape:
                    raise RuntimeError('Observations do not match checkpoint inputs')
                print(f'\n[MIXED {map_no:02d}] {env.level.title} '
                      f'entities={initial_info["full_entities"]} '
                      f'slots={initial_info["visible_full_capacity"]} '
                      f'episodes={args.episodes} limit={args.limit}s', flush=True)
                results = []
                for ep in range(args.episodes):
                    seed = args.seed + map_no*100000 + ep
                    result = one_episode(model, env, seed, greedy=args.greedy)
                    result['episode'] = ep + 1
                    results.append(result)
                    trials.append(result)
                    print(f'  {ep+1:>3}/{args.episodes}: '
                          f'{"CLEAR" if result["success"] else "FAIL "} '
                          f'{result["seconds"]:>7.2f}s  '
                          f'{result["reason"]}  '
                          f'min_goal={result["min_goal_distance"]:.0f}', flush=True)
                wins = sum(bool(r['success']) for r in results)
                failures = Counter(str(r['reason']) for r in results if not r['success'])
                win_seconds = [r['seconds'] for r in results if r['success']]
                summary = {
                    'map': map_no, 'title': env.level.title,
                    'episodes': len(results), 'successes': wins,
                    'success_rate': wins / len(results),
                    'mean_success_seconds': sum(win_seconds)/len(win_seconds) if win_seconds else None,
                    'failures': dict(failures),
                    'mean_min_goal_distance': sum(r['min_goal_distance'] for r in results)/len(results),
                    'full_entities': initial_info['full_entities'],
                    'full_visible_capacity': initial_info['visible_full_capacity']}
                summaries.append(summary)
                print(f'[RESULT MAP {map_no:02d}] {wins}/{len(results)} = {wins/len(results)*100:.1f}% '
                      f'failures={dict(failures)}', flush=True)
            finally:
                env.close()
    finally:
        vec.close()

    payload = {
        'experiment': 'zero_shot_stage2_ppo_to_unseen_mixed_rl_maps',
        'checkpoint': str(ckpt), 'checkpoint_sha256': digest,
        'checkpoint_retrained': False, 'checkpoint_modified': False,
        'map_data': 'assets/rl_maps/layouts.json',
        'observations': ('original milestone schema with dynamically nearest CAPACITY entities, '
                         'nearest LOCAL_COUNT objects, ZERO milestone slots, stage2 onehot; '
                         'out of distribution and not comparable to in-domain 96%'),
        'actions': '0 left,1 idle,2 right; six native physics ticks per decision',
        'reward': 'terminal-only for reporting; no training and no reward shaping',
        'evaluation_seed_base': args.seed, 'per_episode_seed_formula': 'base+map_number*100000+episode_index',
        'stochastic': not args.greedy, 'limit_seconds': args.limit,
        'per_map': summaries, 'trials': trials,
    }
    json_file = out / ('mixed_transfer_seed_' + str(args.seed) + '_n_' + str(args.episodes) +
                       '_limit_' + str(args.limit).replace('.', 'p') +
                       ('_greedy' if args.greedy else '_sampled') + '.json')
    with json_file.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    csv_file = json_file.with_suffix('.csv')
    keys = ['map','episode','seed','success','reason','seconds','decisions','ticks','max_x',
            'end_x','end_y','start_goal_distance','min_goal_distance','final_goal_distance',
            'full_entities','visible_full_capacity','sampled','scaled_eval_terminal_return']
    with csv_file.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(trials)
    total_wins = sum(item['successes'] for item in summaries)
    total_episodes = sum(item['episodes'] for item in summaries)
    print(f'\n[OVERALL] {total_wins}/{total_episodes} ({total_wins/total_episodes:.1%}) '
          f'-- NOTE: transfer adapter, not the native in-domain stage2 evaluation')
    print(f'[SAVED] {json_file}')
    print(f'[SAVED] {csv_file}')


def do_play(args):
    ensure_root()
    if args.fps < 1 or args.fps > 120:
        raise ValueError('--fps must be within [1, 120]')
    from pingu_ball.app import App
    from pingu_ball.rl_maps import make_rl_level
    from .env import MixedMapTransferEnv
    import pygame

    digest, ckpt = checkpoint_hash()
    model, vec = load_policy()
    env = MixedMapTransferEnv(args.map, limit=args.limit)
    try:
        with tempfile.TemporaryDirectory(prefix='pingu_mixed_ppo_watch_') as scratch:
            app = App(save_path=Path(scratch) / 'progress.json')
            try:
                # Run stock renderer. Replace only the screen's World with the
                # transfer evaluation World (auto_reset=False). Never call
                # app.update(), which would move the ball a second time.
                app.levels[args.map] = make_rl_level(args.map)
                app.start(args.map)
                app.world = env.world
                app.map_view = app.map_viewport()
                app.next_stage = app.restart
                app.menu = app.quit
                pygame.display.set_caption(f'Pingu Ball - Mixed map {args.map:02d} - frozen stage2 PPO')
                print(f'[WATCH] Mixed map {args.map}; frozen stage2 PPO '
                      f'({"greedy" if args.greedy else "sampled"}); ESC or close to stop.', flush=True)
                app.render()
                def update_view(_info):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            return True
                    app.render()
                    app.clock.tick(args.fps)
                    return False
                result = one_episode(model, env, args.seed, greedy=args.greedy,
                                     on_reset=lambda: setattr(app, "world", env.world),
                                     on_step=update_view)
                print(f'[WATCH RESULT] map={args.map} success={result["success"]} '
                      f'reason={result["reason"]} seconds={result["seconds"]} '
                      f'decisions={result["decisions"]} min_goal={result["min_goal_distance"]}', flush=True)
                if not result.get('interrupted'):
                    # Keep final frame visible briefly, without changing physics.
                    app.render()
                    deadline = pygame.time.get_ticks() + int(args.hold*1000)
                    while pygame.time.get_ticks() < deadline:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                                return
                        app.clock.tick(30)
            finally:
                pygame.quit()
    finally:
        env.close()
        vec.close()


def main():
    parser = argparse.ArgumentParser(description='Untrained stage2 PPO transfer to new ten mixed RL maps')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect', help='check all mixed maps and PPO input shape WITHOUT loading PPO')
    eval_parser = sub.add_parser('evaluate', help='run the unchanged stage2 PPO on mixed maps')
    eval_parser.add_argument('--maps', default='1-10', help='1-10, all, 1, or e.g. 1,4,8')
    eval_parser.add_argument('--episodes', type=int, default=5)
    eval_parser.add_argument('--seed', type=int, default=120000)
    eval_parser.add_argument('--limit', type=float, default=180., help='seconds per attempt (max 180)')
    eval_parser.add_argument('--greedy', action='store_true', help='greedy argmax policy, default sampled')
    eval_parser.add_argument('--output', type=Path, help='output directory, default tutorial_2/results/mixed_map_transfer')
    play_parser = sub.add_parser('play', help='render one mixed map controlled by the frozen PPO')
    play_parser.add_argument('--map', type=int, choices=range(1,11), default=1)
    play_parser.add_argument('--seed', type=int, default=120000)
    play_parser.add_argument('--limit', type=float, default=180.)
    play_parser.add_argument('--greedy', action='store_true')
    play_parser.add_argument('--fps', type=int, default=30,
                             help='game decision frames per second (each decision has 6 native physics ticks)')
    play_parser.add_argument('--hold', type=float, default=2., help='seconds to show final frame')
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            do_inspect(args)
        elif args.command == 'evaluate':
            if not (0 < args.limit <= 180):
                raise ValueError('--limit must be 0..180s')
            do_evaluate(args)
        elif args.command == 'play':
            if not (0 < args.limit <= 180) or not (0 <= args.hold <= 30):
                raise ValueError('--limit must be 0..180; --hold 0..30')
            do_play(args)
    except (FileNotFoundError, ValueError, RuntimeError, AssertionError) as exc:
        parser.exit(2, f'[ERROR] {exc}\n')


if __name__ == '__main__':
    main()
