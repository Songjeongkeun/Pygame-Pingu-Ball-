"""Run one tutorial with an isolated import path for its historical game runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def play_view(env, stage):
    import pygame
    from pingu_ball.app import App

    with tempfile.TemporaryDirectory() as temporary:
        app = App(Path(temporary) / "progress.json")
        app.sound_on = False
        app.start_special(stage)
        clock = pygame.time.Clock()
        frames = 0

        def render(world, _direction):
            nonlocal frames
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    raise KeyboardInterrupt
            frames += 1
            if frames % 2 == 0 or world.failed or world.cleared:
                app.world = world
                app.render()
            clock.tick(round(1. / world.physics.dt))

        try:
            yield render
        finally:
            pygame.quit()


def run_episodes(env, probabilities, args, render=None):
    import numpy as np
    from pingu_ball.engine import World

    rows = []
    max_error = 0.
    total_decisions = 0
    for offset in range(args.episodes):
        seed = args.seed + offset
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        native = World(env.world.level, auto_reset=False) if args.mode == "verify" else None

        def tick(world, direction):
            nonlocal max_error
            if native is not None:
                dt = native.physics.dt if args.stage == 4 else min(native.physics.dt, 180. - native.time)
                native.step(direction, dt)
                for field in ("x", "y", "vx", "vy"):
                    max_error = max(max_error, abs(getattr(world.ball, field) - getattr(native.ball, field)))
                if (max_error > 1e-8 or abs(world.time - native.time) > 1e-8
                        or (world.cleared, world.failed, world.events) != (native.cleared, native.failed, native.events)):
                    raise RuntimeError(f"Native physics mismatch in stage {args.stage}")
            if render:
                render(world, direction)

        done = False
        while not done:
            p = np.asarray(probabilities(obs), dtype=np.float64)
            if p.shape != (env.action_space.n,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
                raise RuntimeError(f"Invalid policy distribution in stage {args.stage}")
            action = int(p.argmax()) if args.greedy else int(rng.choice(len(p), p=p / p.sum()))
            obs, _, terminated, truncated, info = env.step(action, on_tick=tick if native or render else None)
            total_decisions += 1
            done = terminated or truncated
        rows.append(info)
    successes = sum(row["success"] for row in rows)
    return dict(stage=args.stage, model=args.model, observations=int(env.observation_space.shape[0]),
                actions=int(env.action_space.n), evaluation_seed=args.seed, episodes=len(rows),
                successes=successes, success_rate=successes / len(rows),
                mean_score=sum(row["score"] for row in rows) / len(rows),
                decisions=total_decisions, max_native_error=max_error if args.mode == "verify" else None,
                results=rows)


def train(env, args):
    from stable_baselines3 import PPO
    from rl.milestone_models import MilestoneExtractor, MilestonePPO
    from torch import nn

    if args.output.suffix.lower() != ".zip":
        raise ValueError("Training output must be a .zip file")
    if args.output.exists():
        raise FileExistsError(args.output)
    policy_args = (dict(net_arch=[128, 128], activation_fn=nn.Tanh) if args.stage == 4
                   else dict(features_extractor_class=MilestoneExtractor))
    algorithm = PPO if args.stage == 4 else MilestonePPO
    model = algorithm("MlpPolicy", env, policy_kwargs=policy_args, n_steps=128, batch_size=64,
                      n_epochs=5, gamma=1., gae_lambda=.98, seed=args.seed, device="cpu")
    model.learn(total_timesteps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Stage {args.stage}: {model.num_timesteps} decisions, saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--mode", choices=("verify", "evaluate", "play", "train"), required=True)
    parser.add_argument("--model", default="new")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sha256 and hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.sha256:
        raise ValueError(f"Checkpoint hash mismatch: {args.model}")

    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    sys.path.insert(0, str(args.runtime))
    os.chdir(args.runtime)
    import rl
    rl.__path__.append(str(ROOT / "rl"))

    import torch
    from contextlib import contextmanager, nullcontext
    from rl.tutorials.runtime import TutorialEnv, load_probabilities

    torch.set_num_threads(1)
    env = TutorialEnv(args.stage)
    try:
        if args.mode == "train":
            return train(env, args)
        probabilities = load_probabilities(args.checkpoint, env,
                                           historical_lift=args.stage == 4 and bool(args.sha256))
        view = (contextmanager(play_view)(env, args.stage) if args.mode == "play" and not args.headless
                else nullcontext())
        with view as render:
            result = run_episodes(env, probabilities, args, render)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except KeyboardInterrupt:
        if args.mode != "play":
            raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
