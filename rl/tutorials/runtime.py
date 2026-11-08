"""Keep each checkpoint's observation and reward rules behind one episode API."""

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .lift_env import ACTIONS, OBS_VERSION, Stage4Env
from .spring_rewards import SpringRewardTracker


class TutorialEnv(gym.Env):
    def __init__(self, stage):
        self.stage = stage
        if stage == 4:
            self.env = Stage4Env()
            self.action_space = gym.spaces.Discrete(len(ACTIONS))
            self.observation_space = gym.spaces.Box(-4., 4., (self.env.obs_dim,), dtype=np.float32)
        else:
            from rl.milestone_env import MilestoneEnv
            self.env = MilestoneEnv(stage=stage, repeat=6, limit=180.)
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

    @property
    def world(self):
        return self.env.world if self.stage == 4 else self.env.unwrapped.world

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.score = 0.
        if self.stage == 4:
            return self.env.reset(), {}
        obs, info = self.env.reset(seed=seed, options=options)
        if self.stage == 2:
            decks = (("middle", 3), ("upper", 2), ("finish", 4))
            targets = {f"spring-{i}-0": tuple(f"{deck}-{j}" for j in range(count))
                       for i, (deck, count) in enumerate(decks, 1)}
            self.springs = SpringRewardTracker(self.world, targets)
        return obs, info

    def step(self, action, on_tick=None):
        if self.stage == 4:
            obs, points, done, info = self.env.step(action, on_tick=on_tick)
            self.score = info["reward_total"]
            return obs, points / 100., done, False, self.result(info["reason"])

        bonus = 0.

        def tick(world, _, direction):
            nonlocal bonus
            if self.stage == 2:
                bonus += self.springs.on_motion(world)
                for event, block in world.events:
                    if event == "land":
                        bonus += self.springs.on_landing(world, str(block))
            if on_tick:
                on_tick(world, direction)

        raw = self.env.unwrapped
        obs, reward, done, truncated, info = raw.step(action, on_tick=tick)
        if self.stage == 2:
            terminal = 100. if self.world.cleared else -25. if done else 0.
            self.score += terminal + bonus
            reward = (terminal + bonus) / 100.
        else:
            self.score = info["episode_return"]
        return self.env.observation(obs), reward, done, truncated, self.result()

    def result(self, reason=None):
        w = self.world
        return dict(success=bool(w.cleared), seconds=float(w.time), score=float(self.score),
                    reason=reason or ("success" if w.cleared else w.last_failure or "time_limit"))

    def close(self):
        if self.stage != 4:
            self.env.close()


class LiftPolicy(nn.Module):
    def __init__(self, obs_size, actions, width):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_size, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.actor = nn.Linear(width, actions)
        self.critic = nn.Linear(width, 1)

    def forward(self, observation):
        return self.actor(self.body(observation))


def load_probabilities(path, env, *, historical_lift=False):
    if historical_lift:
        # This pickle is accepted only after the worker verifies its pinned repository digest.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        dimensions = (env.observation_space.shape[0], env.action_space.n)
        if (checkpoint["obs_dim"], checkpoint["action_dim"]) != dimensions or checkpoint["version"] != OBS_VERSION:
            raise ValueError("Incompatible tutorial 4 checkpoint")
        model = LiftPolicy(*dimensions, checkpoint["width"])
        model.load_state_dict(checkpoint["model"])
        model.eval()

        @torch.inference_mode()
        def probabilities(obs):
            return model(torch.as_tensor(obs)).softmax(-1).numpy()

        return probabilities

    from rl.five_models import action_probs
    from rl.milestone_models import MilestonePPO
    from stable_baselines3 import PPO
    loader = PPO if env.stage == 4 else MilestonePPO
    model = loader.load(path, env=env, device="cpu")
    model.policy.set_training_mode(False)
    return lambda obs: action_probs(model, obs[None])[0]
