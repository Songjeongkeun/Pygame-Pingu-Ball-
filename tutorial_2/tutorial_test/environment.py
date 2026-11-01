"""원래 팀 공통 마일스톤 환경에 개인 보상을 선택적으로 합성한다."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rl.milestone_env import MilestoneRawEnv, OBS_SIZE
from .personal_rewards import PersonalRewardTracker


class CombinedRawEnv(MilestoneRawEnv):
    def __init__(self, stage=2, reward='combined', limit=180.0, repeat=6):
        if stage not in (2, 4):
            raise ValueError('이 실험은 튜토리얼 2·4단계만 지원합니다.')
        if reward not in ('common', 'combined'):
            raise ValueError('reward는 common 또는 combined여야 합니다.')
        self.personal_mode = reward
        super().__init__(stage=stage, reward_mode='milestone', limit=limit, repeat=repeat)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.personal_tracker = PersonalRewardTracker(self)
        return self.observe(), self.info()

    def step(self, action, on_tick=None):
        if self.personal_mode == 'common':
            observation, reward, done, truncated, info = super().step(action, on_tick=on_tick)
            info['personal_reward'] = 0.0
            info['personal_total'] = 0.0
            return observation, reward, done, truncated, info

        tick_start = self.ticks
        extra_events = []

        def tick_callback(world, ticks, direction):
            extra_events.extend(self.personal_tracker.on_tick(world))
            if on_tick is not None:
                on_tick(world, ticks, direction)

        observation, common_reward, done, truncated, _ = super().step(action, on_tick=tick_callback)
        extra_points = sum(event['points'] for event in extra_events)
        if extra_points:
            self.bonus_return += extra_points
            self.episode_return += extra_points
            self.reward_events.extend(extra_events)
        info = self.info(self.ticks - tick_start)
        info['personal_reward'] = extra_points
        info['personal_total'] = self.personal_tracker.total
        return observation, common_reward + extra_points / 100.0, done, truncated, info


class CombinedEnv(gym.ObservationWrapper):
    def __init__(self, stage=2, reward='combined', limit=180.0, repeat=6):
        super().__init__(CombinedRawEnv(stage=stage, reward=reward, limit=limit, repeat=repeat))
        self.observation_space = spaces.Box(-8.0, 8.0, (OBS_SIZE,), dtype=np.float32)

    def observation(self, observation):
        return np.concatenate([
            observation['global'],
            observation['entities'].reshape(-1),
            observation['local'].reshape(-1),
        ]).astype(np.float32)
