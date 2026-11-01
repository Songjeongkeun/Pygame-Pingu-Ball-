"""팀원 공통 관측·행동·물리·PPO용 환경을 변경하지 않고 추가 보상만 합산."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rl.milestone_env import MilestoneRawEnv, OBS_SIZE
from .portable_rewards_v4 import PortableBonusTracker, BonusConfig


class RewardRawEnv(MilestoneRawEnv):
    def __init__(self, stage=2, reward='combined', limit=180., repeat=6):
        if stage not in range(1, 6):
            raise ValueError('기존 팀원 공통 환경은 튜토리얼 1~5단계만 지원합니다.')
        if reward not in ('common', 'combined', 'guided'):
            raise ValueError('reward는 common/combined/guided 중 하나입니다.')
        self.bonus_mode = reward
        super().__init__(stage=stage, reward_mode='milestone', limit=limit, repeat=repeat)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.bonus_tracker = PortableBonusTracker(self, BonusConfig(
            progress_per_pixel=.045 if self.bonus_mode == 'guided' else 0.))
        return obs, self.info()

    def step(self, action, on_tick=None):
        if self.bonus_mode == 'common':
            obs, reward, terminated, truncated, info = super().step(action, on_tick=on_tick)
            info.update(personal_reward=0., personal_total=0., personal_counts={k:0 for k in ('spring_height','spring_horizontal','moving_first','moving_high','progress','progress_reversal')})
            return obs, reward, terminated, truncated, info

        start = self.ticks
        earned = []
        def callback(world, ticks, direction):
            earned.extend(self.bonus_tracker.on_tick(world))
            if on_tick is not None:
                on_tick(world, ticks, direction)
        obs, common_reward, terminated, truncated, _ = super().step(action, on_tick=callback)
        if terminated or truncated:
            earned.extend(self.bonus_tracker.on_terminal(self.world))
        extra = float(sum(ev['points'] for ev in earned))
        if extra:
            self.bonus_return += extra
            self.episode_return += extra
            self.reward_events.extend(earned)
        info = self.info(self.ticks - start)
        info.update(personal_reward=extra, personal_total=float(self.bonus_tracker.total),
                    personal_counts=dict(self.bonus_tracker.counts))
        return obs, common_reward + extra / 100., terminated, truncated, info


class RewardEnv(gym.ObservationWrapper):
    def __init__(self, **options):
        super().__init__(RewardRawEnv(**options))
        self.observation_space = spaces.Box(-8., 8., (OBS_SIZE,), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate((obs['global'], obs['entities'].reshape(-1),
                               obs['local'].reshape(-1))).astype(np.float32)
