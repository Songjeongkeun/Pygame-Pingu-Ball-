"""3-action TEAM observation/physics with only stage 2/4 reward replacement.

No pretrained policies, BC, DAgger, or teacher actions. The team environment
performs all physics; on_tick merely reads native physics events and scores them.
"""
from __future__ import annotations
import gymnasium as gym
from rl.milestone_env import MilestoneEnv
from .reward_components import Stage2Reward, Stage4Reward


class RewardOnlyEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, stage: int, *, repeat: int = 12):
        super().__init__()
        if stage not in (2, 4):
            raise ValueError('Only stages 2/4 receive personal rewards')
        self.stage = int(stage)
        if repeat not in (6, 12):
            raise ValueError('Repeat must be 6 (control) or 12 (experiment)')
        self.repeat = repeat
        # The ONLY environment change: 6 vs 12 native physics ticks per action.
        self.team = MilestoneEnv(stage=self.stage, limit=180., repeat=repeat)
        self.observation_space = self.team.observation_space
        self.action_space = self.team.action_space
        if self.action_space.n != 3:
            raise AssertionError('Common 3-action specification changed')
        self.personal_episode_return = 0.
        self.personal_bonus_return = 0.
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.team.reset(seed=seed, options=options)
        world = self.team.unwrapped.world
        if self.stage == 2:
            self.scorer = Stage2Reward(world, first_spring_bonus=0.)
        else:
            from pingu_ball.special_lab import make_lab
            self.scorer = Stage4Reward(world, original_spawn_x=make_lab(4).spawn[0])
        self.personal_episode_return = 0.
        self.personal_bonus_return = 0.
        return obs, dict(info, reward_mode='tutorial_2_4_personal_only')

    def _on_tick(self, world, tick, direction):
        self._step_bonus += float(self.scorer.on_tick(world))

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid team action: {action}')
        self._step_bonus = 0.
        # ObservationWrapper.step() does not expose on_tick; invoke its raw
        # environment once and flatten using the ORIGINAL common adapter.
        raw_obs, _old_reward, done, truncated, info = self.team.unwrapped.step(
            int(action), on_tick=self._on_tick)
        obs = self.team.observation(raw_obs)
        world = self.team.unwrapped.world
        # Terminal base +100/-25 is identical in native and team reward rules.
        terminal = (100. if world.cleared else -25. if done else 0.)
        step_points = terminal + self._step_bonus
        self.personal_episode_return += step_points
        self.personal_bonus_return += self._step_bonus
        details = self.scorer.detail()
        # Preserve original team reward accounting for audits; PPO sees ONLY
        # the replaced reward. Never add old milestone points twice.
        original_return = float(info['episode_return'])
        original_bonus = float(info['bonus_return'])
        info = dict(info,
                    reward_mode='tutorial_2_4_personal_only',
                    original_team_return=original_return,
                    original_team_bonus_return=original_bonus,
                    bonus_return=self.personal_bonus_return,
                    episode_return=self.personal_episode_return,
                    learning_return=self.personal_episode_return/100.,
                    personal_bonus=self._step_bonus,
                    personal_terminal=terminal,
                    **details)
        return obs, float(step_points/100.), done, truncated, info

    def close(self):
        self.team.close()
