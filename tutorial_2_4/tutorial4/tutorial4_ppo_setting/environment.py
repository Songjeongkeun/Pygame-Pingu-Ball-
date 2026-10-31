from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


DURATIONS = (12, 30, 72)
ACTIONS = tuple((direction, frames) for direction in (-1, 0, 1) for frames in DURATIONS)
OBS_VERSION = 'stage4-v1-fixed-geometry-36'


@dataclass(frozen=True)
class RewardConfig:
    success: float = 100.
    fail_or_timeout: float = -25.
    milestone: float = 10.
    lift_first: float = 12.
    lift_high: float = 12.
    upper_first: float = 18.


    approach_per_pixel: float = 0.045
    approach_end_x: float = 384.
    lift_high_y: float = 334.


class Stage4Env:
    def __init__(self, reward: RewardConfig | None = None, max_seconds: float = 180.):
        from pingu_ball.engine import World
        from pingu_ball.special_lab import make_lab
        from pingu_ball.playfield import expand_playfield
        if max_seconds <= 0:
            raise ValueError('max_seconds must be positive')
        original = make_lab(4)
        self.original_spawn = float(original.spawn[0])
        self.world = World(expand_playfield(original), auto_reset=False)
        self.shift_x = self.world.level.spawn[0] - self.original_spawn
        self.reward = reward or RewardConfig()
        self.max_seconds = max_seconds
        self.upper_tiles = tuple(q for q in self.world.platforms if q.id.startswith('finish-'))
        self.start_tiles = tuple(q for q in self.world.platforms if q.id.startswith('start-'))
        assert len(self.upper_tiles) == 8 and len(self.start_tiles) == 4, 'Original tutorial-4 geometry changed: check Git version'
        assert 'lift' in self.world.special.platforms, 'Original tutorial-4 lift not found'
        self.reset()
        self.obs_dim = len(self.observe())

    def reset(self):
        w = self.world
        w.reset()
        self.done = False
        self.total_reward = 0.
        self.max_x = self.original_spawn
        self.highest_approach_x = self.original_spawn
        self.lift_first = False
        self.high_lift = False
        self.upper_first = False
        self.milestone = False
        self.last_lift_time = -1000.
        self.lift_contacts = 0
        self.action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
        self.last_dir = 0
        self.last_duration = 0
        self.last_landing_id = None
        self.rewards_paid = {'approach': 0., 'lift': 0., 'lift_high': 0., 'upper': 0., 'milestone': 0., 'terminal': 0.}
        return self.observe()

    def _lift(self):
        w = self.world
        return w.special.platforms['lift'].rect_at(w.time, w.physics)

    def _approach(self):

        x = min(float(self.world.ball.x - self.shift_x), self.reward.approach_end_x)
        if x <= self.highest_approach_x:
            return 0.
        gain = (x - self.highest_approach_x) * self.reward.approach_per_pixel
        self.highest_approach_x = x
        self.rewards_paid['approach'] += gain
        return gain

    def _landing(self, block: str):
        reward = 0.
        self.last_landing_id = block
        if block == 'lift':
            self.lift_contacts += 1
            self.last_lift_time = self.world.time
            if not self.lift_first:
                self.lift_first = True
                reward += self.reward.lift_first
                self.rewards_paid['lift'] = self.reward.lift_first
        elif block.startswith('finish-') and self.lift_first:
            if not self.milestone:
                self.milestone = True
                reward += self.reward.milestone
                self.rewards_paid['milestone'] = self.reward.milestone
            if not self.upper_first:
                self.upper_first = True
                reward += self.reward.upper_first
                self.rewards_paid['upper'] = self.reward.upper_first
        return reward

    def _lift_high_reward(self):
        if self.high_lift or not self.lift_first:
            return 0.
        w = self.world
        lx, ly, lw, _ = self._lift()
        b = w.ball


        close_x = lx - 12. <= b.x <= lx + lw + 12.
        close_y = ly - 130. <= b.y <= ly + 18.
        recently_lift = w.time - self.last_lift_time <= 1.05
        if ly <= self.reward.lift_high_y and close_x and close_y and recently_lift:
            self.high_lift = True
            self.rewards_paid['lift_high'] = self.reward.lift_high
            return self.reward.lift_high
        return 0.

    def step(self, action: int):
        if self.done:
            raise RuntimeError('Episode finished: call reset()')
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(ACTIONS):
            raise ValueError(f'Action index 0..{len(ACTIONS)-1} expected')
        w = self.world
        direction, duration = ACTIONS[int(action)]
        self.action_counts[int(action)] += 1
        self.last_dir, self.last_duration = direction, duration
        r = 0.
        nframes = 0
        for _ in range(duration):
            if w.failed or w.cleared or w.time >= self.max_seconds - 1e-9:
                break
            w.step(direction)
            nframes += 1
            self.max_x = max(self.max_x, float(w.ball.x - self.shift_x))
            r += self._approach()

            for kind, value in w.events:
                if kind == 'land':
                    r += self._landing(str(value))
            r += self._lift_high_reward()
            if w.failed or w.cleared:
                break
        timeout = bool(not w.failed and not w.cleared and w.time >= self.max_seconds - 1e-9)
        self.done = bool(w.failed or w.cleared or timeout)
        if self.done:
            end_r = self.reward.success if w.cleared else self.reward.fail_or_timeout
            r += end_r
            self.rewards_paid['terminal'] = end_r
        self.total_reward += r
        info = {
            'success': bool(w.cleared),
            'timeout': timeout,
            'failed': bool(w.failed),
            'reason': ('success' if w.cleared else 'time_limit' if timeout else w.last_failure if w.failed else ''),
            'seconds': float(w.time),
            'max_x': float(self.max_x),
            'lift': int(self.lift_first),
            'lift_contacts': int(self.lift_contacts),
            'lift_high': int(self.high_lift),
            'upper': int(self.upper_first),
            'milestone': int(self.milestone),
            'frames': nframes,
            'reward_total': float(self.total_reward),
            'reward_parts': dict(self.rewards_paid),
            'actions': self.action_counts.tolist(),
        }
        return self.observe(), float(r), self.done, info

    def observe(self):
        w, b, p = self.world, self.world.ball, self.world.physics
        lx, ly, lw, lh = self._lift()
        probe = w.special.platforms['lift'].rect_at(w.time + 0.02, p)
        lift_vy = (probe[1] - ly) / 0.02
        ux = min(q.x for q in self.upper_tiles)
        ur = max(q.x + q.w for q in self.upper_tiles)
        us = self.upper_tiles[0].y
        sx = min(q.x for q in self.start_tiles)
        sr = max(q.x + q.w for q in self.start_tiles)
        gx, gy, gw, gh = w.level.goal
        f = lambda v, scale: float(np.clip(v / scale, -4., 4.))
        last = self.last_landing_id or ''


        obs = [
            f(b.x-self.shift_x, 1088.), f(b.y, 640.),
            f(b.vx, p.max_speed), f(b.vy, p.bounce_speed),
            f(gx+gw/2-b.x, 1088.), f(gy+gh/2-b.y, 640.),
            f(lx+lw/2-b.x, 256.), f(ly-b.y, 192.),
            f(lx-b.x, 128.), f(lx+lw-b.x, 128.),
            f(lift_vy, 64.), f(ly-256., 192.),
            f(ux-b.x, 256.), f(ur-b.x, 1088.), f(us-b.y, 192.),
            f(sx-b.x, 256.), f(sr-b.x, 256.),
            f((gx+gw/2)-(ux+ur)/2, 1088.),
            f(w.time, self.max_seconds),
            math.sin(w.time*math.pi/3.), math.cos(w.time*math.pi/3.),
            f((ly-b.y)-p.radius, 192.),
            float(b.vy < 0), float(b.vy >= 0),
            float(last.startswith('start-')), float(last == 'lift'),
            float(last.startswith('finish-')),
            float(self.lift_first), float(self.high_lift),
            float(self.milestone), float(self.upper_first),
            f(w.time-self.last_lift_time, 10.) if self.lift_first else 1.,
            float(self.last_dir), f(self.last_duration, 72.),
            f(self.max_x-(b.x-self.shift_x), 256.),
            f(self.highest_approach_x-(b.x-self.shift_x), 256.),
        ]
        return np.asarray(obs, dtype=np.float32)
