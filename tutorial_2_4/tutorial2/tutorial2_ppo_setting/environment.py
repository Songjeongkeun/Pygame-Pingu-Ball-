from __future__ import annotations
from dataclasses import dataclass
from .spring_rewards import SpringRewardTracker, SpringRewards
import numpy as np

DURATIONS = (12, 30, 72)
ACTIONS = tuple((direction, frames) for direction in (-1, 0, 1) for frames in DURATIONS)
OBS_VERSION = 'tutorial2-spring-target-milestone-40-team-v1'
SPRINGS = ('spring-1-0', 'spring-2-0', 'spring-3-0')
DECKS = ('middle-', 'upper-', 'finish-')


@dataclass(frozen=True)
class RewardConfig:
    success: float = 100.
    fail_or_timeout: float = -25.
    target_landing: float = 10.
    target_height: float = 0.5
    target_horizontal: float = 0.5
    vertical_tolerance: float = 16.
    flight_window_seconds: float = 3.


class Stage2Env:
    def __init__(self, reward: RewardConfig | None = None, max_seconds: float = 180.):
        from pingu_ball.engine import World
        from pingu_ball.special_lab import make_lab
        from pingu_ball.playfield import expand_playfield
        if max_seconds <= 0:
            raise ValueError('max_seconds must be positive')
        original = make_lab(2)
        self.original_spawn = float(original.spawn[0])
        self.shift_x = 0.
        self.world = World(expand_playfield(original), auto_reset=False)
        self.shift_x = float(self.world.level.spawn[0]) - self.original_spawn
        self.reward = reward or RewardConfig()


        valid_targets = {
            'spring-1-0': tuple(f'middle-{i}' for i in range(3)),
            'spring-2-0': tuple(f'upper-{i}' for i in range(2)),
            'spring-3-0': tuple(f'finish-{i}' for i in range(4)),
        }
        self.spring_tracker = SpringRewardTracker(
            self.world, valid_targets, SpringRewards(
                target_landing=self.reward.target_landing,
                target_height=self.reward.target_height,
                target_horizontal=self.reward.target_horizontal,
                vertical_tolerance=self.reward.vertical_tolerance,
                flight_window_seconds=self.reward.flight_window_seconds,
            ))
        self.max_seconds = float(max_seconds)
        self.springs = {}
        for key in SPRINGS:
            matches = [q for q in self.world.platforms if q.id == key and q.kind == 'spring']
            if len(matches) != 1:
                raise RuntimeError(f'Tutorial 2 spring geometry changed: expected {key} kind=spring')
            self.springs[key] = matches[0]
        self.decks = {}
        for prefix, expected in [('middle-', 3), ('upper-', 2), ('finish-', 4)]:
            deck = tuple(q for q in self.world.platforms if q.id.startswith(prefix))
            if len(deck) != expected:
                raise RuntimeError(f'Tutorial 2 deck geometry changed: {prefix} expected {expected}')
            self.decks[prefix] = deck
        self.start_tiles = tuple(q for q in self.world.platforms if q.id.startswith('start-'))
        if len(self.start_tiles) != 4:
            raise RuntimeError('Tutorial 2 start geometry changed')
        self.reset()
        self.obs_dim = len(self.observe())
        if self.obs_dim != 40:
            raise AssertionError(f'Stage2 obs mismatch: {self.obs_dim} != 40')

    def reset(self):
        w = self.world
        w.reset()
        self.done = False
        self.total_reward = 0.
        self.max_x = self.original_spawn
        self.spring_tracker.reset()
        self.spring_seen = [False, False, False]


        self.deck_seen = [False, False, False]
        self.last_landing_id = None
        self.action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
        self.last_dir, self.last_duration = 0, 0
        self.rewards_paid = {
            'landing1': 0., 'landing2': 0., 'landing3': 0.,
            'height1': 0., 'height2': 0., 'height3': 0.,
            'horizontal1': 0., 'horizontal2': 0., 'horizontal3': 0.,
            'terminal': 0.,
        }
        return self.observe()

    def _landing(self, block_id: str):
        self.last_landing_id = block_id
        earned = self.spring_tracker.on_landing(self.world, block_id)
        for i, spring in enumerate(SPRINGS):
            self.spring_seen[i] = spring in self.spring_tracker.target_paid or self.spring_seen[i]

            if block_id == spring:
                self.spring_seen[i] = True
            self.rewards_paid[f'landing{i+1}'] = self.spring_tracker.events_paid.get(('landing',spring), 0.)
            self.rewards_paid[f'height{i+1}'] = self.spring_tracker.events_paid.get(('height',spring), 0.)
            self.rewards_paid[f'horizontal{i+1}'] = self.spring_tracker.events_paid.get(('horizontal',spring), 0.)
        for i, prefix in enumerate(DECKS):
            if block_id.startswith(prefix):
                self.deck_seen[i] = True
        return earned

    def step(self, action: int):
        if self.done:
            raise RuntimeError('Episode finished; reset first')
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < len(ACTIONS):
            raise ValueError('Action index 0..8 expected')
        w = self.world
        direction, duration = ACTIONS[int(action)]
        self.action_counts[int(action)] += 1
        self.last_dir, self.last_duration = direction, duration
        reward = 0.
        nframes = 0
        for _ in range(duration):
            if w.failed or w.cleared or w.time >= self.max_seconds - 1e-9:
                break
            w.step(direction)
            nframes += 1
            self.max_x = max(self.max_x, float(w.ball.x - self.shift_x))
            reward += self.spring_tracker.on_motion(w)

            for kind, value in w.events:
                if kind == 'land':
                    reward += self._landing(str(value))
            if w.failed or w.cleared:
                break
        timeout = bool(not w.failed and not w.cleared and w.time >= self.max_seconds - 1e-9)
        self.done = bool(w.failed or w.cleared or timeout)
        if self.done:
            terminal = self.reward.success if w.cleared else self.reward.fail_or_timeout
            reward += terminal
            self.rewards_paid['terminal'] = terminal
        self.total_reward += reward
        info = {
            'success': bool(w.cleared), 'failed': bool(w.failed), 'timeout': timeout,
            'reason': 'success' if w.cleared else 'time_limit' if timeout else w.last_failure if w.failed else '',
            'seconds': float(w.time), 'max_x': float(self.max_x),
            'spring1': int(self.spring_seen[0]), 'spring2': int(self.spring_seen[1]),
            'spring3': int(self.spring_seen[2]),
            'middle': int(self.deck_seen[0]), 'upper': int(self.deck_seen[1]),
            'finish': int(self.deck_seen[2]),
            'milestone': int(sum(self.deck_seen)),
            'target_landing_count': int(len(self.spring_tracker.target_paid)),
            'height_count': int(len(self.spring_tracker.height_paid)),
            'horizontal_count': int(len(self.spring_tracker.horizontal_paid)),
            'frames': nframes, 'reward_total': float(self.total_reward),
            'reward_parts': dict(self.rewards_paid), 'actions': self.action_counts.tolist(),
        }
        return self.observe(), float(reward), self.done, info

    def observe(self):
        w, b, p = self.world, self.world.ball, self.world.physics
        gx, gy, gw, gh = w.level.goal
        f = lambda v, scale: float(np.clip(v / scale, -4., 4.))
        last = self.last_landing_id or ''


        obs = [
            f(b.x-self.shift_x, 1600.), f(b.y, 900.),
            f(b.vx, p.max_speed), f(b.vy, p.bounce_speed),
            f(gx+gw/2-b.x, 1600.), f(gy+gh/2-b.y, 900.),
        ]
        for key in SPRINGS:
            pad = self.springs[key]
            obs.extend((f(pad.x+pad.w/2-b.x, 400.), f(pad.y-b.y, 256.)))
        for name in DECKS:
            deck = self.decks[name]
            left = min(q.x for q in deck)
            right = max(q.x+q.w for q in deck)
            obs.extend((f(left-b.x, 400.), f(right-b.x, 400.), f(deck[0].y-b.y, 256.)))
        for key in SPRINGS:
            pad = self.springs[key]
            obs.append(f(pad.x-b.x, 256.))
        obs.extend([
            f(w.time, self.max_seconds),
            float(b.vy < 0.), float(b.vy >= 0.),
            float(last.startswith('start-')),
            float(last in SPRINGS),
            float(last.startswith('middle-')),
            float(last.startswith('upper-') or last.startswith('finish-')),
            *(float(v) for v in self.spring_seen),
            *(float(v) for v in self.deck_seen),
            float(self.last_dir), f(self.last_duration, 72.),
            f(self.max_x - (b.x-self.shift_x), 512.),
        ])
        return np.asarray(obs, dtype=np.float32)
