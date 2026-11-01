"""Faithful stage-specific reward calculators from tutorial_2_4.

This module has no Gym/SB3 dependency and changes no game physics, observations,
or actions. Called once per original 120-Hz physics tick.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage4Settings:
    lift_first: float = 12.
    lift_high: float = 12.
    milestone: float = 10.
    upper_first: float = 18.
    approach_per_pixel: float = .045
    approach_end_x: float = 384.
    lift_high_y: float = 334.


class Stage4Reward:
    """4th tutorial reward semantics from tutorial4_ppo_setting.environment.Stage4Env."""
    def __init__(self, world, *, original_spawn_x=96., settings=None):
        self.settings = settings or Stage4Settings()
        self.shift_x = float(world.level.spawn[0]) - float(original_spawn_x)
        self.reset(original_spawn_x)

    def reset(self, original_spawn_x=96.):
        self.lift_first = False
        self.high_lift = False
        self.upper_first = False
        self.milestone = False
        self.last_lift_time = -1000.
        self.lift_contacts = 0
        self.highest_approach_x = float(original_spawn_x)
        self.max_x = float(original_spawn_x)
        self.parts = {'approach': 0., 'lift': 0., 'lift_high': 0.,
                      'upper': 0., 'milestone': 0.}

    def on_tick(self, world):
        """Same order as native Stage4Env: approach -> landing -> high-lift."""
        s = self.settings
        w = world
        x = float(w.ball.x - self.shift_x)
        self.max_x = max(self.max_x, x)
        earned = 0.
        next_x = min(x, s.approach_end_x)
        if next_x > self.highest_approach_x:
            gain = (next_x - self.highest_approach_x) * s.approach_per_pixel
            self.highest_approach_x = next_x
            self.parts['approach'] += gain
            earned += gain
        for kind, platform_id in w.events:
            if kind != 'land':
                continue
            if platform_id == 'lift':
                self.lift_contacts += 1
                self.last_lift_time = w.time
                if not self.lift_first:
                    self.lift_first = True
                    earned += s.lift_first
                    self.parts['lift'] = s.lift_first
            elif str(platform_id).startswith('finish-') and self.lift_first:
                if not self.milestone:
                    self.milestone = True
                    earned += s.milestone
                    self.parts['milestone'] = s.milestone
                if not self.upper_first:
                    self.upper_first = True
                    earned += s.upper_first
                    self.parts['upper'] = s.upper_first
        if self.lift_first and not self.high_lift:
            lx, ly, lw, _ = w.special.platforms['lift'].rect_at(w.time, w.physics)
            b = w.ball
            close_x = lx-12. <= b.x <= lx+lw+12.
            close_y = ly-130. <= b.y <= ly+18.
            recent = w.time-self.last_lift_time <= 1.05
            if ly <= s.lift_high_y and close_x and close_y and recent:
                self.high_lift = True
                earned += s.lift_high
                self.parts['lift_high'] = s.lift_high
        return earned

    def detail(self):
        return {'lift': int(self.lift_first), 'lift_high': int(self.high_lift),
                'upper': int(self.upper_first), 'milestone': int(self.milestone),
                'lift_contacts': int(self.lift_contacts), 'max_x': float(self.max_x),
                'reward_parts': dict(self.parts)}


class Stage2Reward:
    """Reuse EXACT native jump-reward tracker; do not reimplement it."""
    def __init__(self, world, *, first_spring_bonus=0.):
        if first_spring_bonus < 0:
            raise ValueError("first_spring_bonus must be nonnegative")
        self.first_spring_bonus = float(first_spring_bonus)
        self.first_spring_paid = False
        self.spring1_contacts = 0
        # Imported on demand so tests for Stage4Reward need no app imports.
        import sys
        from pathlib import Path
        tutorial2 = Path(__file__).resolve().parents[2] / 'tutorial_2_4' / 'tutorial2'
        if not tutorial2.is_dir():
            raise FileNotFoundError(f'Missing original tutorial 2 reward code: {tutorial2}')
        if str(tutorial2) not in sys.path:
            sys.path.insert(0, str(tutorial2))
        from tutorial2_ppo_setting.spring_rewards import SpringRewardTracker, SpringRewards
        from pingu_ball.special_lab import make_lab
        self.original_spawn_x = float(make_lab(2).spawn[0])
        self.shift_x = float(world.level.spawn[0]) - self.original_spawn_x
        mapping = {
            'spring-1-0': tuple(f'middle-{i}' for i in range(3)),
            'spring-2-0': tuple(f'upper-{i}' for i in range(2)),
            'spring-3-0': tuple(f'finish-{i}' for i in range(4)),
        }
        self.tracker = SpringRewardTracker(world, mapping, SpringRewards(
            target_landing=10., target_height=.5, target_horizontal=.5,
            vertical_tolerance=16., flight_window_seconds=3.))
        self.max_x = self.original_spawn_x
        self.spring_seen = set()
        self.deck_seen = set()

    def on_tick(self, world):
        self.max_x = max(self.max_x, float(world.ball.x - self.shift_x))
        # EXACT Stage2Env ordering: motion before handling each native landing.
        earned = self.tracker.on_motion(world)
        for kind, value in world.events:
            if kind == 'land':
                earned += self.tracker.on_landing(world, str(value))
                if str(value).startswith('spring-'):
                    self.spring_seen.add(str(value))
                if str(value) == 'spring-1-0':
                    self.spring1_contacts += 1
                    if not self.first_spring_paid:
                        self.first_spring_paid = True
                        if self.first_spring_bonus:
                            earned += self.first_spring_bonus
                            self.tracker.parts['first_spring_bonus'] = self.first_spring_bonus
                for prefix in ('middle-', 'upper-', 'finish-'):
                    if str(value).startswith(prefix):
                        self.deck_seen.add(prefix)
        return earned

    def detail(self):
        return {'target_landing_count': len(self.tracker.target_paid),
                'height_count': len(self.tracker.height_paid),
                'horizontal_count': len(self.tracker.horizontal_paid),
                'springs_seen': len(self.spring_seen), 'first_spring': int(self.first_spring_paid),
                'spring2': int('spring-2-0' in self.spring_seen),
                'spring3': int('spring-3-0' in self.spring_seen),
                'middle': int('middle-' in self.deck_seen),
                'upper': int('upper-' in self.deck_seen),
                'finish': int('finish-' in self.deck_seen),
                'spring1_contacts': int(self.spring1_contacts), 'decks_seen': len(self.deck_seen),
                'max_x': float(self.max_x),
                'reward_parts': dict(self.tracker.parts)}
