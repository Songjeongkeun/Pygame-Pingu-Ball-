from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping, Iterable


@dataclass(frozen=True)
class SpringRewards:
    target_landing: float = 10.0
    target_height: float = 0.5
    target_horizontal: float = 0.5
    vertical_tolerance: float = 16.0
    flight_window_seconds: float = 3.0


class SpringRewardTracker:
    def __init__(self, world, targets_by_spring: Mapping[str, Iterable[str]], config: SpringRewards | None = None):
        self.config = config or SpringRewards()
        self.targets = {str(source): frozenset(map(str, targets)) for source, targets in targets_by_spring.items()}
        platforms = {p.id: p for p in world.platforms}
        self.platform_indices = {p.id: i for i, p in enumerate(world.platforms)}
        if not self.targets:
            raise ValueError('Provide valid spring -> destination-platform mappings')
        for source, dests in self.targets.items():
            if source not in platforms or platforms[source].kind != 'spring':
                raise ValueError(f'Unknown or non-spring source: {source}')
            if not dests or source in dests or not dests.issubset(platforms):
                raise ValueError(f'Invalid target set for {source}: {sorted(dests)}')
        if self.config.flight_window_seconds <= 0 or self.config.vertical_tolerance < 0:
            raise ValueError('Invalid spring reward timing/geometry')
        self.reset()

    def reset(self):
        self.active_source = None
        self.active_since = -1.
        self.height_inside = {}
        self.horizontal_inside = {}
        self.target_paid = set()
        self.height_paid = set()
        self.horizontal_paid = set()
        self.parts = {'target_landing': 0., 'target_height': 0., 'target_horizontal': 0.}
        self.events_paid = {}

    def _position(self, world, platform_id):


        idx = self.platform_indices[platform_id]
        pad = world.platforms[idx]
        if pad.id != platform_id:
            raise RuntimeError('Platform ordering changed; rebuild SpringRewardTracker from current World')
        return pad.rect_at(world.time, world.physics)

    def _is_height_inside(self, world, platform_id):
        _x, y, _w, _h = self._position(world, platform_id)
        ball_bottom = world.ball.y + world.physics.radius
        return abs(ball_bottom - y) <= self.config.vertical_tolerance

    def _is_horizontal_inside(self, world, platform_id):
        x, _y, width, _h = self._position(world, platform_id)
        return world.ball.x + world.physics.radius >= x and world.ball.x - world.physics.radius <= x + width

    def on_landing(self, world, platform_id: str) -> float:
        """Call for every ('land', platform_id) event, *after* on_motion per tick."""
        time = float(world.time)
        earned = 0.
        if (self.active_source is not None
                and time - self.active_since <= self.config.flight_window_seconds
                and platform_id in self.targets[self.active_source]
                and self.active_source not in self.target_paid):
            self.target_paid.add(self.active_source)
            earned += self.config.target_landing
            self.parts['target_landing'] += self.config.target_landing
            self.events_paid[('landing', self.active_source)] = self.config.target_landing

        if platform_id in self.targets:
            self.active_source = platform_id
            self.active_since = time


            dests = self.targets[platform_id]
            self.height_inside = {d:self._is_height_inside(world,d) for d in dests}
            self.horizontal_inside = {d:self._is_horizontal_inside(world,d) for d in dests}
        else:
            self.active_source = None
            self.height_inside = {}
            self.horizontal_inside = {}
        return earned

    def on_motion(self, world) -> float:
        """Call every physics tick AFTER world.step and BEFORE processing events."""
        if world.failed or world.cleared:
            return 0.
        source = self.active_source
        if source is None:
            return 0.
        if world.time - self.active_since > self.config.flight_window_seconds:
            self.active_source = None
            return 0.
        earned = 0.
        for dest in self.targets[source]:
            h = self._is_height_inside(world,dest)
            x = self._is_horizontal_inside(world,dest)
            if h and not self.height_inside.get(dest,False) and source not in self.height_paid:
                self.height_paid.add(source)
                earned += self.config.target_height
                self.parts['target_height'] += self.config.target_height
                self.events_paid[('height',source)] = self.config.target_height
            if x and not self.horizontal_inside.get(dest,False) and source not in self.horizontal_paid:
                self.horizontal_paid.add(source)
                earned += self.config.target_horizontal
                self.parts['target_horizontal'] += self.config.target_horizontal
                self.events_paid[('horizontal',source)] = self.config.target_horizontal
            self.height_inside[dest] = h
            self.horizontal_inside[dest] = x
        return earned
