"""개선 실험 v2: 가로 보상은 목표 높이와 가로 영역이 동시에 맞을 때만 지급한다."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonalRewardSettings:
    target_height: float = 0.5
    target_horizontal: float = 0.5
    vertical_tolerance: float = 16.0
    spring_window_seconds: float = 3.0
    lift_first: float = 12.0
    lift_high: float = 12.0
    lift_high_fraction: float = 0.60
    lift_contact_window_seconds: float = 1.05


class PersonalRewardTracker:
    def __init__(self, env, settings: PersonalRewardSettings | None = None):
        self.env = env
        self.settings = settings or PersonalRewardSettings()
        self.stage = env.stage
        self.source_specs = {}
        self.lift_sources = set()
        for spec in env.specs:
            for source in spec.source:
                device = env.world.special.devices.get(source)
                platform = env.platform_by_id[source]
                if self.stage == 2 and platform.kind == 'spring':
                    self.source_specs[source] = spec
                if self.stage == 4 and device is not None and device.type_id == 508:
                    self.lift_sources.add(source)
        if self.stage == 2 and not self.source_specs:
            raise ValueError('공통 마일스톤에 점프 블록 출발 발판이 없습니다.')
        if self.stage == 4 and not self.lift_sources:
            raise ValueError('공통 마일스톤에 상하 이동 블록 출발 발판이 없습니다.')
        self.reset()

    def reset(self):
        self.active = None
        self.active_since = -1.0
        self.height_inside = {}
        self.horizontal_inside = {}
        self.height_paid = set()
        self.horizontal_paid = set()
        self.lift_first_paid = set()
        self.lift_high_paid = set()
        self.last_lift_contact = {}
        self.total = 0.0
        self.events = []

    def _rect(self, platform_id):
        return self.env.bounds((platform_id,))

    def _inside(self, world, platform_id):
        x, y, width, _ = self._rect(platform_id)
        ball = world.ball
        radius = world.physics.radius
        near_height = abs(ball.y + radius - y) <= self.settings.vertical_tolerance
        near_horizontal = ball.x + radius >= x and ball.x - radius <= x + width
        return near_height, near_horizontal

    def _pay(self, kind, source, amount, world):
        event = {'kind': kind, 'source': source, 'points': float(amount), 'time': float(world.time)}
        self.total += float(amount)
        self.events.append(event)
        return event

    def _spring_motion(self, world):
        source = self.active
        if source is None:
            return []
        if world.time - self.active_since > self.settings.spring_window_seconds:
            self.active = None
            return []
        out = []
        for target in self.source_specs[source].target:
            height, horizontal = self._inside(world, target)
            if height and not self.height_inside.get(target, False) and source not in self.height_paid:
                self.height_paid.add(source)
                out.append(self._pay('target_height', source, self.settings.target_height, world))
            # v2: 높이와 상관없이 X만 겹치는 보상을 막는다.
            # '동시 정렬' 상태의 진입을 추적하므로 X가 먼저 맞은 후 높이가 맞아도 지급된다.
            aligned = height and horizontal
            if aligned and not self.horizontal_inside.get(target, False) and source not in self.horizontal_paid:
                self.horizontal_paid.add(source)
                out.append(self._pay('target_horizontal', source, self.settings.target_horizontal, world))
            self.height_inside[target] = height
            self.horizontal_inside[target] = aligned
        return out

    def _spring_land(self, world, platform_id):
        if platform_id in self.source_specs:
            self.active = platform_id
            self.active_since = float(world.time)
            self.height_inside = {}
            self.horizontal_inside = {}
            for target in self.source_specs[platform_id].target:
                h, x = self._inside(world, target)
                self.height_inside[target] = h
                self.horizontal_inside[target] = h and x
        else:
            self.active = None
            self.height_inside = {}
            self.horizontal_inside = {}

    def _lift_motion(self, world):
        out = []
        b = world.ball
        for source in self.lift_sources:
            if source not in self.lift_first_paid or source in self.lift_high_paid:
                continue
            if world.time - self.last_lift_contact.get(source, -1e9) > self.settings.lift_contact_window_seconds:
                continue
            device = world.special.devices[source]
            travel = float(device.options.get('travel', 0.0))
            if travel <= 0:
                continue
            x, y, width, _ = self._rect(source)
            # 절대 좌표 대신 발판의 실제 이동 구간에서 얼마나 올라갔는지 계산한다.
            ascent = (float(device.y) + travel - y) / travel
            near_x = x - 12.0 <= b.x <= x + width + 12.0
            near_y = y - 1.4 * world.physics.bounce_height <= b.y <= y + world.physics.radius + 4.0
            if ascent >= self.settings.lift_high_fraction and near_x and near_y:
                self.lift_high_paid.add(source)
                out.append(self._pay('lift_high', source, self.settings.lift_high, world))
        return out

    def _lift_land(self, world, platform_id):
        if platform_id not in self.lift_sources:
            return []
        self.last_lift_contact[platform_id] = float(world.time)
        if platform_id in self.lift_first_paid:
            return []
        self.lift_first_paid.add(platform_id)
        return [self._pay('lift_first', platform_id, self.settings.lift_first, world)]

    def on_tick(self, world):
        if world.failed:
            return []
        if self.stage == 2:
            out = self._spring_motion(world)
            for event, platform_id in world.events:
                if event == 'land':
                    self._spring_land(world, str(platform_id))
            return out
        out = self._lift_motion(world)
        for event, platform_id in world.events:
            if event == 'land':
                out.extend(self._lift_land(world, str(platform_id)))
        return out
