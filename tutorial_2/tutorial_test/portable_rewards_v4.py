"""공통 1~5단계 환경의 물리를 변경하지 않는 보조 보상(실험용).

원본과 동일한 의미:
- 점프 발판 출발 -> 연계된 목표 발판의 높이 ±16px 최초 진입 +0.5;
  연계된 목표 발판의 x 범위 최초 겹침 +0.5. 둘은 독립 이벤트.
- 상하 이동 발판 최초 착지 +12; 그 발판을 이용한 충분한 상승 +12.
- 공통 환경의 목표 발판 '최초 착지 +10'은 여기서 절대로 중복 지급하지 않음.
제외: 오른쪽 방향으로 전진하는 매 픽셀 보상, 특정 결승 발판 +18,
      맵 번호/고정 좌표에 연결된 보상, 특정 스테이지 전용 관측값.

'일반화'는 발판 유형과 마일스톤 연결을 참조한다는 뜻이며, 새 맵 성능을 보장하지 않는다.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BonusConfig:
    spring_height: float = 0.5
    spring_horizontal: float = 0.5
    vertical_tolerance: float = 16.0
    spring_window_seconds: float = 3.0
    moving_first: float = 12.0
    moving_high: float = 12.0
    moving_high_fraction: float = 0.60
    moving_contact_window: float = 1.05
    # stage 숫자가 아니라 환경에 spring 또는 508 리프트가 있을 때 활성화.
    # 에피소드에서 새로 도달한 최대 x만 가점하며 왕복/정지로 중복 보상 불가.
    progress_per_pixel: float = 0.0
    progress_total_cap: float = 36.0


class PortableBonusTracker:
    """스테이지 번호 대신 공통 마일스톤 출발/목표 발판 ID와 종류로 식별."""

    def __init__(self, env, settings: BonusConfig | None = None):
        self.env = env
        self.settings = settings or BonusConfig()
        self.spring_specs = {}
        self.vertical_sources = set()
        for spec in env.specs:
            for source in spec.source:
                platform = env.platform_by_id[source]
                device = env.world.special.devices.get(source)
                if platform.kind == 'spring':
                    self.spring_specs[source] = spec
                if device is not None and device.type_id == 508:
                    self.vertical_sources.add(source)
        self.progress_capable = bool(self.spring_specs or self.vertical_sources)
        self.reset()

    def reset(self):
        self.active_spring = None
        self.active_since = -1e9
        self.was_in_height = {}
        self.was_in_x = {}
        self.height_paid = set()
        self.x_paid = set()
        self.moving_first_paid = set()
        self.moving_high_paid = set()
        self.moving_contact_time = {}
        self.counts = {k: 0 for k in ('spring_height','spring_horizontal','moving_first','moving_high','progress','progress_reversal')}
        self.total = 0.0
        # 게임이 좌표계를 이동한 경우도 원본 맵 값 대신 현재 World 좌표 사용.
        world = self.env.world
        self.progress_max_x = float(world.ball.x)
        level = getattr(world, 'level', None)
        goal = getattr(level, 'goal', None)
        self.progress_goal_x = (float(goal[0]) + float(goal[2])*0.5) if goal is not None else None
        self.progress_paid = 0.0
        self.progress_balanced = False

    def _rect(self, pid):
        return self.env.bounds((pid,))

    def _alignment(self, world, pid):
        x, y, w, _ = self._rect(pid)
        b, radius = world.ball, world.physics.radius
        height = abs(b.y + radius - y) <= self.settings.vertical_tolerance
        horizontal = (b.x + radius >= x and b.x - radius <= x + w)
        return height, horizontal

    def _pay(self, name, source, score, world):
        self.counts[name] += 1
        self.total += score
        return {'kind': name, 'source': source, 'points': float(score), 'time': float(world.time)}

    def _on_spring_motion(self, world):
        source = self.active_spring
        if source is None:
            return []
        if world.time - self.active_since > self.settings.spring_window_seconds:
            self.active_spring = None
            return []
        results = []
        for target in self.spring_specs[source].target:
            h, x = self._alignment(world, target)
            # 네 최종 GitHub 2단계 보상과 동일하게 높이와 가로를 독립적으로 판정.
            if h and not self.was_in_height.get(target, False) and source not in self.height_paid:
                self.height_paid.add(source)
                results.append(self._pay('spring_height', source, self.settings.spring_height, world))
            if x and not self.was_in_x.get(target, False) and source not in self.x_paid:
                self.x_paid.add(source)
                results.append(self._pay('spring_horizontal', source, self.settings.spring_horizontal, world))
            self.was_in_height[target] = h
            self.was_in_x[target] = x
        return results

    def _on_spring_landing(self, world, pid):
        if pid not in self.spring_specs:
            self.active_spring = None
            self.was_in_height = {}
            self.was_in_x = {}
            return
        self.active_spring = pid
        self.active_since = float(world.time)
        self.was_in_height = {}
        self.was_in_x = {}
        for target in self.spring_specs[pid].target:
            h, x = self._alignment(world, target)
            self.was_in_height[target] = h
            self.was_in_x[target] = x

    def _on_moving_motion(self, world):
        results = []
        b = world.ball
        for source in self.vertical_sources:
            if source not in self.moving_first_paid or source in self.moving_high_paid:
                continue
            if world.time - self.moving_contact_time.get(source, -1e9) > self.settings.moving_contact_window:
                continue
            device = world.special.devices[source]
            travel = float(device.options.get('travel', 128.))
            if travel <= 0:
                continue
            x, y, width, _ = self._rect(source)
            # 4단계의 고정 y=334 대신 이동 발판의 자체 왕복 범위에서 상대 상승량 사용.
            ascent = (float(device.y) + travel - y) / travel
            near_x = x - 12 <= b.x <= x + width + 12
            near_y = y - 1.4 * world.physics.bounce_height <= b.y <= y + world.physics.radius + 4
            if ascent >= self.settings.moving_high_fraction and near_x and near_y:
                self.moving_high_paid.add(source)
                results.append(self._pay('moving_high', source, self.settings.moving_high, world))
        return results

    def _on_progress(self, world):
        """단방향 탐색 힌트: 최초 도달 위치만 지급하고 원래 클리어와 구간 보상은 그대로 둔다."""
        if (not self.progress_capable or self.settings.progress_per_pixel <= 0
                or self.progress_goal_x is None or self.progress_paid >= self.settings.progress_total_cap):
            return []
        x = max(self.progress_max_x, min(float(world.ball.x), self.progress_goal_x))
        delta = x - self.progress_max_x
        if delta <= 0:
            return []
        self.progress_max_x = x
        score = min(delta * self.settings.progress_per_pixel,
                    self.settings.progress_total_cap - self.progress_paid)
        if score <= 0:
            return []
        self.progress_paid += score
        return [self._pay('progress', 'new-horizontal-record', score, world)]

    def on_terminal(self, world):
        """실패/타임아웃에서는 진행 보상을 회수해 자살·낙하 학습을 방지한다.

        실제 클리어는 진척분을 유지한다. 한 에피소드에서 최대 한 번만 호출할 수 있다.
        """
        if self.progress_balanced:
            return []
        self.progress_balanced = True
        if not world.cleared and self.progress_paid:
            return [self._pay('progress_reversal', 'failed-episode',
                              -self.progress_paid, world)]
        return []

    def on_tick(self, world):
        """MilestoneRawEnv의 on_tick callback에서 매 물리 틱 1회 호출.

        원본 2단계와 동일하게 motion 먼저, 착지 events 나중에 처리한다.
        """
        if world.failed or world.cleared:
            return []
        events = list(world.events)
        results = (self._on_spring_motion(world) + self._on_moving_motion(world)
                   + self._on_progress(world))
        for event, raw_pid in events:
            if event != 'land':
                continue
            pid = str(raw_pid)
            self._on_spring_landing(world, pid)
            if pid in self.vertical_sources:
                self.moving_contact_time[pid] = float(world.time)
                if pid not in self.moving_first_paid:
                    self.moving_first_paid.add(pid)
                    results.append(self._pay('moving_first', pid, self.settings.moving_first, world))
        return results
