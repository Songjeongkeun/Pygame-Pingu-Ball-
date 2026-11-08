"""기존 핵심 구간 보상에 더할 2·4번 전용 보조 보상.

성공·실패와 목표 착지 +10은 이 파일에서 계산하지 않는다. 기존
milestone_env의 규칙을 그대로 사용하여 같은 사건의 중복 지급을 막는다.
"""
from collections import Counter


class AuxiliaryRewards:
    """한 에피소드의 최초 달성 여부와 점프 후 비행 상태를 추적한다."""

    def __init__(self, stage, world, specs):
        self.stage = stage
        self.platforms = {p.id: p for p in world.platforms}
        self.targets = {s.source[0]: s.target for s in specs} if stage == 2 else {}
        self.active_source = None
        self.active_since = -1.
        self.inside_height, self.inside_horizontal = {}, {}
        self.height_paid, self.horizontal_paid = set(), set()
        self.lift_first = self.high_lift = self.upper_first = False
        self.last_lift_time = -1000.
        # 원래 4번의 문서 좌표 96→384 구간을 확장된 화면 좌표로 옮긴다.
        self.approach_start = float(world.level.spawn[0])
        self.approach_end = self.approach_start + 288.
        self.highest_x = self.approach_start
        self.parts = Counter()

    def _event(self, kind, points, source=None):
        self.parts[kind] += points
        return dict(kind=kind, points=float(points), **({'id': source} if source else {}))

    def _inside(self, world, target):
        x, y, width, _ = self.platforms[target].rect_at(world.time, world.physics)
        b, radius = world.ball, world.physics.radius
        return (abs(b.y + radius - y) <= 16.,
                b.x + radius >= x and b.x - radius <= x + width)

    def motion(self, world):
        """물리 갱신 후, 실제 착지 사건을 처리하기 전에 호출한다."""
        if world.failed:
            return []
        events = []
        if self.stage == 2 and not world.cleared and self.active_source is not None:
            source = self.active_source
            if world.time - self.active_since > 3.:
                self.active_source = None
                return events
            for target in self.targets[source]:
                height, horizontal = self._inside(world, target)
                if height and not self.inside_height.get(target, False) and source not in self.height_paid:
                    self.height_paid.add(source)
                    events.append(self._event('spring_height', .5, source))
                if horizontal and not self.inside_horizontal.get(target, False) and source not in self.horizontal_paid:
                    self.horizontal_paid.add(source)
                    events.append(self._event('spring_horizontal', .5, source))
                self.inside_height[target], self.inside_horizontal[target] = height, horizontal
        elif self.stage == 4:
            x = min(float(world.ball.x), self.approach_end)
            if x > self.highest_x:
                events.append(self._event('approach', (x - self.highest_x) * .045))
                self.highest_x = x
        return events

    def landing(self, world, platform_id):
        """착지 +10과 독립적인 보조 보상만 반환한다."""
        if world.failed:
            return []
        events = []
        if self.stage == 2:
            if platform_id in self.targets:
                self.active_source, self.active_since = platform_id, float(world.time)
                # 발판을 밟을 때 이미 겹친 영역은 새 진입으로 세지 않는다.
                pairs = {q: self._inside(world, q) for q in self.targets[platform_id]}
                self.inside_height = {q: v[0] for q, v in pairs.items()}
                self.inside_horizontal = {q: v[1] for q, v in pairs.items()}
            else:
                self.active_source = None
                self.inside_height, self.inside_horizontal = {}, {}
        elif self.stage == 4:
            if platform_id == 'lift':
                self.last_lift_time = float(world.time)
                if not self.lift_first:
                    self.lift_first = True
                    events.append(self._event('lift_first', 12., platform_id))
            elif platform_id.startswith('finish-') and self.lift_first and not self.upper_first:
                self.upper_first = True
                events.append(self._event('upper_first', 18., platform_id))
        return events

    def after_landings(self, world):
        """이동 발판에 실제 착지한 이력이 있는 상태에서 높은 위치를 이용했는지 검사한다."""
        if self.stage != 4 or world.failed or self.high_lift or not self.lift_first:
            return []
        lx, ly, width, _ = world.special.platforms['lift'].rect_at(world.time, world.physics)
        b = world.ball
        if (ly <= 334. and lx - 12. <= b.x <= lx + width + 12.
                and ly - 130. <= b.y <= ly + 18.
                and world.time - self.last_lift_time <= 1.05):
            self.high_lift = True
            return [self._event('lift_high', 12., 'lift')]
        return []

    def state(self):
        """보고서용 정보다. 기존 정책 관측의 크기와 값은 바꾸지 않는다."""
        return dict(active_spring=self.active_source, spring_since=self.active_since,
                    height_paid=sorted(self.height_paid), horizontal_paid=sorted(self.horizontal_paid),
                    lift_first=self.lift_first, lift_high=self.high_lift, upper_first=self.upper_first)
