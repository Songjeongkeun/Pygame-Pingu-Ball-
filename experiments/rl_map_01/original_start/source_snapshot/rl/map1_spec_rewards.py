"""얼음계곡 맵1 PDF의 33개 보상. 물리나 정책 관측에는 관여하지 않는다."""
from collections import Counter
from dataclasses import dataclass


def group(name, count):
    return tuple(f'{name}-{i}' for i in range(count))


@dataclass(frozen=True)
class Rule:
    id: str
    label: str
    region: str
    target: tuple[str, ...]
    source: tuple[str, ...] = ()
    max_top_y: float | None = None


def definitions():
    """독립적인 항목이다. 앞 항목을 놓쳐도 뒤 연결의 보상을 받을 수 있다."""
    rows = []
    def add(key, label, region, target, source=(), high=None):
        rows.append(Rule(key, label, region, tuple(target), tuple(source), high))
    add('01-first', '아래 파괴 첫 착지', '아래 파괴', ('z01break-a-0',))
    add('01-link', '아래 파괴 A→B', '아래 파괴', ('z01break-b-0',), ('z01break-a-0',))
    add('01-exit', '아래 파괴→기본', '아래 파괴', group('z01dock', 3), ('z01break-b-0',))
    add('02-first', '아래 점프 첫 착지', '아래 점프', ('z02spring-0',))
    add('02-link', '아래 점프→위 발판', '아래 점프', group('z02upper', 3), ('z02spring-0',))
    add('02-exit', '아래 위 발판→기본', '아래 점프', group('z02dock', 3), group('z02upper', 3))
    add('03-first', '아래 시간 첫 착지', '아래 시간', ('z03clock-0',))
    add('03-exit', '아래 시간→기본', '아래 시간', group('z03dock', 3), ('z03clock-0',))
    add('04-first', '아래 좌우 이동 탑승', '아래 좌우 이동', group('z04ferry', 2))
    add('04-exit', '아래 좌우 이동→기본', '아래 좌우 이동', group('z04dock', 3), group('z04ferry', 2))
    add('05-first', '아래 상하 이동 탑승', '아래 상하 이동', ('z05lift-0',))
    add('05-high', '아래 상하 이동 높은 착지', '아래 상하 이동', ('z05lift-0',), high=832.)
    add('05-link', '아래 상하 이동→위 발판', '아래 상하 이동', group('z05upper', 2), ('z05lift-0',))
    add('05-exit', '아래 위 발판→기본', '아래 상하 이동', group('z05dock', 3), group('z05upper', 2))
    add('turn-first', '회전 점프 첫 착지', '층 전환', ('turn0-spring0-0',))
    for i in range(3):
        add(f'turn-{i+1}', f'회전 점프 {i}→{i+1}', '층 전환',
            (f'turn0-spring{i+1}-0',), (f'turn0-spring{i}-0',))
    add('turn-exit', '회전 점프3→위층 기본', '층 전환', group('turn0buffer', 3), ('turn0-spring3-0',))
    add('06-first', '위층 기본→점프', '위층 점프', ('z06spring-0',), group('turn0buffer', 3))
    add('06-link', '위층 점프→위 발판', '위층 점프', group('z06upper', 3), ('z06spring-0',))
    add('06-exit', '위층 위 발판→기본', '위층 점프', group('z06dock', 3), group('z06upper', 3))
    add('07-first', '위층 기본→파괴 A', '위층 파괴', ('z07break-a-0',), group('z06dock', 3))
    add('07-link', '위층 파괴 A→B', '위층 파괴', ('z07break-b-0',), ('z07break-a-0',))
    add('07-exit', '위층 파괴→기본', '위층 파괴', group('z07dock', 3), ('z07break-b-0',))
    add('08-first', '위층 기본→상하 이동', '위층 상하 이동', ('z08lift-0',), group('z07dock', 3))
    add('08-high', '위층 상하 이동 높은 착지', '위층 상하 이동', ('z08lift-0',), high=320.)
    add('08-link', '위층 상하 이동→위 발판', '위층 상하 이동', group('z08upper', 2), ('z08lift-0',))
    add('08-exit', '위층 위 발판→기본', '위층 상하 이동', group('z08dock', 3), group('z08upper', 2))
    add('09-first', '위층 기본→좌우 이동', '위층 좌우 이동', group('z09ferry', 2), group('z08dock', 3))
    add('09-exit', '위층 좌우 이동→기본', '위층 좌우 이동', group('z09dock', 3), group('z09ferry', 2))
    add('10-first', '위층 기본→시간', '위층 시간', ('z10clock-0',), group('z09dock', 3))
    add('10-exit', '위층 시간→기본', '위층 시간', group('z10dock', 3), ('z10clock-0',))
    assert len(rows) == 33 and len({r.id for r in rows}) == 33
    return tuple(rows)


RULES = definitions()


class RewardTracker:
    """실제 착지만 받는다. 공중에서는 직전 착지를 유지하고 다른 착지로 교체한다."""
    def __init__(self):
        self.paid = set()
        self.previous_landing = None
        self.upper_entered = False
        self.landed = set()
        self.last_region = '출발'
        self.attempts, self.duplicates, self.unpaid = Counter(), Counter(), Counter()
        self.landing_count = 0

    def land(self, platform, top_y):
        events = []
        previous = self.previous_landing
        self.landed.add(platform)
        self.landing_count += 1
        if platform in group('turn0buffer', 3):
            self.upper_entered = True
        for rule in RULES:
            if platform not in rule.target:
                continue
            self.last_region = rule.region
            self.attempts[rule.id] += 1
            # 미지급 원인을 구분해 기록한다. 이미 받은 항목은 다시 지급하지 않는다.
            if rule.id in self.paid:
                self.duplicates[rule.id] += 1
                continue
            if rule.source and previous not in rule.source:
                self.unpaid[f'{rule.id}:직전 발판 불일치'] += 1
                continue
            if rule.max_top_y is not None and top_y > rule.max_top_y:
                self.unpaid[f'{rule.id}:리프트 높이 미달'] += 1
                continue
            self.paid.add(rule.id)
            events.append(dict(kind='intermediate', id=rule.id, points=1.,
                               platform=platform, previous=previous, top_y=float(top_y)))
        self.previous_landing = platform
        return events

    def state(self):
        return dict(reward_items=sorted(self.paid), previous_landing=self.previous_landing,
            upper_entered=self.upper_entered, landed_platforms=sorted(self.landed),
            last_region=self.last_region, landing_count=self.landing_count,
            reward_attempts=dict(self.attempts), duplicate_landings=dict(self.duplicates),
            unpaid_landings=dict(self.unpaid))


def audit_reward(info):
    """종료한 한 판의 금액·중복·종료 보상 단일 지급을 대조한다."""
    parts, items = info['reward_parts'], info['reward_items']
    assert len(items) == len(set(items)) <= 33
    assert set(items) <= {r.id for r in RULES}
    assert set(parts) <= {'intermediate', 'success', 'failure'}
    assert parts.get('intermediate', 0.) == len(items)
    assert parts.get('success', 0.) + parts.get('failure', 0.) == (100. if info['success'] else -25.)
    assert not (parts.get('success') and parts.get('failure'))
    assert abs(sum(parts.values()) - info['episode_return']) < 1e-6
    assert abs(info['learning_return'] * 100 - info['episode_return']) < 1e-6
