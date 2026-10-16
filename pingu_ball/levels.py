"""Eight 25 by 13 tile stages and reproducible input routes."""
from .config import PHYSICS
from .engine import Platform, Collectible, Spike, Level

TITLES = [
    ('첫 바운스', '방향키로 이동하세요. 생선은 선택 사항이며 이글루에 닿으면 성공합니다.'),
    ('금이 간 빙하', '금 간 얼음은 첫 착지 0.5초 후 깨집니다.'),
    ('가시 해협', '넓은 가시 앞에서는 위쪽 우회 발판을 이용하세요.'),
    ('무너지는 길', '깨지는 발판과 가시를 함께 통과하세요.'),
    ('더 높은 바운스', '분홍 점프대는 144px 높이로 튕깁니다.'),
    ('움직이는 얼음', '왕복 발판은 공을 운반하지 않습니다. 착지 시점을 조절하세요.'),
    ('얼음의 시간', '청록 발판은 깜빡인 뒤 사라집니다. 주기를 살펴보세요.'),
    ('집으로 가는 길', '빨간 스위치로 다리를 열고 이글루로 돌아가세요.'),
]


def make_levels():
    return [make_level(i) for i in range(8)]


def make_level(index):
    if index not in range(8):
        raise ValueError('stage must be between 0 and 7')
    blocks = []

    def tile(name, col, row, kind='normal', **kwargs):
        blocks.append(Platform(name, col*64, row*64, kind=kind, **kwargs))

    def strip(name, first, end, row, kind='normal', **kwargs):
        for col in range(first, end):
            tile(f'{name}-{col}', col, row, kind, **kwargs)

    strip('start', 0, 8, 11)
    # One-tile stairs rise 64px at a time, below the 96px normal bounce.
    tile('step-low', 3, 10)
    tile('step-mid', 5, 9)
    tile('step-high', 7, 8)
    strip('upper', 9, 22, 7)
    strip('home', 20, 25, 11)
    spikes = []
    base = []

    if index in (2, 3, 7):
        strip('lower', 8, 14, 11, 'breakable' if index == 3 else 'normal')
        tile('bypass-step', 14, 10)
        strip('bypass', 16, 19, 9)
        strip('landing', 14, 20, 11)
        spikes = [Spike(16*64, 10*64)]
        base += [(928., 'land:bypass-step'), (1056., 'land:bypass'), (1280., 'below:600')]
    elif index == 4:
        strip('approach', 8, 14, 11)
        tile('big-jump', 14, 11, 'spring')
        # 128px gap: spring offers 235px center travel with ample headroom.
        strip('landing', 17, 20, 11)
        base += [(928., 'land:big-jump'), (1120., 'land:landing')]
    elif index == 5:
        strip('approach', 8, 13, 11)
        tile('ferry', 13, 11, 'moving', travel=128.)
        strip('landing', 16, 20, 11)
        base += [(928., 'land:ferry'), (1056., 'land:landing')]
    elif index == 6:
        strip('approach', 8, 14, 11)
        strip('time-bridge', 14, 17, 11, 'timed', period=3.6, active_for=2.8, phase=.2)
        strip('landing', 17, 20, 11)
        base += [(864., 'land:approach'), (1120., 'land:landing')]
    else:
        strip('lower', 8, 20, 11, 'breakable' if index == 1 else 'normal')

    if index == 7:
        # Both routes must open the bridge; it starts hidden and non-solid.
        blocks = [q for q in blocks if q.id != 'start-2' and not q.id.startswith('home-2')]
        tile('switch-A', 2, 11, 'switch', group='A')
        strip('bridge-A', 20, 23, 11, 'linked', group='A')
        strip('goal-floor', 23, 25, 11)
        tile('launcher', 19, 9, 'launcher', arrow=1)
        tile('side-time', 19, 10, 'timed', period=3.2, active_for=2.2)
        tile('side-crumble', 22, 8, 'breakable')
        base.insert(0, (160., 'switch:A'))

    # All fish are airborne. The negative item is near the start, below the spawn apex.
    fish = [Collectible(672., 366.), Collectible(960., 366.), Collectible(1280., 366.),
            Collectible(64., 650., -1)]
    all_fish = [(224., 'land:step-low'), (352., 'land:step-mid'),
                (480., 'land:step-high'), (608., 'land:upper'),
                (672., 'fish0'), (960., 'fish1'), (1280., 'fish2')]
    if index == 7:
        all_fish.insert(0, (160., 'switch:A'))
    if index == 0:
        fish = fish[:1]
        all_fish = all_fish[:5]
    base.append((1504., 'goal'))
    all_fish.append((1504., 'goal'))
    return Level(index, *TITLES[index], blocks, fish, spikes,
                 (96., 11*64-PHYSICS.radius-PHYSICS.bounce_height),
                 (23*64., 10*64., 64., 64.), base, all_fish)


def waypoint_direction(world, target):
    """Supply ordinary left/right input for repeatable route validation."""
    b = world.ball
    distance = target-b.x
    stop = b.vx*abs(b.vx)/(2*world.physics.deceleration)
    if abs(distance) < 1.5 and abs(b.vx) < 8:
        return 0
    if (distance > 0 and b.vx > 0 and distance <= stop+.7) or (distance < 0 and b.vx < 0 and -distance <= -stop+.7):
        return 0
    return 1 if distance > 0 else -1


def condition_met(world, condition):
    if condition == 'goal':
        return world.cleared
    if condition.startswith('fish'):
        return int(condition[4:]) in world.collected
    if condition.startswith('land:'):
        return world.last_landing is not None and world.last_landing.startswith(condition[5:])
    if condition.startswith('switch:'):
        return world.switches.get(condition[7:], False)
    if condition.startswith('below:'):
        return world.ball.y > float(condition[6:])
    return False
