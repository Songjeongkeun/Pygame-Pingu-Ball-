"""Ten document-based tutorials; all previous demonstration rooms are removed."""
from .engine import Level, Platform, PowerUp
from .specials import Device
from .config import PHYSICS

LAB_TYPES = (301, 401, 501, 508, 302, 402, 701, 702, 703, 710)
LAB_NUMBERS = {code: n for n, code in enumerate(LAB_TYPES, 1)}
ENABLED_LAB_TYPES = LAB_TYPES
TITLES = dict(zip(LAB_TYPES, ('파괴 블록', '점프 블록', '이동 블록 (좌우)',
    '이동 블록 (상하)', '시간 블록', '좌우 발사 블록', '태양 블록',
    '고드름', '바다사자', '점프와 대시 아이템')))
HINTS = dict(zip(LAB_TYPES, (
    '밟으면 사라지는 발판을 건너 이글루로 가세요.',
    '화살표 발판은 일반 반동의 1.5배 높이로 튕깁니다.',
    '발판을 따라 움직이세요. 발판의 이동량은 공에 더해지지 않습니다.',
    '상하로 왕복하는 발판을 거쳐 위쪽 이글루로 가세요.',
    '시계가 보일 때만 착지할 수 있습니다. 반투명 상태는 통과합니다.',
    '발사 발판에서 왼쪽으로 날아가세요. 방향키를 누르면 발사가 풀립니다.',
    '태양은 발판이 아닙니다. 닿지 않게 뛰어넘으세요.',
    '생성 중인 0.5초는 안전합니다. 떨어지는 고드름을 피하세요.',
    '블록 둘레를 따라 회전하는 바다사자를 피해 건너세요.',
    '점프는 ↑↑, 대시는 ←←·→→ 또는 더블 터치로 사용하세요.',
)))


def resolve_lab_type(number):
    if type(number) is int and 1 <= number <= len(LAB_TYPES):
        return LAB_TYPES[number - 1]
    if number in LAB_TYPES:
        return number
    raise ValueError('Tutorial number must be between 1 and 10')


def make_lab(number):
    code = resolve_lab_type(number)
    stage = LAB_NUMBERS[code]
    blocks, devices, items = [], [], []

    def row(name, x, y, count, kind='normal', **options):
        for i in range(count):
            blocks.append(Platform(f'{name}-{i}', x + i * 64, y, kind=kind, **options))

    def device(name, kind, x, y, **options):
        devices.append(Device(name, kind, x, y, options=options))

    def apex(x, floor):
        return x, floor - PHYSICS.radius - PHYSICS.bounce_height

    width, height = 1600, 900
    if stage == 1:
        width, height = 1088, 640
        row('start', 64, 384, 4)
        row('finish', 832, 384, 3)
        for i, x in enumerate((320, 448, 576, 704)):
            device(f'break-{i}', 301, x, 384)
        spawn, goal = apex(96, 384), (960, 320, 64, 64)
    elif stage == 2:
        row('start', 192, 704, 4)
        row('spring-1', 448, 704, 1, 'spring')
        row('middle', 576, 576, 3)
        row('spring-2', 768, 576, 1, 'spring')
        row('upper', 896, 448, 2)
        row('spring-3', 1024, 448, 1, 'spring')
        row('finish', 1152, 320, 4)
        for i in range(3): row(f'left-wall-{i}', 128, 512 + i * 64, 1)
        for i in range(2): row(f'right-wall-{i}', 1344, 192 + i * 64, 1)
        spawn, goal = apex(288, 704), (1280, 256, 64, 64)
    elif stage == 3:
        row('start', 184, 176, 3)
        for i in range(3): row(f'wall-{i}', -8, 240 + i * 64, 1)
        steps = [(56, 416, 1), (120, 480, 3), (312, 544, 2),
                 (440, 608, 2), (568, 672, 2), (696, 736, 2), (824, 800, 5)]
        for n, (x, y, count) in enumerate(steps):
            row(f'step-{n}', x, y, count)
            for i in range(count): device(f'spike-{n}-{i}', 201, x + i * 64, y - 64, teeth=6)
        row('end-wall', 1144, 736, 1)
        row('end-wall-base', 1144, 800, 1)
        row('finish', 1280, 800, 3)
        for i in range(3):
            device(f'ferry-{i}', 501, 184 + i * 64, 360, travel=1120., speed=96., phase=144.)
        spawn, goal = apex(293, 176), (1344, 736, 64, 64)
    elif stage == 4:
        width, height = 1088, 640
        row('start', 64, 448, 4)
        row('finish', 512, 256, 8)
        device('lift', 508, 384, 256, travel=192., speed=64., phase=128.)
        spawn, goal = apex(96, 448), (960, 192, 64, 64)
    elif stage == 5:
        height = 1000
        row('start', 0, 480, 4)
        for i in range(3): row(f'wall-{i}', 0, 288 + i * 64, 1)
        row('finish', 960, 480, 4)
        for i, (x, phase) in enumerate(((384, 0.), (576, .45), (768, 2.8))):
            device(f'clock-{i}', 302, x, 480, period=3.6, active_for=2.8, phase=phase)
        spawn, goal = apex(156, 480), (1088, 416, 64, 64)
    elif stage == 6:
        row('finish', 0, 480, 3)
        for i in range(3): row(f'wall-{i}', 0, 288 + i * 64, 1)
        row('lower', 640, 480, 9)
        row('upper', 896, 288, 6)
        for i in range(3): device(f'upper-spike-{i}', 201, 1088 + i * 64, 224, teeth=6)
        device('launch', 402, 1088, 416, direction=-1)
        spawn, goal = apex(992, 288), (64, 416, 64, 64)
    elif stage == 7:
        height = 1000
        row('start', 192, 600, 4)
        row('middle', 576, 600, 2)
        row('launch-deck', 832, 600, 2)
        row('spring', 960, 600, 1, 'spring')
        row('finish', 1216, 600, 4)
        for i, (x, y) in enumerate(((448, 600), (512, 600), (704, 600), (768, 600), (1088, 536))):
            device(f'sun-{i}', 701, x, y)
        spawn, goal = apex(283, 600), (1280, 536, 64, 64)
    elif stage == 8:
        height = 1000
        row('start', 128, 736, 6)
        row('middle', 640, 736, 5)
        row('finish', 1088, 736, 6)
        for i, (x, y, phase) in enumerate(((512, 299, .5), (768, 299, .5),
                                          (1024, 272, 0.), (1024, 272, 1.4))):
            device(f'icicle-{i}', 702, x, y, phase=phase, period=4., delay=.5)
        spawn, goal = apex(220, 736), (1344, 672, 64, 64)
    elif stage == 9:
        height = 1000
        for name, x, count in (('start', 192, 3), ('patrol-a', 512, 3), ('single', 832, 1),
                                ('patrol-b', 1024, 3), ('finish', 1344, 2)):
            row(name, x, 534, count)
        for i, (left, phase) in enumerate(((512, 140.), (1024, 204.))):
            device(f'sea-lion-{i}', 703, left, 534, path=(left, 534., 192., 64.),
                   speed=64., phase=phase, size=56.)
        spawn, goal = apex(283, 534), (1408, 470, 64, 64)
    else:
        # Five empty tiles: no lower bypass. Use jump to reach the high deck,
        # then dash during a bounce to extend its horizontal reach.
        width, height = 1536, 864
        row('start', 64, 576, 4)
        row('dash-deck', 448, 448, 3)
        row('finish', 960, 448, 5)
        for i in range(3): row(f'wall-{i}', 1280, 320 + i * 64, 1)
        items = [PowerUp(288, 542, 'up', 48), PowerUp(608, 414, 'forward', 48)]
        spawn, goal = apex(96, 576), (1152, 384, 64, 64)

    path=[(goal[0]+goal[2]/2,'goal')]
    if stage==1:
        path=[(288.,'land:start-3')]+[(x,f'touch:break-{i}') for i,x in enumerate((352.,480.,608.,736.))]+path
    return Level(0, f'{stage} {TITLES[code]}', HINTS[code], blocks, [], [], spawn,
                 goal, path, path, height=height, width=width, time_limit=60. if stage==1 else 180., specials=devices,
                 powerups=items, preview=f'assets/tutorials/stage_{stage:02d}.png')
