"""Playable adaptations of the six approved map images, in 64px world tiles.

The maps use 15 rows; the view fits the complete map below the HUD. Short
approach platforms bridge gaps that cannot be crossed with the original
96px bounce / 144px spring. Coordinates here are actual collision geometry.
"""
from .engine import Level, Platform, Spike


# Each clock opens before the first ordinary-input landing on its route.
CLOCK_PHASES = {1: {'timed-13-4': 1.0083,
     'timed-15-3': 0.2917,
     'timed-17-2': 4.7833,
     'timed-7-4': 3.6417},
 2: {'timed-18-5': 4.6333, 'timed-19-4': 3.925, 'timed-20-4': 3.025},
 3: {'timed-21-14': 3.4417},
 4: {'timed-10-10': 3.7955,
     'timed-10-13': 3.8402,
     'timed-13-10': 4.6955,
     'timed-13-13': 2.0402,
     'timed-16-10': 0.3955,
     'timed-16-13': 0.2402,
     'timed-19-10': 1.2955,
     'timed-3-8': 0.3238,
     'timed-5-9': 1.0336,
     'timed-7-10': 2.8955,
     'timed-7-13': 4.7402},
 6: {'timed-21-10': 0.225}}


class Layout:
    def __init__(self, index, title, hint):
        self.index, self.title, self.hint = index, title, hint
        self.blocks = {}
        self.spikes = []
        self.route = []

    def block(self, c, r, kind='normal', **kwargs):
        key = (c, r)
        if key in self.blocks:
            raise ValueError(f'duplicate tile: {self.index} {key}')
        name = f'{kind}-{c:g}-{r:g}'
        if kind == 'timed':
            kwargs = dict(period=5.2, active_for=4.2,
                          phase=CLOCK_PHASES.get(self.index, {}).get(name, 0.), **kwargs)
        q = Platform(name, c*64., r*64., kind=kind, **kwargs)
        self.blocks[key] = q
        return q

    def row(self, c, end, r, kind='normal', **kwargs):
        return [self.block(i, r, kind, **kwargs) for i in range(c, end)]

    def column(self, c, r, end):
        return [self.block(c, i) for i in range(r, end)]

    def hazard(self, c, end, r):
        for i in range(c, end):
            self.spikes.append(Spike(i*64., r*64., 64., 48., teeth=6))
            q = self.block(i, r+.75, h=16.)
            q.id = f'spike-base-{i}-{r}'

    def land(self, q, x=None):
        condition = 'touch:' if q.kind == 'launcher' else 'land:'
        self.route.append((q.x+32 if x is None else x, condition+q.id))
        return q

    def step(self, c, r, kind='normal', x=None, **kwargs):
        return self.land(self.block(c, r, kind, **kwargs), x)

    def ferry(self, name, c, r, tiles, travel, phase=0.):
        group = []
        for i in range(tiles):
            q = self.block(c+i, r, 'moving', travel=travel*64., phase=phase)
            q.id = f'{name}-{i}'
            group.append(q)
        return group

    def ride(self, group, source, exit_block, direction, bounds=None):
        prefix = group[0].id.rsplit('-', 1)[0]
        suffix = '' if bounds is None else f':{bounds[0]}:{bounds[1]}'
        self.route.append((source.x+32, f'board:{prefix}:{source.id}:{direction}'+suffix))
        self.route.append((0., f'ride:{prefix}:{exit_block.id}:{direction}'))
        self.land(exit_block)

    def finish(self, start, goal):
        self.route.append((goal.x+32, 'goal'))
        return Level(self.index, self.title, self.hint, list(self.blocks.values()), [], self.spikes,
                     (start.x+32, start.y-110.), (goal.x, goal.y-64., 64., 64.),
                     list(self.route), list(self.route), height=960,
                     preview=f'assets/maps/stage_{self.index:02d}.png', time_limit=120.)


def first():
    a = Layout(1, '빙하 협곡', '위쪽 발판을 건넌 뒤 두 발사 블록으로 내려가 왼쪽 이글루에 도착하세요.')
    a.column(2, 0, 4)
    a.row(3, 6, 4); a.row(3, 6, 5); a.column(3, 6, 11)
    start = a.blocks[4, 4]; a.land(start)
    a.land(a.blocks[5, 4])
    a.step(7, 4, 'timed')
    a.step(9, 4); a.step(10, 3)
    a.block(10, 4)
    a.step(13, 4, 'timed'); a.step(15, 3, 'timed'); a.step(17, 2, 'timed')
    a.column(18, 3, 6); a.land(a.blocks[18, 3])
    a.step(20, 5, 'spring')
    a.step(23, 7, 'launcher', arrow=-1)
    a.route.append((864., 'flyleft:864'))
    a.step(13, 12, 'launcher', arrow=-1)
    a.route.append((224., 'flyleft:224'))
    a.row(2, 5, 14)
    a.hazard(6, 10, 6); a.hazard(11, 18, 6); a.hazard(19, 20, 6)
    a.hazard(4, 7, 11); a.hazard(21, 22, 4)
    a.block(21, 5)
    return a.finish(start, a.blocks[3, 14])


def second():
    a = Layout(2, '얼음 장벽', '깊은 골짜기를 내려가 점프 발판으로 오른쪽 장벽을 올라가세요.')
    a.column(3, 0, 4); a.row(4, 7, 4)
    start = a.blocks[5, 4]; a.land(start); a.land(a.blocks[6, 4])
    a.hazard(7, 8, 4)
    a.step(8, 5); a.column(9, 5, 14); a.land(a.blocks[9, 5])
    a.column(13, 1, 10)
    a.step(11, 12)
    a.step(14, 12, 'spring')
    a.step(16, 13, 'spring')
    a.step(15, 11, 'spring')
    a.step(14, 9, 'spring')
    a.step(15, 7, 'spring')
    a.step(16, 7, 'spring')
    a.column(17, 6, 13)
    a.step(18, 5, 'timed'); a.step(19, 4, 'timed'); a.step(20, 4, 'timed')
    a.row(22, 25, 4)
    a.hazard(10, 13, 14); a.row(13, 16, 14)
    a.hazard(18, 19, 6)
    return a.finish(start, a.blocks[23, 4])


def third():
    a = Layout(3, '갈림길의 시간', '오른쪽으로 내려간 뒤 중앙 점프 발판과 시간 블록을 이용하세요.')
    a.row(23, 25, 3); start = a.blocks[23, 3]; a.land(start)
    q = a.block(22, 4, 'launcher', arrow=-1)
    a.route.append((1456., 'touch:'+q.id))
    a.route.append((1504., 'release'))
    a.block(23, 5)
    for r in (7, 9, 11):
        a.block(22, r-1, 'launcher', arrow=-1)
        a.block(23, r)
    # At 640px/s the release momentum takes the outside descent to the bottom shelf.
    a.block(24, 13)
    a.step(23, 13)
    a.step(22, 13)
    a.step(21, 14, 'timed')
    a.row(17, 20, 14); a.land(a.blocks[19, 14], x=1232.)
    a.step(16, 13, 'spring')
    a.step(17, 11, 'spring')
    a.step(18, 10, 'spring', x=1168.)
    a.step(16, 9, 'spring')
    a.column(15, 8, 13); a.land(a.blocks[15, 8], x=961.)
    goal = a.block(12, 10, 'timed')
    a.block(4, 14, 'timed'); a.block(6, 13, 'timed')
    a.block(8, 12, 'timed'); a.block(10, 11, 'timed')
    for r in (5, 8, 11, 14): a.block(2, r)
    a.block(18, 4, 'launcher', arrow=-1); a.block(19, 4)
    for c, r in ((19,3), (20,5), (19,6), (20,7), (19,8), (20,9), (19,10), (20,11)):
        a.hazard(c, c+1, r)
    a.block(15, 5); a.block(15, 6)
    a.hazard(5, 15, 5)
    a.block(0, 8, 'timed'); a.block(0, 11, 'timed')
    return a.finish(start, goal)


def fourth():
    a = Layout(4, '시간의 디딤돌', '아래 길을 건너 오른쪽에서 올라간 뒤 왼쪽 이글루로 돌아오세요.')
    a.block(3, 13); a.row(3, 6, 14); start = a.blocks[5, 14]; a.land(start)
    for c in (7, 10, 13, 16): a.step(c, 13, 'timed')
    a.step(19, 13); a.step(21, 12, 'spring')
    # The upper row is lowered two tiles so the 144px spring can reach it.
    for c in (19, 16, 13, 10): a.step(c, 10, 'timed')
    a.step(7, 10, 'timed'); a.step(8, 10, 'spring')
    a.step(5, 9, 'timed'); a.step(3, 8, 'timed')
    a.column(0, 6, 9); goal = a.block(1, 8)
    a.block(23, 9, 'launcher', arrow=-1); a.block(21, 7, 'launcher', arrow=-1)
    return a.finish(start, goal)


def fifth():
    a = Layout(5, '가시 사이 왕복', '이동 발판을 반복해서 밟아 왕복 구간을 건너고 아래 길로 돌아오세요.')
    start = a.step(1, 6)
    a.column(0, 7, 13); goal = a.block(1, 13)
    top_exit = a.block(23, 6)
    bottom_exit = a.block(23, 13)
    upper = a.ferry('upper-ferry', 2, 6, 4, 17, phase=192.)
    lower = a.ferry('lower-ferry', 2, 13, 3, 18, phase=192.)
    a.ride(upper, start, top_exit, 1)
    a.route.append((1560., 'below:490'))
    a.land(bottom_exit)
    a.ride(lower, bottom_exit, goal, -1)
    a.hazard(2, 23, 7); a.hazard(2, 23, 14)
    for c, r in ((9,2), (15,2), (8,8), (12,8), (16,8)): a.block(c, r, 'timed')
    return a.finish(start, goal)


def sixth():
    a = Layout(6, '두 빙벽의 틈', '왼쪽 빙벽을 올라가 아래 이동 발판을 탄 뒤 중앙 이글루로 돌아오세요.')
    a.column(9, 4, 12); a.row(6, 9, 11)
    a.column(12, 4, 12)
    start = a.blocks[7, 11]; a.land(start)
    a.step(5, 10, 'spring')
    a.step(3, 8, 'spring', x=240.)
    a.step(6, 6, 'spring', x=432.)
    top = a.land(a.blocks[9, 4])
    ferry = a.ferry('valley-ferry', 2, 14, 3, 18, phase=192.)
    exit_pad = a.block(23, 13, 'spring')
    a.ride(ferry, top, exit_pad, 1, bounds=(688., 736.))
    a.step(22, 11, 'spring', x=1424.)
    a.step(21, 10, 'timed')
    a.step(19, 9, 'spring')
    a.step(18, 10, 'spring', x=1168.)
    goal = a.block(14, 10)
    a.hazard(13, 25, 5); a.hazard(0, 6, 11)
    a.hazard(15, 18, 11); a.hazard(19, 21, 11)
    return a.finish(start, goal)


def make_reference_level(index):
    return (first, second, third, fourth, fifth, sixth)[index-1]()
