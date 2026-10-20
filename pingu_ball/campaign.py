"""The twelve regular stages in 스테이지.docx, using native collision geometry."""
import json
import copy
from pathlib import Path

from .config import CAMPAIGN_ID, REGULAR_STAGE_COUNT
from .engine import Level, Platform, PowerUp
from .specials import Device

ROOT = Path(__file__).resolve().parents[1]
TITLES = ('빙하 협곡', '얼음 계단', '오르막 발판', '움직이는 다리',
          '시계 탑', '태양 사이로', '점프와 대시', '빙하 미로',
          '고드름 다리', '태양 협곡', '바다사자 계곡', '남극 대모험')
HINTS = ('가시를 피해 내려가고 발사 블록을 이용해 오른쪽 이글루로 가세요.',
         '파괴 발판과 점프 발판을 거쳐 왼쪽 위 이글루로 돌아가세요.',
         '이동 발판에서 가시 계단 아래로 내려간 뒤 점프 블록을 따라 올라가세요.',
         '움직이는 발판의 범위를 살펴 왼쪽 아래 이글루로 가세요.',
         '시간 발판이 나타나는 때에 맞춰 가운데 이글루로 가세요.',
         '태양을 피하고 좌우 이동 발판을 이용하세요.',
         '점프는 ↑↑, 대시는 ←←·→→로 사용하세요.',
         '파괴 발판과 엘리베이터를 이용해 왼쪽 이글루에 도착하세요.',
         '고드름을 피하며 점프 아이템과 발사 발판을 이용하세요.',
         '태양 사이의 발판을 건너 오른쪽 아래 이글루로 가세요.',
         '발판 둘레를 도는 바다사자를 피해 오른쪽 아래로 가세요.',
         '블록과 아이템을 조합해 오른쪽 아래 이글루에 도착하세요.')

# Keep the fixed 96/144px jump heights. Append stage 5's extra tiles so existing
# device IDs and timer phases stay stable.
TILE_REMOVALS = {4: {('spike', 3, 13)}}
TILE_ADDITIONS = {5: [{'kind': 'spring', 'c': 18, 'r': 7},
                      {'kind': 'spring', 'c': 16, 'r': 5},
                      {'kind': 'normal', 'c': 15, 'r': 7}]}
POSITION_ADJUSTMENTS = {
    3: {('spring',23,10):(21,9), ('spring',20,12):(19,11),
        ('spring',17,14):(17,13)},
    4: {('spring',23,9):(23,10), ('normal',21,6):(21,8),
        ('normal',20,5):(20,7), ('normal',19,5):(18,6)},
    5: {('spring',18,10):(18,11), ('spring',16,7):(16,9)},
    6: {('normal',10,7):(10,9), ('normal',22,7):(22,9)},
    7: {('spring',7,14):(5,12), ('spring',11,14):(7,10),
        ('spring',15,14):(9,8), ('spring',5,8):(11,6),
        ('spring',2,5):(13,4), ('breakable',7,8):(8,12),
        ('breakable',9,8):(10,12)},
    9: {('breakable',13,9):(13,10), ('breakable',15,9):(15,10),
        ('breakable',17,9):(17,10), ('breakable',20,9):(19,10),
        ('spring',23,9):(21,9), ('launcher_left',21,6):(20,7), ('normal',1,7):(1,8)},
}


def fit_jump_layout(source):
    """Return gameplay positions without modifying the retained document data."""
    data=copy.deepcopy(source)
    removed=TILE_REMOVALS.get(data['number'],set())
    data['tiles']=[tile for tile in data['tiles']
                   if (tile['kind'],tile['c'],tile['r']) not in removed]
    changes=POSITION_ADJUSTMENTS.get(data['number'],{})
    goal_support=(data['goal'][0]/64,(data['goal'][1]+60)/64)
    for tile in data['tiles']:
        key=(tile['kind'],tile['c'],tile['r'])
        if key in changes:
            c,r=changes[key]
            if tile['kind']=='normal' and (tile['c'],tile['r'])==goal_support:
                data['goal']=[c*64,(r-1)*64+4,64,64]
            tile['c'],tile['r']=c,r
        elif data['number']==7 and tile['kind']=='sun' and tile['r']==11:
            # Keep all 13 suns; shift the barrier clear of the rising spring path.
            tile['c']+=9
    if data['number']==7:
        data['items']=[[13.5,3.5,kind] if kind=='up' else [c,r,kind]
                       for c,r,kind in data['items']]
    if data['number']==4:
        # Keep the dash pickup above the repositioned top approach platform.
        data['items']=[[18.5,5.5,kind] if kind=='forward' else [c,r,kind]
                       for c,r,kind in data['items']]
    if data['number']==3:
        # The two-tile deck must extend below the upper ledge's exit (x=768).
        # End its right edge at x=832, before the next spike column begins.
        data['rails'][0]=[[4.5,8.5],[11.5,8.5]]
    data['tiles'].extend(copy.deepcopy(TILE_ADDITIONS.get(data['number'], [])))
    return data


def campaign_layouts():
    data = json.loads((ROOT / 'assets/stages/layouts.json').read_text(encoding='utf-8'))
    if data['campaign'] != CAMPAIGN_ID or len(data['maps']) != REGULAR_STAGE_COUNT:
        raise ValueError('Invalid document campaign data')
    return data['maps']


def make_campaign_level(index):
    if type(index) is not int or not 1 <= index <= REGULAR_STAGE_COUNT:
        raise ValueError(f'stage must be between 1 and {REGULAR_STAGE_COUNT}')
    data = fit_jump_layout(campaign_layouts()[index - 1])
    blocks, devices = [], []
    movers = [q for q in data['tiles'] if q['kind'] == 'moving']
    horizontal = [rail for rail in data['rails'] if rail[0][1] == rail[-1][1]]
    # All tiles of a joined moving deck keep the same speed and phase.
    if movers:
        initial = min(q['c'] for q in movers)
        rail = horizontal[0] if horizontal else [[26.5, movers[0]['r'] + .5], [32.5, movers[0]['r'] + .5]]
        low, high = sorted((rail[0][0] - .5, rail[-1][0] - .5))
        travel = (high - low) * 64
        phase = (initial - low) * 64
    for n, q in enumerate(data['tiles']):
        c, r, kind = q['c'], q['r'], q['kind']
        x, y = c * 64., r * 64.
        name = f'{kind}-{n}'
        if kind == 'normal':
            blocks.append(Platform(name, x, y))
        elif kind == 'breakable':
            devices.append(Device(name, 301, x, y))
        elif kind == 'spring':
            devices.append(Device(name, 401, x, y))
        elif kind in ('timed', 'timed_off'):
            # Each block runs its own fixed phase from stage start, on every retry.
            offset = (n % 3) * .15
            devices.append(Device(name, 302, x, y, options={
                'period': 3.6, 'active_for': 2.6, 'phase': (.2 if kind == 'timed' else 2.7) + offset}))
        elif kind.startswith('launcher_'):
            devices.append(Device(name, 402, x, y, options={'direction': -1 if kind.endswith('left') else 1}))
        elif kind == 'spike':
            devices.append(Device(name, 201, x, y - 64, options={'teeth': 6}))
        elif kind == 'sun':
            devices.append(Device(name, 701, x, y))
        elif kind.startswith('icicle'):
            devices.append(Device(name, 702, x, y, options={
                'period': 4., 'delay': .5, 'phase': .2 if kind.endswith('forming') else .5}))
        elif kind == 'moving':
            devices.append(Device(f'ferry-{n}', 501, (low + c - initial) * 64, y,
                                  options={'travel': travel, 'phase': phase, 'speed': 96.}))
        elif kind == 'moving_vertical':
            rail = next(rail for rail in data['rails'] if rail[0][0] == rail[-1][0])
            top, bottom = sorted((rail[0][1] - .5, rail[-1][1] - .5))
            devices.append(Device(name, 508, x, top * 64, options={
                'travel': (bottom - top) * 64, 'phase': (r - top) * 64, 'speed': 64.}))
        else:
            raise ValueError(f'Unknown stage tile {kind}')
    for n, patrol in enumerate(data.get('patrols', [])):
        c, r, w, h = patrol['bounds']
        along = patrol['along']
        phase = w * along * 64 if patrol['edge'] == 'top' else (w + h + w * (1 - along)) * 64
        devices.append(Device(f'sea-lion-{n}', 703, c * 64, r * 64, options={
            'path': (c * 64, r * 64, w * 64, h * 64), 'phase': phase,
            'speed': 64., 'size': patrol['size']}))
    items = [PowerUp(c * 64, r * 64, kind, 48.) for c, r, kind in data['items']]
    movement_markers=[(c*64,r*64) for c,r in data['markers']]
    if index==3:
        # Mark the outermost occupied tiles of the whole two-tile deck, rather
        # than using the first tile's right endpoint for both ends.
        deck=[d for d in devices if d.type_id==501]
        left=min(d.x for d in deck)
        right=max(d.x+d.options['travel']+d.w for d in deck)
        movement_markers=[(left,deck[0].y),(right-64,deck[0].y)]
    # Rendering adds the same 4px snow overlap as the document image.
    gx, gy, gw, gh = data['goal']
    return Level(index, TITLES[index - 1], HINTS[index - 1], blocks, [], [],
                 tuple(data['spawn']), (gx, gy - 4, gw, gh),
                 height=data['height'], width=data['width'], time_limit=180.,
                 preview=f'assets/stages/stage_{index:02d}.png', powerups=items, specials=devices,
                 movement_markers=movement_markers,show_movement_rails=index!=3)
