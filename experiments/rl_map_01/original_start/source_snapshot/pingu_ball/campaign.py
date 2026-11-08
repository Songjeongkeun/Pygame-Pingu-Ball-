"""스테이지.docx의 일반 스테이지 12개를 게임의 실제 충돌 지형으로 구성한다."""
import json
import copy
from pathlib import Path

from .config import CAMPAIGN_ID, REGULAR_STAGE_COUNT
from .engine import Collectible, Level, Platform, PowerUp
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
         '이동 발판으로 왼쪽에 돌아가 위쪽 발사 블록을 타고 이글루로 가세요.',
         '점프는 ↑↑, 대시는 ←←·→→로 사용하세요.',
         '파괴 발판과 엘리베이터를 이용해 왼쪽 이글루에 도착하세요.',
         '고드름을 피하며 점프 아이템과 발사 발판을 이용하세요.',
         '태양 사이의 발판을 건너 오른쪽 아래 이글루로 가세요.',
         '발판 둘레를 도는 바다사자를 피해 오른쪽 아래로 가세요.',
         '블록과 아이템을 조합해 오른쪽 아래 이글루에 도착하세요.')

# 일반·점프 블록의 높이는 96·144px로 유지한다. 기존 장치 식별자와
# 타이머의 시작 위상이 바뀌지 않도록 추가 타일은 뒤에 붙인다.
TILE_REMOVALS = {4: {('spike', 3, 13)}}
TILE_ADDITIONS = {5: [{'kind': 'spring', 'c': 18, 'r': 7},
                      {'kind': 'spring', 'c': 16, 'r': 5},
                      {'kind': 'normal', 'c': 15, 'r': 7}],
                  7: [{'kind': 'normal', 'c': c, 'r': 14} for c in (7,11,15)],
                  11: [{'kind': 'normal', 'c': 23, 'r': 8}]}
POSITION_ADJUSTMENTS = {
    3: {('spring',23,10):(21,9), ('spring',20,12):(19,11),
        ('spring',17,14):(17,13)},
    4: {('spring',23,9):(23,10), ('normal',21,6):(21,8),
        ('normal',20,5):(20,7), ('normal',19,5):(18,6)},
    5: {('spring',18,10):(18,11), ('spring',16,7):(16,9)},
    6: {('normal',10,7):(10,8), ('normal',22,7):(22,8),
        ('launcher_right',12,6):(12,7),
        ('moving',12.15,10):(12.15,11), ('moving',13.15,10):(13.15,11)},
    7: {('spring',7,14):(5,12), ('spring',11,14):(7,10),
        ('spring',15,14):(9,8), ('spring',5,8):(11,6),
        ('spring',2,5):(13,4),
        ('moving',12.8,5):(13,5), ('moving',13.8,5):(14,5)},
    8: {('spring',4,9):(4,10), ('breakable',6,6):(6,8),
        ('breakable',7,6):(7,7)},
    9: {('breakable',13,9):(13,10), ('breakable',15,9):(15,10),
        ('breakable',17,9):(17,10), ('breakable',20,9):(19,10),
        ('spring',23,9):(21,9), ('launcher_left',21,6):(20,7), ('normal',1,7):(1,8)},
    # 시작점 아래 블록과 옆의 64px 높이 접근 계단을 유지한다.
    # 추가한 천장 타일로 시작점에서 이글루로 바로 떨어지는 경로를 막는다.
    11: {('spring',6,9):(6,10), ('spring',10,12):(10,11),
         ('normal',20,12):(19,11), ('normal',23,8):(21,9)},
    12: {('spring',13,12.5):(13,11.5), ('spring',34,6.25):(34,5),
         # 의도한 반 칸 높이를 유지하면서 다섯 열 모두를
         # 인접한 점프 블록보다 128px 위에 맞춘다.
         **{('normal',c,r):(c,aligned_r) for c in (14,16,18,20,22)
            for r,aligned_r in ((9.4,9.5),(10.4,10.5))}},
}


def fit_jump_layout(source):
    """보관한 문서 원본 데이터는 수정하지 않고 게임에 사용할 배치를 반환한다."""
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
            # 태양 13개는 모두 유지하되, 상승하는 점프 경로를 가리지 않게 장벽을 옮긴다.
            tile['c']+=9
    if data['number']==7:
        data['items']=[[13.5,3.5,kind] if kind=='up' else [c,r,kind]
                       for c,r,kind in data['items']]
        # 오른쪽 아래 생선 옆에 대시 아이템 하나를 추가한다.
        data['items'].append([22.5,13.5,'forward'])
        # 이어진 이동 발판의 왼쪽 반환점을 위쪽 점프 블록과 맞춘다.
        rail=next(rail for rail in data['rails'] if rail[0][1]==rail[-1][1])
        min(rail,key=lambda point:point[0])[0]=13.5
    if data['number']==4:
        # 위치를 조정한 최상단 접근 발판 위에 대시 아이템을 유지한다.
        data['items']=[[18.5,5.5,kind] if kind=='forward' else [c,r,kind]
                       for c,r,kind in data['items']]
        # 늘린 왼쪽 반환점을 32px 단위의 반 칸 격자에 맞춘다.
        # 기존 발판에서 위상을 계산해 처음 시작하는 위치를 보존한다.
        rail=next(rail for rail in data['rails'] if rail[0][1]==rail[-1][1])
        left=min(rail,key=lambda point:point[0])
        left[0]=7
    if data['number']==3:
        # 두 칸짜리 발판은 위쪽 선반의 출구(x=768) 아래까지 뻗어야 한다.
        # 다음 가시 열이 시작되기 전인 x=832에서 오른쪽 가장자리가 끝나게 한다.
        data['rails'][0]=[[4.5,8.5],[11.5,8.5]]
    if data['number']==6:
        # 발판 전체와 이동 범위를 함께 내려 오른쪽 점프 블록보다 128px 위에 둔다.
        data['rails']=[[[c,r+1] for c,r in rail] for rail in data['rails']]
        data['markers']=[[c,r+1] for c,r in data['markers']]
    if data['number']==8:
        # 원본에는 승강기의 이동 경로만 기록되어 있다. 가로 이동 발판은 왼쪽에 두어
        # 시작 쪽 지형과 x=768의 섬을 연결한다.
        data['rails'].append([[3.5,13.5],[11.5,13.5]])
        data['markers'].extend([[3,13],[11,13]])
    data['tiles'].extend(copy.deepcopy(TILE_ADDITIONS.get(data['number'], [])))
    return data


def campaign_layouts():
    data = json.loads((ROOT / 'assets/stages/layouts.json').read_text(encoding='utf-8'))
    if data['campaign'] != CAMPAIGN_ID or len(data['maps']) != REGULAR_STAGE_COUNT:
        raise ValueError('Invalid document campaign data')
    return data['maps']


def collectible_spec():
    return json.loads((ROOT / 'assets/stages/collectibles.json').read_text(encoding='utf-8'))


def collectible_layouts():
    """파괴 블록 안에 놓인 수집물까지 포함한 승인된 중심 좌표를 반환한다."""
    return collectible_spec()['maps']


def make_campaign_level(index):
    if type(index) is not int or not 1 <= index <= REGULAR_STAGE_COUNT:
        raise ValueError(f'stage must be between 1 and {REGULAR_STAGE_COUNT}')
    data = fit_jump_layout(campaign_layouts()[index - 1])
    blocks, devices = [], []
    movers = [q for q in data['tiles'] if q['kind'] == 'moving']
    horizontal = [rail for rail in data['rails'] if rail[0][1] == rail[-1][1]]
    # 이어진 이동 발판을 구성하는 모든 타일은 속도와 위상을 같게 유지한다.
    if movers:
        initial = min(q['c'] for q in movers)
        rail = horizontal[0] if horizontal else [[26.5, movers[0]['r'] + .5], [32.5, movers[0]['r'] + .5]]
        low, high = sorted((rail[0][0] - .5, rail[-1][0] - .5))
        travel = (high - low) * 64
        phase = (initial - low) * 64
    for n, q in enumerate(data['tiles']):
        c, r, kind = q['c'], q['r'], q['kind']
        # 7번 생선뼈 위의 블록 두 개를 제거하되, 이후 장치의 식별자와
        # 시간 블록의 위상은 변경하지 않는다.
        if index==7 and kind=='breakable' and (c,r) in ((7,8),(9,8)):
            continue
        x, y = c * 64., r * 64.
        name = f'{kind}-{n}'
        if kind == 'normal':
            blocks.append(Platform(name, x, y))
        elif kind == 'breakable':
            devices.append(Device(name, 301, x, y))
        elif kind == 'spring':
            devices.append(Device(name, 401, x, y))
        elif kind in ('timed', 'timed_off'):
            # 재시작할 때마다 각 블록은 스테이지 시작 기준의 고정 위상으로 동작한다.
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
        # 첫 타일의 오른쪽 끝을 양쪽에 사용하는 대신, 두 칸짜리 발판 전체에서
        # 가장 바깥쪽을 차지하는 타일 위치를 이동 범위의 끝으로 표시한다.
        deck=[d for d in devices if d.type_id==501]
        left=min(d.x for d in deck)
        right=max(d.x+d.options['travel']+d.w for d in deck)
        movement_markers=[(left,deck[0].y),(right-64,deck[0].y)]
    # 문서 이미지와 동일하게 렌더링할 때 눈을 4px 겹쳐 표시한다.
    gx, gy, gw, gh = data['goal']
    pickups=collectible_spec()
    collectibles=[Collectible(q['x'],q['y'],q['value'],pickups['size'],pickups['height'])
                  for q in pickups['maps'][index-1]['collectibles']]
    return Level(index, TITLES[index - 1], HINTS[index - 1], blocks, collectibles, [],
                 tuple(data['spawn']), (gx, gy - 4, gw, gh),
                 height=data['height'], width=data['width'], time_limit=180.,
                 preview=f'assets/stages/stage_{index:02d}.png', powerups=items, specials=devices,
                 movement_markers=movement_markers,show_movement_rails=index not in (3,6))
