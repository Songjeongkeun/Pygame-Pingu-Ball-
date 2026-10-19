"""Interactive per-device test rooms reachable from the game's stage menu."""
from .engine import Level,Platform,Collectible
from .specials import CATALOG,Device,SOLID


HINTS = {
    301:'착지 0.5초 후 파괴됩니다. R로 복구합니다.',401:'착지하면 144px 높이로 자동 반동합니다.',
    302:'활성 2초 / 전체 2.8초 주기. 사라지면 통과합니다.',403:'착지 0.1초 후 발판 자체가 낙하합니다.',
    307:'착지 후 2초 동안 좌우 최대 속도가 1.5배입니다.',501:'128px 구간을 좌우 왕복합니다.',
    508:'발판이 128px 구간을 위아래 왕복합니다.',507:'벽 옆면을 누르면 낙하 속도가 초당 48px로 제한됩니다.',
    511:'재착지할 때마다 오른쪽 블록을 당기기 / 밀기로 전환합니다.',512:'재착지하면 연결 블록 위치를 128px 전환합니다.',
    304:'금색 생선 1개를 얻은 뒤 착지하면 연결 장벽이 열립니다.',404:'착지할 때마다 오른쪽 대포의 전원을 전환합니다.',
    305:'왼쪽 스위치에 착지하면 장벽이 열립니다.',415:'착지하면 오른쪽 연결 발판을 켜고 끕니다.',
    416:'왼쪽 스위치로 켜면 실제 발판이 됩니다.',510:'전기가 흐르는 발판입니다. 접촉 자체는 치명적이지 않습니다.',
    306:'빨강은 접촉 0.85초 뒤 치명적, 파랑은 0.5초 뒤 반발합니다.',
    402:'화살표 방향 수평 발사. 방향키를 누르면 발사를 해제합니다.',
    405:'재착지할 때마다 위 / 오른쪽 / 아래 / 왼쪽으로 발사합니다.',
    406:'재착지할 때마다 8방향으로 발사합니다. 방향키로 해제합니다.',
    407:'블랙홀에 닿으면 짝 화이트홀 출구로 이동합니다.',408:'화이트홀도 짝 블랙홀로 이동합니다.',
    410:'짝 포털로 이동합니다. 출구가 막히면 전송하지 않습니다.',
    303:'1초 활성 / 1초 비활성. 활성 중 접촉하면 재시작합니다.',
    201:'보이는 삼각형에 접촉하면 재시작합니다.',202:'발판 폭의 1/4인 작은 삼각형만 위험합니다.',
    211:'반투명한 유리 가시도 접촉하면 재시작합니다.',221:'밝게 빛나는 가시도 접촉하면 재시작합니다.',
    502:'빨강 레이저는 치명적이며 파랑 레이저는 공을 밀어냅니다.',
    503:'가로에서 시작해 초당 90도로 회전합니다. 빨강 / 파랑 효과가 다릅니다.',
    504:'세로에서 시작해 초당 90도로 회전합니다.',601:'소행성이 실제 궤도로 이동하며 충돌하면 재시작합니다.',
    603:'1.5초마다 발사체를 생성합니다. 발사체도 연속 충돌을 검사합니다.',
    602:'좌우로 이동하면서 튀는 위험물입니다. 발판이 아닙니다.',
    604:'단검이 자유 이동하고 지정 경계에서 반사합니다.',605:'단검이 지정된 벽의 둘레를 순환합니다.',
    606:'위에서 하강하며 밟으면 제거합니다. 옆면 접촉은 치명적입니다.',
}


def make_lab(type_id):
    if type_id not in CATALOG: raise ValueError('Unknown device')
    floor=[Platform(f'floor-{i}',i*64.,704.) for i in range(25)]
    floor += [Platform('step-1',256.,640.),Platform('step-2',320.,576.)]
    solid=type_id in SOLID
    x,y=(384.,512.) if solid else (640.,640.)
    options={}
    group='demo'
    if type_id==501: options={'travel':128.}
    elif type_id==508: options={'travel':128.,'speed':64.}
    elif type_id==507: x,y=448.,384.; options={'slide_speed':48.}
    elif type_id in (511,512): options={'radius':400.,'offset':(128.,0.)}
    elif type_id==510: options={'power_group':'demo'}
    elif type_id in (502,503,504): x,y=640.,576.; options={'length':192.,'color':'red'}
    elif type_id==601: x,y=1000.,450.; options={'velocity':(-96.,32.)}
    elif type_id==604: options={'bounds':(576.,480.,1056.,640.),'velocity':(96.,64.)}
    elif type_id==605: options={'path':(640.,480.,192.,160.)}
    elif type_id==606: y=656.
    primary=Device('main',type_id,x,y,group=group,options=options)
    devices=[primary]
    fish=[]
    if type_id in (511,512):
        primary.target='movable'; floor.append(Platform('movable',640.,512.))
    if type_id==304:
        fish=[Collectible(416.,410.)]
        devices.append(Device('gate',305,768.,640.,group=group))
    elif type_id==404:
        devices.append(Device('connected',603,768.,576.,group=group))
    elif type_id==305:
        devices.append(Device('switch',415,192.,640.,group=group))
    elif type_id==415:
        devices.append(Device('linked',416,640.,512.,group=group))
    elif type_id==416:
        devices.append(Device('switch',415,192.,640.,group=group))
    elif type_id==510:
        devices.append(Device('power',404,256.,576.,group=group,options={'on':True}))
    elif type_id==306:
        devices.append(Device('blue',306,640.,512.,options={'color':'blue'}))
    elif type_id in (407,408,410):
        paired=408 if type_id==407 else 407 if type_id==408 else 410
        devices.append(Device('exit',paired,960.,512.,group=group))
    elif type_id in (502,503,504):
        devices.append(Device('blue',type_id,1024.,512.,options={'length':160.,'color':'blue'}))
    if type_id==507: spawn=(434.,400.)
    else: spawn=(x+32,y-110.) if solid else (288.,594.)
    if type_id==501: spawn=(x+128,y-110.)
    return Level(0,f'{type_id} {CATALOG[type_id][0]}',HINTS[type_id],floor,fish,[],spawn,
                 (1472.,640.,64.,64.),specials=devices)
