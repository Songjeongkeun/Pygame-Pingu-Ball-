"""Eight authored stages. Coordinates are multiples of the 20px tile."""
from .engine import Platform, Collectible, Spike, Level


TITLES = [
    ('첫 바운스', '좌우 방향키로 움직이고, 생선은 선택해서 모으세요.'),
    ('금이 간 빙하', '착지한 얼음은 0.5초 뒤 깨집니다. 멈추지 말고 이어 가세요.'),
    ('가시 해협', '가시 위를 건너고, 위쪽 길에서 금색 생선을 모으세요.'),
    ('무너지는 길', '깨지는 발판과 가시를 함께 통과하세요.'),
    ('더 높은 바운스', '분홍 점프대는 90px. 긴 틈 앞에서 점프대에 착지하세요.'),
    ('움직이는 얼음', '왕복 발판과 공은 독립적으로 움직입니다. 다음 착지를 예측하세요.'),
    ('얼음의 시간', '청록 발판은 사라집니다. 점멸을 보고 안전한 곳에서 기다리세요.'),
    ('집으로 가는 길', '빨간 스위치로 A 다리를 열고, 모든 규칙을 활용해 집으로 돌아가세요.'),
]


def make_levels():
    return [make_level(i) for i in range(8)]


def make_level(index):
    blocks=[]
    def block(name,x,y,w=20,kind='normal',**kwargs):
        blocks.append(Platform(name,x,y,w,20,kind,**kwargs))
    def strip(name,x,end,y,kind='normal',**kwargs):
        if kind=='breakable':
            for n,at in enumerate(range(x,end,20)):
                block(f'{name}-{n}',at,y,20,kind,**kwargs)
        else:
            block(name,x,y,end-x,kind,**kwargs)

    if index==0:
        strip('start',40,260,500)
        strip('island',320,540,500)
        strip('home',600,1160,500)
        block('practice-step',400,460,80)
        fish=[Collectible(440,405)]
        return Level(0,*TITLES[0],blocks,fish,[],(70.,435.5),(1120.,480.,20.,20.),
                     [(1130.,'goal')],[(350.,'land:island'),(440.,'fish0'),(1130.,'goal')])

    # Main lower route and a higher optional route containing all three bonuses.
    if index==4:
        strip('start',40,420,500)
        block('big-jump',420,500,20,'spring')
        strip('home',700,1160,500)
        base=[(430.,'land:big-jump'),(1130.,'goal')]
    elif index==6:
        strip('start',40,420,500)
        block('time-bridge',420,500,240,'timed',period=3.2,active_for=2.2,phase=.2)
        strip('home',660,1160,500)
        base=[(1130.,'goal')]
    elif index==7:
        strip('start',40,180,500)
        block('switch-A',180,500,20,'switch',group='A')
        strip('approach',200,480,500)
        block('bridge-A',480,500,180,'linked',group='A')
        strip('home',660,1160,500)
        base=[(190.,'switch:A'),(1130.,'goal')]
    elif index in (1,3):
        strip('start',40,380,500)
        strip('crumble-main',380,460,500,'breakable')
        strip('middle',460,860,500)
        strip('crumble-end',860,940,500,'breakable')
        strip('home',940,1160,500)
        base=[(1130.,'goal')]
    elif index==5:
        strip('start',40,420,500)
        block('ferry',420,500,20,'moving',travel=160)
        strip('home',600,1160,500)
        base=[(410.,'land:start'),(1130.,'goal')]
    else:
        strip('start',40,1160,500)
        base=[(1130.,'goal')]

    # Staircase rises 40px per bounce, with generous landing areas.
    block('step-low',180,460,80)
    block('step-high',300,420,100)
    strip('upper-left',440,620,380)
    if index in (1,3):
        strip('crumble-upper',620,780,380,'breakable')
    elif index==6:
        block('upper-time',620,380,160,'timed',period=3.2,active_for=2.4,phase=.1)
    elif index==7:
        block('upper-launch',620,380,20,'launcher',arrow=1)
        strip('upper-mid',640,780,380)
    else:
        strip('upper-mid',620,780,380)
    strip('upper-right',780,1040,380)
    block('return-step',1060,440,40)
    if index==4:
        block('bonus-pad',300,420,100,'spring')
        blocks=[q for q in blocks if q.id!='step-high']
    if index==5:
        block('upper-ferry',760,440,20,'moving',travel=200,phase=90)
        block('small-ferry',500,440,20,'moving',travel=100,phase=40)
    if index==7:
        next(q for q in blocks if q.id=='home').w=420
        strip('home-last',1120,1160,500)
        block('final-crumble',1080,500,40,'breakable')
        block('time-side',920,440,80,'timed',period=2.6,active_for=1.8,phase=.4)
        block('ferry-side',680,440,20,'moving',travel=120,phase=50)
        # The final bonus route climbs through three distinct tiers.
        blocks=[q for q in blocks if q.id not in ('upper-mid','upper-right')]
        strip('peak-step',640,760,340)
        strip('high-ridge',780,900,300)
        strip('summit',920,1040,260)

    spikes=[]
    if index in (2,3):
        spikes=[Spike(600,480),Spike(840,480)]
        base=[(490.,'land:start' if index==2 else 'land:middle'),(1130.,'goal')]
    if index==4:
        # Below the spring-assisted jump, rather than an invisible fail region.
        spikes=[Spike(x,560) for x in range(460,660,40)]
    if index==7:
        spikes=[Spike(740,480),Spike(980,480)]

    fish=[Collectible(480,325),Collectible(720,325),Collectible(960,325),Collectible(540,470,-1)]
    # First bonus is on the second step, so the upper path can be entered naturally.
    fish_path=[(120.,'land:start'),(220.,'land:step-low'),(350.,'land:step-high' if index!=4 else 'land:bonus-pad'),
               (480.,'fish0'),(720.,'fish1'),(960.,'fish2'),(1130.,'goal')]
    if index==7:
        fish=[Collectible(480,325),Collectible(820,245),Collectible(960,205),Collectible(540,470,-1)]
        fish_path=[(190.,'switch:A'),(120.,'land:start'),(220.,'land:step-low'),
                   (350.,'land:step-high'),(480.,'fish0'),(580.,'land:upper-left'),(700.,'land:peak-step'),
                   (820.,'fish1'),(960.,'fish2'),(1130.,'goal')]
    return Level(index,*TITLES[index],blocks,fish,spikes,(70.,435.5),(1120.,480.,20.,20.),base,fish_path)


def waypoint_direction(world, target):
    """Feedback controller used by the validation CLI, not by gameplay."""
    b=world.ball
    distance=target-b.x
    stop=b.vx*abs(b.vx)/(2*world.physics.deceleration)
    if abs(distance)<1.5 and abs(b.vx)<8:
        return 0
    if (distance>0 and b.vx>0 and distance<=stop+.7) or (distance<0 and b.vx<0 and -distance<=-stop+.7):
        return 0
    return 1 if distance>0 else -1


def condition_met(world, condition):
    if condition=='goal':return world.cleared
    if condition.startswith('fish'):return int(condition[4:]) in world.collected
    if condition.startswith('land:'):return world.last_landing==condition[5:]
    if condition.startswith('switch:'):return world.switches.get(condition[7:],False)
    return False
