"""강화학습 탭 1번 맵에 A의 행동·관측·블록별 보상을 적용한다.

원래 맵과 물리를 그대로 사용한다. 커진 맵의 모든 객체·구간을 전달하기 위해
입력 용량만 확장한다. 시연, 이전 가중치, 중간 시작점, 생존 보상은 사용하지 않는다.
"""
from collections import Counter
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pingu_ball.engine import World
from pingu_ball.playfield import expand_playfield
from pingu_ball.rl_maps import make_rl_level
from .all_tutorial_env import AllTutorialEnv, ENTITY_DIM, GRID, merge_rectangles
from .map_input_env import MapInputEnv, LOCAL_COUNT
from .milestone_env import Milestone, MilestoneTracker, MilestoneRawEnv, terminal_points
from .stabilized_env import ACTIONS


def definitions():
    """연결된 이동 발판은 한 그룹으로 취급하고 다음 실제 착지에만 +10을 준다."""
    group = lambda name, n: tuple(f'{name}-{i}' for i in range(n))
    return (
        Milestone('01-break-a-b', ('z01break-a-0',), ('z01break-b-0',)),
        Milestone('01-break-dock', ('z01break-b-0',), group('z01dock', 3)),
        Milestone('02-spring-upper', ('z02spring-0',), group('z02upper', 3)),
        Milestone('03-clock-dock', ('z03clock-0',), group('z03dock', 3)),
        Milestone('04-ferry-dock', group('z04ferry', 2), group('z04dock', 3)),
        Milestone('05-lift-upper', ('z05lift-0',), group('z05upper', 2)),
        *(Milestone(f'turn-spring-{i}', (f'turn0-spring{i}-0',),
            (f'turn0-spring{i+1}-0',) if i < 3 else group('turn0buffer', 3)) for i in range(4)),
        Milestone('06-spring-upper', ('z06spring-0',), group('z06upper', 3)),
        Milestone('07-break-a-b', ('z07break-a-0',), ('z07break-b-0',)),
        Milestone('07-break-dock', ('z07break-b-0',), group('z07dock', 3)),
        Milestone('08-lift-upper', ('z08lift-0',), group('z08upper', 2)),
        Milestone('09-ferry-dock', group('z09ferry', 2), group('z09dock', 3)),
        Milestone('10-clock-dock', ('z10clock-0',), group('z10dock', 3)))


SPECS = definitions()
SPRINGS = tuple(s.source[0] for s in SPECS if 'spring' in s.id)
LIFTS = ('z05lift-0', 'z08lift-0')
CAPACITY = 35
FIELDS = ENTITY_DIM + 6
GLOBALS = 29 + len(SPECS)*11
TARGET_CAPACITY = max(len(s.target) for s in SPECS if 'spring' in s.id)
EXTRA_SIZE = len(SPRINGS)*3 + 2 + TARGET_CAPACITY*2 + len(LIFTS)*6 + 2
OBS_SIZE = GLOBALS + (CAPACITY+LOCAL_COUNT)*FIELDS + EXTRA_SIZE
LABELS = ('아래 파괴①', '아래 파괴②', '아래 점프', '아래 시간', '아래 좌우 이동',
          '아래 상하 이동', '회전 점프①', '회전 점프②', '회전 점프③', '회전 점프④',
          '위층 점프', '위층 파괴①', '위층 파괴②', '위층 상하 이동', '위층 좌우 이동', '위층 시간')


class MapAuxiliary:
    """A의 금액과 최초 지급 규칙을 이 맵의 6개 점프·2개 리프트에 확장한다."""
    def __init__(self, world):
        self.targets = {s.source[0]:s.target for s in SPECS if 'spring' in s.id}
        self.platforms = {p.id:p for p in world.platforms}
        self.active_source = None
        self.active_since = -1.
        self.inside_height, self.inside_horizontal = {}, {}
        self.height_paid, self.horizontal_paid = set(), set()
        self.first, self.high, self.upper, self.approach_armed = set(), set(), set(), set()
        self.last_lift_time = dict.fromkeys(LIFTS, -1000.)
        self.approach_progress = dict.fromkeys(LIFTS, 0.)
        self.lift_targets = {s.source[0]:s.target for s in SPECS if 'lift' in s.id}
        # 왕복 맵에서 반대 층의 x좌표만으로 보상을 받지 않도록 앞쪽 발판 착지로 활성화한다.
        self.approach_groups = {LIFTS[0]:tuple(f'z04dock-{i}' for i in range(3)),
                                LIFTS[1]:tuple(f'z07dock-{i}' for i in range(3))}
        self.directions = dict(zip(LIFTS, (1, -1)))

    @staticmethod
    def event(kind, points, source):
        return dict(kind=kind, points=float(points), id=source)

    def inside(self, world, target):
        x,y,width,_ = self.platforms[target].rect_at(world.time, world.physics)
        b,r = world.ball,world.physics.radius
        return abs(b.y+r-y)<=16., b.x+r>=x and b.x-r<=x+width

    def motion(self, world):
        if world.failed:
            return []
        events = []
        source = self.active_source
        if source is not None and world.time-self.active_since > 3.:
            self.active_source = None
        elif source is not None and not world.cleared:
            for target in self.targets[source]:
                height,horizontal = self.inside(world,target)
                if height and not self.inside_height.get(target,False) and source not in self.height_paid:
                    self.height_paid.add(source);events.append(self.event('spring_height',.5,source))
                if horizontal and not self.inside_horizontal.get(target,False) and source not in self.horizontal_paid:
                    self.horizontal_paid.add(source);events.append(self.event('spring_horizontal',.5,source))
                self.inside_height[target],self.inside_horizontal[target] = height,horizontal
        for lift in self.approach_armed:
            device = world.special.devices[lift]
            direction = self.directions[lift]
            start = device.x+device.w/2-direction*288.
            progress = float(np.clip((world.ball.x-start)*direction,0.,288.))
            if progress > self.approach_progress[lift]:
                events.append(self.event('approach',(progress-self.approach_progress[lift])*.045,lift))
                self.approach_progress[lift] = progress
        return events

    def landing(self, world, platform):
        if world.failed:
            return []
        events = []
        if platform in self.targets:
            self.active_source,self.active_since = platform,float(world.time)
            pairs = {q:self.inside(world,q) for q in self.targets[platform]}
            self.inside_height = {q:v[0] for q,v in pairs.items()}
            self.inside_horizontal = {q:v[1] for q,v in pairs.items()}
        else:
            self.active_source = None
            self.inside_height,self.inside_horizontal = {},{}
        for lift in LIFTS:
            if platform in self.approach_groups[lift]:
                self.approach_armed.add(lift)
            if platform == lift:
                self.last_lift_time[lift] = float(world.time)
                if lift not in self.first:
                    self.first.add(lift);events.append(self.event('lift_first',12.,lift))
            elif platform in self.lift_targets[lift] and lift in self.first and lift not in self.upper:
                self.upper.add(lift);events.append(self.event('upper_first',18.,lift))
        return events

    def after_landings(self, world):
        if world.failed:
            return []
        events = []
        for lift in LIFTS:
            if lift not in self.first or lift in self.high:
                continue
            d = world.special.devices[lift]
            lx,ly,width,_ = world.special.platforms[lift].rect_at(world.time,world.physics)
            b = world.ball
            if (ly<=d.y+14. and lx-12.<=b.x<=lx+width+12. and ly-130.<=b.y<=ly+18.
                    and world.time-self.last_lift_time[lift]<=1.05):
                self.high.add(lift);events.append(self.event('lift_high',12.,lift))
        return events

    def observation(self, world):
        active = self.active_source
        values = [float(active==s) for s in SPRINGS]
        age = max(0.,world.time-self.active_since) if active is not None else 0.
        values += [min(age/3.,1.),max(0.,1.-age/3.) if active is not None else 0.]
        values += [float(s in self.height_paid) for s in SPRINGS]
        values += [float(s in self.horizontal_paid) for s in SPRINGS]
        for history in (self.inside_height,self.inside_horizontal):
            targets = self.targets[active] if active is not None else ()
            values += [float(history.get(targets[i],False)) if i<len(targets) else 0.
                       for i in range(TARGET_CAPACITY)]
        for lift in LIFTS:
            values += [float(lift in h) for h in (self.first,self.high,self.upper)]
            values += [min(max(world.time-self.last_lift_time[lift],0.)/3.,1.) if lift in self.first else 0.,
                       self.approach_progress[lift]/288.,float(lift in self.approach_armed)]
        return values

    def state(self):
        return dict(active_spring=self.active_source, height_paid=sorted(self.height_paid),
            horizontal_paid=sorted(self.horizontal_paid), lift_first=sorted(self.first),
            lift_high=sorted(self.high), upper_first=sorted(self.upper),
            approach_armed=sorted(self.approach_armed),approach_progress=dict(self.approach_progress))


class MapARawEnv(MapInputEnv):
    bounds = MilestoneRawEnv.bounds

    def __init__(self, limit=180.):
        gym.Env.__init__(self)
        self.stage,self.limit,self.repeat = 1,float(limit),12
        self.variant = 'local'
        self.level = expand_playfield(make_rl_level(1))
        self.terrain = merge_rectangles(self.level.platforms)
        self.n_entities = len(self.terrain)+len(self.level.specials)+len(self.level.powerups)
        # 맵1의 오른쪽 발판 삭제 후 남는 빈 관측 슬롯은 0으로 채워 입력 크기를 유지한다.
        assert self.n_entities <= CAPACITY
        self.max_entities = CAPACITY
        self.specs = SPECS
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict(dict(
            global_state=spaces.Box(-8.,8.,(GLOBALS,),dtype=np.float32),
            entities=spaces.Box(-8.,8.,(CAPACITY,FIELDS),dtype=np.float32),
            local=spaces.Box(-8.,8.,(LOCAL_COUNT,FIELDS),dtype=np.float32),
            extra=spaces.Box(-8.,8.,(EXTRA_SIZE,),dtype=np.float32)))
        self.reset()

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self,seed=seed)
        self.world = World(self.level,auto_reset=False)
        self.done = False
        self.ticks = self.decisions = 0
        self.previous_action = 1
        self.last_direction = self.last_duration = 0
        self.raw_base_return = self.shaping_return = self.bonus_return = self.episode_return = 0.
        self.bonus_credit = self.pickups = self.power_uses = 0
        self.visited = np.zeros(GRID[0]*GRID[1],np.float32)
        self.visited[self.cell()] = 1.
        self.min_goal_distance = self.goal_distance()
        self.static_hazard_bounds = {}
        self.tracker = MilestoneTracker(self.specs)
        self.auxiliary = MapAuxiliary(self.world)
        self.reward_parts = Counter()
        self.reward_events = []
        self.platform_by_id = {p.id:p for p in self.world.platforms}
        self.platform_by_id.update(self.world.special.platforms)
        assert all(p in self.platform_by_id for s in self.specs for p in s.source+s.target)
        self.touched,self.landing_index,self.support_lookup = set(),None,{}
        for p in self.level.platforms:
            for i,(x,y,w,h,kind) in enumerate(self.terrain):
                if p.kind==kind and x<=p.x and y<=p.y and p.x+p.w<=x+w and p.y+p.h<=y+h:
                    self.support_lookup[p.id] = i
                    break
        self.support_lookup.update({d.id:i+len(self.terrain) for i,d in enumerate(self.level.specials)})
        self.touched.add(self.support_lookup['start-0'])
        return self.observe(),self.info()

    def observe(self):
        base = AllTutorialEnv.observe(self)
        records,b = self.entity_records(),self.world.ball
        entities = np.zeros((CAPACITY,FIELDS),np.float32)
        entities[:,:ENTITY_DIM] = base['entities']
        from .all_tutorial_env import distance_feature
        for i,row in enumerate(records):
            if row['endpoints']:
                (x1,y1),(x2,y2) = row['endpoints']
                entities[i,ENTITY_DIM:ENTITY_DIM+4] = [distance_feature(x1-b.x),distance_feature(y1-b.y),
                                                       distance_feature(x2-b.x),distance_feature(y2-b.y)]
            entities[i,-2:] = float(i in self.touched),float(i==self.landing_index)
        local = np.zeros((LOCAL_COUNT,FIELDS),np.float32)
        for j,i in enumerate(self.local_indices(records)):
            local[j] = entities[i]
            x,y,w,h = records[i]['rect']
            local[j,:6] = (x-b.x)/256,(x+w-b.x)/256,(y-b.y)/256,(y+h-b.y)/256,w/64,h/64
        global_state = np.zeros(GLOBALS,np.float32)
        global_state[:24] = base['global'][:24]
        global_state[15] = self.previous_action/2
        global_state[20] = len(self.tracker.completed)/len(self.specs)
        global_state[24] = 1.
        for i,spec in enumerate(self.specs):
            values = []
            for ids in (spec.source,spec.target):
                x,y,w,h = self.bounds(ids)
                values += [distance_feature(x-b.x),distance_feature(y-b.y),
                           distance_feature(x+w-b.x),distance_feature(y+h-b.y)]
            values += [1.,float(spec.id in self.tracker.armed),float(spec.id in self.tracker.completed)]
            global_state[29+i*11:29+(i+1)*11] = values
        extra = np.asarray(self.auxiliary.observation(self.world)+[self.last_direction,self.last_duration/72.],np.float32)
        assert extra.shape == (EXTRA_SIZE,)
        return dict(global_state=np.clip(global_state,-8,8),entities=np.clip(entities,-8,8),
                    local=np.clip(local,-8,8),extra=np.clip(extra,-8,8))

    def info(self, tick_delta=0):
        result = AllTutorialEnv.info(self,tick_delta)
        result.update(map_kind='reinforcement',map_number=1,reward_variant='map-A',
            milestones=sorted(self.tracker.completed),armed=sorted(self.tracker.armed),
            milestone_events=list(self.tracker.last_events),reward_events=list(self.reward_events),
            reward_parts=dict(self.reward_parts),auxiliary_state=self.auxiliary.state(),
            learning_return=self.episode_return/100.,direction=self.last_direction,requested_ticks=self.last_duration)
        return result

    def step(self, action, on_tick=None):
        if self.done:
            raise RuntimeError('종료 후 reset()이 필요하다')
        if not self.action_space.contains(action):
            raise ValueError(action)
        direction,frames = ACTIONS[int(action)]
        self.last_direction,self.last_duration = direction,frames
        before = self.ticks
        self.reward_events = []
        for _ in range(frames):
            dt = min(self.world.physics.dt,self.limit-self.world.time)
            if dt<1e-10 or self.world.failed or self.world.cleared:
                break
            self.world.step(direction,dt)
            self.ticks += 1
            self.min_goal_distance = min(self.min_goal_distance,self.goal_distance())
            self.reward_events.extend(self.auxiliary.motion(self.world))
            for event,value in self.world.events:
                if event=='land' and not self.world.failed:
                    if value in self.support_lookup:
                        self.landing_index = self.support_lookup[value]
                        self.touched.add(self.landing_index)
                    for milestone in self.tracker.land(value,self.ticks):
                        self.reward_events.append(dict(kind='milestone',id=milestone,points=10.))
                    self.reward_events.extend(self.auxiliary.landing(self.world,value))
            self.reward_events.extend(self.auxiliary.after_landings(self.world))
            if on_tick:
                on_tick(self.world,self.ticks,direction)
        self.previous_action,self.decisions = direction+1,self.decisions+1
        timeout = self.world.time>=self.limit-1e-9
        self.done = bool(self.world.cleared or self.world.failed or timeout)
        base = terminal_points(self.world.cleared,self.world.failed,timeout)
        bonus = sum(e['points'] for e in self.reward_events)
        if self.done:
            self.reward_events.append(dict(kind='success' if base>0 else 'failure',points=base))
        for event in self.reward_events:
            self.reward_parts[event['kind']] += event['points']
        self.raw_base_return += base
        self.bonus_return += bonus
        self.episode_return += base+bonus
        self.visited[self.cell()] = 1.
        return self.observe(),(base+bonus)/100.,self.done,False,self.info(self.ticks-before)


class MapAEnv(gym.ObservationWrapper):
    def __init__(self, **kwargs):
        super().__init__(MapARawEnv(**kwargs))
        self.observation_space = spaces.Box(-8.,8.,(OBS_SIZE,),dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs['global_state'],obs['entities'].ravel(),obs['local'].ravel(),obs['extra']]).astype(np.float32)


def audit_reward(info):
    parts = info['reward_parts']
    assert abs(sum(parts.values())-info['episode_return'])<1e-6
    assert abs(info['learning_return']*100-info['episode_return'])<1e-6
    assert parts.get('milestone',0.)==10.*len(info['milestones'])
    assert len(info['milestones'])==len(set(info['milestones']))
    assert parts.get('success',0.)+parts.get('failure',0.)==(100. if info['success'] else -25.)
    assert not (parts.get('success') and parts.get('failure'))
    for key,cap in dict(spring_height=len(SPRINGS)*.5,spring_horizontal=len(SPRINGS)*.5,
                        lift_first=24.,lift_high=24.,upper_first=36.,approach=25.92).items():
        assert -1e-8<=parts.get(key,0.)<=cap+1e-6
