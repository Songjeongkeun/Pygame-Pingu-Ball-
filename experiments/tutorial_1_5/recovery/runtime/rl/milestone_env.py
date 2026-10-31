"""튜토리얼 1~5에서 공통으로 사용하는 강화학습 환경과 보상 규칙.

보상 규칙:
    이글루 도착: +100점.
    실패 또는 제한 시간 초과: -25점. 동시에 발생해도 한 번만 적용한다.
    핵심 구간 최초 달성: 구간마다 +10점. 한 판에서 같은 구간은 중복 인정하지 않는다.
    일반 이동, 자동 반동, 단순 생존: 그 자체로는 보상이 없다.

핵심 구간은 시작 특수 블록에 실제 착지한 뒤, 지정된 목표 발판 그룹에
실제로 착지해야 달성한다. 특수 블록만 밟거나 목표 좌표를 지나가는 것은 제외한다.
definitions()가 현재 스테이지의 구간만 선택하므로, 없는 블록의 구간은 검사하지 않는다.

보고서와 누적 점수는 원래 점수를 사용한다. 학습기에 반환하는 보상만 /100으로
조정한다. 예를 들어 구간 달성과 성공이 같은 행동에서 발생하면 +110점이며,
학습기에는 1.1을 반환한다. 이전 행동에서 지급한 구간 보상을 다시 지급하지 않는다.
"""
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .five_env import FiveRawEnv, GLOBALS as BASE_GLOBALS, CAPACITY, FIELDS, LOCAL_COUNT
from .all_tutorial_env import distance_feature

# 맵마다 구간 수가 달라도 신경망 입력 크기를 같게 유지하기 위한 최대 구간 수다.
MAX_MILESTONES = 4
# 구간당 입력: 시작·목표 영역 좌표 8개 + 존재 여부 + 시작 착지 여부 + 달성 여부.
MILESTONE_FIELDS = 11
# 기존 공·게임 상태 입력에 핵심 구간 정보를 추가한 크기다.
GLOBALS = BASE_GLOBALS + MAX_MILESTONES*MILESTONE_FIELDS
# 최종 입력은 전체 상태, 전체 맵 객체, 가까운 맵 객체 정보를 차례로 연결한다.
OBS_SIZE = GLOBALS + (CAPACITY+LOCAL_COUNT)*FIELDS


@dataclass(frozen=True)
class Milestone:
    """보상 대상 구간 하나를 구성하는 고정 식별자와 발판 그룹."""

    id: str  # 중복 지급 여부를 구분하는 구간 식별자.
    source: tuple  # 이 그룹의 발판 중 하나에 착지하면 구간 도전을 시작한 것으로 기록한다.
    target: tuple  # 시작 착지 이후 이 그룹의 발판 중 하나에 착지하면 구간을 달성한다.


def definitions(stage):
    """현재 스테이지에 실제로 있는 핵심 구간만 반환한다.

    발판은 좌표 대신 고유 식별자로 연결한다. 이동 발판도 위치가 바뀌어도
    같은 발판으로 추적할 수 있다. 한 그룹의 여러 발판은 하나의 목표로 취급한다.
    각 구간은 독립적으로 판정하며, 목록의 모든 구간을 순서대로 강제하지는 않는다.
    """
    # 예: group('finish', 3)은 finish-0, finish-1, finish-2를 묶는다.
    group = lambda name, n: tuple(f'{name}-{i}' for i in range(n))
    if stage == 1:
        # 파괴 블록 사이의 이동 3구간과 마지막 파괴 블록→도착 지형의 총 4구간.
        return tuple(Milestone(f'break-crossing-{i}', (f'break-{i}',),
                     (f'break-{i+1}',) if i < 3 else group('finish', 3)) for i in range(4))
    if stage == 2:
        # 각 점프 블록을 밟은 뒤 중간층→윗층→도착층에 착지하는 총 3구간.
        return (Milestone('spring-to-middle', ('spring-1-0',), group('middle', 3)),
                Milestone('spring-to-upper', ('spring-2-0',), group('upper', 2)),
                Milestone('spring-to-finish', ('spring-3-0',), group('finish', 4)))
    if stage == 3:
        # 나란한 이동 발판 3개는 같은 그룹이다. 그룹 안에서 이동해도 보상을 주지 않는다.
        return (Milestone('ferry-to-finish', group('ferry', 3), group('finish', 3)),)
    if stage == 4:
        # 상하 이동 발판을 밟은 뒤 도착 지형에 착지하는 1구간.
        return (Milestone('lift-to-finish', ('lift',), group('finish', 8)),)
    if stage == 5:
        # 시간 블록 사이의 이동 2구간과 마지막 시간 블록→도착 지형의 총 3구간.
        return tuple(Milestone(f'clock-crossing-{i}', (f'clock-{i}',),
                     (f'clock-{i+1}',) if i < 2 else group('finish', 4)) for i in range(3))
    raise ValueError('Only tutorials 1..5')


class MilestoneTracker:
    """한 판에서 구간별 시작 착지와 최초 달성을 추적한다."""

    def __init__(self, specs):
        self.specs = specs
        # armed: 시작 발판을 밟은 구간. completed: +10점을 이미 지급한 구간.
        # 두 집합은 다음 판의 reset()에서 새 추적기를 만들 때 초기화된다.
        self.armed, self.completed = set(), set()
        # 현재 판에서 달성한 구간·목표 발판·물리 틱을 기록한다. 보상 계산과 별도인 진단 자료다.
        self.last_events = []
        for spec in specs:
            # 시작과 목표에 같은 발판을 넣어 제자리 착지로 보상을 받는 설정을 막는다.
            if set(spec.source) & set(spec.target):
                raise ValueError('A connected support cannot reward moving within itself')

    def land(self, platform_id, tick):
        """실제 착지 한 번을 처리하고, 이번 착지로 처음 달성한 구간 목록을 반환한다."""
        awarded = []
        for spec in self.specs:
            # 이전에 시작 발판을 밟았고, 아직 보상을 받지 않았으며, 이번에 목표에 착지했는가?
            if (spec.id in self.armed and spec.id not in self.completed
                    and platform_id in spec.target):
                self.completed.add(spec.id)
                self.last_events.append(dict(id=spec.id, target=platform_id, tick=tick))
                awarded.append(spec.id)
            # 시작 발판을 밟은 사실만 기록한다. 이 단계에서는 +10점을 지급하지 않는다.
            # 이후 다른 발판에 착지해도 이 기록은 같은 판이 끝날 때까지 유지된다.
            if platform_id in spec.source:
                self.armed.add(spec.id)
        return awarded


def terminal_points(cleared, failed, timeout):
    """이번 상태의 종료 보상만 계산한다. 진행 중이면 0점을 반환한다."""
    # 실패와 시간 초과가 겹쳐도 -25점 한 번이다.
    # 판정 우선순위는 실패→성공→시간 초과다. 실패 없이 제한 시각에 도착하면 성공이다.
    return -25. if failed else 100. if cleared else -25. if timeout else 0.


class MilestoneRawEnv(FiveRawEnv):
    """원래 게임 물리를 실행하고, 실제 착지·성공·실패에 따라 공통 보상을 계산한다."""

    def __init__(self, stage=1, reward_mode='milestone', limit=180., repeat=6):
        # 기본 제한 시간은 게임 시간 180초다.
        # 한 행동을 물리 6틱 유지한다. 물리 주기 120Hz에서 행동 주기는 0.05초다.
        if reward_mode != 'milestone':
            raise ValueError('Use milestone reward mode')
        # 해당 스테이지의 구간만 선택한다. 다른 맵의 특수 블록은 검사 대상에 넣지 않는다.
        self.specs = definitions(stage)
        self.tracker = MilestoneTracker(self.specs)
        self.reward_events = []
        # 부모의 맵·관측 기능을 재사용한다. 실제 보상은 아래 step()에서 계산하며,
        # 부모의 일반 착지 보상이나 거리·생존 보상을 추가로 사용하지 않는다.
        super().__init__(stage, reward_mode='terminal', limit=limit, repeat=repeat)
        self.observation_space = spaces.Dict({
            'global': spaces.Box(-8., 8., (GLOBALS,), dtype=np.float32),
            'entities': spaces.Box(-8., 8., (CAPACITY, FIELDS), dtype=np.float32),
            'local': spaces.Box(-8., 8., (LOCAL_COUNT, FIELDS), dtype=np.float32)})
        self.reset()

    def reset(self, *, seed=None, options=None):
        """새 판을 시작하고, 구간 시작·달성 여부와 점수 기록을 초기화한다."""
        self.tracker = MilestoneTracker(self.specs)
        self.reward_events = []
        super().reset(seed=seed, options=options)
        self.platform_by_id = {p.id: p for p in self.world.platforms}
        self.platform_by_id.update(self.world.special.platforms)
        for spec in self.specs:
            # 잘못된 발판 식별자를 보상 규칙에 넣었으면 학습 시작 전에 발견한다.
            assert all(i in self.platform_by_id for i in spec.source+spec.target)
        return self.observe(), self.info()

    def bounds(self, ids):
        """발판 그룹 전체를 감싸는 현재 영역을 (왼쪽, 위, 너비, 높이)로 반환한다."""
        # 부모 생성 중에는 platform_by_id가 이전 판을 가리킬 수 있어 현재 월드에서 다시 찾는다.
        platforms = {p.id:p for p in self.world.platforms}
        platforms.update(self.world.special.platforms)
        # 이동 발판은 게임 시간에 따른 실제 현재 위치를 사용한다.
        # 이 영역은 정책 입력용이며, 원래 게임의 충돌 지형을 바꾸지는 않는다.
        rects = [self.world.special.rect(self.world.special.devices[i], self.world.time)
                 if i in self.world.special.devices else platforms[i].rect_at(self.world.time, self.world.physics)
                 for i in ids]
        left, top = min(r[0] for r in rects), min(r[1] for r in rects)
        return left, top, max(r[0]+r[2] for r in rects)-left, max(r[1]+r[3] for r in rects)-top

    def observe(self):
        """정책이 볼 게임 상태에 핵심 구간의 위치와 진행 상태를 추가한다."""
        obs = super().observe()
        ball = self.world.ball
        # 실제 구간이 없는 나머지 칸은 0으로 남긴다. 존재 여부 0으로 빈 칸을 구분한다.
        extra = np.zeros((MAX_MILESTONES, MILESTONE_FIELDS), dtype=np.float32)
        for i, spec in enumerate(self.specs):
            for j, ids in enumerate((spec.source, spec.target)):
                x,y,w,h = self.bounds(ids)
                # 시작·목표 영역의 양 모서리를 공 기준 상대 좌표로 바꾸고 크기를 조정한다.
                extra[i, j*4:j*4+4] = [distance_feature(x-ball.x), distance_feature(y-ball.y),
                                      distance_feature(x+w-ball.x), distance_feature(y+h-ball.y)]
            extra[i, 8:] = (1., float(spec.id in self.tracker.armed), float(spec.id in self.tracker.completed))
        # 맵별 구간 수가 아니라 공통 최대 구간 수 4로 나눠 입력 기준을 통일한다.
        obs['global'][20] = len(self.tracker.completed)/MAX_MILESTONES
        obs['global'] = np.concatenate([obs['global'], extra.reshape(-1)])
        return obs

    def info(self, tick_delta=0):
        """평가·보고서용 기록을 반환한다. 이 기록 자체가 추가 보상은 아니다."""
        info = super().info(tick_delta)
        info.update(milestones=sorted(self.tracker.completed), armed=sorted(self.tracker.armed),
                    milestone_events=list(self.tracker.last_events),
                    reward_events=list(self.reward_events), learning_return=self.episode_return/100.)
        return info

    def step(self, action, on_tick=None):
        """행동 하나를 실행하고 관측, 학습용 보상, 종료 여부, 진단 기록을 반환한다."""
        # 종료 뒤에는 반드시 reset()해야 한다. 종료 보상을 반복해서 받는 것을 막는다.
        if self.done:
            raise RuntimeError('Reset before stepping a terminal episode')
        if not self.action_space.contains(action):
            raise ValueError(action)
        # bonus는 이번 행동 중 새로 달성한 구간의 점수만 담는다.
        before, bonus = self.ticks, 0.
        self.reward_events = []
        for _ in range(self.repeat):
            # 남은 제한 시간을 넘지 않도록 마지막 물리 틱의 길이를 조정한다.
            dt = min(self.world.physics.dt, self.limit-self.world.time)
            if dt < 1e-10 or self.world.failed or self.world.cleared:
                break
            # 정책 행동 0·1·2를 게임 입력 -1·0·1(왼쪽·무입력·오른쪽)로 변환한다.
            self.world.step(int(action)-1, dt)
            self.ticks += 1
            # 목표에 가장 가까웠던 거리는 분석용으로만 기록하며 거리 보상을 주지 않는다.
            self.min_goal_distance = min(self.min_goal_distance, self.goal_distance())
            for event, value in self.world.events:
                # 원래 물리 엔진의 실제 착지만 인정한다. 실패한 틱의 착지는 인정하지 않는다.
                if event == 'land' and not self.world.failed:
                    if value in self.support_lookup:
                        # 관측에 사용할 마지막 착지와 방문 발판을 기록한다. 착지만으로 가점은 없다.
                        self.landing_index = self.support_lookup[value]
                        self.touched.add(self.landing_index)
                    for milestone in self.tracker.land(value, self.ticks):
                        # 추적기가 처음 달성했다고 반환한 구간에만 +10점을 지급한다.
                        bonus += 10.
                        self.reward_events.append(dict(kind='milestone', id=milestone, points=10.))
            if on_tick:
                # 재생·물리 대조 등에서 매 틱의 상태를 확인할 때 사용하는 선택적 콜백이다.
                on_tick(self.world, self.ticks, int(action)-1)
        self.previous_action, self.decisions = int(action), self.decisions+1
        timeout = self.world.time >= self.limit-1e-9
        self.done = bool(self.world.cleared or self.world.failed or timeout)
        # 종료 조건 여러 개가 참이어도 terminal_points()가 종료 점수를 한 번만 선택한다.
        base = terminal_points(self.world.cleared, self.world.failed, timeout)
        if self.done:
            self.reward_events.append(dict(kind='success' if base>0 else 'failure', points=base))
        self.raw_base_return += base
        self.bonus_return += bonus
        self.episode_return += base+bonus  # 보고서용 누적 점수는 사용자가 정한 원래 점수다.
        self.visited[self.cell()] = 1  # 방문 위치는 진단용으로만 기록한다. 탐색 보상은 없다.
        # 학습기에 줄 때만 /100으로 조정한다. 같은 행동의 구간 점수와 종료 점수는 합산한다.
        # 180초 제한도 이 과제의 종료 조건이므로 종료 여부는 done, 외부 중단 여부는 False다.
        return self.observe(), (base+bonus)/100., self.done, False, self.info(self.ticks-before)


class MilestoneEnv(gym.ObservationWrapper):
    """항목별로 나뉜 관측을 신경망에 넣을 하나의 실수 배열로 연결한다."""

    def __init__(self, **options):
        super().__init__(MilestoneRawEnv(**options))
        self.observation_space = spaces.Box(-8., 8., (OBS_SIZE,), dtype=np.float32)

    def observation(self, obs):
        # 전체 상태→전체 맵 객체→가까운 객체 순서를 유지하며 정보를 빠짐없이 펼친다.
        return np.concatenate([obs['global'], obs['entities'].reshape(-1), obs['local'].reshape(-1)]).astype(np.float32)
