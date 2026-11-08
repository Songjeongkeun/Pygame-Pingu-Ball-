"""PDF 보상 전용 환경. 기존 게임 물리·9행동·1,984차원 관측을 유지한다."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pingu_ball.engine import World
from .map_a_env import MapARawEnv, MapAuxiliary, OBS_SIZE
from .milestone_env import terminal_points
from .stabilized_env import ACTIONS
from .map1_spec_rewards import RULES, RewardTracker


class ContactWorld(World):
    """원래 착지 함수를 실행하며 충돌 순간의 발판 좌표만 추가로 기록한다."""
    def reset(self):
        super().reset()
        self.reward_landings = []

    def step(self, *args, **kwargs):
        self.reward_landings = []
        return super().step(*args, **kwargs)

    def _land(self, platform):
        contact = dict(platform=platform.id, top_y=float(platform.rect_at(self.time, self.physics)[1]),
                       time=float(self.time))
        super()._land(platform)
        self.reward_landings.append(contact)


class SpecRawEnv(MapARawEnv):
    def reset(self, *, seed=None, options=None):
        # 부모 초기화가 info()도 호출하므로 새 보상 기록을 먼저 만든다.
        self.spec_tracker = RewardTracker()
        super().reset(seed=seed, options=options)
        self.world = ContactWorld(self.level, auto_reset=False)
        self.auxiliary = MapAuxiliary(self.world)
        self.platform_by_id = {p.id: p for p in self.world.platforms}
        self.platform_by_id.update(self.world.special.platforms)
        missing = {p for r in RULES for p in r.source+r.target} - self.platform_by_id.keys()
        if missing:
            raise RuntimeError(f'보상 발판이 현재 맵에 없습니다: {sorted(missing)}')
        if any(p.startswith('turn0landing-') for p in self.platform_by_id):
            raise RuntimeError('오른쪽 위 발판 3개가 삭제된 최신 맵을 사용해야 합니다')
        return self.observe(), self.info()

    def info(self, tick_delta=0):
        result = super().info(tick_delta)
        result.update(self.spec_tracker.state())
        result.update(reward_variant='map1-pdf-33', start_kind='original')
        return result

    def _consume_landings(self):
        """실패 확정 프레임에서는 새 양수 보상을 전부 억제한다."""
        if self.world.failed:
            return
        # 아래 기존 이력은 정책 입력 호환용이다. 기존 함수가 반환한 보상은 버린다.
        # 새 33항목의 지급 기억이나 직전 발판은 관측 벡터에 추가하지 않는다.
        self.auxiliary.motion(self.world)
        for contact in self.world.reward_landings:
            value = contact['platform']
            if value in self.support_lookup:
                self.landing_index = self.support_lookup[value]
                self.touched.add(self.landing_index)
            self.tracker.land(value, self.ticks)
            self.auxiliary.landing(self.world, value)
            # 시간 블록은 엔진이 인정한 착지가 기준이다. 프레임 끝의 활성 상태로 취소하지 않는다.
            self.reward_events.extend(self.spec_tracker.land(value, contact['top_y']))
        self.auxiliary.after_landings(self.world)

    def step(self, action, on_tick=None):
        if self.done:
            raise RuntimeError('종료 후 reset()이 필요합니다')
        if not self.action_space.contains(action):
            raise ValueError(action)
        direction, frames = ACTIONS[int(action)]
        self.last_direction, self.last_duration = direction, frames
        before = self.ticks
        self.reward_events = []
        for _ in range(frames):
            dt = min(self.world.physics.dt, self.limit-self.world.time)
            if dt < 1e-10 or self.world.failed or self.world.cleared:
                break
            self.world.step(direction, dt)
            self.ticks += 1
            self.min_goal_distance = min(self.min_goal_distance, self.goal_distance())
            self._consume_landings()
            if on_tick:
                on_tick(self.world, self.ticks, direction)
        self.previous_action, self.decisions = direction+1, self.decisions+1
        timeout = self.world.time >= self.limit-1e-9
        self.done = bool(self.world.cleared or self.world.failed or timeout)
        base = terminal_points(self.world.cleared, self.world.failed, timeout)
        middle = sum(e['points'] for e in self.reward_events)
        if self.done:
            self.reward_events.append(dict(kind='success' if base > 0 else 'failure', points=base))
        for event in self.reward_events:
            self.reward_parts[event['kind']] += event['points']
        self.raw_base_return += base
        self.bonus_return += middle
        self.episode_return += base+middle
        self.visited[self.cell()] = 1.
        return self.observe(), (base+middle)/100., self.done, False, self.info(self.ticks-before)


class SpecEnv(gym.ObservationWrapper):
    def __init__(self, **kwargs):
        super().__init__(SpecRawEnv(**kwargs))
        self.observation_space = spaces.Box(-8., 8., (OBS_SIZE,), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs['global_state'], obs['entities'].ravel(),
                               obs['local'].ravel(), obs['extra']]).astype(np.float32)


def preflight():
    """현재 맵의 ID와 계단 마지막 점프→왼쪽 기본 발판의 실제 물리 연결을 검사한다."""
    env = SpecRawEnv()
    try:
        world = env.world
        spring = env.platform_by_id['turn0-spring3-0']
        # 이미 마지막 점프 블록에 착지한 합법적인 접촉 상태에서 한 번의 점프만 검사한다.
        # 본학습 초기 상태나 정책에 이 경로·상태를 전달하지 않는다.
        world.ball.x, world.ball.vx = spring.x+32., 0.
        world._land(spring)
        for tick in range(180):
            world.step(-1)
            landed = [c['platform'] for c in world.reward_landings]
            if landed:
                if landed[0] not in {f'turn0buffer-{i}' for i in range(3)}:
                    raise RuntimeError(f'층 전환 물리 연결 검사 실패: {landed[0]}')
                return dict(platform_ids_valid=True, reward_count=len(RULES),
                    deleted_platforms_absent=True, turn_jump_reachable=True,
                    source=spring.id, target=landed[0], ticks=tick+1,
                    scope='마지막 계단 착지 후 한 번의 실제 점프. 전체 맵 완주 보장 아님')
            if world.failed or world.cleared:
                break
        raise RuntimeError('층 전환 마지막 점프에서 왼쪽 위 기본 발판에 도달하지 못했습니다')
    finally:
        env.close()
