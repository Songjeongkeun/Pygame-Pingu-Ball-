"""행동 지속 시간·보상 추적 상태·축소 보상을 단계별로 비교하는 환경.

A: 9개 행동과 관측 상태 추가. B: A에서 4번의 보조 보상만 축소.
C: 환경은 B와 같고 학습기의 갱신·탐색 설정을 안정화한다.
기존 combined_* 파일과 원래 게임 물리는 수정하지 않는다.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .combined_env import CombinedRawEnv, audit_reward as audit_original
from .combined_rewards import AuxiliaryRewards
from .milestone_env import OBS_SIZE as BASE_SIZE

DURATIONS = (12, 30, 72)
ACTIONS = tuple((direction, frames) for direction in (-1, 0, 1) for frames in DURATIONS)
VARIANTS = ('A', 'B', 'C')
EXTRA_FIELDS = (
    'active_spring_1', 'active_spring_2', 'active_spring_3',
    'spring_elapsed', 'spring_remaining',
    'height_paid_1', 'height_paid_2', 'height_paid_3',
    'horizontal_paid_1', 'horizontal_paid_2', 'horizontal_paid_3',
    'height_inside_0', 'height_inside_1', 'height_inside_2', 'height_inside_3',
    'horizontal_inside_0', 'horizontal_inside_1', 'horizontal_inside_2', 'horizontal_inside_3',
    'lift_first', 'lift_high', 'upper_first', 'since_last_lift',
    'approach_remaining', 'previous_direction', 'previous_duration')
EXTRA_SIZE = len(EXTRA_FIELDS)
OBS_SIZE = BASE_SIZE + EXTRA_SIZE


class ScaledAuxiliaryRewards(AuxiliaryRewards):
    """판정 조건은 그대로 두고 4번 보조 보상의 금액만 줄인다."""
    def __init__(self, stage, world, specs, reduced=False):
        self.reduced = reduced and stage == 4
        super().__init__(stage, world, specs)

    def _event(self, kind, points, source=None):
        if self.reduced:
            factor = {'lift_first': 2/12, 'lift_high': 2/12,
                      'upper_first': 3/18, 'approach': 2/12.96}.get(kind, 1.)
            points *= factor
        return super()._event(kind, points, source)


class StabilizedRawEnv(CombinedRawEnv):
    def __init__(self, stage=2, variant='A', reward_mode=None, limit=180.):
        if variant not in VARIANTS:
            raise ValueError(variant)
        # 부모 MapInputEnv의 variant='local'과 실험 조건을 별도 이름으로 보관한다.
        self.experiment_variant = variant
        self.last_direction = self.last_duration = 0
        self.last_requested_action = -1
        super().__init__(stage=stage, limit=limit, repeat=DURATIONS[0])
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict({**self.observation_space.spaces,
            'extra': spaces.Box(-8., 8., (EXTRA_SIZE,), dtype=np.float32)})
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.last_direction = self.last_duration = 0
        self.last_requested_action = -1
        self.repeat = DURATIONS[0]
        super().reset(seed=seed, options=options)
        self.auxiliary = ScaledAuxiliaryRewards(self.stage, self.world, self.specs,
                                               self.experiment_variant in ('B', 'C'))
        return self.observe(), self.info()

    def extra_observation(self):
        """보상 지급에 사용한 내부 상태를 유효 여부와 함께 전달한다."""
        values = np.zeros(EXTRA_SIZE, dtype=np.float32)
        aux = getattr(self, 'auxiliary', None)
        if aux is not None:
            springs = tuple(aux.targets)
            active = aux.active_source
            if active is not None:
                values[springs.index(active)] = 1.
                age = max(0., self.world.time - aux.active_since)
                values[3:5] = min(age/3., 1.), max(0., 1.-age/3.)
                # 목표 그룹의 타일별 진입 이력도 빠짐없이 관측한다.
                for i, target in enumerate(aux.targets[active]):
                    values[11+i] = float(aux.inside_height.get(target, False))
                    values[15+i] = float(aux.inside_horizontal.get(target, False))
            for i, spring in enumerate(springs):
                values[5+i] = float(spring in aux.height_paid)
                values[8+i] = float(spring in aux.horizontal_paid)
            if self.stage == 4:
                values[19:22] = aux.lift_first, aux.high_lift, aux.upper_first
                values[22] = min(max(self.world.time-aux.last_lift_time, 0.)/3., 1.) if aux.lift_first else 0.
                values[23] = max(0., aux.approach_end-aux.highest_x)/288.
        values[24:26] = self.last_direction, self.last_duration/72.
        return values

    def observe(self):
        obs = super().observe()
        obs['extra'] = self.extra_observation()
        return obs

    def info(self, tick_delta=0):
        info = super().info(tick_delta)
        info.update(variant=self.experiment_variant, action_index=self.last_requested_action,
                    direction=self.last_direction, requested_ticks=self.last_duration,
                    reduced_auxiliary=self.stage == 4 and self.experiment_variant in ('B', 'C'))
        return info

    def step(self, action, on_tick=None):
        if not self.action_space.contains(action):
            raise ValueError(action)
        direction, frames = ACTIONS[int(action)]
        self.last_requested_action = int(action)
        self.last_direction, self.last_duration = direction, frames
        self.repeat = frames
        # 부모는 방향 0/1/2를 받아 원래 120Hz 물리를 선택한 틱 수만큼 진행한다.
        return super().step(direction+1, on_tick=on_tick)


class StabilizedEnv(gym.ObservationWrapper):
    def __init__(self, **options):
        super().__init__(StabilizedRawEnv(**options))
        self.observation_space = spaces.Box(-8., 8., (OBS_SIZE,), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs['global'], obs['entities'].ravel(),
                               obs['local'].ravel(), obs['extra']]).astype(np.float32)


def audit_reward(info):
    audit_original(info)
    if info['reduced_auxiliary']:
        parts = info['reward_parts']
        for key, cap in dict(lift_first=2., lift_high=2., upper_first=3., approach=2.).items():
            assert parts.get(key, 0.) <= cap+1e-6
        if not info['success']:
            assert info['episode_return'] <= -6.+1e-6
