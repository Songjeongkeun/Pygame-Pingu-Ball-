"""에피소드 기준 커리큘럼과 위층 연결 구간 연습을 위한 맵1 환경.

기존 물리·보상·관측·9행동은 유지한다. 체크포인트 선택과 진단만 확장하고,
목표 발판에 도달했다는 진단 자체로 보상을 추가하거나 판을 끝내지 않는다.
"""
from collections import Counter
import copy
from dataclasses import dataclass, field
import gzip
from pathlib import Path
import pickle

import numpy as np
from gymnasium import spaces
import gymnasium as gym

from .map_a_curriculum_env import (
    CurriculumRawEnv, Snapshot, SnapshotBank, STATE_FIELDS, audit_training_reward,
)
from .map_a_env import MapAEnv, OBS_SIZE


# 실제 동선을 세분화하는 분류이며, 이 순서대로 행동하도록 정책을 강제하지 않는다.
GROUPS = {
    'lower_lift_approach': ('아래 리프트 앞', tuple(f'z04dock-{i}' for i in range(3))),
    'lower_lift_exit': ('아래 리프트 이후', tuple(f'z05upper-{i}' for i in range(2))),
    'right_turn': ('오른쪽 점프 계단 이후', tuple(f'turn0buffer-{i}' for i in range(3))),
    'upper_return': ('위층 점프 이후', tuple(f'z06upper-{i}' for i in range(3))),
    'upper_bridge': ('위층 연결 기본 발판', tuple(f'z06dock-{i}' for i in range(3))),
    'upper_break_a': ('위층 첫 파괴 이후', ('z07break-a-0',)),
    'upper_break_b': ('위층 두번째 파괴 이후', ('z07break-b-0',)),
    'upper_break_exit': ('위층 파괴 이후 기본 발판', tuple(f'z07dock-{i}' for i in range(3))),
    'upper_lift_exit': ('위층 리프트 이후', tuple(f'z08upper-{i}' for i in range(2))),
    'upper_ferry_exit': ('위층 좌우 이동 이후', tuple(f'z09dock-{i}' for i in range(3))),
}

# 다음 실제 착지율을 측정하기 위한 목표다. 보상 함수와 정책 입력은 바꾸지 않는다.
LOCAL_GOALS = {
    'lower_lift_approach': ('z05lift-0',),
    'lower_lift_exit': ('turn0-spring0-0',),
    'right_turn': ('z06spring-0',),
    'upper_return': tuple(f'z06dock-{i}' for i in range(3)),
    'upper_bridge': ('z07break-a-0',),
    'upper_break_a': ('z07break-b-0',),
    'upper_break_b': tuple(f'z07dock-{i}' for i in range(3)),
    'upper_break_exit': ('z08lift-0',),
    'upper_lift_exit': tuple(f'z08dock-{i}' for i in range(3)),
    'upper_ferry_exit': ('z10clock-0',),
}
RARE_MILESTONES = {'07-break-a-b': 'upper_break_b', '07-break-dock': 'upper_break_exit'}


@dataclass
class EpisodeSchedule:
    """판 수로 진행률을 계산하고 원래 시작 검증이 좋아질 때만 중간 시작을 줄인다."""
    budget: int = 10_000
    middle_probability: float = .5
    ticks: int = 0
    completed: int = 0
    validation_rates: list = field(default_factory=list)
    current_probability: float = field(init=False)

    def __post_init__(self):
        if self.budget <= 0 or not 0 <= self.middle_probability <= 1:
            raise ValueError('에피소드 수는 양수, 중간 시작 확률은 0~1이어야 한다')
        self.current_probability = self.middle_probability

    @property
    def probability(self):
        return self.current_probability if self.completed < self.budget else 0.

    def observe_validation(self, success_rate):
        """연속 두 검증이 50%/80% 이상이면 목표 확률을 절반/4분의1로 낮춘다."""
        if not 0 <= success_rate <= 1:
            raise ValueError('검증 성공률은 0~1이어야 한다')
        self.validation_rates.append(float(success_rate))
        target = self.middle_probability
        if len(self.validation_rates) >= 2:
            recent = min(self.validation_rates[-2:])
            if recent >= .8:
                target *= .25
            elif recent >= .5:
                target *= .5
        # 성능 하락 시에도 되돌릴 수 있으며, 한 검증당 최대 0.05만 변경한다.
        self.current_probability += float(np.clip(target-self.current_probability, -.05, .05))


class EpisodeBank(SnapshotBank):
    """새로 도달한 구간과 경험이 부족한 구간을 우선하되 모든 구간의 추출 기회를 남긴다."""
    def __init__(self, seed=1, capacity=24, protected_capacity=8):
        super().__init__(seed, capacity)
        if protected_capacity <= 0:
            raise ValueError('보호 상태 저장 개수는 양수여야 한다')
        self.groups = {name: [] for name in GROUPS}
        self.protected = {name: [] for name in GROUPS}
        self.protected_capacity = protected_capacity
        self.completed, self.goal_hits, self.frames = Counter(), Counter(), Counter()
        self.rejections = Counter()

    def add(self, snapshot, protect=False):
        super().add(snapshot)
        # 드문 새 구간 달성 상태를 일반 저수지 교체만으로 잃지 않도록 따로 보존한다.
        if protect:
            items = self.protected[snapshot.group]
            items.append(snapshot)
            del items[:-self.protected_capacity]

    def __bool__(self):
        return any(self.groups.values()) or any(self.protected.values())

    def sample(self):
        available = [name for name in GROUPS if self.groups[name] or self.protected[name]]
        if not available:
            raise RuntimeError('사용할 저장 상태가 없다')
        # 20% 균등 + 40% 어려움/프레임 부족 + 40% 가장 뒤쪽의 도달 구간을 섞는다.
        difficulty = np.asarray([
            (1-(self.goal_hits[name]+1)/(self.completed[name]+2))
            / np.sqrt(1+self.frames[name]/1200.) for name in available])
        weights = .2/len(available) + .4*difficulty/difficulty.sum()
        weights[-1] += .4
        name = available[int(self.rng.choice(len(available), p=weights/weights.sum()))]
        self.selected[name] += 1
        regular, protected = self.groups[name], self.protected[name]
        items = protected if protected and (not regular or self.rng.random() < .5) else regular
        return items[int(self.rng.integers(len(items)))]

    def manifest(self):
        return dict(seed=self.seed, capacity_per_group=self.capacity,
            protected_capacity_per_group=self.protected_capacity,
            count=sum(len(v)+len(self.protected[k]) for k, v in self.groups.items()),
            rejections=dict(self.rejections),
            groups={name: dict(label=GROUPS[name][0], stored=len(items),
                protected=len(self.protected[name]), collected=self.seen[name], selected=self.selected[name],
                ended=self.completed[name], local_goal_hits=self.goal_hits[name],
                local_goal_rate=self.goal_hits[name]/self.completed[name] if self.completed[name] else None,
                physics_ticks=self.frames[name]) for name, items in self.groups.items()})

    def save(self, path):
        """상태 저장소는 검토용이다. 이 파일만으로 정확한 PPO 재개를 지원하지 않는다."""
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.tmp')
        with gzip.open(temporary, 'wb') as stream:
            pickle.dump(dict(schema=2, manifest=self.manifest(), groups=self.groups,
                protected=self.protected, rng_state=self.rng.bit_generator.state), stream,
                protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)


class EpisodeRawEnv(CurriculumRawEnv):
    """에피소드 종료·다음 발판 착지 진단만 추가하는 환경. 물리는 원래 A를 호출한다."""
    def __init__(self, bank=None, schedule=None, limit=180.):
        super().__init__(bank=bank if bank is not None else EpisodeBank(),
                         schedule=schedule if schedule is not None else EpisodeSchedule(), limit=limit)

    def _clear_segment(self):
        super()._clear_segment()
        self.local_goal_ids, self.landed_platforms = (), set()
        self.local_goal_reached = False
        self._completed_before_step = set()

    def restore_state(self, snapshot):
        super().restore_state(snapshot)
        self.local_goal_ids = LOCAL_GOALS.get(self.start_group, ())
        self.local_goal_reached, self.landed_platforms = False, set()

    def capture_state(self, group):
        """새 분류를 지원하며, 월드와 보상 기억을 한 번에 복사해 참조를 유지한다."""
        if group not in GROUPS or self.done or self.world.failed or self.world.cleared or self.world.time >= self.limit:
            raise ValueError('유효한 진행 중 저장 구간이 아니다')
        prefix = Counter(self.initial_reward_parts)
        prefix.update(self.reward_parts)
        if abs(prefix.get('milestone', 0)-10*len(self.tracker.completed)) > 1e-6:
            raise ValueError('달성 구간과 보상 이력이 일치하지 않는다')
        return Snapshot(1, self.fingerprint, group,
            copy.deepcopy({name: getattr(self, name) for name in STATE_FIELDS}),
            dict(prefix), self.start_snapshot_id)

    def _collect(self, landings):
        # 시작 확률이 낮아져도 상태 수집은 계속한다. 평가에는 이 환경을 사용하지 않는다.
        if self.done or self.limit-self.world.time < 5:
            return
        b, radius = self.world.ball, self.world.physics.radius
        if not all(np.isfinite(v) for v in (b.x, b.y, b.vx, b.vy)):
            return
        newly = self.tracker.completed-self._completed_before_step
        protected_groups = {RARE_MILESTONES[m] for m in newly if m in RARE_MILESTONES}
        for name, (_, ids) in GROUPS.items():
            if name in self.captured_groups or (not landings.intersection(ids) and name not in protected_groups):
                continue
            valid = False
            for pid in ids:
                p = self.platform_by_id[pid]
                x, y, w, _ = p.rect_at(self.world.time, self.world.physics)
                # 파괴 발판은 이미 반동을 준 뒤 사라져도 그 직후 공중 상태를 보존한다.
                if name in ('upper_break_a', 'upper_break_b') or name in protected_groups:
                    valid |= b.y <= y-radius+1e-6
                else:
                    valid |= (p.active(self.world.time, self.world.switches)
                              and x-radius <= b.x <= x+w+radius and y-256 <= b.y <= y-radius+1e-6)
            if valid:
                self.bank.add(self.capture_state(name), protect=name in protected_groups)
                self.captured_groups.add(name)
            else:
                self.bank.rejections[name+':행동 종료 시 후보 영역 밖'] += 1

    def info(self, tick_delta=0):
        result = super().info(tick_delta)
        result.update(local_goal_ids=list(self.local_goal_ids), local_goal_reached=self.local_goal_reached,
                      landed_platforms=sorted(self.landed_platforms))
        return result

    def step(self, action, on_tick=None):
        self._completed_before_step = set(self.tracker.completed)
        def tick(world, number, direction):
            landed = {value for event, value in world.events if event == 'land'}
            self.landed_platforms.update(landed)
            self.local_goal_reached |= bool(landed.intersection(self.local_goal_ids))
            if on_tick is not None:
                on_tick(world, number, direction)
        observation, reward, done, truncated, info = super().step(action, on_tick=tick)
        self.bank.frames[self.start_group or 'original'] += info['tick_delta']
        if done:
            self.schedule.completed += 1
            if self.start_group is not None:
                self.bank.completed[self.start_group] += 1
                self.bank.goal_hits[self.start_group] += int(self.local_goal_reached)
        return observation, reward, done, truncated, self.info(info['tick_delta'])


class EpisodeEnv(gym.ObservationWrapper):
    """관측의 크기와 의미를 기존 A와 같은 1,984요소로 유지한다."""
    def __init__(self, **options):
        super().__init__(EpisodeRawEnv(**options))
        self.observation_space = spaces.Box(-8., 8., (OBS_SIZE,), dtype=np.float32)

    observation = MapAEnv.observation
