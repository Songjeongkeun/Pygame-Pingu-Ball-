"""강화학습 맵1에서 직접 도달한 상태를 저장하고 구간 연습에 사용하는 환경.

기존 A 환경의 물리·관측·보상 계산은 그대로 호출한다. 게임 상태와 보상 수령
기억만 복원하며, 정책 가중치나 행동 선택 난수는 저장 상태로 되돌리지 않는다.
"""
from collections import Counter
import copy
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .map_a_env import MapAEnv, MapARawEnv, OBS_SIZE, audit_reward


# 사람이 공을 배치하는 좌표가 아니라, 실제 착지한 발판을 분류하는 이름이다.
GROUPS = {
    'lower_lift_approach': ('아래 리프트 앞', tuple(f'z04dock-{i}' for i in range(3))),
    'lower_lift_exit': ('아래 리프트 이후', tuple(f'z05upper-{i}' for i in range(2))),
    'right_turn': ('오른쪽 점프 계단 이후', tuple(f'turn0buffer-{i}' for i in range(3))),
    'upper_return': ('위층 점프 이후', tuple(f'z06upper-{i}' for i in range(3))),
    'upper_break_exit': ('위층 파괴 이후', tuple(f'z07dock-{i}' for i in range(3))),
    'upper_lift_exit': ('위층 리프트 이후', tuple(f'z08upper-{i}' for i in range(2))),
    'upper_ferry_exit': ('위층 좌우 이동 이후', tuple(f'z09dock-{i}' for i in range(3))),
}

# 한 번의 deepcopy로 묶어서 복사해야 월드·장치·발판 사이의 참조도 함께 유지된다.
STATE_FIELDS = (
    'world', 'tracker', 'auxiliary', 'platform_by_id', 'support_lookup',
    'touched', 'landing_index', 'ticks', 'decisions', 'previous_action',
    'last_direction', 'last_duration', 'visited', 'min_goal_distance',
    'static_hazard_bounds',
)


@dataclass
class StartSchedule:
    """네 환경이 실제로 계산한 물리 프레임을 합산해 시작 확률을 결정한다."""
    budget: int = 6_000_000
    middle_probability: float = .5
    final_original_fraction: float = .2
    ticks: int = 0

    def __post_init__(self):
        if self.budget <= 0 or not 0 <= self.middle_probability <= 1:
            raise ValueError('프레임 예산은 양수, 중간 시작 확률은 0~1이어야 한다')
        if not 0 <= self.final_original_fraction <= 1:
            raise ValueError('마지막 원래 시작 전용 구간 비율은 0~1이어야 한다')

    @property
    def probability(self):
        # 이미 진행 중인 판은 끝까지 진행하고, 다음 reset부터 이 확률을 사용한다.
        boundary = self.budget * (1 - self.final_original_fraction)
        return self.middle_probability if self.ticks < boundary else 0.


@dataclass
class Snapshot:
    """행동 종료 경계에서 저장한 게임 상태. 학습된 모델은 포함하지 않는다."""
    schema: int
    fingerprint: str
    group: str
    state: dict
    prefix_parts: dict
    parent_id: str | None
    id: str = ''


class SnapshotBank:
    """같은 시드의 네 환경이 공유하는 구간별 제한 크기 저장소."""
    def __init__(self, seed=0, capacity=24):
        if capacity <= 0:
            raise ValueError('구간별 저장 개수는 양수여야 한다')
        self.seed, self.capacity = int(seed), int(capacity)
        # 정책의 numpy/torch 난수와 독립적이다. 상태 복원 때 이 난수를 되감지 않는다.
        self.rng = np.random.default_rng(self.seed + 832_001)
        self.groups = {name: [] for name in GROUPS}
        self.seen, self.selected = Counter(), Counter()
        self.serial = 0

    def __bool__(self):
        return any(self.groups.values())

    def add(self, snapshot):
        """구간마다 저수지 표본 추출을 사용해 오래된 상황과 새 상황을 섞는다."""
        if snapshot.group not in self.groups:
            raise ValueError('정의되지 않은 저장 구간이다')
        name = snapshot.group
        self.serial += 1
        snapshot.id = f'seed{self.seed}-state{self.serial}'
        self.seen[name] += 1
        items = self.groups[name]
        if len(items) < self.capacity:
            items.append(snapshot)
        else:
            index = int(self.rng.integers(self.seen[name]))
            if index < self.capacity:
                items[index] = snapshot

    def sample(self):
        """상태 개수가 많은 구간이 독점하지 않도록 구간부터 균등하게 뽑는다."""
        available = [name for name, items in self.groups.items() if items]
        if not available:
            raise RuntimeError('사용할 저장 상태가 없다')
        name = available[int(self.rng.integers(len(available)))]
        self.selected[name] += 1
        items = self.groups[name]
        return items[int(self.rng.integers(len(items)))]

    def manifest(self):
        return dict(seed=self.seed, capacity_per_group=self.capacity,
                    count=sum(map(len, self.groups.values())),
                    groups={name: dict(label=GROUPS[name][0], stored=len(items),
                        collected=self.seen[name], selected=self.selected[name])
                        for name, items in self.groups.items()})

    def save(self, path):
        """검토용 저장소를 남긴다. 이 파일만으로 PPO 학습 재개를 지원하지는 않는다."""
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with gzip.open(temporary, 'wb') as stream:
            pickle.dump(dict(schema=1, manifest=self.manifest(), groups=self.groups,
                             rng_state=self.rng.bit_generator.state), stream,
                        protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)


class CurriculumRawEnv(MapARawEnv):
    """기존 A의 한 행동을 실행한 뒤 상태 수집과 프레임 집계만 추가한다."""
    def __init__(self, bank=None, schedule=None, limit=180.):
        # 비어 있는 저장소도 유효하므로 `bank or SnapshotBank()`로 교체하면 안 된다.
        self.bank = bank if bank is not None else SnapshotBank()
        self.schedule = schedule if schedule is not None else StartSchedule()
        self.fingerprint = None
        self._clear_segment()
        super().__init__(limit=limit)

    def _clear_segment(self):
        """연습 구간의 시작 정보와 집계 기준을 초기화한다."""
        self.start_kind, self.start_group, self.start_snapshot_id = 'original', None, None
        self.start_time = self.start_tick = self.start_decisions = 0
        self.initial_completed, self.initial_armed = set(), set()
        self.initial_reward_parts = Counter()
        self.captured_groups = set()
        self.start_middle_probability = self.schedule.probability

    def reset(self, *, seed=None, options=None):
        self._clear_segment()
        super().reset(seed=seed, options=options)
        if self.fingerprint is None:
            # 다른 맵·물리 설정·시간 제한으로 만든 상태가 섞이면 복원을 거부한다.
            content = json.dumps(dict(level=asdict(self.level), physics=asdict(self.world.physics),
                                      limit=self.limit), sort_keys=True, ensure_ascii=False)
            self.fingerprint = hashlib.sha256(content.encode('utf-8')).hexdigest()
        force_original = bool((options or {}).get('original_start', False))
        if (not force_original and self.bank and self.schedule.probability > 0
                and self.bank.rng.random() < self.schedule.probability):
            self.restore_state(self.bank.sample())
        return self.observe(), self.info()

    def capture_state(self, group):
        """현재 정책이 실제 진행한 비종료 상태를 참조 관계까지 통째로 복사한다."""
        if self.done or self.world.failed or self.world.cleared or self.world.time >= self.limit:
            raise ValueError('종료된 상태는 체크포인트로 저장할 수 없다')
        if group not in GROUPS:
            raise ValueError('정의되지 않은 저장 구간이다')
        prefix = Counter(self.initial_reward_parts)
        prefix.update(self.reward_parts)
        if abs(prefix.get('milestone', 0) - 10 * len(self.tracker.completed)) > 1e-6:
            raise ValueError('달성 구간과 보상 이력이 일치하지 않는다')
        state = copy.deepcopy({name: getattr(self, name) for name in STATE_FIELDS})
        return Snapshot(1, self.fingerprint, group, state, dict(prefix), self.start_snapshot_id)

    def restore_state(self, snapshot):
        """보상을 받을 자격은 복원하고, 이번 연습의 누적 점수는 0부터 센다."""
        if snapshot.schema != 1 or snapshot.fingerprint != self.fingerprint:
            raise ValueError('현재 맵·물리와 호환되지 않는 저장 상태다')
        state = copy.deepcopy(snapshot.state)
        world = state['world']
        if world.failed or world.cleared or not 0 <= world.time < self.limit:
            raise ValueError('유효한 진행 중 상태가 아니다')
        for name in STATE_FIELDS:
            setattr(self, name, state[name])
        self.done = False
        self.start_kind, self.start_group = 'middle', snapshot.group
        self.start_snapshot_id = snapshot.id
        self.start_time, self.start_tick = self.world.time, self.ticks
        self.start_decisions = self.decisions
        self.initial_completed = set(self.tracker.completed)
        self.initial_armed = set(self.tracker.armed)
        self.initial_reward_parts = Counter(snapshot.prefix_parts)
        self.captured_groups = {snapshot.group}
        # 월드의 시각과 구간별 최초 지급 기억은 유지한다. 이미 얻은 점수를 재지급하지 않는다.
        self.reward_parts, self.reward_events = Counter(), []
        self.raw_base_return = self.shaping_return = self.bonus_return = self.episode_return = 0.
        self.tracker.last_events = []

    def info(self, tick_delta=0):
        result = super().info(tick_delta)
        result.update(start_kind=self.start_kind, start_group=self.start_group,
            start_snapshot_id=self.start_snapshot_id, start_world_time=self.start_time,
            start_middle_probability=self.start_middle_probability,
            segment_seconds=self.world.time-self.start_time,
            segment_ticks=self.ticks-self.start_tick,
            segment_decisions=self.decisions-self.start_decisions,
            initial_milestones=sorted(self.initial_completed), initial_armed=sorted(self.initial_armed),
            initial_reward_parts=dict(self.initial_reward_parts),
            new_milestones=sorted(self.tracker.completed-self.initial_completed))
        return result

    def _collect(self, landings):
        """후보 발판에 실제 착지했고 행동 종료 시에도 그 위쪽인 상태만 저장한다."""
        # 이 판정은 저장 후보만 선택하며 물리 상태나 보상을 바꾸지 않는다.
        if self.done or self.limit-self.world.time < 5 or self.schedule.probability <= 0:
            return
        ball, radius = self.world.ball, self.world.physics.radius
        if not all(np.isfinite(v) for v in (ball.x, ball.y, ball.vx, ball.vy)):
            return
        for name, (_, ids) in GROUPS.items():
            if name in self.captured_groups or not landings.intersection(ids):
                continue
            for platform_id in ids:
                p = self.platform_by_id[platform_id]
                x, y, w, _ = p.rect_at(self.world.time, self.world.physics)
                if (p.active(self.world.time, self.world.switches)
                        and x-radius <= ball.x <= x+w+radius and y-256 <= ball.y <= y-radius+1e-6):
                    self.bank.add(self.capture_state(name))
                    self.captured_groups.add(name)
                    break

    def step(self, action, on_tick=None):
        landings = set()
        def tick(world, number, direction):
            # 마지막 틱만 보면 도중의 착지를 놓칠 수 있어 매 틱의 이벤트를 모은다.
            landings.update(value for event, value in world.events if event == 'land')
            if on_tick is not None:
                on_tick(world, number, direction)
        observation, reward, done, truncated, info = super().step(action, on_tick=tick)
        self.schedule.ticks += info['tick_delta']
        # 모든 보상과 직전 행동 기록이 확정된 뒤 저장해 중간 상태의 불일치를 막는다.
        self._collect(landings)
        return observation, reward, done, truncated, self.info(info['tick_delta'])


class CurriculumEnv(gym.ObservationWrapper):
    """입력의 순서와 크기를 기존 A와 같은 1,984요소로 유지하는 래퍼."""
    def __init__(self, **options):
        super().__init__(CurriculumRawEnv(**options))
        self.observation_space = spaces.Box(-8., 8., (OBS_SIZE,), dtype=np.float32)

    observation = MapAEnv.observation


def audit_training_reward(info):
    """이번 보상과 복원 전까지 포함한 최초 수령 상한을 모두 검사한다."""
    initial, completed = set(info['initial_milestones']), set(info['milestones'])
    new = set(info['new_milestones'])
    assert initial <= completed and new == completed-initial
    parts = info['reward_parts']
    assert abs(sum(parts.values())-info['episode_return']) < 1e-6
    assert abs(info['learning_return']*100-info['episode_return']) < 1e-6
    assert parts.get('milestone', 0.) == 10*len(new)
    prefix = info['initial_reward_parts']
    assert not prefix.get('success', 0.) and not prefix.get('failure', 0.)
    assert prefix.get('milestone', 0.) == 10*len(initial)
    # 기존 검사에 넘길 때만 통산 기록으로 만든다. 학습 보상은 이번 연습분뿐이다.
    total = Counter(prefix)
    total.update(parts)
    cumulative = dict(info, reward_parts=dict(total), episode_return=sum(total.values()),
                      learning_return=sum(total.values())/100.)
    audit_reward(cumulative)
