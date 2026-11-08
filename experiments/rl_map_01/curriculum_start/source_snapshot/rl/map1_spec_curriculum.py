"""PDF 보상 전용 상태 저장·구간 시작·행동 진단. 정답 행동이나 이전 가중치는 사용하지 않는다."""
from collections import Counter
import copy
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .map_a_env import OBS_SIZE
from .map1_spec_env import SpecRawEnv, SpecEnv
from .stabilized_env import ACTIONS


GROUPS = {
    'pre_turn': ('마지막 계단 접근', ('turn0-spring2-0',), ('turn0-spring3-0',)),
    'last_stair': ('마지막 계단 이후', ('turn0-spring3-0',), tuple(f'turn0buffer-{i}' for i in range(3))),
    'upper_buffer': ('2층 기본 발판', tuple(f'turn0buffer-{i}' for i in range(3)), ('z06spring-0',)),
}
ROLES = ('original', 'original', 'turn', 'upper')
# 함께 복사해야 월드·장치·발판·관측 이력 사이의 참조가 유지된다.
STATE_FIELDS = ('world', 'spec_tracker', 'tracker', 'auxiliary', 'platform_by_id', 'support_lookup',
    'touched', 'landing_index', 'ticks', 'decisions', 'previous_action', 'last_direction',
    'last_duration', 'visited', 'min_goal_distance', 'static_hazard_bounds')


def is_turn_decision(env):
    """끝의 두 계단을 밟은 뒤2초 이내의 행동 선택을 전환 구간으로 정의한다."""
    w = env.world
    return (env.spec_tracker.previous_landing in ('turn0-spring2-0', 'turn0-spring3-0')
            and 0 <= w.time-w.last_landing_time <= 2.
            and w.ball.y <= env.platform_by_id['turn0-spring2-0'].y+64.)


class DecisionLog:
    """전환 구간에서 선택한 행동과 선택 직전 확률을 합산한다. 정책 입력에는 넣지 않는다."""
    def __init__(self):
        self.count = 0
        self.actions = np.zeros(9, dtype=np.int64)
        self.probabilities = np.zeros(9, dtype=np.float64)

    def add(self, action, probabilities):
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (9,) or not np.all(np.isfinite(p)) or np.any(p < 0) or abs(p.sum()-1) > 1e-5:
            raise ValueError('행동 확률이 유효하지 않습니다')
        self.count += 1
        self.actions[int(action)] += 1
        self.probabilities += p

    def summary(self):
        return dict(decisions=self.count, actions=self.actions.tolist(),
            left_actions=int(self.actions[:3].sum()), stay_actions=int(self.actions[3:6].sum()),
            right_actions=int(self.actions[6:].sum()),
            mean_left_probability=float(self.probabilities[:3].sum()/self.count) if self.count else None,
            mean_stay_probability=float(self.probabilities[3:6].sum()/self.count) if self.count else None,
            mean_right_probability=float(self.probabilities[6:].sum()/self.count) if self.count else None)


@dataclass
class Snapshot:
    schema: int
    fingerprint: str
    group: str
    state: dict
    prefix_parts: dict
    parent_id: str | None
    id: str = ''


class SnapshotBank:
    """구간별 저수지 표본과 별도 난수로 다양한 실제 도달 상태를 보관한다."""
    def __init__(self, seed=2026, capacity=32):
        if capacity <= 0:
            raise ValueError('구간별 보관 개수는 양수여야 합니다')
        self.seed, self.capacity = int(seed), int(capacity)
        self.rng = np.random.default_rng(self.seed+832001)
        self.groups = {g: [] for g in GROUPS}
        self.seen, self.selected, self.rejections = Counter(), Counter(), Counter()
        self.serial = 0

    def available(self, role):
        turns = [g for g in ('pre_turn', 'last_stair') if self.groups[g]]
        return ['upper_buffer'] if role == 'upper' and self.groups['upper_buffer'] else turns if role in ('turn', 'upper') else []

    def add(self, snapshot):
        if snapshot.group not in GROUPS:
            raise ValueError('알 수 없는 저장 구간입니다')
        def key(item):
            world = item.state['world']
            return (round(world.time, 2), *(round(getattr(world.ball, k), 1) for k in ('x', 'y', 'vx', 'vy')),
                    tuple(sorted(item.state['spec_tracker'].paid)), item.state['spec_tracker'].previous_landing)
        if any(key(old) == key(snapshot) for old in self.groups[snapshot.group]):
            self.rejections[snapshot.group+':같은 상태 중복'] += 1
            return
        self.serial += 1
        snapshot.id = f'seed{self.seed}-state{self.serial}'
        group = snapshot.group
        self.seen[group] += 1
        items = self.groups[group]
        if len(items) < self.capacity:
            items.append(snapshot)
        else:
            index = int(self.rng.integers(self.seen[group]))
            if index < self.capacity:
                items[index] = snapshot

    def sample(self, role, *, rng=None, record=True):
        groups = self.available(role)
        if not groups:
            return None
        chooser = self.rng if rng is None else rng
        group = groups[int(chooser.integers(len(groups)))]
        items = self.groups[group]
        snapshot = items[int(chooser.integers(len(items)))]
        if record:
            self.selected[group] += 1
        return snapshot

    def manifest(self):
        return dict(seed=self.seed, capacity_per_group=self.capacity, count=sum(map(len, self.groups.values())),
            rejections=dict(self.rejections), groups={g: dict(label=GROUPS[g][0], stored=len(items),
                collected=self.seen[g], selected=self.selected[g]) for g, items in self.groups.items()})

    def save(self, path):
        """검토용 상태를 저장한다. 정확한 중단 지점 재개 기능은 아니다."""
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.tmp')
        with gzip.open(temporary, 'wb') as stream:
            pickle.dump(dict(schema=1, manifest=self.manifest(), groups=self.groups,
                             rng_state=self.rng.bit_generator.state), stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)


class SpecCurriculumRawEnv(SpecRawEnv):
    def __init__(self, bank=None, role='original', collect=True, limit=180.):
        if role not in ROLES:
            raise ValueError(role)
        self.bank = bank if bank is not None else SnapshotBank()
        self.role, self.collect_enabled = role, bool(collect)
        self.fingerprint = None
        self._clear_segment()
        super().__init__(limit=limit)

    def _clear_segment(self):
        self.start_group = self.start_snapshot_id = None
        self.start_kind = 'original'
        self.start_time = self.start_tick = self.start_decisions = 0
        self.initial_reward_items, self.initial_reward_parts = set(), Counter()
        self.initial_duplicates, self.initial_unpaid = Counter(), Counter()
        self.local_goal_ids, self.local_goal_reached = (), False
        self.upper_entry_after_start = False
        self.segment_landed, self.captured = set(), Counter()
        self.segment_actions, self.turn_actions = np.zeros(9, dtype=np.int64), np.zeros(9, dtype=np.int64)
        self.decision_in_turn = False

    def reset(self, *, seed=None, options=None):
        self._clear_segment()
        super().reset(seed=seed, options=options)
        if self.fingerprint is None:
            content = json.dumps(dict(level=asdict(self.level), physics=asdict(self.world.physics),
                                      limit=self.limit), sort_keys=True, ensure_ascii=False)
            self.fingerprint = hashlib.sha256(content.encode('utf-8')).hexdigest()
        options = options or {}
        snapshot = options.get('snapshot')
        if snapshot is None and not options.get('original_start', False):
            snapshot = self.bank.sample(self.role)
        if snapshot is not None:
            self.restore_state(snapshot)
        return self.observe(), self.info()

    def capture_state(self, group):
        """행동 종료 후 확정된 월드·관측·최초 보상 기억을 한 번에 복사한다."""
        if group not in GROUPS or self.done or self.world.failed or self.world.cleared or self.world.time >= self.limit:
            raise ValueError('저장할 수 있는 진행 중 상태가 아닙니다')
        prefix = Counter(self.initial_reward_parts)
        prefix.update(self.reward_parts)
        if set(prefix)-{'intermediate'} or prefix.get('intermediate', 0) != len(self.spec_tracker.paid):
            raise ValueError('보상 지급 이력이 일치하지 않습니다')
        return Snapshot(1, self.fingerprint, group,
            copy.deepcopy({name: getattr(self, name) for name in STATE_FIELDS}), dict(prefix), self.start_snapshot_id)

    def restore_state(self, snapshot):
        """환경 시간과 지급 기억을 복원한다. 정책 가중치·행동 난수는 되감지 않는다."""
        if snapshot.schema != 1 or snapshot.fingerprint != self.fingerprint or snapshot.group not in GROUPS:
            raise ValueError('현재 맵·물리·시간 제한과 호환되지 않는 상태입니다')
        state = copy.deepcopy(snapshot.state)
        world = state['world']
        if world.failed or world.cleared or not 0 <= world.time < self.limit:
            raise ValueError('종료되었거나 시간 제한을 벗어난 상태입니다')
        if set(snapshot.prefix_parts)-{'intermediate'} or snapshot.prefix_parts.get('intermediate', 0) != len(state['spec_tracker'].paid):
            raise ValueError('저장 상태의 보상 이력이 일치하지 않습니다')
        self._clear_segment()
        for name in STATE_FIELDS:
            setattr(self, name, state[name])
        self.done = False
        self.start_kind, self.start_group, self.start_snapshot_id = 'checkpoint', snapshot.group, snapshot.id
        self.start_time, self.start_tick, self.start_decisions = self.world.time, self.ticks, self.decisions
        self.initial_reward_items = set(self.spec_tracker.paid)
        self.initial_reward_parts = Counter(snapshot.prefix_parts)
        self.initial_duplicates, self.initial_unpaid = Counter(self.spec_tracker.duplicates), Counter(self.spec_tracker.unpaid)
        self.local_goal_ids = GROUPS[snapshot.group][2]
        self.reward_parts, self.reward_events = Counter(), []
        self.raw_base_return = self.shaping_return = self.bonus_return = self.episode_return = 0.
        # 마지막 물리 프레임의 오래된 접촉을 새로운 착지로 처리하지 않는다.
        self.world.reward_landings = []
        self.tracker.last_events = []

    def _collect(self):
        if not self.collect_enabled or self.done or self.limit-self.world.time < 5:
            return
        b, w = self.world.ball, self.world
        if not all(np.isfinite(v) for v in (b.x, b.y, b.vx, b.vy)):
            return
        previous = self.spec_tracker.previous_landing
        for group, (_, ids, _) in GROUPS.items():
            if previous not in ids or self.captured[group] >= 2 or w.time-w.last_landing_time > 1.2:
                continue
            p = self.platform_by_id[previous]
            x, y, width, _ = p.rect_at(w.time, w.physics)
            # 낙하 중 목표 아래로 지나간 상태를 보관하지 않는다. 다른 속도·시각은 그대로 둔다.
            margin = 256. if group == 'last_stair' else 128.
            if y-256 <= b.y <= y-w.physics.radius+1e-6 and x-128 <= b.x <= x+width+margin:
                self.bank.add(self.capture_state(group))
                self.captured[group] += 1
            else:
                self.bank.rejections[group+':행동 종료 위치 범위 밖'] += 1

    def info(self, tick_delta=0):
        result = super().info(tick_delta)
        new = self.spec_tracker.paid-self.initial_reward_items
        result.update(start_role=self.role, start_kind=self.start_kind, start_group=self.start_group,
            start_snapshot_id=self.start_snapshot_id, start_world_time=self.start_time,
            segment_seconds=self.world.time-self.start_time, segment_ticks=self.ticks-self.start_tick,
            segment_decisions=self.decisions-self.start_decisions,
            initial_reward_items=sorted(self.initial_reward_items), initial_reward_parts=dict(self.initial_reward_parts),
            new_reward_items=sorted(new), segment_landed_platforms=sorted(self.segment_landed),
            local_goal_ids=list(self.local_goal_ids), local_goal_reached=self.local_goal_reached,
            upper_entry_after_start=self.upper_entry_after_start, decision_in_turn=self.decision_in_turn,
            segment_actions=self.segment_actions.tolist(), turn_actions=self.turn_actions.tolist(),
            segment_duplicate_landings=dict(Counter(self.spec_tracker.duplicates)-self.initial_duplicates),
            segment_unpaid_landings=dict(Counter(self.spec_tracker.unpaid)-self.initial_unpaid))
        return result

    def step(self, action, on_tick=None):
        if self.done:
            raise RuntimeError('종료 후 reset()이 필요합니다')
        if not self.action_space.contains(action):
            raise ValueError(action)
        self.decision_in_turn = is_turn_decision(self)
        self.segment_actions[int(action)] += 1
        if self.decision_in_turn:
            self.turn_actions[int(action)] += 1
        def tick(world, number, direction):
            if not world.failed:
                landings = {c['platform'] for c in world.reward_landings}
                self.segment_landed.update(landings)
                self.local_goal_reached |= bool(landings.intersection(self.local_goal_ids))
                self.upper_entry_after_start |= any(p.startswith('turn0buffer-') for p in landings)
            if on_tick is not None:
                on_tick(world, number, direction)
        _, reward, done, truncated, info = super().step(action, on_tick=tick)
        self._collect()
        return self.observe(), reward, done, truncated, self.info(info['tick_delta'])


class SpecCurriculumEnv(gym.ObservationWrapper):
    """구간 이름과 보상 기억을 정책 입력에 추가하지 않고 기존1,984차원을 유지한다."""
    def __init__(self, **options):
        super().__init__(SpecCurriculumRawEnv(**options))
        self.observation_space = spaces.Box(-8., 8., (OBS_SIZE,), dtype=np.float32)

    observation = SpecEnv.observation
