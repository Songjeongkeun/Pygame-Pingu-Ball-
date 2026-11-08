"""튜토리얼 1~5의 기존 보상에 2·4번 보조 보상만 더한 비교 환경.

물리·맵·행동·관측·기존 핵심 구간 판정은 바꾸지 않는다. 2번의 기존
+10 착지 보상에는 시간 제한을 새로 추가하지 않는다. 보조 진입 보상만
점프 발판 반동 후 3초 이내라는 tutorial_2_4 조건을 사용한다.
"""
from collections import Counter
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .milestone_env import MilestoneRawEnv, MilestoneEnv, terminal_points, OBS_SIZE
from .combined_rewards import AuxiliaryRewards


class CombinedRawEnv(MilestoneRawEnv):
    def __init__(self, stage=1, reward_mode='combined', limit=180., repeat=6):
        if reward_mode not in ('combined', 'milestone'):
            raise ValueError('통합 보상 모드만 지원한다')
        self.auxiliary = None
        self.reward_parts = Counter()
        super().__init__(stage=stage, reward_mode='milestone', limit=limit, repeat=repeat)

    def reset(self, *, seed=None, options=None):
        self.auxiliary = None
        self.reward_parts = Counter()
        super().reset(seed=seed, options=options)
        self.auxiliary = AuxiliaryRewards(self.stage, self.world, self.specs)
        return self.observe(), self.info()

    def info(self, tick_delta=0):
        info = super().info(tick_delta)
        info.update(reward_variant='combined', reward_parts=dict(self.reward_parts),
                    auxiliary_state=self.auxiliary.state() if self.auxiliary else {})
        return info

    def step(self, action, on_tick=None):
        """기존 환경과 같은 물리 갱신 순서로 모든 보상 항목을 한 번씩 합산한다."""
        if self.done:
            raise RuntimeError('종료 후에는 reset()을 호출해야 한다')
        if not self.action_space.contains(action):
            raise ValueError(action)
        before = self.ticks
        self.reward_events = []
        for _ in range(self.repeat):
            dt = min(self.world.physics.dt, self.limit - self.world.time)
            if dt < 1e-10 or self.world.failed or self.world.cleared:
                break
            self.world.step(int(action) - 1, dt)
            self.ticks += 1
            self.min_goal_distance = min(self.min_goal_distance, self.goal_distance())
            self.reward_events.extend(self.auxiliary.motion(self.world))
            for event, value in self.world.events:
                if event == 'land' and not self.world.failed:
                    if value in self.support_lookup:
                        self.landing_index = self.support_lookup[value]
                        self.touched.add(self.landing_index)
                    # +10은 기존 추적기에서만 지급한다. 보조 추적기에는 같은 보상이 없다.
                    for milestone in self.tracker.land(value, self.ticks):
                        self.reward_events.append(dict(kind='milestone', id=milestone, points=10.))
                    self.reward_events.extend(self.auxiliary.landing(self.world, value))
            self.reward_events.extend(self.auxiliary.after_landings(self.world))
            if on_tick:
                on_tick(self.world, self.ticks, int(action) - 1)
        self.previous_action, self.decisions = int(action), self.decisions + 1
        timeout = self.world.time >= self.limit - 1e-9
        self.done = bool(self.world.cleared or self.world.failed or timeout)
        base = terminal_points(self.world.cleared, self.world.failed, timeout)
        bonus = sum(e['points'] for e in self.reward_events)
        if self.done:
            self.reward_events.append(dict(kind='success' if base > 0 else 'failure', points=base))
        for event in self.reward_events:
            self.reward_parts[event['kind']] += event['points']
        self.raw_base_return += base
        self.bonus_return += bonus
        self.episode_return += base + bonus
        self.visited[self.cell()] = 1
        return self.observe(), (base + bonus) / 100., self.done, False, self.info(self.ticks - before)


class CombinedEnv(MilestoneEnv):
    """기존 1,664개 관측을 같은 순서로 반환하므로 같은 신경망을 사용할 수 있다."""
    def __init__(self, **options):
        gym.ObservationWrapper.__init__(self, CombinedRawEnv(**options))
        self.observation_space = spaces.Box(-8., 8., (OBS_SIZE,), dtype=np.float32)


def audit_reward(info):
    """평가가 끝난 매 에피소드의 중복 지급과 점수 합계를 검사한다."""
    parts = info['reward_parts']
    assert abs(sum(parts.values()) - info['episode_return']) < 1e-6
    assert abs(info['learning_return'] * 100. - info['episode_return']) < 1e-6
    assert parts.get('milestone', 0.) == 10. * len(info['milestones'])
    assert len(info['milestones']) == len(set(info['milestones']))
    assert parts.get('success', 0.) + parts.get('failure', 0.) == (100. if info['success'] else -25.)
    assert not (parts.get('success') and parts.get('failure'))
    caps = {'spring_height': 1.5, 'spring_horizontal': 1.5,
            'approach': 12.96, 'lift_first': 12., 'lift_high': 12., 'upper_first': 18.}
    for key, cap in caps.items():
        assert -1e-8 <= parts.get(key, 0.) <= cap + 1e-6, (key, parts)
    if info['stage'] not in (2, 4):
        assert not any(parts.get(k) for k in caps)
