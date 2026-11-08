"""구간 시작 학습의 행동 확률·실제 선택·진단 평가를 원래 출발점 평가와 분리한다."""
from collections import Counter
import json

import numpy as np
import torch

from .five_models import action_probs
from .map1_spec_curriculum import GROUPS, DecisionLog, SpecCurriculumEnv, is_turn_decision
from .map1_spec_rewards import audit_reward


def new_group_record():
    return dict(steps=0, physics_ticks=0, episodes=0, local_goal_hits=0, stage_successes=0,
                upper_entries_after_start=0, new_reward_items=0)


class CurriculumStatsMixin:
    """기존 스텝 콜백 앞에서 확률과 시작 유형별 기록을 덧붙인다."""
    def setup_curriculum(self, bank, count=4):
        self.bank = bank
        self.pending_logs = [DecisionLog() for _ in range(count)]
        self.group_logs = {g: DecisionLog() for g in ('original', *GROUPS)}
        self.group_records = {g: new_group_record() for g in ('original', *GROUPS)}
        self.role_steps = Counter()

    def curriculum_status(self):
        return dict(snapshot_bank=self.bank.manifest(), steps_by_role=dict(self.role_steps),
            by_start={g: dict(row, local_goal_rate=row['local_goal_hits']/row['episodes'] if row['episodes'] else None,
                              turn_policy=self.group_logs[g].summary()) for g, row in self.group_records.items()})

    def record_curriculum_step(self):
        # obs_tensor는 자동 리셋 전, 행동을 선택한 원래 상태다. 종료 후 관측을 사용하면 안 된다.
        with torch.no_grad():
            probabilities = action_probs(self.model, self.locals['obs_tensor'])
        for i, (done, info, action, p) in enumerate(zip(self.locals['dones'], self.locals['infos'],
                                                       self.locals['actions'], probabilities)):
            group = info['start_group'] or 'original'
            row = self.group_records[group]
            row['steps'] += 1
            row['physics_ticks'] += info['tick_delta']
            self.role_steps[info['start_role']] += 1
            if info['decision_in_turn']:
                self.pending_logs[i].add(action, p)
                self.group_logs[group].add(action, p)
            if done:
                policy = self.pending_logs[i].summary()
                assert policy['actions'] == info['turn_actions']
                info['turn_policy'] = policy
                self.pending_logs[i] = DecisionLog()
                row['episodes'] += 1
                row['local_goal_hits'] += int(info['local_goal_reached'])
                row['stage_successes'] += int(info['success'])
                row['upper_entries_after_start'] += int(info['upper_entry_after_start'])
                row['new_reward_items'] += len(info['new_reward_items'])


def evaluate_local(model, bank, count=20, seed=21000):
    """학습 저장 상태에서의 진단이다. 별도 난수를 사용하며 모델 선택에는 사용하지 않는다."""
    before = (json.dumps(bank.manifest(), sort_keys=True), json.dumps(bank.rng.bit_generator.state, sort_keys=True))
    training = model.policy.training
    model.policy.set_training_mode(False)
    results = {}
    try:
        for gi, group in enumerate(GROUPS):
            states = tuple(bank.groups[group])
            if not states:
                results[group] = dict(available=False, episodes=0, local_goal_rate=None)
                continue
            rng = np.random.default_rng(seed+gi*104729)
            env = SpecCurriculumEnv(collect=False)
            rows, logs = [], DecisionLog()
            try:
                for _ in range(count):
                    snapshot = states[int(rng.integers(len(states)))]
                    obs, _ = env.reset(options={'snapshot': snapshot})
                    per_episode = DecisionLog()
                    while True:
                        p = action_probs(model, obs[None])[0]
                        action = int(rng.choice(9, p=p/p.sum()))
                        if is_turn_decision(env.unwrapped):
                            per_episode.add(action, p)
                            logs.add(action, p)
                        obs, _, done, truncated, info = env.step(action)
                        if done or truncated:
                            audit_reward(info)
                            info['turn_policy'] = per_episode.summary()
                            rows.append(info)
                            break
                results[group] = dict(available=True, episodes=len(rows),
                    local_goal_rate=float(np.mean([r['local_goal_reached'] for r in rows])),
                    stage_success_rate=float(np.mean([r['success'] for r in rows])),
                    mean_new_reward_items=float(np.mean([len(r['new_reward_items']) for r in rows])),
                    mean_segment_return=float(np.mean([r['episode_return'] for r in rows])),
                    turn_policy=logs.summary(), raw=rows,
                    protocol='학습 저장 상태의 진단 평가. 원래 출발점 시험과 별도. 모델 선택에 사용하지 않음')
            finally:
                env.close()
    finally:
        model.policy.set_training_mode(training)
    after = (json.dumps(bank.manifest(), sort_keys=True), json.dumps(bank.rng.bit_generator.state, sort_keys=True))
    assert before == after
    return results
