"""완료된 비교 실험의 보상 기록·검증 선택·학습 파일을 재검토한다."""
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from .five_models import action_probs
from .five_train import read, write
from .stabilized_env import StabilizedEnv, audit_reward
from .stabilized_load import load_policy
from .stabilized_train import sources, score


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    """JSON에 비정상적인 부동소수점 값이 없는지 확인한다."""
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def audit_evaluation(evaluation, count):
    rows = evaluation['raw']
    assert len(rows) == evaluation['episodes'] == count
    wins = sum(bool(row['success']) for row in rows)
    assert evaluation['successes'] == wins
    assert abs(evaluation['success_rate'] - wins/count) < 1e-12
    assert evaluation['failures'] == dict(Counter(row['reason'] for row in rows if not row['success']))
    for row in rows:
        audit_reward(row)
    assert abs(evaluation['mean_return'] - np.mean([r['episode_return'] for r in rows])) < 1e-8
    if evaluation['native_comparison']:
        assert evaluation['max_native_error'] == 0.
    finite(evaluation)
    return len(rows)


def failure_buckets(evaluation, stage):
    """실패한 판에서 마지막으로 실제 달성한 구간을 집계한다."""
    result = Counter()
    for row in evaluation['raw']:
        if row['success']:
            label = '성공'
        elif stage == 4:
            aux = row['auxiliary_state']
            label = ('리프트 착지 전' if not aux['lift_first'] else
                     '리프트에서 윗층까지' if not aux['upper_first'] else '윗층에서 이글루까지')
        else:
            milestones = row['milestones']
            label = ('특수 발판 착지 전' if not row['armed'] else
                     '첫 목표 착지 전' if not milestones else
                     f'핵심 구간 {len(milestones)}개 달성 후')
        result[label] += 1
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = read(root/'plan.json')
    completion = read(root/'completion.json')
    aggregate = read(root/'aggregate.json')
    initialization = read(root/'initialization_audit.json')
    implementation = read(root/'implementation_checks.json')
    assert completion['complete'] and aggregate['all_complete']
    assert completion['models'] == plan['expected_models'] == aggregate['completed_models']
    assert initialization['passed'] and initialization['models'] == plan['expected_models']
    assert implementation['passed']
    assert plan['sources'] == completion['sources'] == sources()
    for name, expected in plan['sources'].items():
        assert digest(root/'source_snapshot'/name) == expected

    torch.set_num_threads(1)
    total = Counter()
    rows = []
    paths = sorted(root.glob('*/stage_*/seed_*/summary.json'))
    assert len(paths) == plan['expected_models']
    for path in paths:
        data = read(path)
        config = data['config']
        stage, variant, seed = config['stage'], config['variant'], config['seed']
        assert config['sources'] == data['sources_after'] == plan['sources']
        assert config['from_scratch'] and not config['previous_models_loaded']
        assert data['test_used_for_selection'] is False
        history = read(path.parent/'metrics.json')
        assert history[0]['decisions'] == history[0]['physics_ticks'] == 0
        for before, after in zip(history, history[1:]):
            assert before['decisions'] < after['decisions']
            assert before['physics_ticks'] < after['physics_ticks']
        assert history[-1]['decisions'] == data['decisions']
        assert history[-1]['physics_ticks'] == data['physics_ticks']
        expected = max(enumerate(history), key=lambda item: (score(item[1]['validation']), item[0]))[1]
        assert data['best_step'] == expected['decisions']
        assert data['best_validation']['evaluation'] == expected['validation']
        assert data['physics_ticks'] <= plan['physics_budget'] + 128*4*72
        if data['stop_reason'] == 'validation_drop':
            assert variant == 'C' and history[-1]['consecutive_drops'] >= 3
        else:
            assert data['stop_reason'] == 'physics_budget'
            assert data['physics_ticks'] >= plan['physics_budget']
        for point in history:
            total['validation_records'] += audit_evaluation(point['validation'], config['validation_count'])
        for key in ('final', 'latest_test'):
            total['test_records'] += audit_evaluation(data[key], config['test_count'])
        for key in ('selected_greedy', 'latest_greedy'):
            total['native_diagnostic_records'] += audit_evaluation(data[key], 1)

        diagnostics = {}
        env = StabilizedEnv(stage=stage, variant=variant)
        try:
            obs = env.reset()[0][None]
            for key in ('best', 'latest'):
                checkpoint = path.parent/f'{key}.zip'
                model = load_policy(checkpoint)
                assert model.num_timesteps == (data['best_step'] if key == 'best' else data['decisions'])
                assert model.action_space.n == 9 and model.observation_space.shape == (1690,)
                assert model.gamma == 1. and model.gae_lambda == .98
                assert all(torch.isfinite(value).all().item() for value in model.policy.state_dict().values())
                probabilities = action_probs(model, obs)[0]
                assert np.isfinite(probabilities).all() and abs(float(probabilities.sum())-1.) < 1e-6
                diagnostics[key] = dict(checkpoint_sha256=digest(checkpoint),
                    initial_probabilities=probabilities.tolist(),
                    initial_entropy=float(-np.sum(probabilities*np.log(np.maximum(probabilities, 1e-12)))))
                del model
        finally:
            env.close()
        parts = data['training_reward_parts']
        assert abs(parts.get('success', 0.) - data['training_successes']*100) < 1e-6
        assert abs(parts.get('failure', 0.) + (data['training_episodes']-data['training_successes'])*25) < 1e-6
        for key in ('decisions', 'physics_ticks', 'training_episodes', 'training_successes'):
            total[key] += data[key]
        finite(data)
        rows.append(dict(variant=variant, stage=stage, seed=seed,
            selected_rate=data['final']['success_rate'], latest_rate=data['latest_test']['success_rate'],
            selected_failures=failure_buckets(data['final'], stage),
            latest_failures=failure_buckets(data['latest_test'], stage),
            first_nonzero_validation_ticks=next((p['physics_ticks'] for p in history if p['validation']['successes']), None),
            first_half_success_ticks=next((p['physics_ticks'] for p in history if p['validation']['success_rate'] >= .5), None),
            stop_reason=data['stop_reason'], checkpoints=diagnostics))
        print(f'검토 완료 {variant} {stage}번 시드{seed}', flush=True)

    artifacts = ['plan.json', 'completion.json', 'initialization_audit.json',
                 'implementation_checks.json', 'aggregate.json', 'comparison.csv',
                 '결과보고서.md', 'success_comparison.png', 'learning_curves.png', 'phase_comparison.png']
    result = dict(passed=True, models=len(rows), totals=dict(total), rows=rows,
        sources_unchanged=True, source_snapshots_verified=True,
        previous_experiment_sources_unchanged=initialization['previous_sources_unchanged'],
        original_native_physics_max_error=0., selected_using_validation_only=True,
        initialization_audit_passed=True, implementation_checks=implementation,
        results_are_same_spawn_tests=True, git_actions=False,
        evaluation_record_note='검증 기록·선택본·마지막본·진단의 저장 항목 수. 같은 가중치의 평가를 재사용한 항목도 포함한다.',
        artifact_sha256={name:digest(root/name) for name in artifacts})
    write(root/'final_review.json', result)
    print(dict(total), flush=True)
    print(root/'final_review.json', flush=True)


if __name__ == '__main__':
    main()
