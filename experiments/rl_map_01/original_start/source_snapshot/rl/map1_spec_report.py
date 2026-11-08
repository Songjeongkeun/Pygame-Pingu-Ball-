"""PDF 실험의 스텝 기준 곡선·33항목 지급률·PPO 안정성과 한글 보고서를 생성한다."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .five_train import read, write
from .map1_spec_rewards import RULES, audit_reward
from .map1_spec_train import selection_score


def episodes(folder):
    path = folder/'episodes.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def review(root, plan, summary, history, records):
    """실제 종료 판·검증 선택·보상 합계를 검사한다. 학습 성능과 검사 통과는 다르다."""
    for row in records:
        audit_reward(row)
    assert len(records) == summary['training_episodes']
    if plan['budget_unit'] == 'steps':
        assert summary['training_steps'] == summary['decisions'] == plan['expected_training_steps']
        assert summary['steps_over_budget'] == summary['training_steps']-plan['steps']
        assert 0 <= summary['steps_over_budget'] < 512
        assert summary['last_updated_decisions'] == summary['training_steps']
        assert summary['tail_decisions_not_updated'] == 0
    else:
        # 이전 에피소드 기준 실행 결과도 덮어쓰지 않고 보고서를 다시 만들 수 있다.
        assert summary['episodes_over_budget'] == len(records)-plan['episodes']
        assert 0 <= summary['episodes_over_budget'] <= 3
    assert 0 <= summary['tail_decisions_not_updated'] <= 512
    assert summary['config']['sources'] == plan['sources'] == summary['sources_after']
    assert read(root/'completion.json')['sources'] == plan['sources']
    assert not summary['test_used_for_selection']
    assert all(row['episode'] == i+1 for i, row in enumerate(records))
    parts = Counter()
    payouts = Counter()
    for row in records:
        parts.update(row['reward_parts'])
        payouts.update(row['reward_items'])
    assert dict(parts) == summary['training_reward_parts']
    assert dict(payouts) == summary['training_reward_payouts']
    selected = max(enumerate(history), key=lambda pair: (selection_score(pair[1]['validation']), pair[0]))[1]
    assert selected['decisions'] == summary['best_step']
    assert selected['validation'] == summary['best_validation']['evaluation']
    for name in ('final', 'latest_test', 'selected_greedy', 'latest_greedy'):
        result = summary[name]
        assert result['episodes'] == len(result['raw'])
        assert result['successes'] == sum(row['success'] for row in result['raw'])
        assert abs(result['success_rate']-result['successes']/result['episodes']) < 1e-9
        for row in result['raw']:
            audit_reward(row)
        for rule in RULES:
            rate = sum(rule.id in row['reward_items'] for row in result['raw'])/result['episodes']
            assert abs(rate-result['reward_item_rates'][rule.id]) < 1e-9
    return dict(passed=True, checked_episodes=len(records), reward_items=33,
                selection_uses_validation_only=True, success_is_separate_from_checks=True)


def plots(root, history, summary=None):
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Malgun Gothic', 'NanumGothic', 'Noto Sans CJK KR'):
        if name in fonts:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False
    step_budget = history[0].get('budget_unit') == 'steps'
    def coordinate(row):
        return row['training_steps'] if step_budget else row['training_episodes']
    xlabel = '학습 스텝 (환경4개 합계)' if step_budget else '종료 학습 에피소드'
    x = [coordinate(r) for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    curves = [('success_rate', '검증 완주율 (%)', 100), ('mean_return', '검증 평균 원점수', 1),
              ('mean_reward_items', '검증 평균 획득 항목 (최대33)', 1), ('upper_entry_rate', '위층 진입률 (%)', 100)]
    for ax, (key, label, scale) in zip(axes.flat, curves):
        ax.plot(x, [r['validation'][key]*scale for r in history], marker='o', markersize=3)
        ax.set(xlabel=xlabel, ylabel=label)
        ax.grid(alpha=.25)
    fig.savefig(root/'learning_metrics.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, key in zip(axes.flat, ('train/approx_kl', 'train/clip_fraction',
                                 'train/entropy_loss', 'train/explained_variance')):
        points = [(coordinate(row), row['diagnostics'][key])
                  for row in history if key in row['diagnostics']]
        if points:
            ax.plot(*zip(*points), marker='o', markersize=3)
        else:
            ax.text(.5, .5, '아직 PPO 갱신 지표 없음', ha='center', transform=ax.transAxes)
        ax.set(xlabel=xlabel, ylabel=key)
        ax.grid(alpha=.25)
    fig.savefig(root/'ppo_stability.png', dpi=150)
    plt.close(fig)
    evaluations = [r['validation'] for r in history]
    labels = [f'{coordinate(r):,}{"스텝" if step_budget else "판"} 검증' for r in history]
    if summary:
        evaluations = [summary['final'], summary['latest_test']]
        labels = ['검증 선택본 · 독립 시험', '마지막본 · 독립 시험']
    arrays = [np.array([[ev[key][rule.id]*100 for rule in RULES] for ev in evaluations])
              for key in ('reward_item_rates', 'target_reach_rates')]
    height = max(6., 3.+len(labels)*.55)
    fig, axes = plt.subplots(2, 1, figsize=(22, height), constrained_layout=True)
    for ax, data, title in zip(axes, arrays, ('항목별 +1 보상 지급률 (%)', '대상 발판 실제 착지율 (%) · 연결 조건과 별도')):
        im = ax.imshow(data, aspect='auto', vmin=0, vmax=100, cmap='Blues')
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(33), [f'{i+1}. {r.label}' for i, r in enumerate(RULES)], rotation=65, ha='right', fontsize=8)
        ax.set_title(title)
        if len(labels) <= 3:
            for y, row in enumerate(data):
                for col, value in enumerate(row):
                    ax.text(col, y, f'{value:.0f}', ha='center', va='center', fontsize=7,
                            color='white' if value > 55 else '#23354a')
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.savefig(root/'reward_items.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--live', action='store_true', help='진행 중 검증 그래프만 생성')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    plan = read(root/'plan.json')
    folder = root/f'seed_{plan["seed"]}'
    history = read(folder/'metrics.json')
    if args.live:
        plots(root, history)
        print(f'진행 중 검증 그래프: {root}')
        return
    if not (root/'completion.json').exists():
        raise RuntimeError('완료 기록이 없습니다. 진행 중에는 --live를 사용해 주세요')
    summary = read(folder/'summary.json')
    records = episodes(folder)
    checked = review(root, plan, summary, history, records)
    plots(root, history, summary)
    write(root/'final_review.json', checked)
    keys = ('final', 'latest_test', 'training_episodes', 'physics_ticks', 'training_reward_parts',
            'training_reward_payouts', 'tail_decisions_not_updated', 'elapsed_seconds', 'budget_unit')
    keys += ('training_steps', 'requested_steps', 'steps_over_budget', 'active_episodes_at_stop') \
        if plan['budget_unit'] == 'steps' else ('episodes_over_budget',)
    write(root/'aggregate.json', {k: summary[k] for k in keys})
    final = summary['final']
    budget_line = (f'- 요청 {plan["steps"]:,}스텝 / 실제 {summary["training_steps"]:,}스텝. '
                   f'완성된512스텝 묶음 단위로 수집·갱신 후 종료. 종료 에피소드 {summary["training_episodes"]:,}판.'
                   if plan['budget_unit'] == 'steps' else
                   f'- 요청 {plan["episodes"]:,}판 / 실제 종료 {summary["training_episodes"]:,}판.')
    lines = ['# 얼음계곡 맵1 · PDF 보상 실험 결과', '',
        f'- 시드: {plan["seed"]} 하나. 모든 판은 원래 출발점에서 시작.',
        budget_line,
        f'- 선택본 독립 시험: {final["successes"]}/{final["episodes"]}판 완주 ({final["success_rate"]:.1%}).',
        f'- 마지막본 시험 완주율: {summary["latest_test"]["success_rate"]:.1%}.',
        f'- 선택본 평균 원점수: {final["mean_return"]:.2f}, 평균 획득 항목 {final["mean_reward_items"]:.2f}/33.',
        f'- 위층 진입률: {final["upper_entry_rate"]:.1%}.',
        f'- 실제 물리 프레임: {summary["physics_ticks"]:,}. 종료 기준은 {"행동 스텝" if plan["budget_unit"] == "steps" else "에피소드"}.',
        f'- PPO 갱신에 사용되지 않은 수집 스텝: {summary["tail_decisions_not_updated"]}.',
        '- 기존 +10·접근·높이 통과 보상은 사용하지 않음. 33항목 각+1, 성공+100, 실패/시간초과-25, 입력 보상은 /100.',
        '- 선택은 검증 결과만 사용. 시험 성공률은 모델 선택에 사용하지 않음.', '',
        '## 항목별 실제 착지와 지급', '',
        '| 항목 | 실제 대상 발판 착지율 | +1 지급률 |', '|---|---:|---:|']
    lines += [f'| {r.label} | {final["target_reach_rates"][r.id]:.1%} | {final["reward_item_rates"][r.id]:.1%} |' for r in RULES]
    lines += ['', '## 실패·미지급 진단', '',
        '실패 구간은 실패 직전까지 마지막으로 실제 착지가 확인된 보상 구간이다.', '',
        '```json', json.dumps({k: final[k] for k in ('failures', 'failure_regions', 'unpaid_landings',
                                                   'duplicate_landings')}, ensure_ascii=False, indent=2), '```', '',
        '중복 착지 수는 이미 지급한 항목의 대상 발판에 다시 착지한 횟수이며 중복 지급이 아니다.',
        '착지율이 높고 지급률이 낮다면 직전 발판 연결 또는 리프트 접촉 높이 조건을 확인한다.',
        '학습 지표와 코드 검사는 완주 보장을 의미하지 않는다. 단일 시드 결과만으로 일반적인 우열을 단정하지 않는다.']
    (root/'결과보고서.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'검사·그래프·결과보고서 생성 완료: {root}')


if __name__ == '__main__':
    main()
