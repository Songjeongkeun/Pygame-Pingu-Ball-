"""에피소드 기준 맵1 학습의 구간별 진단과 원래 시작 평가를 그린다. 추가 학습은 없다."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil

import numpy as np
import torch

from .five_train import read, write
from .map_a_curriculum_report import episodes
from .map_a_episode_env import GROUPS, LOCAL_GOALS, audit_training_reward
from .map_a_episode_train import sources, parameter_hash
from .map_a_models import load_policy
from .map_a_env import SPECS, LABELS
from .map_a_report import setup_plot, audit_evaluation
from .map_a_train import selection_score


def learning_plot(root, history, records, live):
    """가로축은 완료한 학습 에피소드다. 검증·시험 판은 더하지 않는다."""
    plt = setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    x = [r['training_episodes'] for r in history]
    axes[0, 0].plot(x, [r['validation']['success_rate']*100 for r in history], marker='o')
    axes[0, 0].set(title='원래 출발점 검증 성공률', ylabel='완주 성공률 (%)', ylim=(-2, 102))
    axes[0, 1].plot(x, [r['validation']['mean_milestones'] for r in history], marker='o')
    axes[0, 1].set(title='원래 출발점 검증 평균 달성 구간', ylabel='구간 수 / 16', ylim=(-.2, 16.5))
    for kind, label, style in [('original', '원래 시작', '-'), ('middle', '중간 시작 연습', '--')]:
        selected = [r for r in records if r['start_kind'] == kind]
        if not selected:
            continue
        wins = np.asarray([r['success'] for r in selected], float)
        prefix = np.r_[0., wins.cumsum()]
        at = np.arange(1, len(wins)+1)
        begin = np.maximum(0, at-200)
        values = (prefix[at]-prefix[begin])/(at-begin)*100
        take = np.unique(np.r_[np.arange(0, len(wins), max(1, len(wins)//700)), len(wins)-1])
        axes[1, 0].plot([selected[i]['episode'] for i in take], values[take], linestyle=style, label=label)
    axes[1, 0].set(title='시작 방식별 최근 최대 200판 성공률', ylabel='성공률 (%)', ylim=(-2, 102))
    if axes[1, 0].lines:
        axes[1, 0].legend(fontsize=9)
    axes[1, 1].plot(x, [r['middle_probability']*100 for r in history], marker='o', label='중간 시작 확률')
    axes[1, 1].plot(x, [r['entropy']*1000 for r in history], linestyle='--', label='엔트로피 계수 × 1,000')
    axes[1, 1].set(title='검증 기반 시작 확률과 에피소드 기반 엔트로피 일정', ylim=(-2, 102))
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.set_xlabel('종료한 학습 에피소드 수')
        ax.grid(alpha=.2)
    fig.suptitle('진행 중 · 최종 시험 결과 아님' if live else '맵1 · 시드 하나 · 에피소드 기준 커리큘럼')
    path = root/('learning_metrics_live.png' if live else 'learning_metrics.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def final_plots(root, summary, records):
    """전체 성공률, 구간별 목표 달성률, 실제 연습량을 서로 구분해서 그린다."""
    plt = setup_plot()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout='constrained')
    bars = axes[0].bar(['검증 선택본', '마지막 저장본'],
        [summary[k]['success_rate']*100 for k in ('final', 'latest_test')], color=['#0284c7', '#f59e0b'])
    axes[0].bar_label(bars, fmt='%.0f%%', padding=3)
    axes[0].set(ylim=(-2, 110), ylabel='성공률 (%)', title='원래 출발점 독립 시험')
    matrix = np.asarray([[summary[k]['milestone_rates'][s.id]*100 for s in SPECS]
                         for k in ('final', 'latest_test')])
    im = axes[1].imshow(matrix, aspect='auto', vmin=0, vmax=100, cmap='Blues')
    for i in range(2):
        for j in range(len(SPECS)):
            axes[1].text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center', fontsize=8,
                         color='white' if matrix[i,j] > 55 else '#334155')
    axes[1].set(yticks=[0, 1], yticklabels=['선택본', '마지막본'],
        xticks=range(len(SPECS)), xticklabels=[f'{i+1}. {v}' for i, v in enumerate(LABELS)],
        title='원래 출발점에서의 구간 달성률 (%)')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    fig.colorbar(im, ax=axes[1], shrink=.7)
    fig.savefig(root/'test_results.png', dpi=150)
    plt.close(fig)
    rows = []
    for name, (label, _) in GROUPS.items():
        selected = [r for r in records if r['start_group'] == name]
        n = len(selected)
        rows.append(dict(group=name, label=label, episodes=n,
            goal_ids=list(LOCAL_GOALS[name]), local_goal_hits=sum(r['local_goal_reached'] for r in selected),
            local_goal_rate=sum(r['local_goal_reached'] for r in selected)/n if n else None,
            clears=sum(r['success'] for r in selected),
            ended_episode_ticks=sum(r['segment_ticks'] for r in selected)))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout='constrained')
    axes[0].bar(range(len(rows)), [r['episodes'] for r in rows], color='#38bdf8')
    axes[0].set(title='구간별 종료한 연습 에피소드', ylabel='판 수')
    axes[1].bar(range(len(rows)), [r['local_goal_rate']*100 if r['episodes'] else np.nan for r in rows], color='#10b981')
    for i, row in enumerate(rows):
        if not row['episodes']:
            axes[1].text(i, 3, '경험 없음', ha='center', fontsize=8)
    axes[1].set(title='해당 시작 구간에서 다음 지정 발판에 실제 착지한 비율', ylabel='착지율 (%)', ylim=(0, 110))
    for ax in axes:
        ax.set(xticks=range(len(rows)), xticklabels=[r['label'] for r in rows])
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right', fontsize=9)
    fig.savefig(root/'local_progress.png', dpi=150)
    plt.close(fig)
    return rows


def main(argv=None):
    """종료 판 수·보상·모델·소스를 대조한 뒤 결과를 작성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--reference', type=Path, help='이전 결과를 참고 표로 표시. 예산이 달라 통제 비교는 아님')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    plan = read(root/'plan.json')
    if plan.get('variant') != 'A-map1-episode-v2':
        raise ValueError('에피소드 기준 v2 결과 폴더를 지정해 주세요')
    folder = root/f'seed_{plan["seed"]}'
    history, records = read(folder/'metrics.json'), episodes(folder)
    if args.live:
        print(learning_plot(root, history, records, True))
        return
    complete, summary = read(root/'completion.json'), read(folder/'summary.json')
    assert complete['complete'] and complete['models'] == 1
    assert plan['sources'] == sources() == complete['sources'] == summary['sources_after']
    assert summary['config']['sources'] == plan['sources']
    assert plan['episode_budget'] <= summary['training_episodes'] <= plan['episode_budget']+3
    assert summary['training_episodes'] == len(records)
    assert sum(summary['episodes_by_start'].values()) == len(records)
    assert sum(summary['frames_by_start'].values()) == summary['physics_ticks']
    assert 0 <= summary['tail_decisions_not_updated'] <= 512
    counts, wins, parts = Counter(), Counter(), Counter()
    for row in records:
        audit_training_reward(row)
        assert row['start_kind'] in ('original', 'middle')
        assert row['local_goal_reached'] == bool(set(row['local_goal_ids']) & set(row['landed_platforms']))
        assert abs(row['seconds']-row['start_world_time']-row['segment_seconds']) < 1e-8
        if row['start_kind'] == 'original':
            assert row['start_world_time'] == 0 and not row['initial_milestones']
        else:
            assert row['local_goal_ids'] == list(LOCAL_GOALS[row['start_group']])
        counts[row['start_kind']] += 1
        wins[row['start_kind']] += int(row['success'])
        parts.update(row['reward_parts'])
    assert dict(counts) == summary['episodes_by_start'] and dict(wins) == summary['successes_by_start']
    for key in parts.keys() | summary['training_reward_parts'].keys():
        assert abs(parts[key]-summary['training_reward_parts'].get(key, 0.)) < 1e-6
    best = max(history, key=lambda r: (selection_score(r['validation']), r['decisions']))
    assert summary['best_step'] == best['decisions'] and not summary['test_used_for_selection']
    for row in history:
        audit_evaluation(row['validation'], plan['validation_count'])
    for key in ('final', 'latest_test'):
        audit_evaluation(summary[key], plan['test_count'])
    for key in ('selected_greedy', 'latest_greedy'):
        audit_evaluation(summary[key], 1)
    initialization = read(folder/'initialization.json')
    assert initialization['steps'] == 0 and not initialization['previous_models_loaded']
    np.testing.assert_allclose(initialization['probabilities'], np.full(9, 1/9), rtol=1e-6)
    hashes = {}
    for name, step in [('initial', 0), ('best', summary['best_step']), ('latest', summary['decisions'])]:
        path = folder/f'{name}.zip'
        model = load_policy(path)
        assert model.num_timesteps == step and all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
        if name == 'initial':
            assert parameter_hash(model) == initialization['parameter_sha256']
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, expected in plan['sources'].items():
        assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest() == expected
    learning_plot(root, history, records, False)
    local = final_plots(root, summary, records)
    for row in local:
        logged = summary['snapshot_bank']['groups'][row['group']]
        assert logged['ended'] == row['episodes'] and logged['local_goal_hits'] == row['local_goal_hits']
    lines = ['# 맵1 · 에피소드 기준 커리큘럼 v2', '',
        f'시드 **{plan["seed"]}** 하나. 목표 **{plan["episode_budget"]:,}판**, 실제 종료 **{len(records):,}판**.',
        f'실제 사용 물리 프레임 **{summary["physics_ticks"]:,}**. 프레임은 종료 기준이 아니다.',
        f'원래 시작 시험: 선택본 **{summary["final"]["success_rate"]:.1%}**, 마지막본 **{summary["latest_test"]["success_rate"]:.1%}**.', '',
        '검증·시험 판은 학습 에피소드 수에 포함하지 않는다. 원래 시작과 중간 시작 종료 판을 합산한다.',
        '네 내부 환경이 함께 종료할 때 목표보다 최대3판 초과할 수 있다.',
        f'표준 PPO 콜백 중단으로 마지막 미완성 수집 묶음의 {summary["tail_decisions_not_updated"]}개 행동은 가중치 갱신에서 제외됐다.', '',
        '위층 연결 기본 발판과 파괴 이후 상태를 수집하며, 드문 구간 달성 상태를 보호한다.',
        '저장 구간은 균등20%·어려움/경험부족40%·도달한 최후방 구간40%를 혼합해 선택한다.',
        '중간 시작은 검증 성과에 따라 서서히 줄이며, 완주0%인데 예산80%에서 중단하지 않는다.',
        'PPO·보상 금액·관측·9행동은 기존 A를 유지한다. 엔트로피 일정은 완료 에피소드 비율 기준이다.', '',
        '| 시작 구간 | 연습 판 | 다음 발판 실제 착지 | 착지율 | 이글루 도착 |',
        '|---|---:|---:|---:|---:|']
    for row in local:
        rate = f'{row["local_goal_rate"]:.1%}' if row['episodes'] else '경험 없음'
        lines.append(f'| {row["label"]} | {row["episodes"]} | {row["local_goal_hits"]} | {rate} | {row["clears"]} |')
    if args.reference:
        previous = read(args.reference/f'seed_{plan["seed"]}/summary.json')
        lines += ['', '## 이전 실험 참고', '',
            f'이전 선택본 성공률 {previous["final"]["success_rate"]:.1%}, 평균 구간 {previous["final"]["mean_milestones"]:.2f}/16.',
            f'이전 학습 종료 {previous["training_episodes"]:,}판, 실제 물리 프레임 {previous["physics_ticks"]:,}.',
            '**예산 기준과 커리큘럼이 달라 수정 효과만을 분리한 통제 비교는 아니다.**']
    lines += ['', '## 주의해서 해석할 내용', '',
        '- 다음 발판 착지율은 진단 지표다. 이 지표를 위해 보상을 추가하지 않았다.',
        '- 중간 상태에서 이글루에 도착한 결과와 원래 출발점 완주 성공률을 구분한다.',
        '- 방문한 적 없는 구간은 0% 대신 경험 없음으로 표시한다.',
        '- 원래 맵·물리·기존 실험 소스를 수정하지 않았다. 모든 상태는 같은 시드가 직접 방문했다.',
        '- 모델과 상태 저장소만으로 중단된 PPO 실행의 정확한 재개를 지원하지 않는다.', '',
        '![학습 지표](learning_metrics.png)', '', '![원래 시작 시험](test_results.png)', '',
        '![구간별 다음 발판 착지](local_progress.png)', '']
    (root/'결과보고서.md').write_text('\n'.join(lines), encoding='utf-8')
    write(root/'aggregate.json', dict(seed=plan['seed'], budget_unit='episodes',
        requested_episodes=plan['episode_budget'], actual_episodes=len(records),
        physics_ticks=summary['physics_ticks'], selected=summary['final'], latest=summary['latest_test'], local=local))
    analysis = {}
    for name in ('rl/map_a_episode_report.py', 'tests/test_map_a_episode.py'):
        target = root/'analysis_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        analysis[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    names = ['plan.json', 'completion.json', 'aggregate.json', '결과보고서.md',
             'learning_metrics.png', 'test_results.png', 'local_progress.png']
    write(root/'final_review.json', dict(passed=True, models=1, sources=sources(),
        checkpoint_sha256=hashes, analysis_sources=analysis, training_records_audited=len(records),
        artifact_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}))
    print(root/'결과보고서.md')


if __name__ == '__main__':
    main()
