"""커리큘럼의 구간 연습과 원래 시작 완주를 분리해 그래프·보고서를 만든다.

이 명령은 저장된 기록과 모델만 읽으며 추가 학습을 하지 않는다. --live는
진행 중 검증 곡선용이고, 최종 결과 검사는 completion.json이 있을 때 수행한다.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .five_train import read, write
from .map_a_curriculum_env import GROUPS, audit_training_reward
from .map_a_curriculum_train import sources, parameter_hash
from .map_a_env import SPECS, LABELS
from .map_a_models import load_policy
from .map_a_report import setup_plot, audit_evaluation
from .map_a_train import selection_score


def episodes(folder):
    """실행 중 쓰다 만 마지막 줄은 제외하고 완전히 기록된 학습판만 읽는다."""
    path = folder/'episodes.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines(keepends=True)
            if line.endswith('\n')]


def reference(baseline, seed, plan, folder):
    """같은 학습 코드와 초기 가중치인 기존 A가 있을 때만 비교에 포함한다."""
    if baseline is None or not (baseline/f'seed_{seed}/summary.json').exists():
        return None
    previous = read(baseline/'plan.json')
    if previous.get('variant') != 'A-map1' or previous.get('curriculum'):
        raise ValueError('비교 기준은 원래 시작 전용 A-map1 실험이어야 한다')
    if previous['physics_budget'] != plan['physics_budget']:
        raise ValueError('비교 기준과 이번 실험의 물리 프레임 예산이 다르다')
    if previous['test_count'] != plan['test_count']:
        raise ValueError('비교 기준과 이번 실험의 최종 시험 횟수가 다르다')
    if any(plan['sources'].get(name) != value for name, value in previous['sources'].items()):
        raise ValueError('비교 기준과 기존 A 학습 코드가 달라 직접 비교할 수 없다')
    before = read(baseline/f'seed_{seed}/initialization.json')['parameter_sha256']
    after = read(folder/'initialization.json')['parameter_sha256']
    if before != after:
        raise ValueError('같은 시드의 초기 가중치가 비교 기준과 다르다')
    return dict(summary=read(baseline/f'seed_{seed}/summary.json'),
                history=read(baseline/f'seed_{seed}/metrics.json'))


def learning_plot(root, plan, data, live):
    """평가 성공률과 구간 수, 시작 방식별 학습 성공률과 실제 프레임 비율을 그린다."""
    plt = setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    colors = ['#0284c7', '#f59e0b', '#059669', '#8b5cf6']
    for i, (seed, history, records, baseline) in enumerate(data):
        color = colors[i % len(colors)]
        x = [r['physics_ticks']/1e6 for r in history]
        axes[0, 0].plot(x, [r['validation']['success_rate']*100 for r in history],
                        marker='o', color=color, label=f'{seed}: 커리큘럼')
        axes[0, 1].plot(x, [r['validation']['mean_milestones'] for r in history],
                        marker='o', color=color, label=f'{seed}: 커리큘럼')
        if baseline:
            old = baseline['history']
            ox = [r['physics_ticks']/1e6 for r in old]
            axes[0, 0].plot(ox, [r['validation']['success_rate']*100 for r in old],
                            linestyle=':', color=color, alpha=.6, label=f'{seed}: 기존 A')
            axes[0, 1].plot(ox, [r['validation']['mean_milestones'] for r in old],
                            linestyle=':', color=color, alpha=.6, label=f'{seed}: 기존 A')
        for kind, style, label in [('original', '-', '원래'), ('middle', '--', '구간 연습')]:
            selected = [r for r in records if r['start_kind'] == kind]
            if not selected:
                continue
            wins = np.asarray([r['success'] for r in selected], float)
            prefix = np.r_[0., wins.cumsum()]
            at = np.arange(1, len(wins)+1)
            begin = np.maximum(0, at-200)
            moving = (prefix[at]-prefix[begin])/(at-begin)*100
            take = np.unique(np.r_[np.arange(0, len(wins), max(1, len(wins)//600)), len(wins)-1])
            axes[1, 0].plot([selected[j]['physics_ticks']/1e6 for j in take], moving[take],
                            linestyle=style, color=color, label=f'{seed}: {label}')
        used = [r['frames_by_start'].get('middle', 0)/max(1, r['physics_ticks'])*100 for r in history]
        axes[1, 1].plot(x, used, marker='o', color=color, label=f'시드 {seed}')
    axes[0, 0].set(title='원래 출발점 검증 성공률', ylabel='완주 성공률 (%)', ylim=(-2, 102))
    axes[0, 1].set(title='원래 출발점 검증 평균 구간', ylabel='실제 착지 구간 수 / 16', ylim=(-.2, 16.5))
    axes[1, 0].set(title='학습 중 최근 최대 200판 · 시작 방식별 분리', ylabel='성공률 (%)', ylim=(-2, 102))
    axes[1, 1].set(title='누적 물리 프레임 중 중간 시작이 사용한 비율', ylabel='사용 프레임 비율 (%)', ylim=(-2, 102))
    boundary = plan['physics_budget']*(1-plan['final_original_fraction'])/1e6
    for ax in axes.flat:
        ax.axvline(boundary, color='#94a3b8', linestyle='--', lw=.8)
        ax.set_xlabel('실제로 실행한 학습 물리 프레임 (백만)')
        ax.grid(alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle('진행 중 · 최종 시험 결과 아님' if live else '맵1 · 시작 상태 커리큘럼 · 원래 시작 완주와 구간 연습을 구분')
    path = root/('learning_metrics_live.png' if live else 'learning_metrics.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def final_plots(root, plan, summaries, references, all_records):
    """선택본·마지막본 시험과 구간별 연습 결과를 서로 다른 그림으로 만든다."""
    plt = setup_plot()
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), layout='constrained')
    x = np.arange(len(summaries))
    for offset, key, label, color in [(-.2, 'final', '커리큘럼 선택본', '#0284c7'),
                                     (.2, 'latest_test', '커리큘럼 마지막본', '#f59e0b')]:
        bars = axes[0].bar(x+offset, [s[key]['success_rate']*100 for s in summaries], .35,
                           label=label, color=color)
        axes[0].bar_label(bars, fmt='%.0f%%', padding=3)
    for i, previous in enumerate(references):
        if previous:
            axes[0].scatter(i, previous['summary']['final']['success_rate']*100,
                            color='#111827', marker='x', s=65,
                            label='기존 A 선택본' if not any(references[:i]) else None)
    axes[0].set(xticks=x, xticklabels=[f'시드 {s["config"]["seed"]}' for s in summaries],
                ylim=(-2, 110), ylabel='완주 성공률 (%)', title=f'원래 시작점 독립 시험 · 모델별 {plan["test_count"]}판')
    axes[0].legend(fontsize=9)
    matrix = np.asarray([[s[key]['milestone_rates'][spec.id]*100 for spec in SPECS]
                         for s in summaries for key in ('final', 'latest_test')])
    labels = [f'{s["config"]["seed"]} · {label}' for s in summaries for label in ('선택본', '마지막')]
    im = axes[1].imshow(matrix, aspect='auto', vmin=0, vmax=100, cmap='Blues')
    for i in range(len(matrix)):
        for j in range(len(SPECS)):
            axes[1].text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center', fontsize=8,
                         color='white' if matrix[i,j] > 55 else '#334155')
    axes[1].set(yticks=range(len(labels)), yticklabels=labels,
                xticks=range(len(SPECS)), xticklabels=[f'{i+1}. {v}' for i, v in enumerate(LABELS)],
                title='원래 출발점 시험의 구간별 달성률 (%)')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    fig.colorbar(im, ax=axes[1], shrink=.7)
    fig.savefig(root/'test_results.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    counts, rates = [], []
    for name in GROUPS:
        rows = [r for r in all_records if r['start_kind'] == 'middle' and r['start_group'] == name]
        counts.append(len(rows))
        rates.append(sum(r['success'] for r in rows)/len(rows)*100 if rows else np.nan)
    ax.bar(range(len(GROUPS)), counts, color='#38bdf8', label='종료된 구간 연습 판 수')
    ax.set(xticks=range(len(GROUPS)), xticklabels=[v[0] for v in GROUPS.values()],
           ylabel='연습 판 수', title='중간 시작 구간별 연습량 · 전체 맵 성공률과 구분')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    right = ax.twinx()
    right.plot(range(len(GROUPS)), rates, color='#f97316', marker='o', label='해당 상태부터 이글루 도착률')
    right.set(ylabel='구간 연습에서 이글루 도착률 (%)', ylim=(-2, 105))
    ax.legend(loc='upper left', fontsize=9)
    right.legend(loc='upper right', fontsize=9)
    fig.savefig(root/'curriculum_usage.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    """저장 기록을 검사한 뒤 그래프를 만들며 모델을 추가 학습하지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='비교할 기존 A 결과 폴더. 생략하면 이번 결과만 표시')
    parser.add_argument('--live', action='store_true', help='학습 중 검증 곡선만 생성')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    plan = read(root/'plan.json')
    if plan.get('variant') != 'A-map1-curriculum':
        raise ValueError('커리큘럼 학습 결과 폴더를 지정해 주세요')
    data = []
    for seed in plan['seeds']:
        folder = root/f'seed_{seed}'
        if not (folder/'metrics.json').exists():
            continue
        history = read(folder/'metrics.json')
        previous = reference(args.baseline, seed, plan, folder)
        data.append((seed, history, episodes(folder), previous))
    if not data:
        raise RuntimeError('아직 완료된 검증 기록이 없습니다. 첫 검증 이후 다시 실행해 주세요')
    if args.live:
        print(learning_plot(root, plan, data, True))
        return
    completion = read(root/'completion.json')
    assert completion['complete'] and completion['models'] == len(plan['seeds']) == len(data)
    # 실행 당시 소스와 현재 소스가 다르면 같은 실험으로 오해하지 않도록 중단한다.
    assert plan['sources'] == sources() == completion['sources']
    summaries, references, records_all, checkpoint_hashes = [], [], [], {}
    for seed, history, records, previous in data:
        folder = root/f'seed_{seed}'
        summary = read(folder/'summary.json')
        assert summary['sources_after'] == summary['config']['sources'] == plan['sources']
        assert not summary['test_used_for_selection']
        assert plan['physics_budget'] <= summary['physics_ticks'] <= plan['physics_budget']+512*72
        assert sum(summary['frames_by_start'].values()) == summary['physics_ticks']
        assert summary['training_episodes'] == len(records)
        parts, by_start, wins = Counter(), Counter(), Counter()
        for row in records:
            audit_training_reward(row)
            assert row['start_kind'] in ('original', 'middle')
            assert row['segment_ticks'] >= 0 and row['segment_seconds'] >= 0
            assert abs(row['seconds']-row['start_world_time']-row['segment_seconds']) < 1e-8
            if row['start_kind'] == 'original':
                assert not row['initial_milestones'] and row['start_world_time'] == 0.
            parts.update(row['reward_parts'])
            by_start[row['start_kind']] += 1
            wins[row['start_kind']] += int(row['success'])
        assert dict(by_start) == summary['episodes_by_start']
        assert dict(wins) == summary['successes_by_start']
        for key in parts.keys() | summary['training_reward_parts'].keys():
            assert abs(parts[key]-summary['training_reward_parts'].get(key, 0.)) < 1e-6
        # 최종 시험이 아닌 검증 기록으로만 선택본을 골랐는지 다시 확인한다.
        best = max(history, key=lambda r: (selection_score(r['validation']), r['decisions']))
        assert summary['best_step'] == best['decisions']
        assert summary['best_validation']['evaluation'] == best['validation']
        for row in history:
            audit_evaluation(row['validation'], plan['validation_count'])
        for key in ('final', 'latest_test'):
            audit_evaluation(summary[key], plan['test_count'])
        for key in ('selected_greedy', 'latest_greedy'):
            audit_evaluation(summary[key], 1)
        initialization = read(folder/'initialization.json')
        assert initialization['steps'] == 0 and not initialization['previous_models_loaded']
        np.testing.assert_allclose(initialization['probabilities'], np.full(9, 1/9), rtol=1e-6)
        # 실제 저장 모델의 단계 수와 수치도 검사한다. 가중치는 읽기만 한다.
        for name, step in [('initial', 0), ('best', summary['best_step']), ('latest', summary['decisions'])]:
            path = folder/f'{name}.zip'
            model = load_policy(path)
            assert model.num_timesteps == step
            assert all(torch.isfinite(value).all() for value in model.policy.state_dict().values())
            if name == 'initial':
                assert parameter_hash(model) == initialization['parameter_sha256']
            checkpoint_hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        records_all.extend(records)
        summaries.append(summary)
        references.append(previous)
    for name, expected in plan['sources'].items():
        assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest() == expected
    learning_plot(root, plan, data, False)
    final_plots(root, plan, summaries, references, records_all)
    rows = []
    lines = ['# 맵1 · 시작 상태 커리큘럼 실험', '',
        '최종 성공률은 원래 출발점에서만 평가했다. 중간 시작의 이글루 도착은 구간 연습 성과다.', '',
        f'중간 시작 확률 {plan["middle_probability"]:.0%}, 마지막 원래 시작 전용 예산 {plan["final_original_fraction"]:.0%}.',
        '저장소가 비어 있으면 원래 시작한다. 비율 전환은 다음 에피소드부터 적용된다.',
        'PPO·관측·9행동·보상·학습률·할인율·엔트로피 일정은 기존 A와 같다.', '',
        '| 시드 | 기존 A 선택본 | 이번 선택본 | 이번 마지막본 | 선택본 평균 구간/16 | 원래 시작 학습 성공/판 | 구간 연습 성공/판 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for summary, previous in zip(summaries, references):
        seed = summary['config']['seed']
        baseline_rate = previous['summary']['final']['success_rate'] if previous else None
        row = dict(seed=seed, selected=summary['final'], latest=summary['latest_test'],
                   baseline_selected_rate=baseline_rate, episodes_by_start=summary['episodes_by_start'],
                   successes_by_start=summary['successes_by_start'], physics_ticks=summary['physics_ticks'])
        rows.append(row)
        old = f'{baseline_rate:.1%}' if baseline_rate is not None else '비교 없음'
        count, wins = summary['episodes_by_start'], summary['successes_by_start']
        lines.append(f'| {seed} | {old} | {summary["final"]["success_rate"]:.1%} | '
            f'{summary["latest_test"]["success_rate"]:.1%} | {summary["final"]["mean_milestones"]:.2f} | '
            f'{wins.get("original",0)}/{count.get("original",0)} | {wins.get("middle",0)}/{count.get("middle",0)} |')
    lines += ['', '## 해석과 한계', '',
        '- 위쪽 학습 곡선과 최종 시험은 모두 원래 시작점 기준이다. 구간 연습 성공률과 합치지 않는다.',
        '- 구간 연습이 성공해도 원래 시작 완주가 늘지 않았다면 전체 경로 연결은 아직 부족하다.',
        '- 이미 받은 보상은 기억만 복원하고 새로 지급하지 않는다. 연습 점수는 복원 이후 획득분이다.',
        '- 게임 시각과 남은 제한 시간은 저장 상태를 따른다. 학습량은 복원 후 새로 실행한 틱만 센다.',
        '- 초기 모델부터 다시 학습했다. 저장 상태는 같은 시드의 학습 환경이 직접 방문한 상태뿐이다.',
        '- 검증으로 모델을 선택했으며 최종 시험 결과로 선택하지 않았다.',
        '- states.pkl.gz는 검토용 상태 저장소다. 종료한 실행을 정확히 이어 학습하는 기능은 제공하지 않는다.',
        '- Git 작업과 게임 맵·물리 변경은 수행하지 않는다.', '',
        '![학습 지표](learning_metrics.png)', '', '![원래 시작 최종 시험](test_results.png)', '',
        '![구간 연습량](curriculum_usage.png)', '']
    (root/'결과보고서.md').write_text('\n'.join(lines), encoding='utf-8')
    write(root/'aggregate.json', dict(rows=rows, training_records_audited=len(records_all),
        selected_mean=float(np.mean([s['final']['success_rate'] for s in summaries])),
        latest_mean=float(np.mean([s['latest_test']['success_rate'] for s in summaries]))))
    analysis = {}
    for name in ('rl/map_a_curriculum_report.py', 'tests/test_map_a_curriculum.py'):
        target = root/'analysis_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        analysis[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    artifacts = ['plan.json', 'completion.json', 'aggregate.json', '결과보고서.md',
                 'learning_metrics.png', 'test_results.png', 'curriculum_usage.png']
    write(root/'final_review.json', dict(passed=True, models=len(summaries), sources=sources(),
        source_snapshots_verified=True, training_records_audited=len(records_all),
        original_start_evaluation_only=True, checkpoint_sha256=checkpoint_hashes, analysis_sources=analysis,
        artifact_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in artifacts}))
    print(root/'결과보고서.md')


if __name__ == '__main__':
    main()
