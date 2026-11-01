"""Graphs and independent native-physics audit for the five-tutorial experiment."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import json
import numpy as np
import torch
from .five_train import read, write, sources
from .five_env import FiveEnv
from .five_models import ALGORITHMS, action_probs
from pingu_ball.engine import World


def native_verify(payload):
    checkpoint, algorithm, stage, mode, expected = payload
    torch.set_num_threads(1)
    model = ALGORITHMS[algorithm].load(checkpoint, device='cpu')
    model.policy.set_training_mode(False)
    env = FiveEnv(stage=stage, reward_mode=mode)
    raw = env.unwrapped
    native = World(raw.level, auto_reset=False)
    obs, _ = env.reset()
    error = 0.
    actions = []
    while not raw.done:
        action = int(action_probs(model, obs[None])[0].argmax())
        actions.append(action)
        obs, _, _, _, info = env.step(action)
        for _ in range(raw.repeat):
            dt = min(native.physics.dt, raw.limit-native.time)
            if dt < 1e-10 or native.failed or native.cleared:
                break
            native.step(action-1, dt)
        for name in ('x', 'y', 'vx', 'vy'):
            delta = abs(getattr(native.ball, name)-getattr(raw.world.ball, name))
            error = max(error, delta)
            assert delta < 1e-8
        assert native.failed == raw.world.failed and native.cleared == raw.world.cleared
        assert native.time == raw.world.time
        assert native.last_landing == raw.world.last_landing
        for key, p in native.special.platforms.items():
            assert p.destroyed == raw.world.special.platforms[key].destroyed
    assert native.cleared == expected['success'] and abs(native.time-expected['seconds']) < 1e-8
    env.close()
    return dict(checkpoint=str(Path(checkpoint).resolve()), stage=stage, success=native.cleared,
                checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                seconds=native.time, max_ball_error=error, decisions=len(actions),
                action_changes=sum(a != b for a, b in zip(actions, actions[1:])),
                live_feedback=True, native_physics=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.root
    plan = read(root/'plan.json')
    assert plan['sources'] == sources()
    rows, tasks = [], []
    summaries = list((root/'experts').rglob('summary.json'))
    expected_experts = len(plan['seeds'])*10
    for path in summaries:
        r = read(path)
        c = r['config']
        for m in r['final']['maps']:
            rows.append(dict(phase='expert', algorithm=c['algorithm'], seed=c['seed'], stage=m['stage'],
                             success_rate=m['success_rate'], successes=m['successes'], episodes=m['episodes'],
                             greedy=int(m['greedy']['success']), decisions=r['decisions'],
                             checkpoint=str((path.parent/'best.zip').resolve())))
            tasks.append((str(path.parent/'best.zip'), c['algorithm'], m['stage'], c['reward_mode'], m['greedy']))
    completions = []
    for seed in plan['seeds']:
        folder = root/'consolidated'/f'seed_{seed}'
        if not (folder/'completion.json').exists():
            continue
        completion = read(folder/'completion.json')
        completions.append(dict(seed=seed, **completion))
        if completion['status'] != 'completed':
            continue
        selection = read(folder/'teachers.json')
        for s, teacher in selection.items():
            training_seed = teacher.get('training_seed', seed)
            expert = next(r for r in rows if r['phase']=='expert' and r['seed']==training_seed
                          and r['stage']==int(s) and r['algorithm']==teacher['algorithm'])
            rows.append(dict(expert, phase='teacher', algorithm='selected', seed=seed))
        for phase, path, checkpoint in [('distilled', folder/'distilled_test.json', folder/'distilled.zip'),
                                         ('joint', folder/'summary.json', folder/'best.zip')]:
            r = read(path)
            for m in r['final']['maps']:
                rows.append(dict(phase=phase, algorithm='ppo', seed=seed, stage=m['stage'],
                                 success_rate=m['success_rate'], successes=m['successes'], episodes=m['episodes'],
                                 greedy=int(m['greedy']['success']), decisions=r.get('decisions', 0),
                                 checkpoint=str(checkpoint.resolve())))
                tasks.append((str(checkpoint), 'ppo', m['stage'], plan['reward_mode'], m['greedy']))
    if not rows:
        raise RuntimeError('No finished models to report')
    with (root/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    for phase, algorithm in [('expert', 'a2c'), ('expert', 'ppo'), ('teacher', 'selected'), ('distilled', 'ppo'), ('joint', 'ppo')]:
        for stage in range(1, 6):
            group = [r for r in rows if (r['phase'], r['algorithm'], r['stage']) == (phase, algorithm, stage)]
            if phase=='teacher' and plan.get('pooled_teachers'):
                group = list({r['checkpoint']:r for r in group}.values())
            if group:
                rates = [r['success_rate'] for r in group]
                aggregate.append(dict(phase=phase, algorithm=algorithm, stage=stage,
                    mean=float(np.mean(rates)), seed_sd=float(np.std(rates, ddof=1)) if len(rates)>1 else None,
                    seeds=len(group), greedy_models=sum(r['greedy'] for r in group),
                    successes=sum(r['successes'] for r in group), episodes=sum(r['episodes'] for r in group)))
    write(root/'aggregate.json', dict(groups=aggregate, completions=completions, source_sha256=sources()))
    if args.verify:
        results, pending = [], []
        cache = {}
        cache_paths = [root/'native_verification.json']
        if plan.get('expert_models_reused_from'):
            cache_paths.append(Path(plan['expert_models_reused_from'])/'native_verification.json')
        for path in cache_paths:
            if not path.exists():
                continue
            old = read(path)
            if old.get('sources') != sources():
                continue
            for row in old.get('rows', []):
                if row.get('checkpoint_sha256'):
                    cache[(row['checkpoint_sha256'], row['stage'])] = row
        for task in tasks:
            checkpoint, algorithm, stage, mode, expected = task
            key = (hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(), stage)
            old = cache.get(key)
            if old and old['success']==expected['success'] and abs(old['seconds']-expected['seconds'])<1e-8:
                results.append(dict(old, checkpoint=str(Path(checkpoint).resolve()),
                                    verification_reused_from=old['checkpoint']))
            else:
                pending.append(task)
        reused = len(results)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(native_verify, t) for t in pending]):
                results.append(future.result())
                write(root/'native_verification.json', dict(complete=False, rows=results))
        write(root/'native_verification.json', dict(complete=True, rows=results,
            max_ball_error=max(r['max_ball_error'] for r in results), sources=sources(),
            reused_verifications=reused, new_verifications=len(pending)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#f8fafc', 'axes.facecolor': '#f8fafc'})
    colors = {'a2c': '#2563eb', 'ppo': '#ea580c', 'distilled': '#8b5cf6', 'joint': '#059669'}
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    for j, algorithm in enumerate(('a2c', 'ppo')):
        group = [r for r in aggregate if r['phase']=='expert' and r['algorithm']==algorithm]
        ax.bar(np.array([r['stage'] for r in group])+(j-.5)*.32,
               [100*r['mean'] for r in group], width=.30,
               color=colors[algorithm], label=algorithm.upper())
        for stage in range(1, 6):
            rates = [100*r['success_rate'] for r in rows if r['phase']=='expert'
                     and r['algorithm']==algorithm and r['stage']==stage]
            if rates:
                offsets = np.linspace(-.045, .045, len(rates)) if len(rates)>1 else [0.]
                ax.scatter(stage+(j-.5)*.32+np.asarray(offsets), rates, s=32,
                           facecolors='white', edgecolors='#0f172a', linewidths=1., zorder=4)
    ax.set(xticks=range(1, 6), xlabel='Tutorial stage', ylabel='Sampled-policy success (%)',
           ylim=(-3, 105), yticks=range(0, 101, 20), title='Individual policies: A2C vs PPO')
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, .5))
    fig.text(.5, -.02, 'Validation-selected checkpoints. Bars: mean; dots: training seeds. Held-out action RNGs.', ha='center', fontsize=9)
    fig.savefig(root/'expert_comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for stage, ax in zip(range(1, 6), axes.flat):
        for algorithm in ('a2c', 'ppo'):
            curves = []
            for seed in plan['seeds']:
                p = root/'experts'/algorithm/f'seed_{seed}'/f'stage_{stage:02d}'/'metrics.json'
                if p.exists():
                    curves.append(read(p))
            common = sorted(set.intersection(*[set(r['decisions'] for r in curve) for curve in curves])) if curves else []
            if common:
                values = np.array([[next(r['validation']['mean_success_rate'] for r in curve if r['decisions']==x)
                                    for x in common] for curve in curves])*100
                mean = values.mean(0)
                sd = values.std(0, ddof=1) if len(curves)>1 else np.zeros_like(mean)
                x = np.array(common)/1000
                ax.plot(x, mean, color=colors[algorithm], label=algorithm.upper())
                ax.fill_between(x, np.clip(mean-sd, 0, 100), np.clip(mean+sd, 0, 100), color=colors[algorithm], alpha=.15)
        ax.set(title=f'Tutorial {stage}', ylim=(-2, 102), xlabel='Action decisions (thousands)', ylabel='Validation success (%)')
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, .9, 'Original spawn, native game physics\n3 controls, no route/expert demonstrations\n20 validation episodes / checkpoint\nOne model per map and training seed\nShading: training-seed standard deviation', va='top', linespacing=1.8)
    fig.savefig(root/'learning_curves.png', dpi=150)
    plt.close(fig)
    if any(r['phase']=='joint' for r in rows):
        fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
        for j, phase in enumerate(('teacher', 'distilled', 'joint')):
            means, values = [], []
            for s in range(1, 6):
                group = [r for r in rows if r['phase']==phase and r['stage']==s]
                if phase=='teacher' and plan.get('pooled_teachers'):
                    group = list({r['checkpoint']:r for r in group}.values())
                rates = [r['success_rate'] for r in group]
                means.append(np.mean(rates)*100 if rates else np.nan)
                values.append(rates)
            ax.bar(np.arange(1, 6)+(j-1)*.25, means, width=.23,
                   label={'teacher':'Selected expert', 'distilled':'Distilled student (selected)', 'joint':'Joint PPO/KL (selected)'}[phase],
                   color={'teacher':'#64748b', 'distilled':colors['distilled'], 'joint':colors['joint']}[phase])
            for s, rates in enumerate(values, 1):
                if rates:
                    offsets = np.linspace(-.035, .035, len(rates)) if len(rates)>1 else [0.]
                    ax.scatter(s+(j-1)*.25+np.asarray(offsets), np.array(rates)*100,
                               facecolors='white', edgecolors='#0f172a', s=28, linewidths=1., zorder=4)
        ax.set(xticks=range(1, 6), xlabel='Tutorial stage', ylabel='Sampled-policy success (%)', ylim=(-3, 105), yticks=range(0,101,20),
               title='Diagnostic consolidation (weak teachers allowed)' if plan.get('diagnostic') else 'Consolidation: individual teachers to one shared policy')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3, fontsize=9)
        ax.text(.5, -.24, 'Bars: mean; dots: individual policies. Pooled students share one selected teacher per map.'
                if plan.get('pooled_teachers') else 'Bars: mean; dots: individual training seeds.',
                transform=ax.transAxes, ha='center', va='top', fontsize=9)
        ax.grid(axis='y', alpha=.2)
        fig.savefig(root/'consolidation_comparison.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
        curves = []
        for seed in plan['seeds']:
            path = root/'consolidated'/f'seed_{seed}'/'metrics.json'
            if path.exists():
                curves.append(read(path))
        common = sorted(set.intersection(*[set(r['decisions'] for r in c) for c in curves])) if curves else []
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
        for stage, ax in zip(range(1,6), axes.flat):
            values = np.array([[next(m['success_rate'] for m in next(r for r in c if r['decisions']==x)['validation']['maps']
                                    if m['stage']==stage) for x in common] for c in curves])*100
            if common:
                mean = values.mean(0)
                sd = values.std(0, ddof=1) if len(curves)>1 else np.zeros_like(mean)
                x = np.array(common)/1000
                ax.plot(x, mean, color=colors['joint'], marker='o', markersize=4)
                ax.fill_between(x, np.clip(mean-sd, 0, 100), np.clip(mean+sd, 0, 100), color=colors['joint'], alpha=.15)
            ax.set(title=f'Tutorial {stage}', xlabel='Joint decisions, all maps (thousands)',
                   ylabel='Validation success (%)', ylim=(-2,102))
            ax.grid(alpha=.2)
        axes.flat[-1].axis('off')
        axes.flat[-1].text(0,.9, f'Starts AFTER policy distillation\nPPO + teacher-KL auxiliary update\n{len(curves)} student training seeds\nEach map: 20% of joint transitions\nShading: seed SD, not a confidence interval\nFinal held-out tests: separate bar chart',
                           va='top', linespacing=1.8)
        fig.suptitle('Shared-policy fine-tuning from the distilled initialization', fontsize=15)
        fig.savefig(root/'joint_learning_curves.png', dpi=150)
        plt.close(fig)
    lines = ['# 튜토리얼 1~5 개별 학습 및 정책 통합', '',
             f"- 개별 학습 완료: {len(summaries)}/{expected_experts}개. 알고리즘별·맵별 초기 난수 {len(plan['seeds'])}개.",
             f"- 개별 학습량: 모델당 {plan['expert_steps']:,} 행동 선택. 공동 학습량: 통합 모델당 총 {plan['joint_steps']:,} 행동 선택.",
             '- 범위: 튜토리얼 1~5만. 기존 모델·사람의 정답 행동·작성한 경로를 개별 학습에 넣지 않았다.',
             '- 행동: 왼쪽 / 방향 입력 없음 / 오른쪽. 게임 물리와 맵 배치를 수정하지 않았다.',
             '- 주 평가: 학습한 확률대로 행동 선택. 모델당 맵별 독립 행동 난수로 최종 시험. 잡음 없는 탐욕 실행은 별도 1회.',
             ('- 저장본은 검증 20판, 전체 후보의 교사는 별도 검증 50판으로 선택한다. 최종 시험은 선택에 사용하지 않는다.'
              if plan.get('pooled_teachers') else '- 검증 20판으로 저장본과 교사를 선택하며 최종 시험은 선택에 사용하지 않는다.'),
             '- 공동 학습의 검증 후보에는 증류 직후(0전이) 정책도 포함한다. 표의 공동 학습 결과는 검증 선택 저장본이며 마지막 가중치와 다를 수 있다.',
             '- A2C와 PPO의 네트워크·보상·입력·행동·환경 상호작용 예산은 같고 알고리즘별 업데이트 구조는 다르다.',
             '- 보상: 클리어 +1 / 사망·40초 제한 -1. contact 모드만 최초 유효 발판 착지 +.01, 판당 총 .1 상한. 시작 발판 제외.',
             '- 접촉 보상용 발판은 빈틈을 메우지 않고 병합한 정적 직사각형 또는 개별 특수 발판 단위다. 기록은 관측에도 포함한다.',
             '- 증류는 스스로 강화학습한 교사의 확률을 모방하는 별도 지도학습이다. 처음부터 독립적으로 학습한 PG와 구분한다.',
             '- 통합 정책은 맵 ID를 포함한 상태로 판단한다. 새로운 맵으로의 일반화는 시험하지 않았다.', '',
             '| 단계 | 알고리즘 | 맵 | 평균 성공률 | 시드 표준편차 | 탐욕 클리어 모델 |',
             '|---|---|---:|---:|---:|---:|']
    for r in aggregate:
        sd = f"{100*r['seed_sd']:.1f}%p" if r['seed_sd'] is not None else '반복 1개'
        phase_name = {'expert':'개별 강화학습', 'teacher':'선택된 교사',
                      'distilled':'증류 후 검증 선택', 'joint':'공동 학습 후 검증 선택'}[r['phase']]
        algorithm_name = '맵별 선택' if r['algorithm']=='selected' else r['algorithm'].upper()
        lines.append(f"| {phase_name} | {algorithm_name} | {r['stage']} | {100*r['mean']:.1f}% | {sd} | {r['greedy_models']}/{r['seeds']} |")
    lines += ['', '## 정책 통합 상태', '']
    for r in completions:
        lines.append(f"- 시드 {r['seed']}: {r['status']}; 기준 미달 맵 {r.get('failed_stages', r.get('weak_teacher_stages', []))}.")
    if plan.get('diagnostic'):
        lines += ['', '**이 통합은 품질 기준 미달 교사를 포함한 추가 진단 실험이다. 검증된 전문가 5개를 통합한 결과로 해석하면 안 된다.**',
                  f"- 개별 모델은 `{plan['expert_models_reused_from']}` 결과를 그대로 재사용했다. 개별 학습을 새로 30회 한 것으로 세지 않는다.",
                  '- 증류와 공동 PPO만 새로 실행했다. 기준 미달 맵을 제외하거나 성공 결과만 골라 평균내지 않았다.']
    if plan.get('pooled_teachers'):
        lines += ['', '## 전체 후보 중 교사 선택', '',
                  '- 이 결과는 개별 모델 30개를 재사용한 별도의 정책 통합 실험이다. 새 개별 학습 30회로 세지 않는다.',
                  '- 맵마다 A2C/PPO × 학습 난수 3개의 후보를 새 검증 난수로 50판씩 평가해 선택했다. 기존 최종 시험은 선택 코드가 읽지 않는다.',
                  '- 학생 난수 3개는 같은 교사 집합을 공유한다. 학생 편차는 교사 학습 난수의 편차를 포함하지 않는다.',
                  '- 배포 후보 학생도 새 검증 난수 4510000으로 맵별 50판 평가해 선택한다. 학생의 최종 100판 시험은 선택에 사용하지 않는다.',
                  '- teacher 행은 맵마다 선택된 교사 1개 결과다. expert 행은 모든 독립 개별 모델의 평균이다.',
                  '- 학습·증류·공동 학습 알고리즘은 동일하며 교사 선택 진입점만 five_pool.py에서 명시적으로 대체했다.']
        if plan.get('weak_teacher_stages') and plan.get('allow_weak_teachers'):
            lines += ['- 품질 기준에 미달한 교사도 포함하도록 한 진단 실행이다. 검증된 전문가 5개를 확보했다고 해석하지 않는다.']
    if completions and all(r['status']!='completed' for r in completions):
        lines += ['', '**개별 교사 품질 기준에 미달해 증류·공동 학습을 실행하지 않았다. 실행하지 않은 통합 성능을 0%로 표시하지 않는다.**']
    completed_students = [r for r in completions if r['status']=='completed']
    if completed_students:
        lines += ['', '## 공동 학습의 저장본 선택', '',
                  '선택 전이 0은 증류 직후 정책이 선택됐다는 뜻이다. 마지막 검증은 맵별 20판이며, 위 표의 별도 최종 100판 시험과 구분한다.', '',
                  '| 학생 난수 | 공동 학습 전이 | 선택 전이 | 마지막 검증 평균 | 마지막 검증 최저 맵 |',
                  '|---:|---:|---:|---:|---:|']
        for row in completed_students:
            folder = root/'consolidated'/f"seed_{row['seed']}"
            summary = read(folder/'summary.json')
            latest = read(folder/'metrics.json')[-1]['validation']
            lines.append(f"| {row['seed']} | {summary['decisions']:,} | {summary['best_step']:,} | {100*latest['mean_success_rate']:.1f}% | {100*latest['min_success_rate']:.1f}% |")
    lines += ['', '## 그래프', '', '![개별 정책](expert_comparison.png)', '', '![학습 곡선](learning_curves.png)']
    if (root/'consolidation_comparison.png').exists():
        lines += ['', '![정책 통합](consolidation_comparison.png)']
        lines += ['', '아래 곡선은 증류된 정책에서 시작하는 공동 학습이다. 처음부터 무작위 정책으로 학습한 곡선과 구분한다.',
                  '', '![공동 학습 곡선](joint_learning_curves.png)']
    (root/'결과보고서.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(experts=len(summaries), groups=aggregate, completions=completions), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
