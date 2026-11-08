"""통합 보상의 학습 곡선·원래 보상 비교·항목별 감사 결과를 생성한다."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from pathlib import Path
import numpy as np
from .combined_train import install, sources, read, write
from .combined_env import audit_reward


def verify(payload):
    install()
    from . import five_report
    from .combined_env import CombinedEnv
    from .milestone_models import ALGORITHMS
    five_report.FiveEnv, five_report.ALGORITHMS = CombinedEnv, ALGORITHMS
    return five_report.native_verify(payload)


def review_latest(payload):
    """선택에는 쓰지 않는 마지막 가중치 시험으로 학습 후 성능 저하를 구분한다."""
    import torch
    from .milestone_models import ALGORITHMS
    from . import five_train
    from .combined_env import CombinedEnv
    from .five_models import action_probs
    install()
    torch.set_num_threads(1)
    folder, stage, seed, count = payload
    folder = Path(folder)
    digest = hashlib.sha256((folder/'latest.zip').read_bytes()).hexdigest()
    cache = folder/'latest_review.json'
    if cache.exists():
        result=read(cache)
        if (result.get('review_version')==2 and result['checkpoint_sha256']==digest
                and result['sources']==sources()):
            return result
    latest=ALGORITHMS['ppo'].load(folder/'latest.zip',device='cpu')
    best=ALGORITHMS['ppo'].load(folder/'best.zip',device='cpu')
    latest.policy.set_training_mode(False)
    env=CombinedEnv(stage=stage)
    initial=action_probs(latest,env.reset()[0][None])[0]
    initial_probabilities=initial.tolist()
    initial_entropy=float(-(initial*np.log(np.maximum(initial,1e-12))).sum())
    env.close()
    same=all(torch.equal(v,best.policy.state_dict()[k]) for k,v in latest.policy.state_dict().items())
    evaluation=(read(folder/'summary.json')['final'] if same else
                five_train.evaluate(latest,[stage],count,910000+seed,'combined'))
    for row in evaluation['maps']:
        for episode in row['raw']:
            audit_reward(episode)
    result=dict(review_version=2,checkpoint_sha256=digest,stage=stage,seed=seed,matches_selected_weights=same,
                initial_action_probabilities=initial_probabilities,initial_action_entropy=initial_entropy,
                evaluation=evaluation,used_for_selection=False,sources=sources())
    write(cache,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('experiments/tutorial_1_5/baseline/results'))
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = read(root / 'plan.json')
    assert plan['sources'] == sources(), '학습한 소스와 현재 코드가 다르다'
    paths = sorted((root / 'experts').rglob('summary.json'))
    with ProcessPoolExecutor(max_workers=min(5,max(1,len(paths)))) as pool:
        latest_rows=list(pool.map(review_latest,[(str(p.parent),read(p)['config']['stage'],
                    read(p)['config']['seed'],plan['test_count']) for p in paths]))
    latest_lookup={(r['stage'],r['seed']):r for r in latest_rows}
    rows, tasks = [], []
    for path in paths:
        summary = read(path)
        config = summary['config']
        for result in summary['final']['maps']:
            def failure_buckets(episodes):
                buckets=Counter()
                if result['stage']==4:
                    for e in episodes:
                        a=e['auxiliary_state']
                        label=('성공' if e['success'] else '리프트 착지 전 실패' if not a['lift_first']
                               else '리프트→윗층 실패' if not a['upper_first'] else '윗층→이글루 실패')
                        buckets[label]+=1
                return dict(buckets)
            parts = Counter()
            for episode in result['raw']:
                audit_reward(episode)
                parts.update(episode['reward_parts'])
            parts = {k: v / result['episodes'] for k,v in parts.items()}
            baseline_path = args.baseline / 'experts' / 'ppo' / f'seed_{config["seed"]}' / f'stage_{result["stage"]:02d}' / 'summary.json'
            previous = read(baseline_path)['final']['maps'][0] if baseline_path.exists() else None
            rows.append(dict(stage=result['stage'], seed=config['seed'],
                successes=result['successes'], episodes=result['episodes'], success_rate=result['success_rate'],
                baseline_success_rate=previous['success_rate'] if previous else None,
                mean_points=result['mean_return'],
                mean_common_points=sum(parts.get(k,0.) for k in ('success','failure','milestone')),
                mean_auxiliary_points=sum(v for k,v in parts.items() if k not in ('success','failure','milestone')),
                baseline_mean_common_points=previous['mean_return'] if previous else None,
                mean_success_seconds=result['mean_success_seconds'],
                last_success_rate=latest_lookup[(result['stage'],config['seed'])]['evaluation']['mean_success_rate'],
                last_initial_probabilities=latest_lookup[(result['stage'],config['seed'])].get('initial_action_probabilities'),
                best_step=summary['best_step'],
                mean_milestones=float(np.mean([len(e['milestones']) for e in result['raw']])),
                source_landing_rate=float(np.mean([bool(e['armed']) for e in result['raw']])),
                milestone_rates={spec:float(np.mean([spec in e['milestones'] for e in result['raw']]))
                    for spec in sorted(set(s for e in result['raw'] for s in e['armed']))},
                auxiliary_rates={k:float(np.mean([e['reward_parts'].get(k,0.)>0 for e in result['raw']]))
                    for k in ('spring_height','spring_horizontal','lift_first','lift_high','upper_first','approach')},
                failures=result['failures'],
                failure_buckets=failure_buckets(result['raw']),
                last_failure_buckets=failure_buckets(latest_lookup[(result['stage'],config['seed'])]['evaluation']['maps'][0]['raw']),
                greedy_success=result['greedy']['success'],
                mean_reward_parts=parts, decisions=summary['decisions'],
                training_episodes=summary['episodes'], training_successes=summary['training_successes'],
                checkpoint=str(path.parent / 'best.zip')))
            tasks.append((str(path.parent/'best.zip'),'ppo',result['stage'],'combined',result['greedy']))
    expected = len(plan['seeds']) * 5
    write(root / 'aggregate.json', dict(rows=rows, expected_models=expected, completed_models=len(rows),
          all_complete=len(rows)==expected, sources=sources(), historical_baseline=str(args.baseline)))
    if not rows:
        raise RuntimeError('아직 완료된 모델이 없다')
    fields = [k for k in rows[0] if k not in ('mean_reward_parts','milestone_rates','auxiliary_rates','failures','last_initial_probabilities','failure_buckets','last_failure_buckets')]
    with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    if args.verify:
        audit, pending, cache = [], [], {}
        cache_path=root/'native_verification.json'
        if cache_path.exists():
            previous=read(cache_path)
            if previous.get('sources')==sources():
                cache={(r['checkpoint_sha256'],r['stage']):r for r in previous['rows']}
        for task in tasks:
            key=(hashlib.sha256(Path(task[0]).read_bytes()).hexdigest(),task[2])
            if key in cache:
                audit.append(cache[key])
            else:
                pending.append(task)
        if pending:
            with ProcessPoolExecutor(max_workers=min(5,len(pending))) as pool:
                audit.extend(pool.map(verify,pending))
        write(root/'native_verification.json',dict(passed=True,rows=audit,
              max_ball_error=max(r['max_ball_error'] for r in audit), sources=sources()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
    fig, axes = plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for stage,ax in enumerate(axes.flat[:5],1):
        for seed in plan['seeds']:
            for directory,label,color in [(args.baseline,'기존 보상','#94a3b8'),(root,'통합 보상','#0284c7')]:
                path=directory/'experts'/'ppo'/f'seed_{seed}'/f'stage_{stage:02d}'/'metrics.json'
                if path.exists():
                    data=read(path)
                    ax.plot([r['decisions']/1000 for r in data],
                            [r['validation']['mean_success_rate']*100 for r in data],
                            label=f'{label} · 시드 {seed}',color=color,marker='o',markersize=3)
        ax.set(title=f'튜토리얼 {stage}',xlabel='누적 행동 선택 (천 회)',ylabel='검증 20판 성공률 (%)',ylim=(-2,102))
        ax.grid(alpha=.2)
        if stage==1 and ax.lines: ax.legend(fontsize=9)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,.95,'모든 정책은 무작위 초기화에서 출발\nPPO · 행동 0.05초 · 관측 1,664개\nγ=1.0 · 학습률 0.0003 · 제한 180초\n기존 핵심 구간 +10 유지\n2·4번에만 보조 보상 추가\n확률 분포에서 행동을 추출해 평가\n한 학습 시드의 탐색적 비교',va='top',linespacing=1.8)
    fig.suptitle('튜토리얼 1~5: 보상 통합 후 처음부터 재학습')
    fig.savefig(root/'learning_curves.png',dpi=160);plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    stages=sorted(set(r['stage'] for r in rows))
    for offset,field,label,color in [(-.25,'baseline_success_rate','기존 선택본','#94a3b8'),
                                   (0.,'success_rate','통합 선택본','#0284c7'),
                                   (.25,'last_success_rate','통합 마지막','#f59e0b')]:
        values=[np.mean([r[field] for r in rows if r['stage']==s and r[field] is not None])*100 for s in stages]
        bars=axes[0].bar(np.asarray(stages)+offset,values,.23,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.0f',padding=3,fontsize=9)
    for offset,field,label,color in [(-.18,'baseline_mean_common_points','기존','#94a3b8'),(.18,'mean_common_points','통합','#0284c7')]:
        values=[np.mean([r[field] for r in rows if r['stage']==s and r[field] is not None]) for s in stages]
        axes[1].bar(np.asarray(stages)+offset,values,.33,label=label,color=color)
    axes[0].set(title=f'독립 시험 {plan["test_count"]}판: 선택본·마지막 정책',ylabel='성공률 (%)',ylim=(0,115))
    axes[0].legend(fontsize=9,ncol=3,loc='upper center',frameon=False)
    axes[1].set(title='검증 선택본의 공통 평균 점수',ylabel='성공 +100 / 실패 -25 / 핵심 구간 +10')
    axes[1].axhline(0,color='#94a3b8',lw=1)
    for ax in axes:
        ax.set_xticks(range(1,6)); ax.set_xlabel('튜토리얼 스테이지'); ax.grid(axis='y',alpha=.2)
    fig.suptitle('보조 점수 증가와 실제 클리어 향상을 구분하여 비교')
    fig.savefig(root/'success_comparison.png',dpi=160);plt.close(fig)
    keys=['milestone','spring_height','spring_horizontal','approach','lift_first','lift_high','upper_first']
    labels=['핵심 구간','목표 높이','목표 가로 구간','리프트 접근','리프트 첫 착지','높은 위치 이용','윗층 첫 착지']
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    bottom=np.zeros(len(stages))
    for key,label in zip(keys,labels):
        values=np.array([np.mean([r['mean_reward_parts'].get(key,0.) for r in rows if r['stage']==s]) for s in stages])
        ax.bar(stages,values,bottom=bottom,label=label)
        bottom+=values
    ax.set(xticks=range(1,6),xlabel='튜토리얼 스테이지',ylabel='한 판당 평균 점수',title='독립 시험의 핵심 구간·보조 보상 구성 (종료 점수 제외)')
    ax.legend(fontsize=9,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.15),frameon=False)
    ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'reward_components.png',dpi=160);plt.close(fig)
    lines=['# 튜토리얼 1~5 통합 보상 실험','',
        f'- 완료 모델: {len(rows)}/{expected}. 기존 가중치를 읽지 않고 각 맵을 처음부터 PPO 학습.',
        '- 공통: 성공 +100, 실패/180초 제한 -25 한 번, 기존 시작 특수 발판→목표 발판 최초 착지 +10.',
        '- 2번: 점프 발판 반동 후 3초 안에 목표 높이 ±16px 첫 진입 +0.5, 가로 범위 첫 진입 +0.5. 발판별 한 번.',
        '- 2번의 기존 핵심 구간 +10에는 3초 제한을 추가하지 않았다. tutorial_1_5의 기존 규칙을 기준으로 유지한 통합이다.',
        '- 4번: 리프트 첫 착지 +12, 높은 위치에서 이용 +12, 윗층 첫 착지 +18, 첫 전진 거리 1px당 +0.045 (최대 12.96).',
        '- 1·3·5번 보상은 기존과 완전히 동일. 실패 틱에는 새로운 보조·착지 보상을 지급하지 않는다.',
        '- 원점수를 기록하고 학습기에만 /100. 2번 성공 최대 133, 4번 성공 최대 164.96.',
        '- 학습 변수: 4개 환경, 128스텝 수집, 미니배치 256, 최대 5회 갱신, 학습률 0.0003, γ=1, GAE λ=0.98, clip=0.2, target KL=0.02.',
        '- 행동 3개(좌/무입력/우), 물리 120Hz, 행동 유지 6틱=0.05초, 관측 1,664개, 기존 특징 추출기와 신경망 유지.',
        '- 엔트로피: 0~20%까지 0.02, 80%까지 0.005로 감소, 종료까지 0.001로 감소.',
        '- 매 4만 행동마다 독립 행동 난수로 20판 검증. 검증으로 선택한 저장본을 별도 난수 100판 시험. 최종 시험으로 저장본을 고르지 않는다.',
        '- 학습 마지막 가중치도 같은 별도 시험 난수로 평가한다. 최선 저장본과 가중치가 같으면 같은 시험 결과를 재사용한다. 마지막 가중치 시험도 선택에는 쓰지 않는다.',
        '- 주 결과는 정책 확률에서 100% 추출한 행동이다. Greedy 재생과 Greedy+5% 무작위 입력은 별도 진단으로만 저장한다.',
        '- 시험 100판은 원래 시작점에서 행동 추출 난수를 바꾼 반복이다. 시작 좌표·맵·물리를 무작위로 바꾼 일반화 시험은 아니다.',
        '- 추가 보상의 지급 플래그와 점프 후 경과 시간은 진단 로그에 기록하며 정책 입력에는 추가하지 않았다. 관측을 고정한 비교이므로 보상 추적 상태 일부는 정책에서 직접 볼 수 없는 한계가 있다.',
        '- 기존 비교 자료는 보관된 baseline의 시드 0 PPO 결과다. 한 학습 시드로 일반적인 성능 향상이나 통계적 우월성을 확정하지 않는다.',
        '- tutorial_2_4의 원래 실행은 맵당 100만 행동이고 행동당 12/30/72틱을 진행했다. 이번 실행은 약 40만 행동·행동당 6틱이다. 원래 2·4번 전용 실험과는 물리 경험량도 다르므로 성공률을 보상만의 효과로 직접 비교하지 않는다.',
        '- 맵별 정책을 각각 학습하는 실험이며 정책 증류나 5개 맵 공통 정책은 이번에 학습하지 않는다.',
        '- 3번은 핵심 구간을 밟지 않고 성공하는 경로도 기존 맵에 있다. 성공률과 핵심 구간 달성률을 같은 뜻으로 해석하지 않는다.',
        '- 특히 4번은 보조 보상을 많이 받은 뒤 실패하면 누적 점수가 양수일 수 있다. 점수 상승만으로 학습 성공을 판단하지 않는다.',
        '- Git 커밋·푸시는 하지 않는다. 기존 보관 실험은 수정하지 않는다.','',
        '| 맵 | 기존 성공률 | 통합 최선 성공/시험 | 통합 최선 성공률 | 마지막 성공률 | 공통 점수 | 보조 점수 | 핵심 구간 수 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        oldrate=f'{r["baseline_success_rate"]*100:.0f}%' if r['baseline_success_rate'] is not None else '없음'
        lines.append(f'| {r["stage"]} | {oldrate} | {r["successes"]}/{r["episodes"]} | {r["success_rate"]*100:.0f}% | {r["last_success_rate"]*100:.0f}% | {r["mean_common_points"]:.2f} | {r["mean_auxiliary_points"]:.2f} | {r["mean_milestones"]:.2f} |')
    lines += ['', '## 구간별 검토', '']
    for r in rows:
        rates=', '.join(f'{k}: {v*100:.0f}%' for k,v in r['milestone_rates'].items()) or '핵심 구간 시작 착지 없음'
        extras=', '.join(f'{k}: {v*100:.0f}%' for k,v in r['auxiliary_rates'].items() if v) or '보조 보상 획득 없음'
        lines += [f'- {r["stage"]}번 핵심 구간 달성률: {rates}.', f'  보조 보상 획득률: {extras}. 실패 사유: {r["failures"]}.']
        if r.get('last_initial_probabilities'):
            p=r['last_initial_probabilities']
            lines += [f'  마지막 정책 시작점 행동 확률: 왼쪽 {p[0]*100:.4f}%, 무입력 {p[1]*100:.4f}%, 오른쪽 {p[2]*100:.4f}%.']
        if r['failure_buckets']:
            lines += [f'  검증 선택본 구간별 결과: {r["failure_buckets"]}.',
                      f'  마지막 가중치 구간별 결과: {r["last_failure_buckets"]}.']
    lines += ['', '## 결과 해석', '']
    for r in rows:
        if r['stage']==2 and r['success_rate']==0 and r['source_landing_rate']==0:
            lines += ['- 2번: 시험에서 첫 점프 발판에도 착지하지 못했다. 새 보조 보상은 발판 반동 후에만 발생하므로, 보상 구간에 들어가기 전의 탐색 실패를 해결하지 못했다.']
        if r['success_rate']>r['last_success_rate']+.1:
            lines += [f'- {r["stage"]}번: 검증 선택본 {r["success_rate"]*100:.0f}%와 마지막 가중치 {r["last_success_rate"]*100:.0f}%를 구분해야 한다. 성공 경로를 발견했어도 후속 갱신에서 성능이 유지되지 않았다.']
    lines += ['- 1·3·5번은 보상·행동·관측을 바꾸지 않은 대조군이다. 기존 기록과 같은 학습 시드·같은 초기화로 재학습했다.',
              '- 다음 원인 분리 실험은 2번의 행동 유지 시간 확대와 3·4번의 학습률/갱신 안정화를 별도로 비교하는 것이다. 이번 실행 중에는 해당 조건을 바꾸지 않았다.',
              '- 보상 통합만으로 다섯 맵의 안정적인 성공을 달성했다고 결론 내릴 수 없다.']
    lines += ['', '![학습 곡선](learning_curves.png)', '', '![독립 시험 비교](success_comparison.png)', '',
              '![보상 항목](reward_components.png)', '']
    if (root/'stage4_last_policy_failure.png').exists():
        lines += ['## 4번 마지막 정책의 실패 사례', '',
                  '최대 확률 행동으로 재생한 한 판에서는 리프트→윗층까지 이동한 뒤 왼쪽으로 되돌아가 낙하했다. 확률 선택 100판의 성공률과 구분하는 진단 사례이며, 이 궤적을 학습 입력이나 시연으로 사용하지 않았다.', '',
                  '![4번 마지막 정책의 실제 실패 궤적](stage4_last_policy_failure.png)', '']
    lines += ['## 재현 명령', '', '```powershell',
              r'.venv\Scripts\python.exe -B -m rl.combined_train --root output/새_실험_폴더 --steps 400000 --test-count 100 --workers 5',
              f'.venv\\Scripts\\python.exe -B -m rl.combined_report --root "{root}" --verify', '```', '']
    (root/'결과보고서.md').write_text('\n'.join(lines),encoding='utf-8')
    print(str(root/'결과보고서.md'),flush=True)


if __name__ == '__main__':
    main()
