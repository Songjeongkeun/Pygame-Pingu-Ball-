"""2·4번 단계별 실험을 시드별 성공률·학습량·실패 구간으로 검토한다."""
import argparse
import csv
from pathlib import Path
import numpy as np
from .five_train import read, write
from .stabilized_train import sources
from .stabilized_env import audit_reward

LABELS={'A':'행동·관측 변경','B':'보조 보상 축소 추가','C':'보수적 PPO 추가'}


def plot_learning(root,plan,live=False):
    """미완료 모델도 실제 검증 시점까지만 표시한다."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for i,stage in enumerate((2,4)):
        for j,variant in enumerate(('A','B','C')):
            ax=axes[i,j]
            if stage==2 and variant=='B':
                ax.axis('off');ax.text(.1,.6,'2번은 B에서도 보상이 같아\nA의 결과를 공유합니다.\n중복 학습·중복 표본 집계 없음',va='center',linespacing=1.8)
                continue
            for seed in plan['seeds']:
                path=root/variant/f'stage_{stage:02d}'/f'seed_{seed}'/'metrics.json'
                if path.exists():
                    data=read(path)
                    ax.plot([r['physics_ticks']/1000000 for r in data],
                            [r['validation']['success_rate']*100 for r in data],marker='o',markersize=3,label=f'시드 {seed}')
            ax.set(title=f'{stage}번 · {variant}: {LABELS[variant]}',xlabel='학습 물리 프레임 (백만)',
                   ylabel='검증 성공률 (%)',ylim=(-2,102),xlim=(0,plan['physics_budget']/1000000*1.02))
            ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=8)
            else:ax.text(.5,.5,'학습 대기 중',ha='center',va='center',transform=ax.transAxes,color='#64748b')
    title='진행 중인 중간 검증 · 최종 시험 결과 아님' if live else '원래 시작점에서 새로 학습한 과정 · C는 반복적인 검증 성능 하락 시 조기 종료'
    fig.suptitle(title)
    path=root/('learning_curves_live.png' if live else 'learning_curves.png')
    fig.savefig(path,dpi=160);plt.close(fig)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve();plan=read(root/'plan.json')
    assert plan['sources']==sources()
    if args.live:
        print(plot_learning(root,plan,live=True),flush=True)
        return
    rows=[];audited=0;max_error=0.
    for p in sorted(root.glob('*/stage_*/seed_*/summary.json')):
        d=read(p);c=d['config'];selected=d['final'];last=d['latest_test']
        assert c['sources']==plan['sources']==d['sources_after']
        for evaluation in (selected,last,d['selected_greedy'],d['latest_greedy']):
            for episode in evaluation['raw']:
                audit_reward(episode);audited+=1
            if evaluation['native_comparison']:
                max_error=max(max_error,evaluation['max_native_error'])
        rows.append(dict(variant=c['variant'],stage=c['stage'],seed=c['seed'],
            selected_success_rate=selected['success_rate'],latest_success_rate=last['success_rate'],
            selected_successes=selected['successes'],test_episodes=selected['episodes'],
            mean_common_return=selected['mean_common_return'],mean_total_return=selected['mean_return'],
            mean_success_seconds=selected['mean_success_seconds'],
            source_rate=selected['source_rate'],mean_milestones=selected['mean_milestones'],
            latest_source_rate=last['source_rate'],
            selected_phase_rates=selected.get('phase_rates',selected.get('milestone_rates')),
            latest_phase_rates=last.get('phase_rates',last.get('milestone_rates')),
            selected_failures=selected['failures'],latest_failures=last['failures'],
            selected_greedy_success=bool(d['selected_greedy']['successes']),
            latest_greedy_success=bool(d['latest_greedy']['successes']),
            decisions=d['decisions'],physics_ticks=d['physics_ticks'],
            training_episodes=d['training_episodes'],training_successes=d['training_successes'],
            best_step=d['best_step'],stop_reason=d['stop_reason'],
            elapsed_seconds=d['elapsed_seconds'],checkpoint=str(p.parent/'best.zip')))
    if not rows:
        raise RuntimeError('아직 완료된 모델이 없다')
    groups=[]
    for stage,variants in [(2,['A','C']),(4,['A','B','C'])]:
        for variant in variants:
            group=[r for r in rows if r['stage']==stage and r['variant']==variant]
            if not group:continue
            groups.append(dict(stage=stage,variant=variant,seeds=[r['seed'] for r in group],
                mean_selected=float(np.mean([r['selected_success_rate'] for r in group])),
                sd_selected=float(np.std([r['selected_success_rate'] for r in group],ddof=1)) if len(group)>1 else 0.,
                mean_latest=float(np.mean([r['latest_success_rate'] for r in group])),
                sd_latest=float(np.std([r['latest_success_rate'] for r in group],ddof=1)) if len(group)>1 else 0.,
                minimum_latest=min(r['latest_success_rate'] for r in group),
                mean_physics_ticks=float(np.mean([r['physics_ticks'] for r in group])),
                mean_decisions=float(np.mean([r['decisions'] for r in group])),
                mean_training_successes=float(np.mean([r['training_successes'] for r in group]))))
    write(root/'aggregate.json',dict(rows=rows,groups=groups,
        completed_models=len(rows),expected_models=plan['expected_models'],
        all_complete=len(rows)==plan['expected_models'],evaluation_records_audited=audited,
        max_native_error=max_error,sources=sources()))
    fields=[k for k in rows[0] if not isinstance(rows[0][k],dict)]
    with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for stage,ax in zip((2,4),axes):
        group=[g for g in groups if g['stage']==stage]
        x=np.arange(len(group))
        for offset,key,sd,label,color in [(-.18,'mean_selected','sd_selected','검증 선택본','#0284c7'),
                                          (.18,'mean_latest','sd_latest','마지막 저장본','#f59e0b')]:
            values=[g[key]*100 for g in group]
            field='selected_success_rate' if key=='mean_selected' else 'latest_success_rate'
            limits=[]
            for g in group:
                points=[r[field]*100 for r in rows if r['stage']==stage and r['variant']==g['variant']]
                mean=float(np.mean(points));limits.append((mean-min(points),max(points)-mean))
            bars=ax.bar(x+offset,values,.33,yerr=np.asarray(limits).T,capsize=4,label=label,color=color)
            ax.bar_label(bars,fmt='%.1f',padding=4,fontsize=10)
            for i,g in enumerate(group):
                seed_rows=[r for r in rows if r['stage']==stage and r['variant']==g['variant']]
                field='selected_success_rate' if key=='mean_selected' else 'latest_success_rate'
                ax.scatter([i+offset+(j-(len(seed_rows)-1)/2)*.035 for j in range(len(seed_rows))],
                           [r[field]*100 for r in seed_rows],s=18,c='#0f172a',zorder=4)
        ax.set(xticks=x,xticklabels=[g['variant'] for g in group],ylim=(0,125),
               ylabel='독립 시험 성공률 (%)',title=f'튜토리얼 {stage}')
        ax.legend(loc='upper center',ncol=2,fontsize=9,frameon=False)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('A: 행동·관측 / B: 4번 보상 축소 / C: 보수적 PPO\n막대는 학습 시드 평균, 오차막대는 시드 최저~최고, 점은 각 시드',fontsize=12)
    fig.savefig(root/'success_comparison.png',dpi=160);plt.close(fig)
    plot_learning(root,plan)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    colors={'A':'#0284c7','B':'#8b5cf6','C':'#059669'}
    for i,stage in enumerate((2,4)):
        labels=(['특수 발판','중간층','윗층','도착 발판','이글루'] if stage==2 else
                ['리프트 착지','높은 위치','윗층 착지','이글루'])
        for j,latest in enumerate((False,True)):
            ax=axes[i,j]
            for variant in (['A','C'] if stage==2 else ['A','B','C']):
                group=[r for r in rows if r['stage']==stage and r['variant']==variant]
                if not group:continue
                values=[]
                for r in group:
                    phase=r['latest_phase_rates' if latest else 'selected_phase_rates']
                    goal=r['latest_success_rate' if latest else 'selected_success_rate']
                    if stage==2:
                        values.append([r['latest_source_rate' if latest else 'source_rate'],
                            *(phase[k] for k in ('spring-to-middle','spring-to-upper','spring-to-finish')),goal])
                    else:
                        values.append([*(phase[k] for k in ('lift_first','lift_high','upper_first')),goal])
                ax.plot(range(len(labels)),np.mean(values,axis=0)*100,marker='o',
                        label=variant,color=colors[variant])
            ax.set(xticks=range(len(labels)),xticklabels=labels,ylim=(0,112),
                   ylabel='시드 평균 달성률 (%)',title=f'{stage}번 · {"마지막 정책" if latest else "검증 선택본"}')
            ax.grid(alpha=.2);ax.legend(loc='upper center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('특수 발판 이후 어느 구간에서 성공률이 떨어지는지 비교')
    fig.savefig(root/'phase_comparison.png',dpi=160);plt.close(fig)
    lines=['# 튜토리얼 2·4 단계별 안정화 실험','',
        f'- 완료 모델 {len(rows)}/{plan["expected_models"]}. 모든 정책은 무작위 초기화에서 시작하며 이전 가중치를 읽지 않았다.',
        '- A: 방향 3개×유지 시간 0.1/0.25/0.6초=9행동, 기존 1,664개 관측에 보상 추적 상태 26개 추가.',
        '- B: A와 같은 행동·관측에서 4번 보조 보상만 첫 리프트 +2, 높은 위치 +2, 윗층 추가 +3, 접근 합계 최대 +2로 축소.',
        '- C: B의 환경에 학습률 0.0001→0.00003, clip 0.1, target KL 0.01, 갱신 반복 3회를 적용.',
        '- C의 엔트로피 계수는 검증 성공률 50% 전까지 0.03, 이후 0.01→0.003으로 감소한다. 계수는 무작위 행동 비율이 아니다.',
        '- C에서 최고 검증 성공률이 50% 이상인 뒤 20%p 이상 하락이 3회 연속이면 조기 종료한다. 최종 시험은 이 판단에 쓰지 않는다.',
        '- 모든 조건: 원래 게임 물리 120Hz, 원래 시작점, 판당 180초, 성공 +100/실패·시간 초과 -25 한 번/핵심 구간 최초 착지 +10, 학습 보상 /100.',
        '- 2번의 기존 핵심 구간 +10에는 시간 제한을 새로 붙이지 않았다. 높이·가로 진입 보조 보상에만 기존 3초 조건을 유지한다.',
        '- 4번 B·C에서 중간 보상을 전부 받고 실패해도 합계는 최대 -6이다. 2번에 동일한 상한을 적용한다는 뜻은 아니다.',
        f'- 조건별 최대 학습 물리 프레임 {plan["physics_budget"]:,}. 완전한 롤아웃까지 진행하므로 소폭 초과할 수 있다. C는 조기 종료하면 실제 경험량이 적다.',
        '- 행동 길이는 정책이 상태를 보고 선택한다. 고정 시간표·정답 행동·사람의 시연·체크포인트 시작을 사용하지 않는다.',
        '- 2번 B는 A와 동일한 조건이라 별도 학습하지 않았다. 실제 모델은 시드당 A2/A4/B4/C2/C4의 5개다.',
        '- 검증 성공률을 우선하여 모델을 선택하고, 별도 행동 난수로 선택본과 마지막 정책을 각각 시험한다. 같은 가중치라면 같은 평가를 재사용한다.',
        '- 주 결과는 확률 선택 100%이며, 최대 확률 행동 1판은 별도 원래 물리 대조용 진단이다.',
        '- 기존 0.05초·40만 행동 실험은 학습량과 행동·관측이 다르므로 참고 자료다. 직접적인 단계별 비교는 이번 A/B/C 사이에서 한다.',
        '- B는 보상 규모도 바뀌므로 원점수만으로 우열을 판단하지 않는다. 공통 보상 점수와 성공률을 기록한다.',
        '- 동일한 중간 보상을 받은 경로라면 성공과 실패의 종료 점수 차이는 A·B 모두 125점이다. 실패 누적 점수를 음수로 만드는 것 자체가 성공을 보장하지는 않으며, 탐색·가치 추정·실제 성공률을 함께 확인한다.',
        '- 시작 위치·맵이 바뀐 일반화 시험이나 1~5 공통 정책 증류 실험은 아니다. Git 커밋·푸시를 하지 않는다.','',
        '| 맵 | 조건 | 학습 시드 수 | 선택본 평균 성공률 | 마지막 평균 성공률 | 마지막 최저 시드 | 평균 실제 물리 프레임 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f'| {g["stage"]} | {g["variant"]} | {len(g["seeds"])} | {g["mean_selected"]*100:.1f}% | {g["mean_latest"]*100:.1f}% | {g["minimum_latest"]*100:.1f}% | {g["mean_physics_ticks"]:,.0f} |')
    lines+=['','## 시드별 결과','','| 맵 | 조건 | 시드 | 선택본 | 마지막 | 성공 경험 | 물리 프레임 | 행동 선택 | 종료 사유 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f'| {r["stage"]} | {r["variant"]} | {r["seed"]} | {r["selected_success_rate"]*100:.0f}% | {r["latest_success_rate"]*100:.0f}% | {r["training_successes"]} | {r["physics_ticks"]:,} | {r["decisions"]:,} | {r["stop_reason"]} |')
    lines+=['','## 설정과 비교 범위','',
        '| 항목 | A·B | C |',
        '|---|---|---|',
        '| 학습률 | 0.0003 고정 | 0.0001 → 0.00003 |',
        '| PPO clip | 0.2 | 0.1 |',
        '| 목표 KL | 0.02 | 0.01 |',
        '| 롤아웃당 갱신 반복 | 5 | 3 |',
        '| 엔트로피 계수 | 예산 처음 20%는 0.02, 80%까지 0.005로, 마지막에 0.001로 감소 | 검증 50% 전 0.03, 이후 0.01 → 0.003 |',
        '| 할인율 / GAE | 1.0 / 0.98 | 1.0 / 0.98 |',
        '| 환경 수 / 환경당 수집 행동 | 4 / 128 | 4 / 128 |',
        '| 배치 / 가치 손실 계수 / 기울기 상한 | 256 / 0.5 / 0.5 | 256 / 0.5 / 0.5 |','',
        'A와 이전 실험은 행동 길이·관측·경험 예산이 함께 달라졌다. 따라서 이번 개선을 특정 항목 하나의 효과로 단정할 수 없다. '
        'A→B는 4번 보조 보상 규모 비교이며, B→C는 여러 PPO 설정을 묶어 변경한 비교다. '
        'C가 낮다고 해서 작은 학습률이나 높은 엔트로피가 언제나 나쁘다는 결론은 아니다.','',
        '검증은 매 시점 20판으로 표본이 작다. 선택본의 독립 시험 성공률이 마지막본보다 낮을 수 있으며, '
        '최종 시험 결과를 보고 체크포인트를 다시 선택하지 않는다. 학습 시드 3개는 초기화에 따른 편차를 보는 용도이고, '
        '오차막대는 신뢰구간이 아니라 이 세 시드의 최솟값~최댓값이다.']
    if len(rows)==plan['expected_models']:
        lookup={(g['stage'],g['variant']):g for g in groups}
        c_histories=[read(root/'C'/f'stage_{r["stage"]:02d}'/f'seed_{r["seed"]}'/'metrics.json')
                     for r in rows if r['variant']=='C']
        c_triggered=sum(any(p['entropy_trigger_ticks'] is not None for p in h) for h in c_histories)
        lines+=['','## 이번 결과의 해석','',
            f'- A의 마지막 정책은 2번 평균 {lookup[2,"A"]["mean_latest"]*100:.1f}%, '
            f'4번 평균 {lookup[4,"A"]["mean_latest"]*100:.1f}%다. 기존 짧은 행동 실험의 실패를 반복하는 대신 실제 완주 정책을 학습했다.',
            f'- 4번 B의 선택본/마지막 평균은 {lookup[4,"B"]["mean_selected"]*100:.1f}%/'
            f'{lookup[4,"B"]["mean_latest"]*100:.1f}%다. 보조 보상 축소 효과는 이 값과 시드별 변동을 함께 평가해야 한다.',
            f'- C의 마지막 평균은 2번 {lookup[2,"C"]["mean_latest"]*100:.1f}%, '
            f'4번 {lookup[4,"C"]["mean_latest"]*100:.1f}%다. 보수적인 설정이 이번 예산에서도 더 좋은지는 A·B와 실제 수치를 비교해 판단한다.',
            f'- C의 {len(c_histories)}개 모델 중 {c_triggered}개가 검증 50% 조건을 만족해 탐색 계수 감소를 시작했다. '
            '조건에 도달하지 못한 모델은 계수 0.03을 끝까지 유지했다. 이것은 설정의 실제 작동 기록이며, '
            '학습 지연을 어느 한 계수만의 원인으로 입증한 것은 아니다.',
            '- 일반 맵·달라진 시작 위치·1~5단계 공통 정책의 성공을 확인한 실험은 아니다. 이번 결과를 바로 모든 맵의 성공률로 확대하지 않는다.']
    lines+=['','## 최대 확률 행동 재생 진단','',
            '원래 시작점에서 최대 확률 행동으로 한 판씩 재생한 결과다. 확률 선택 100판의 성공률이나 시작점 변화에 대한 강건성으로 해석하지 않는다.','',
            '| 맵 | 조건 | 선택본 성공/시드 수 | 마지막 성공/시드 수 |',
            '|---|---|---:|---:|']
    for g in groups:
        group=[r for r in rows if r['stage']==g['stage'] and r['variant']==g['variant']]
        lines.append(f'| {g["stage"]} | {g["variant"]} | {sum(r["selected_greedy_success"] for r in group)}/{len(group)} | {sum(r["latest_greedy_success"] for r in group)}/{len(group)} |')
    lines+=['','![독립 시험 비교](success_comparison.png)','','![학습 곡선](learning_curves.png)','',
            '![구간별 달성률](phase_comparison.png)','',
            f'- 기록된 평가 보상 감사: {audited}판 통과. 최대 물리 대조 오차: {max_error}.','',
            '## 실행 명령','','```powershell',
            r'.venv\Scripts\python.exe -B -m rl.stabilized_train --root output/새_안정화_실험 --physics-ticks 6000000 --seeds 0,1,2 --workers 8',
            f'.venv\\Scripts\\python.exe -B -m rl.stabilized_report --root "{root}"','```','']
    (root/'결과보고서.md').write_text('\n'.join(lines),encoding='utf-8')
    print(str(root/'결과보고서.md'),flush=True)


if __name__=='__main__':main()
