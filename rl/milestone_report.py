"""사용자 보상 실험의 원래 점수 그래프와 변경하지 않은 월드의 재생 검증을 생성한다."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import numpy as np
from .milestone_train import install, read, write, sources


def verify(payload):
    install()
    from . import five_report
    from .milestone_env import MilestoneEnv
    from .milestone_models import ALGORITHMS
    five_report.FiveEnv, five_report.ALGORITHMS = MilestoneEnv, ALGORITHMS
    return five_report.native_verify(payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--verify',action='store_true')
    p.add_argument('--workers',type=int,default=3)
    a = p.parse_args()
    install()
    root = a.root.resolve()
    plan = read(root/'plan.json')
    assert plan['sources']==sources()
    summaries = list((root/'experts').rglob('summary.json'))
    rows,tasks = [],[]
    for path in summaries:
        r = read(path)
        c = r['config']
        for m in r['final']['maps']:
            checkpoint = path.parent/'best.zip'
            rows.append(dict(phase='expert',algorithm=c['algorithm'],seed=c['seed'],stage=m['stage'],
                success_rate=m['success_rate'],successes=m['successes'],episodes=m['episodes'],
                mean_points=m['mean_return'],mean_seconds=m['mean_success_seconds'],greedy=m['greedy']['success'],
                mean_milestones=float(np.mean([len(x['milestones']) for x in m['raw']])),
                decisions=r['decisions'],checkpoint=str(checkpoint)))
            tasks.append((str(checkpoint),c['algorithm'],m['stage'],'milestone',m['greedy']))
    status = []
    for folder in sorted((root/'consolidated').glob('seed_*')):
        if not (folder/'completion.json').exists(): continue
        c = read(folder/'completion.json')
        status.append(dict(folder=str(folder),**c))
        if c['status']!='completed': continue
        for phase,name,modelname in [('distilled','distilled_test.json','distilled.zip'),('joint_selected','summary.json','best.zip')]:
            r = read(folder/name)
            for m in r['final']['maps']:
                rows.append(dict(phase=phase,algorithm='ppo',seed=int(folder.name.split('_')[-1]),stage=m['stage'],
                    success_rate=m['success_rate'],successes=m['successes'],episodes=m['episodes'],
                    mean_points=m['mean_return'],mean_seconds=m['mean_success_seconds'],greedy=m['greedy']['success'],
                    mean_milestones=float(np.mean([len(x['milestones']) for x in m['raw']])),
                    decisions=r.get('decisions',0),checkpoint=str(folder/modelname)))
                tasks.append((str(folder/modelname),'ppo',m['stage'],'milestone',m['greedy']))
    if a.verify and tasks:
        old = read(root/'native_verification.json') if (root/'native_verification.json').exists() else {}
        cache = {(r['checkpoint_sha256'],r['stage']):r for r in old.get('rows',[])} if old.get('sources')==sources() else {}
        audit,pending = [],[]
        for t in tasks:
            key = (hashlib.sha256(Path(t[0]).read_bytes()).hexdigest(),t[2])
            if key in cache: audit.append(cache[key])
            else: pending.append(t)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            audit.extend(pool.map(verify,pending))
        write(root/'native_verification.json',dict(complete=True,rows=audit,sources=sources(),
              max_ball_error=max(r['max_ball_error'] for r in audit)))
    write(root/'aggregate.json',dict(rows=rows,consolidation=status,expected_experts=10*len(plan['seeds']),
                                   completed_experts=len(summaries),sources=sources()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
    groups = [('expert','a2c','A2C','#2563eb'),('expert','ppo','PPO','#ea580c'),
              ('distilled','ppo','균등 증류','#8b5cf6'),('joint_selected','ppo','공동 학습 검증 선택본','#059669')]
    present = [g for g in groups if any((r['phase'],r['algorithm'])==g[:2] for r in rows)]
    if present:
        fig,axes = plt.subplots(1,2,figsize=(13,5),layout='constrained')
        width = .8/len(present)
        for j,(phase,algorithm,label,color) in enumerate(present):
            for stage in range(1,6):
                group = [r for r in rows if (r['phase'],r['algorithm'],r['stage'])==(phase,algorithm,stage)]
                if not group: continue
                x = stage+(j-(len(present)-1)/2)*width
                axes[0].bar(x,np.mean([r['success_rate'] for r in group])*100,width*.9,color=color,label=label if stage==1 else None)
                axes[1].bar(x,np.mean([r['mean_points'] for r in group]),width*.9,color=color)
        for ax in axes:
            ax.set_xticks(range(1,6));ax.set_xlabel('튜토리얼 스테이지');ax.grid(axis='y',alpha=.2)
        axes[0].set(ylim=(0,105),ylabel='별도 시험 성공률 (%)',title='성공률: 맵별 100판 평가')
        axes[0].legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,frameon=False)
        axes[1].set(ylabel='한 판의 평균 원래 보상 점수',title='성공 +100 / 실패 -25 / 핵심 구간 +10')
        axes[1].axhline(0,color='#94a3b8',lw=1)
        fig.suptitle(f'사용자 보상 실험 · γ=1 · 행동 0.05초 · 제한 180초 · 학습 시드 {plan["seeds"]}')
        fig.savefig(root/'reward_success_comparison.png',dpi=150);plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for stage,ax in enumerate(axes.flat[:5],1):
        for algorithm,color in [('a2c','#2563eb'),('ppo','#ea580c')]:
            for seed in plan['seeds']:
                f = root/'experts'/algorithm/f'seed_{seed}'/f'stage_{stage:02d}'/'metrics.json'
                if f.exists():
                    data=read(f)
                    ax.plot([r['decisions']/1000 for r in data],
                            [r['validation']['mean_success_rate']*100 for r in data],
                            label=f'{algorithm.upper()} seed {seed}',color=color,marker='o',markersize=3)
        ax.set(title=f'튜토리얼 {stage}',xlabel='행동 선택 (천 회)',ylabel='검증 성공률 (%)',ylim=(-2,102))
        ax.grid(alpha=.2)
        if stage==1 and ax.lines: ax.legend(fontsize=9)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,.9,'무작위 정책에서 시작\n각 저장본: 별도 행동 난수로 검증 20판\n핵심 구간의 좌표·순서 정보 제공\n시간/거리/일반 착지 보상 없음\n최종 100판 시험으로 저장본을 고르지 않음\n한 학습 시드 결과: 반복 실험의 평균이 아님',va='top',linespacing=1.8)
    fig.suptitle('새 보상으로 처음부터 학습한 과정')
    fig.savefig(root/'learning_curves.png',dpi=150);plt.close(fig)
    lines=['# 사용자 보상 실험 결과','','- 범위: 튜토리얼 1~5, 로컬 실행, 기존 가중치를 로드하지 않음.',
        '- 할인율 γ=1.0이며 신경망 학습률은 0.0003. 두 값은 다르다.',
        '- 성공 +100, 실패/180초 제한 −25 한 번, 핵심 구간 최초 달성 +10. 그 외 0.',
        '- 핵심 구간은 특수 블록 착지 후 지정 다음 발판에 실제 착지해야 달성한다. 연결된 이동 발판 내부 이동은 제외.',
        '- 사용자 점수는 원래 단위로 기록하고 학습기에만 고정적으로 /100 전달한다.',
        '- 행동 0.05초(120Hz 물리 6틱), 남은 시간과 핵심 구간 활성/달성 상태도 관측한다.',
        '- 개별 엔트로피 0.02 유지(0~20%) → 0.005(80%) → 0.001(100%). 공동 단계 0.005→0.001.',
        f'- 개별 전문가 완료 {len(summaries)}/{10*len(plan["seeds"])}. 학습 시드 {plan["seeds"]}.',
        '- 검증 20판으로 개별 저장본 선택, 새 검증 50판으로 맵별 교사 선택. 최종 시험은 선택에 사용하지 않는다.',
        '- 학습 곡선은 각 시점의 현재 정책이다. 결과표와 막대그래프는 검증에서 선택한 최선 저장본의 최종 시험이며, 마지막 가중치의 성능과 다를 수 있다.',
        '- 이전 실험과 보상·입력·행동 주기·제한 시간이 함께 바뀌었다. 보상만의 인과 효과로 비교할 수 없다.',
        '- 교사 기준 미달 시 그 통합은 실행하지 않는다. 통합 성공률 0으로 표시하지 않는다.','',
        '| 단계 | 알고리즘 | 맵 | 성공/시험 | 성공률 | 평균 점수 | 평균 핵심 구간 수 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['phase'],r['algorithm'],r['seed'],r['stage'])):
        phase_name={'expert':'개별 학습','distilled':'정책 증류','joint_selected':'공동 학습 검증 선택'}[r['phase']]
        lines.append(f"| {phase_name} | {r['algorithm'].upper()} seed{r['seed']} | {r['stage']} | {r['successes']}/{r['episodes']} | {100*r['success_rate']:.1f}% | {r['mean_points']:.2f} | {r['mean_milestones']:.2f} |")
    lines+=['','## 통합 상태','']
    for r in status:
        label='교사 품질 기준 미달로 통합 보류' if r['status']=='blocked_by_teacher_quality' else r['status']
        lines.append(f"- {Path(r['folder']).name}: {label}; 기준 미달 맵 {r.get('failed_stages',[])}.")
        f=Path(r['folder'])/'summary.json'
        if f.exists():
            summary=read(f)
            lines.append(f"  공동 학습 {summary['decisions']:,} 전이, 선택 저장본 {summary['best_step']:,} 전이(0이면 증류 직후).")
    example=root/'experts/ppo/seed_0/stage_03/summary.json'
    if example.exists():
        sample=read(example)['final']['maps'][0]['raw']
        wins=[r for r in sample if r['success']]
        without=sum(not r['milestones'] for r in wins)
        lines+=['','## 성공과 핵심 구간 보상의 차이','',
                f'3번 PPO seed0의 성공 {len(wins)}판 중 {without}판은 핵심 구간 보상 없이 성공했다. 현재 규칙은 핵심 구간 달성을 성공의 필수 조건으로 강제하지 않는다. 특수 발판을 이용해도 다음 목표 발판에 착지하기 전에 이글루에 도착하면 성공 +100만 받는다. 보상 조건을 충족하지 않은 +10을 추가 지급하지 않았다.']
    lines+=['','![성공률과 원래 보상 점수](reward_success_comparison.png)','','![학습 과정](learning_curves.png)','',
            '정의된 핵심 구간의 그룹 ID와 실제 월드 좌표는 `milestones.json`, 설정과 소스 해시는 `plan.json`에 기록했다.']
    (root/'결과보고서.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(experts=len(summaries),total=10*len(plan['seeds']),rows=len(rows),consolidation=[r['status'] for r in status])))


if __name__=='__main__': main()
