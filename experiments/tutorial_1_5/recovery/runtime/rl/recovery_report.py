"""2·4번 재학습의 초기값별 결과, 중간 진행과 탐색량을 그래프와 보고서로 정리한다."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from .five_train import read, write
from .recovery_train import LABELS, audit_episode, sources


def native_verify(summary):
    """선택한 정책의 최대확률 입력을 원래 월드에도 넣어 물리 결과가 같은지 검사한다."""
    import torch
    from pingu_ball.engine import World
    from .milestone_env import MilestoneEnv
    from .milestone_models import MilestonePPO
    from .five_models import action_probs
    torch.set_num_threads(1)
    c=summary['config']
    model=MilestonePPO.load(summary['selection']['checkpoint'],device='cpu')
    model.policy.set_training_mode(False)
    env=MilestoneEnv(stage=c['stage'],repeat=c['settings']['repeat'])
    raw=env.unwrapped
    native=World(raw.level,auto_reset=False)
    obs,_=env.reset()
    max_error=0.
    def tick(world,count,direction):
        nonlocal max_error
        dt=min(native.physics.dt,180.-native.time)
        native.step(direction,dt)
        for field in ('x','y','vx','vy'):
            max_error=max(max_error,abs(getattr(world.ball,field)-getattr(native.ball,field)))
        assert world.failed==native.failed and world.cleared==native.cleared
    while not raw.done:
        action=int(action_probs(model,obs[None])[0].argmax())
        raw_obs,_,_,_,info=raw.step(action,on_tick=tick)
        obs=env.observation(raw_obs)
    audit_episode(info)
    env.close()
    assert max_error<1e-6,max_error
    return dict(stage=c['stage'],seed=c['seed'],variant=c['variant'],
                max_position_velocity_error=max_error,ticks=info['ticks'],success=info['success'])


def report(root,verify=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    root=Path(root)
    paths=sorted(root.glob('*/stage_*/seed_*/summary.json'))
    rows=[read(p) for p in paths]
    groups=defaultdict(list)
    initial=defaultdict(set)
    audited=0
    for p,r in zip(paths,rows):
        c=r['config']
        assert c['sources']==r['sources_after']==sources()
        groups[(c['stage'],c['variant'])].append(r)
        initial[(c['stage'],c['seed'])].add(c['initial_parameter_hash'])
        for ev in [r['final'],r['greedy']]+[x['validation'] for x in read(p.parent/'selection.json')['candidates']]:
            for ep in ev['raw']:audit_episode(ep);audited+=1
        for m in read(p.parent/'metrics.json'):
            for ep in m['validation']['raw']:audit_episode(ep);audited+=1
    assert all(len(hashes)==1 for hashes in initial.values())
    aggregate=[]
    for (stage,variant),items in sorted(groups.items()):
        rates=[r['final']['success_rate'] for r in items]
        aggregate.append(dict(stage=stage,variant=variant,seeds=[r['config']['seed'] for r in items],
            final_success_rates=rates,mean_success_rate=float(np.mean(rates)),
            seed_standard_deviation=float(np.std(rates,ddof=1)) if len(rates)>1 else 0.,
            mean_milestones=float(np.mean([r['final']['mean_milestones'] for r in items])),
            source_landing_rate=float(np.mean([r['final']['source_landing_rate'] for r in items])),
            physics_ticks=[r['physics_ticks'] for r in items],decisions=[r['decisions'] for r in items]))
    write(root/'aggregate.json',aggregate)
    # 조건과 초기값은 최종 시험이 아니라 별도 검증 50판 점수로만 선택한다.
    selected={}
    for stage in (2,4):
        candidates=[r for r in rows if r['config']['stage']==stage]
        best=max(candidates,key=lambda r:tuple(r['selection']['score']))
        selected[str(stage)]=dict(checkpoint=best['selection']['checkpoint'],
            variant=best['config']['variant'],seed=best['config']['seed'],
            validation_success_rate=best['selection']['validation']['success_rate'],
            independent_test_success_rate=best['final']['success_rate'],
            repeat=best['config']['settings']['repeat'])
    write(root/'deployment.json',dict(maps=selected,selection='held_out_validation_only',
          shared_policy=False,other_stages_unchanged=True))
    colors={'stable':'#1475c9','long':'#ee8c16','hold':'#299b68'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True)
    for ax,stage in zip(axes,(2,4)):
        labels=['이전 실험\n초기값 1개'];means=[0.];errs=[0.];cs=['#a8afb8']
        for variant in ('stable','long','hold'):
            found=next((r for r in aggregate if r['stage']==stage and r['variant']==variant),None)
            if found:
                labels.append(LABELS[variant]+f'\n초기값 {len(found["seeds"])}개')
                means.append(found['mean_success_rate']*100)
                errs.append(found['seed_standard_deviation']*100)
                cs.append(colors[variant])
        bars=ax.bar(range(len(means)),means,color=cs,width=.6)
        ax.errorbar(range(len(means)),means,yerr=errs,fmt='none',ecolor='#263238',capsize=4)
        for b,m in zip(bars,means):ax.text(b.get_x()+b.get_width()/2,min(m+4,108),f'{m:.1f}%',ha='center')
        ax.set(xticks=range(len(labels)),xticklabels=labels,ylim=(0,115),title=f'튜토리얼 {stage}번',ylabel='별도 시험 성공률 (%)')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('동일 보상으로 재학습한 결과 - 막대: 초기값 평균, 오차막대: 초기값 간 표준편차')
    fig.text(.5,.01,'이전 실험은 초기값 1개여서 분산 비교가 불가능합니다. 후속 조건은 검증 기준에 미달한 맵만 실행합니다.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.93));fig.savefig(root/'success_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7.6),sharex=True)
    for row,stage in enumerate((2,4)):
        for variant in ('stable','long','hold'):
            for item in groups.get((stage,variant),[]):
                c=item['config'];folder=root/variant/f'stage_{stage:02d}'/f'seed_{c["seed"]}'
                ms=read(folder/'metrics.json');x=[r['physics_ticks']/1e6 for r in ms]
                for col,key,mult in ((0,'success_rate',100),(1,'mean_milestones',1),(2,'mean_entropy',1)):
                    axes[row,col].plot(x,[r['validation'][key]*mult for r in ms],
                        color=colors[variant],alpha=.7,linestyle=['-','--',':'][c['seed']%3],
                        label=LABELS[variant]+f' / 초기값 {c["seed"]}')
        for col,title in enumerate(('검증 성공률 (%)','평균 핵심 구간 달성 수','실제 행동 엔트로피')):
            axes[row,col].set(title=f'{stage}번 - {title}',xlabel='학습 물리 틱 (백만)')
            axes[row,col].grid(alpha=.2)
        axes[row,0].set_ylim(-2,103);axes[row,2].set_ylim(-.02,1.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    h2,l2=axes[1,0].get_legend_handles_labels()
    unique=dict(zip(labels+l2,handles+h2))
    fig.legend(unique.values(),unique.keys(),loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('학습 도중의 독립 검증 - 마지막 점과 최종 선택 모델의 시험 점수는 다를 수 있습니다')
    fig.tight_layout(rect=(0,.14,1,.94));fig.savefig(root/'learning_curves.png',dpi=160);plt.close(fig)
    native=[native_verify(r) for r in rows] if verify else []
    write(root/'audit.json',dict(episodes_reward_checked=audited,paired_initial_weights=True,
          native_verification=native,all_native_passed=bool(native) and all(r['max_position_velocity_error']<1e-6 for r in native),
          total_decisions=sum(r['decisions'] for r in rows),total_physics_ticks=sum(r['physics_ticks'] for r in rows)))
    lines=['# 튜토리얼 2·4 재학습 결과','',
        '보상은 성공 +100, 실패·180초 제한 -25, 특수 블록 착지 후 목표 발판 최초 착지 +10으로 유지했다.',
        '맵·시작점·물리·관측 크기는 그대로이며, 사람이 작성한 행동이나 이전 가중치는 사용하지 않았다.',
        '1·3·5번 모델과 이전 학습 폴더는 수정하지 않았다. 통합 정책은 새로 학습하지 않았다.','',
        '| 맵 | 조건 | 초기값별 별도 시험 성공률 | 평균 |', '|---|---|---|---|']
    for r in aggregate:
        lines.append(f'| {r["stage"]} | {LABELS[r["variant"]]} | '+', '.join(f'{x*100:.0f}%' for x in r['final_success_rates'])+f' | {r["mean_success_rate"]*100:.1f}% |')
    lines += ['', '각 초기값에서 최종 시험은 별도 100판이며, 모델 선택은 그전에 별도 검증 50판으로 수행했다.',
        '주기적 검증은 20판이다. 그래프의 오차막대는 초기값 간 표준편차이며 신뢰구간은 아니다.',
        '이전 실험은 초기값 하나의 0/100 결과여서 여러 초기값의 평균과 동일한 강도로 비교할 수 없다.','',
        '## 수정 조건','',
        '- PPO 학습률 0.0001, 갱신 3회, 변경 제한 0.1, KL 조기 중단 0.01, 할인율 1.0.',
        '- 초기 엔트로피 계수 0.03. 주기 검증 성공률 60% 이상을 세 번 연속 확인한 뒤 0.01까지 점진적으로 감소.',
        '- 안정화: 수집 길이 128, GAE 0.98, 행동 0.05초.',
        '- 긴 수집: 수집 길이 512, GAE 0.995, 행동 0.05초.',
        '- 긴 행동: 긴 수집 설정에서 행동만 0.1초로 변경.',
        '- 조건별 실제 물리 틱 목표 240만. 완전한 수집 묶음까지 학습하므로 최대 한 묶음의 초과량이 있다.',
        '- 성공률과 중간 진행도 최선 저장본을 분리하고, 동점인 최신 모델로 덮어쓰지 않는다.',
        '- 시작 발판 착지는 보상 없이 선택·진단 지표로만 기록한다.','',
        '검증 평균 80% 이상이고 모든 초기값이 50% 이상이면 그 맵의 후속 조건을 건너뛴다. 시험 점수로 후속 조건을 결정하지 않는다.',
        '조건 간 변화가 여러 개이므로 모든 성능 차이를 단일 설정의 인과 효과라고 단정할 수 없다.','',
        '## 검사','',f'- 평가 기록 {audited:,}판의 보상 계산을 검사했다.',
        f'- 조건 간 같은 초기값의 최초 신경망 가중치가 일치한다.',
        f'- 원래 게임 물리 대조: {len(native)}개 모델.',
        f'- 총 학습 행동 선택: {sum(r["decisions"] for r in rows):,}회.',
        f'- 총 실제 학습 물리 틱: {sum(r["physics_ticks"] for r in rows):,}.','',
        '세부 자료는 plan.json, 각 실행의 config.json·metrics.json·selection.json·summary.json·episodes.jsonl에 있다.',
        '그래프: success_comparison.png, learning_curves.png. 재생 선택 자료: deployment.json.']
    (root/'결과보고서.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT COMPLETE',root,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--verify',action='store_true')
    args=p.parse_args();report(args.root,args.verify)


if __name__=='__main__':main()
