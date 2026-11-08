"""강화학습 맵1 A 실험의 곡선·구간 달성률·최종 시험을 그린다."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .five_train import read,write
from .map_a_env import SPECS,LABELS,audit_reward
from .map_a_train import sources,selection_score


def setup_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
    return plt


def histories(root):
    return [(int(p.parent.name.split('_')[-1]),read(p)) for p in sorted(root.glob('seed_*/metrics.json'))]


def plot_learning(root,live):
    plt = setup_plot()
    fig,axes = plt.subplots(2,2,figsize=(14,9),layout='constrained')
    colors = ['#0284c7','#f59e0b','#059669','#8b5cf6']
    largest_tick = 0
    for index,(seed,data) in enumerate(histories(root)):
        largest_tick = max(largest_tick,max(d['physics_ticks'] for d in data))
        color = colors[index%len(colors)]
        x = [d['physics_ticks']/1e6 for d in data]
        values = [d['validation'] for d in data]
        axes[0,0].plot(x,[v['success_rate']*100 for v in values],marker='o',color=color,label=f'시드 {seed}')
        axes[0,1].plot(x,[v['mean_return'] for v in values],marker='o',color=color,label=f'{seed}: 전체 보상')
        axes[0,1].plot(x,[v['mean_common_return'] for v in values],linestyle='--',color=color,label=f'{seed}: 공통 보상')
        axes[1,0].plot(x,[v['mean_milestones'] for v in values],marker='o',color=color,label=f'시드 {seed}')
        path = root/f'seed_{seed}'/'episodes.jsonl'
        if path.exists():
            # 실행 중에는 마지막 행을 쓰고 있을 수 있으므로 개행까지 기록된 행만 읽는다.
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines(keepends=True) if line.endswith('\n')]
            if rows:
                wins = np.asarray([r['success'] for r in rows],float)
                prefix = np.r_[0.,wins.cumsum()]
                at = np.arange(1,len(rows)+1)
                begin = np.maximum(0,at-200)
                moving = (prefix[at]-prefix[begin])/(at-begin)*100
                take = np.unique(np.r_[np.arange(0,len(rows),max(1,len(rows)//1000)),len(rows)-1])
                axes[1,1].plot([rows[i]['physics_ticks']/1e6 for i in take],moving[take],color=color,label=f'시드 {seed}')
    plan = read(root/'plan.json')
    axes[0,0].set(title=f'원래 시작점 검증 성공률 · 시점당 {plan["validation_count"]}판',ylabel='성공률 (%)',ylim=(-2,102))
    axes[0,1].set(title='검증 평균 보상 · 보조 보상을 구분',ylabel='원점수')
    axes[0,1].axhline(0,color='#64748b',lw=.8,alpha=.5)
    axes[1,0].set(title='검증 평균 핵심 구간 달성 수',ylabel='달성 구간 수 / 16',ylim=(-.2,16.5))
    axes[1,1].set(title='학습 중 최근 최대 200판 성공률',ylabel='성공률 (%)',ylim=(-2,102))
    for ax in axes.flat:
        ax.set_xlabel('학습 물리 프레임 (백만)');ax.set_xlim(0,max(plan['physics_budget'],largest_tick)/1e6*1.02)
        ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8,loc='best')
    axes[0,1].legend(fontsize=8,ncol=2)
    fig.suptitle('진행 중 · 최종 시험 결과 아님' if live else f'강화학습 맵 1 · A 설정 PPO · {len(plan["seeds"])}개 시드 모두 처음부터 학습')
    path = root/('learning_metrics_live.png' if live else 'learning_metrics.png')
    fig.savefig(path,dpi=155);plt.close(fig)
    return path


def plot_map(root,trajectories=None):
    plt = setup_plot()
    from matplotlib.patches import Rectangle,Patch
    level = read(root/'plan.json')['level']
    fig,ax = plt.subplots(figsize=(16,5.5),layout='constrained')
    palette = dict(normal='#52718f',spring='#f59e0b',**{'301':'#8b5cf6','302':'#22d3ee','501':'#38bdf8','508':'#2563eb'})
    lookup = {}
    from .map_a_env import MapAEnv
    env = MapAEnv();world = env.unwrapped.world
    for original in level['platforms']+level['specials']:
        row = dict(original)
        if 'type_id' in row:
            row['x'],row['y'],row['w'],row['h'] = world.special.rect(world.special.devices[row['id']],0.)
        lookup[row['id']] = row
        kind = row.get('kind',str(row.get('type_id')))
        ax.add_patch(Rectangle((row['x'],row['y']),row['w'],row['h'],facecolor=palette[kind],edgecolor='#0f172a',lw=.5))
        if row.get('type_id') in (501,508):
            travel = row['options']['travel'];vertical = row['type_id']==508
            ax.add_patch(Rectangle((original['x'],original['y']),row['w']+(0 if vertical else travel),
                row['h']+(travel if vertical else 0),fill=False,edgecolor='#2563eb',linestyle='--',lw=1))
    for i,spec in enumerate(SPECS,1):
        targets = [lookup[k] for k in spec.target]
        x = (min(t['x'] for t in targets)+max(t['x']+t['w'] for t in targets))/2
        y = min(t['y'] for t in targets)-36
        ax.text(x,y,str(i),ha='center',va='center',fontsize=9,color='#fff',
                bbox=dict(boxstyle='circle,pad=.22',fc='#be123c',ec='none'))
    sx,sy = level['spawn'];gx,gy,gw,gh = level['goal']
    ax.scatter([sx],[sy],s=45,color='#111827',zorder=4);ax.text(sx,sy-45,'출발',ha='center',color='#111827')
    ax.add_patch(Rectangle((gx,gy),gw,gh,facecolor='#10b981',edgecolor='#065f46'))
    ax.text(gx+gw/2,gy-40,'이글루',ha='center',color='#065f46')
    ax.set(xlim=(0,level['width']),ylim=(level['height'],0),aspect='equal',title='얼음 계곡 왕복길 · 원래 맵 배치 · 숫자는 보상 목표 착지 구간')
    ax.axis('off')
    handles=[Patch(color=palette[k],label=v) for k,v in [('normal','기본'),('spring','점프'),('301','파괴'),('302','시간'),('501','좌우 이동'),('508','상하 이동')]]
    if trajectories:
        for data in trajectories:
            line, = ax.plot([p['x'] for p in data['trajectory']],[p['y'] for p in data['trajectory']],
                           lw=1.2,alpha=.85,label=f'시드{data["seed"]} 재생')
            handles.append(line)
        ax.set_title('최대 확률 행동 한 판씩의 실제 재생 궤적 · 배경은 초기 배치 · 성공률 시험과 구분')
    ax.legend(handles=handles,loc='lower center',ncol=6,frameon=False,fontsize=9)
    path = root/('policy_paths.png' if trajectories else 'map_milestones.png')
    fig.savefig(path,dpi=150);plt.close(fig);env.close()


def plot_behavior(root):
    """생존 시간 증가나 행동 확률 집중을 클리어와 구분해서 확인한다."""
    plt = setup_plot()
    fig,axes = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for seed,data in histories(root):
        x = [d['physics_ticks']/1e6 for d in data]
        axes[0].plot(x,[d['validation']['mean_episode_seconds'] for d in data],marker='o',label=f'시드 {seed}')
        axes[1].plot(x,[-d['diagnostics'].get('train/entropy_loss',np.nan) for d in data],marker='o',label=f'시드 {seed}')
        axes[2].plot(x,[d['validation']['source_rate']*100 for d in data],marker='o',label=f'시드 {seed}')
    axes[0].set(title='검증 평균 한 판 길이',ylabel='게임 시간 (초)',ylim=(0,185))
    axes[1].set(title='실제 정책 엔트로피 · 학습 롤아웃',ylabel='행동 확률의 분산 정도',ylim=(-.05,np.log(9)+.1))
    axes[1].axhline(np.log(9),linestyle='--',color='#64748b',lw=.8)
    axes[2].set(title='특수 발판을 하나 이상 이용한 판',ylabel='검증 판 비율 (%)',ylim=(-2,102))
    for ax in axes:
        ax.set_xlabel('학습 물리 프레임 (백만)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('오래 살아 있거나 행동이 한쪽으로 집중되는 것과 클리어를 구분')
    fig.savefig(root/'behavior_diagnostics.png',dpi=155);plt.close(fig)


def audit_evaluation(evaluation,count):
    assert evaluation['episodes']==len(evaluation['raw'])==count
    assert evaluation['successes']==sum(r['success'] for r in evaluation['raw'])
    assert abs(evaluation['success_rate']-evaluation['successes']/count)<1e-12
    for row in evaluation['raw']:audit_reward(row)
    if evaluation['native_comparison']:assert evaluation['max_native_error']==0.
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--live',action='store_true')
    args = parser.parse_args();root = args.root.resolve();plan = read(root/'plan.json')
    assert plan['sources']==sources()
    if args.live:
        print(plot_learning(root,True),flush=True);return
    completion = read(root/'completion.json')
    assert completion['complete'] and completion['models']==len(plan['seeds'])
    rows,records,training_count = [],Counter(),0
    checkpoint_hashes = {}
    import torch
    from .map_a_models import load_policy
    torch.set_num_threads(1)
    initial_hashes = []
    for seed in plan['seeds']:
        folder = root/f'seed_{seed}';data = read(folder/'summary.json')
        assert data['config']['sources']==data['sources_after']==plan['sources']
        assert data['test_used_for_selection'] is False
        history = read(folder/'metrics.json')
        best = max(enumerate(history),key=lambda item:(selection_score(item[1]['validation']),item[0]))[1]
        assert best['decisions']==data['best_step']
        assert data['best_validation']['evaluation']==best['validation']
        assert plan['physics_budget']<=data['physics_ticks']<=plan['physics_budget']+512*72
        for point in history:records['validation'] += audit_evaluation(point['validation'],plan['validation_count'])
        for key in ('final','latest_test'):records['test'] += audit_evaluation(data[key],plan['test_count'])
        for key in ('selected_greedy','latest_greedy'):records['native_diagnostic'] += audit_evaluation(data[key],1)
        initialized = read(folder/'initialization.json')
        assert initialized['steps']==0 and not initialized['previous_models_loaded']
        assert np.allclose(initialized['probabilities'],np.full(9,1/9))
        initial_hashes.append(initialized['parameter_sha256'])
        training_parts = Counter();training_milestones = Counter();training_sources = Counter()
        wins = count = maximum_milestones = 0
        with (folder/'episodes.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                e = json.loads(line);audit_reward(e);count += 1;wins += int(e['success']);training_parts.update(e['reward_parts'])
                training_milestones.update(e['milestones']);training_sources.update(e['armed'])
                maximum_milestones = max(maximum_milestones,len(e['milestones']))
        assert count==data['training_episodes'] and wins==data['training_successes']
        for k,v in training_parts.items():assert abs(v-data['training_reward_parts'][k])<1e-5
        training_count += count
        for name in ('initial.zip','best.zip','latest.zip'):
            checkpoint_hashes[(folder/name).relative_to(root).as_posix()] = hashlib.sha256((folder/name).read_bytes()).hexdigest()
            model = load_policy(folder/name)
            expected_step = 0 if name=='initial.zip' else data['best_step'] if name=='best.zip' else data['decisions']
            assert model.num_timesteps==expected_step
            assert model.action_space.n==9 and model.observation_space.shape==(plan['observation_size'],)
            assert model.gamma==1. and model.gae_lambda==.98
            assert all(torch.isfinite(v).all().item() for v in model.policy.state_dict().values())
            if name=='initial.zip':
                digest = hashlib.sha256()
                for key,value in sorted(model.policy.state_dict().items()):
                    digest.update(key.encode());digest.update(value.cpu().numpy().tobytes())
                assert digest.hexdigest()==initialized['parameter_sha256']
            del model
        rows.append(dict(seed=seed,selected=data['final'],latest=data['latest_test'],
            selected_greedy_success=bool(data['selected_greedy']['successes']),
            latest_greedy_success=bool(data['latest_greedy']['successes']),
            training_episodes=count,training_successes=wins,physics_ticks=data['physics_ticks'],
            maximum_training_milestones=maximum_milestones,
            training_milestone_episodes={s.id:training_milestones[s.id] for s in SPECS},
            training_source_episodes={s.id:training_sources[s.id] for s in SPECS},
            decisions=data['decisions'],best_step=data['best_step'],elapsed_seconds=data['elapsed_seconds']))
    assert len(set(initial_hashes))==len(plan['seeds'])
    replays = []
    for path in sorted(root.glob('replay_seed_*.json')):
        replay = read(path)
        assert replay['sources']==plan['sources'] and replay['max_native_error']==0.
        assert replay['checkpoint_sha256']==checkpoint_hashes[f'seed_{replay["seed"]}/best.zip']
        audit_reward(replay['result']);replays.append(replay)
        records['replay_diagnostic'] += 1
    selected = float(np.mean([r['selected']['success_rate'] for r in rows]))
    last = float(np.mean([r['latest']['success_rate'] for r in rows]))
    progress = {}
    for key in ('selected','latest'):
        episodes = [episode for r in rows for episode in r[key]['raw']]
        progress[key] = []
        for spec in SPECS:
            entered = sum(spec.id in e['armed'] for e in episodes)
            completed = sum(spec.id in e['milestones'] for e in episodes)
            assert completed<=entered
            progress[key].append(dict(id=spec.id,source_episodes=entered,completed_episodes=completed,
                completion_given_source=completed/entered if entered else None))
    write(root/'aggregate.json',dict(complete=True,models=len(rows),selected_mean=selected,latest_mean=last,
        rows=rows,milestone_progress=progress,evaluation_records=dict(records),training_records_audited=training_count,sources=sources()))
    plot_learning(root,False);plot_map(root);plot_behavior(root)
    plt = setup_plot()
    fig,(ax,heat) = plt.subplots(2,1,figsize=(15,9),layout='constrained',gridspec_kw={'height_ratios':[1,1.35]})
    x = np.arange(len(rows))
    for offset,key,label,color in [(-.18,'selected','검증 선택본','#0284c7'),(.18,'latest','마지막 저장본','#f59e0b')]:
        bars = ax.bar(x+offset,[r[key]['success_rate']*100 for r in rows],.33,label=label,color=color)
        ax.bar_label(bars,fmt='%.0f%%',padding=4)
    ax.set(xticks=x,xticklabels=[f'시드 {r["seed"]}' for r in rows],ylim=(0,110),ylabel='성공률 (%)',title=f'별도 시험 · 모델별 {plan["test_count"]}판 확률 선택')
    ax.legend(loc='upper center',ncol=2,frameon=False);ax.grid(axis='y',alpha=.2)
    matrix,labels = [],[]
    for r in rows:
        for key,label in [('selected','선택본'),('latest','마지막')]:
            matrix.append([r[key]['milestone_rates'][s.id]*100 for s in SPECS]);labels.append(f'{r["seed"]} · {label}')
    matrix = np.asarray(matrix)
    im = heat.imshow(matrix,aspect='auto',vmin=0,vmax=100,cmap='Blues')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            heat.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>55 else '#334155')
    heat.set(yticks=range(len(labels)),yticklabels=labels,xticks=range(len(SPECS)),
        xticklabels=[f'{i+1}. {label}' for i,label in enumerate(LABELS)],title='독립 시험의 구간별 실제 착지 달성률 (%)')
    plt.setp(heat.get_xticklabels(),rotation=45,ha='right',fontsize=9)
    fig.colorbar(im,ax=heat,shrink=.75,label='달성률 (%)')
    fig.suptitle(f'강화학습 맵1 · A · 선택본 평균 {selected*100:.1f}% / 마지막 평균 {last*100:.1f}%')
    fig.savefig(root/'test_results.png',dpi=155);plt.close(fig)
    lines=['# 강화학습 맵1 · A 설정 PPO 실험','',
        f'대상은 강화학습 탭의 **{plan["map_title"]}**다. 튜토리얼 1번과 구분한다.','',
        f'선택본 평균 성공률 **{selected*100:.1f}%**, 마지막 저장본 평균 **{last*100:.1f}%**.',
        f'학습 시드 {plan["seeds"]}, 모델마다 최대 {plan["physics_budget"]:,} 물리 프레임, 별도 시험 각각 {plan["test_count"]}판.',
        '성공률은 원래 시작 위치·원래 맵의 확률 선택 평가다. Greedy·무작위 비율을 따로 섞지 않았다.','',
        '## 적용 설정','',
        '- A와 동일: PPO, 방향×유지시간 9행동(0.1/0.25/0.6초), 물리 120Hz, 원래 시작점, 판당180초.',
        '- 학습률0.0003, gamma1.0, GAE0.98, 환경4, n_steps128, batch256, epochs최대5, clip0.2, target KL0.02, vf0.5, grad상한0.5.',
        '- 엔트로피 계수: 예산 처음20%는0.02, 80%까지0.005로, 끝에0.001로 감소. 보상은 학습에만 /100.',
        '- 기본 지형을 직사각형으로 병합한 관측35객체·핵심구간16개·점프6개·리프트2개를 빠짐없이 관측하도록 입력을1,984개로 확장했다. 기존 A는1,690개였다. 물리는 원래 기본플랫폼55개와 특수객체12개를 그대로 계산한다.',
        '- 특징 추출 방식과224특징→정책/가치망 각각128×128 구조는 A와 같다. 입력층 크기 변화에 따라 가중치 수는 달라진다.',
        '- 모든 모델은 무작위 초기화(행동별1/9)에서 학습했다. 기존 정책·시연·중간 시작·커리큘럼·증류를 사용하지 않았다.',
        '- 이전 튜토리얼 실험의 소스와 결과, 게임 맵과 물리를 수정하지 않았다. Git 커밋·푸시 없음.','',
        '## 보상','',
        '- 공통: 성공+100, 실패·시간초과−25 한 번, 특수 발판→지정 목표 실제 착지 최초+10. 일반 생존·왕복 자체 보상 없음.',
        '- 점프6개: 반동 후 비행 중 최대3초, 목표 높이/가로범위 새 진입 각각+0.5, 발판마다 최초1회.',
        '- 리프트2개: 최초착지+12, 높은 위치 이용+12, 목표 윗층 첫착지+18, 지정 접근288px에0.045/px로 최대12.96씩.',
        '- 왕복 맵의 다른 층에서 x좌표만 겹쳐 접근 보상을 받지 않도록 각 리프트 앞 발판 실제 착지로 접근 보상을 활성화했다. 이는 큰 맵에 맞춘 명시적 보상 적응이다.',
        '- 핵심구간은 독립 판정하며 번호 순서를 강제하지 않는다. 연결된 이동 발판 그룹 안에서 옮겨 다녀도+10을 주지 않는다.',
        '- 최대 성공 원점수는375.92다. 중간 보상을 모두 받은 뒤 실패한 점수는 최대250.92로 양수일 수 있다. 따라서 점수 상승과 클리어를 구분해서 해석한다.','',
        '## 시드별 결과','',
        '| 시드 | 선택본 성공 | 마지막 성공 | 선택본 평균 구간/16 | 마지막 평균 구간/16 | 학습 중 성공/전체 판 | 물리 프레임 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["seed"]} | {r["selected"]["success_rate"]*100:.0f}% | {r["latest"]["success_rate"]*100:.0f}% | '
            f'{r["selected"]["mean_milestones"]:.2f} | {r["latest"]["mean_milestones"]:.2f} | {r["training_successes"]}/{r["training_episodes"]} | {r["physics_ticks"]:,} |')
    lines+=['','## 선택본의 구간별 이용과 실제 목표 착지','',
        '선택본 시험을 모든 시드에서 합산했다. 출발 특수 발판에 착지한 판을 분모로 한 조건부 달성률이다. '
        '출발 경험 자체가 없으면 0% 대신 경험 없음으로 구분한다. 번호 순서를 강제하지 않는 원래 보상 판정과 같다.','',
        '| 번호 | 구간 | 출발 특수 발판 이용 판 | 목표 실제 착지 판 | 이용한 판 중 목표 달성 |',
        '|---|---|---:|---:|---:|']
    for i,(label,d) in enumerate(zip(LABELS,progress['selected']),1):
        rate = f'{d["completion_given_source"]*100:.1f}%' if d['completion_given_source'] is not None else '경험 없음'
        lines.append(f'| {i} | {label} | {d["source_episodes"]} | {d["completed_episodes"]} | {rate} |')
    lines+=['','## 기록에서 확인한 결과','',
        f'- 학습 중 성공 경험은 총 {sum(r["training_successes"] for r in rows):,}회다.',
        f'- 선택본 평균 핵심구간 달성 수는 {np.mean([r["selected"]["mean_milestones"] for r in rows]):.2f}/16, '
        f'마지막본은 {np.mean([r["latest"]["mean_milestones"] for r in rows]):.2f}/16이다.']
    for r in rows:
        lines.append(f'- 시드{r["seed"]}는 종료된 학습판 중 최대 {r["maximum_training_milestones"]}/16구간을 달성했다. '
                     '한 번의 최고 기록과 독립 시험의 평균·성공률을 구분한다.')
    if not any(r['selected']['successes'] or r['latest']['successes'] for r in rows):
        lines.append('- 이번 선택본·마지막본 시험에서는 완주가 한 번도 관측되지 않았다. 중간 보상이나 부분 구간 달성이 늘었어도 안정적인 완주 정책을 학습했다고 말할 수 없다.')
    for r in rows:
        for key,name in [('selected','선택본'),('latest','마지막본')]:
            ev = r[key]
            reached = [(i+1,LABELS[i],ev['milestone_rates'][s.id]) for i,s in enumerate(SPECS) if ev['milestone_rates'][s.id]>0]
            last_reached = f'{reached[-1][0]}번 {reached[-1][1]} ({reached[-1][2]*100:.0f}%)' if reached else '없음'
            lines.append(f'- 시드{r["seed"]} {name}: 관측된 최고 번호 목표는 {last_reached}. 실패 사유 {ev["failures"]}. '
                         f'한 판 평균 {ev["mean_episode_seconds"]:.2f}초, 성공판 평균 시간 '+
                         (f'{ev["mean_success_seconds"]:.2f}초.' if ev['mean_success_seconds'] is not None else '값 없음.'))
    lines+=['','## 그래프 해석','',
        f'- 왼쪽 위: 시점당{plan["validation_count"]}판 검증 성공률. 전체 맵을 끝까지 클리어한 비율이다.',
        '- 오른쪽 위: 전체 보상(실선)과 공통 보상(점선). 전체 보상만 오르면 보조 보상 획득이 늘어난 것일 수 있다.',
        '- 왼쪽 아래: 실제 착지로 달성한 핵심구간 수.16개 달성 여부와 이글루 도착은 별도로 판정한다.',
        '- 오른쪽 아래: 학습 중 최근 최대200판 성공률. 탐색 중 수집한 기록이며 독립 시험 성공률과 구분한다.',
        '- 행동 진단: 한 판이 길어져도 클리어를 뜻하지 않는다. 실제 정책 엔트로피가 낮으면 특정 행동에 집중한 상태이며, 정답 행동에 집중했는지는 성공률로 확인해야 한다. 이 엔트로피 값과 설정한 엔트로피 계수는 서로 다르다.',
        '- 구간 열지도: 어느 목표 착지에서 도달률이 낮아지는지 확인한다. 각 셀은 해당 구간을 달성한 판의 비율이며 조건부 전이 성공률은 아니다.',
        '- 검증으로 선택한 모델과 마지막 모델을 함께 공개한다. 최종 시험은 모델 선택에 사용하지 않았다.',
        '- 성공 시간 평균은 성공한 판만 계산한다. 성공판이 없을 때는 0초 대신 값 없음으로 기록한다.',
        '- 이번 맵은 튜토리얼보다 길고 여러 블록을 연속 사용한다. 기존 튜토리얼 성공률과 과제가 달라 단순한 알고리즘 우열 비교로 해석하지 않는다.','',
        '![학습 곡선](learning_metrics.png)','', '![독립 시험](test_results.png)','',
        '![행동 진단](behavior_diagnostics.png)','', '![맵과 구간](map_milestones.png)','',
        f'평가 기록 검사: {dict(records)}. 학습 종료 기록 {training_count:,}개 보상 검사 통과. 원래 물리 대조 오차0.','']
    if replays:
        lines+=['## 최대 확률 행동 재생 진단','',
                '선택본을 시드별 한 판씩 재생한 실제 좌표다. 확률 선택100판의 성공률과 구분한다. 배경의 이동 발판은 초기 위치이며 점선은 전체 이동 범위다.','',
                '| 시드 | 선택본 최대 확률 1판 | 마지막본 최대 확률 1판 |',
                '|---|---|---|']
        for r in rows:
            lines.append(f'| {r["seed"]} | {"성공" if r["selected_greedy_success"] else "실패"} | '
                         f'{"성공" if r["latest_greedy_success"] else "실패"} |')
        lines.append('')
        for replay in replays:
            e = replay['result']
            lines.append(f'- 시드{replay["seed"]}: 성공 {e["success"]}, 구간 {len(e["milestones"])}/16, '
                         f'{e["seconds"]:.2f}초, 종료 사유 {e["reason"]}.')
        lines+=['','![실제 정책 재생 좌표](policy_paths.png)','']
    (root/'결과보고서.md').write_text('\n'.join(lines),encoding='utf-8')
    for name,expected in plan['sources'].items():
        assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()==expected
    artifacts = ['plan.json','completion.json','aggregate.json','learning_metrics.png','test_results.png','behavior_diagnostics.png','map_milestones.png','결과보고서.md']
    if replays:artifacts += ['policy_paths.png']+[f'replay_seed_{r["seed"]}.json' for r in replays]
    analysis_sources = {}
    for name in ('rl/map_a_report.py','rl/map_a_replay.py','rl/map_a_status.py','tests/test_map_a_rl.py'):
        target = root/'analysis_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(name,target)
        analysis_sources[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    for name in ('initialization_audit.json','implementation_checks.json'):
        if (root/name).exists():
            assert read(root/name)['passed'];artifacts.append(name)
    write(root/'final_review.json',dict(passed=True,models=len(rows),evaluation_records=dict(records),
        training_records_audited=training_count,original_native_physics_max_error=0.,
        model_weights_finite=True,model_step_counts_verified=True,initial_model_hashes_verified=True,
        selected_using_validation_only=True,source_snapshots_verified=True,sources_unchanged=True,
        checkpoint_sha256=checkpoint_hashes,sources=sources(),analysis_sources=analysis_sources,
        artifact_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in artifacts},
        record_note='동일한 가중치의 평가를 재사용한 저장 항목도 포함한다.',git_actions=False))
    print(root/'결과보고서.md',flush=True)


if __name__=='__main__':main()
