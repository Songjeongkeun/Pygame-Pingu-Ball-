"""보상을 유지하고 튜토리얼 2·4의 탐색과 정책 갱신 조건을 단계적으로 비교한다."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from .five_train import read, write
from .five_models import action_probs
from .milestone_env import MilestoneEnv
from .milestone_models import MilestonePPO, MilestoneExtractor


VARIANTS = {
    'stable': dict(n_steps=128, gae_lambda=.98, repeat=6),
    'long': dict(n_steps=512, gae_lambda=.995, repeat=6),
    'hold': dict(n_steps=512, gae_lambda=.995, repeat=12),
}
LABELS = {'stable': '탐색·갱신 안정화', 'long': '긴 경험 수집', 'hold': '0.1초 행동 유지'}


def sources():
    """실행 폴더의 실제 코드 해시를 고정해 학습 중 수정과 결과 혼합을 막는다."""
    files = [p for folder in ('pingu_ball', 'rl') for p in Path(folder).glob('*.py')]
    return {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def vector(stage, repeat, count=4):
    return DummyVecEnv([lambda: MilestoneEnv(stage=stage, repeat=repeat) for _ in range(count)])


def make_model(vec, seed, settings):
    """새 정책을 균등한 행동 확률로 시작한다. 이전 가중치와 사람의 입력은 사용하지 않는다."""
    model = MilestonePPO('MlpPolicy', vec, seed=seed, device='cpu', verbose=0,
        gamma=1., learning_rate=.0001, gae_lambda=settings['gae_lambda'],
        ent_coef=.03, vf_coef=.5, max_grad_norm=.5,
        n_steps=settings['n_steps'], batch_size=min(256,settings['n_steps']*4), n_epochs=3,
        target_kl=.01, clip_range=.1,
        policy_kwargs=dict(features_extractor_class=MilestoneExtractor,
                           net_arch=dict(pi=[128, 128], vf=[128, 128])))
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    return model


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def audit_episode(info):
    """수집한 판의 원래 점수가 공통 보상 규칙과 정확히 일치하는지 검사한다."""
    completed = info['milestones']
    assert len(completed) == len(set(completed))
    expected = (100. if info['success'] else -25.) + 10*len(completed)
    assert abs(info['episode_return']-expected) < 1e-7
    assert abs(info['learning_return']-expected/100.) < 1e-7
    assert len(info['milestone_events']) == len(completed)


def statistics(rows):
    for row in rows:
        audit_episode(row)
    wins = [r['seconds'] for r in rows if r['success']]
    return dict(episodes=len(rows), successes=len(wins), success_rate=len(wins)/len(rows),
        mean_success_seconds=float(np.mean(wins)) if wins else None,
        mean_return=float(np.mean([r['episode_return'] for r in rows])),
        mean_milestones=float(np.mean([len(r['milestones']) for r in rows])),
        milestone_rate=float(np.mean([bool(r['milestones']) for r in rows])),
        source_landing_rate=float(np.mean([bool(r['armed']) for r in rows])),
        mean_sources=float(np.mean([len(r['armed']) for r in rows])),
        mean_seconds=float(np.mean([r['seconds'] for r in rows])),
        failures=dict(Counter(r['reason'] for r in rows if not r['success'])))


def evaluate(model, stage, repeat, count=20, seed=11000, greedy=False):
    """독립 난수로 평가한다. 원래 시작점·180초·동일 보상 규칙을 사용한다."""
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    vec = vector(stage, repeat)
    rows, entropies, actions_count = [], [], np.zeros(3, dtype=np.int64)
    try:
        rngs = [np.random.default_rng(seed+stage*100003+i*104729) for i in range(4)]
        remaining = [count//4+int(i<count%4) for i in range(4)]
        obs = vec.reset()
        while sum(remaining):
            probabilities = action_probs(model, obs)
            actions = []
            for i, (rng, p) in enumerate(zip(rngs, probabilities)):
                p = p/p.sum()
                action = int(p.argmax()) if greedy else int(rng.choice(3, p=p))
                actions.append(action)
                if remaining[i]:
                    entropies.append(float(-(p*np.log(p.clip(1e-12))).sum()))
                    actions_count[action] += 1
            obs, _, done, infos = vec.step(np.asarray(actions))
            for i in range(4):
                if done[i] and remaining[i]:
                    rows.append({k:v for k,v in infos[i].items()
                                 if k not in ('terminal_observation','TimeLimit.truncated')})
                    remaining[i] -= 1
    finally:
        vec.close()
        model.policy.set_training_mode(was_training)
    return dict(stage=stage, seed=seed, mode='greedy' if greedy else 'sampled',
                **statistics(rows), mean_entropy=float(np.mean(entropies)),
                action_counts=actions_count.tolist(), raw=rows)


def success_score(ev):
    """클리어를 최우선으로 두고, 동률일 때 실제 구간 진행도를 비교한다."""
    return (ev['success_rate'], ev['mean_milestones'], ev['milestone_rate'],
            ev['source_landing_rate'], -(ev['mean_success_seconds'] or 180.))


def progress_score(ev):
    """중간 발전을 별도로 보존한다. 시작 착지는 선택 지표이며 학습 보상이 아니다."""
    return (ev['mean_milestones'], ev['milestone_rate'], ev['mean_sources'],
            ev['success_rate'], -(ev['mean_success_seconds'] or 180.))


class RecoveryMonitor(BaseCallback):
    def __init__(self, folder, stage, settings, tick_budget, eval_every, eval_count):
        super().__init__()
        self.folder, self.stage, self.settings = Path(folder), stage, settings
        self.tick_budget, self.eval_every, self.eval_count = tick_budget, eval_every, eval_count
        self.ticks = self.episodes = self.wins = self.next_eval = self.streak = 0
        self.best_success = self.best_progress = None
        self.rows = []
        self.started = time.perf_counter()
        self.episode_log = (self.folder/'episodes.jsonl').open('a',encoding='utf-8',buffering=1)

    def _on_step(self):
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            self.ticks += info['tick_delta']
            if done:
                audit_episode(info)
                self.episodes += 1
                self.wins += int(info['success'])
                row = {k:info[k] for k in ('stage','success','reason','seconds',
                    'episode_return','learning_return','armed','milestones','milestone_events')}
                row.update(training_ticks=self.ticks, decisions=self.model.num_timesteps)
                self.episode_log.write(json.dumps(row,ensure_ascii=False)+'\n')
        return True

    def checkpoint(self, model):
        ev = evaluate(model, self.stage, self.settings['repeat'], self.eval_count, 11000)
        good = ev['success_rate'] >= .6
        self.streak = self.streak+1 if good else 0
        # 정해진 학습 시간 대신, 세 번 이상 연속으로 성공을 확인한 뒤에만 탐색을 줄인다.
        model.ent_coef = .03-.02*min(1.,max(0,self.streak-2)/3)
        scores = dict(best_success=success_score(ev), best_progress=progress_score(ev))
        for name,score in scores.items():
            old = getattr(self,name)
            if old is None or score > tuple(old['score']):
                value=dict(score=score,step=model.num_timesteps,ticks=self.ticks,evaluation=ev)
                setattr(self,name,value)
                model.save(self.folder/name)
                write(self.folder/(name+'_validation.json'),value)
        model.save(self.folder/'latest')
        model.save(self.folder/'checkpoints'/f'step_{model.num_timesteps:08d}')
        diagnostics={k:float(v) for k,v in model.logger.name_to_value.items()
                     if k.startswith('train/') and np.isscalar(v) and np.isfinite(v)}
        row=dict(decisions=model.num_timesteps,physics_ticks=self.ticks,episodes=self.episodes,
            training_successes=self.wins,entropy_coefficient=model.ent_coef,validation=ev,
            diagnostics=diagnostics,elapsed_seconds=time.perf_counter()-self.started)
        self.rows.append(row)
        write(self.folder/'metrics.json',self.rows)
        write(self.folder/'live.json',{k:v for k,v in row.items() if k!='validation'} |
              dict(validation={k:v for k,v in ev.items() if k!='raw'}))
        model.logger.dump(step=model.num_timesteps)
        print(json.dumps(dict(step=model.num_timesteps,ticks=self.ticks,
            wins=self.wins,validation=ev['success_rate'],milestones=ev['mean_milestones'],
            source=ev['source_landing_rate'],entropy=ev['mean_entropy']),ensure_ascii=False),flush=True)

    def after_update(self,model):
        if self.ticks >= self.next_eval:
            self.checkpoint(model)
            self.next_eval = (self.ticks//self.eval_every+1)*self.eval_every


def run_job(payload):
    root,variant,stage,seed,tick_budget,eval_every,eval_count,test_count,settings = payload
    torch.set_num_threads(1)
    folder=Path(root)/variant/f'stage_{stage:02d}'/f'seed_{seed}'
    if (folder/'summary.json').exists():
        summary=read(folder/'summary.json')
        assert summary['config']['sources']==sources()
        return str(folder/'summary.json')
    if (folder/'config.json').exists():
        raise RuntimeError(f'Incomplete run retained: {folder}')
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'checkpoints').mkdir()
    with (folder/'console.log').open('w',encoding='utf-8',buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        vec=vector(stage,settings['repeat'])
        monitor=model=None
        try:
            model=make_model(vec,seed,settings)
            model.set_logger(configure(str(folder),['csv']))
            config=dict(algorithm='ppo',variant=variant,stage=stage,seed=seed,
                reward_mode='milestone',from_scratch=True,previous_models_loaded=[],
                original_spawn=True,limit=180.,settings=settings,sources=sources(),
                gamma=1.,learning_rate=.0001,n_epochs=3,clip_range=.1,target_kl=.01,
                entropy_initial=.03,entropy_min=.01,entropy_decay='3 consecutive validations >=60%',
                physics_tick_budget=tick_budget,physics_hz=120,action_seconds=settings['repeat']/120,
                validation_count=eval_count,selection_validation_count=50,test_count=test_count,
                reward=dict(success=100,failure_or_timeout=-25,milestone=10,scale=.01),
                initial_parameter_hash=parameter_hash(model),final_test_used_for_selection=False)
            write(folder/'config.json',config)
            model.save(folder/'initial')
            monitor=RecoveryMonitor(folder,stage,settings,tick_budget,eval_every,eval_count)
            model.after_update=monitor.after_update
            monitor.checkpoint(model)
            monitor.next_eval=eval_every
            # 완성한 수집 묶음마다 갱신을 마친다. 마지막 일부 경험을 버리지 않는다.
            while monitor.ticks < tick_budget:
                model.learn(total_timesteps=settings['n_steps']*4,reset_num_timesteps=False,
                            callback=monitor,log_interval=None)
            if monitor.rows[-1]['decisions']!=model.num_timesteps:
                monitor.checkpoint(model)
            # 최종 시험을 보기 전에 별도 50판 검증으로 저장본을 선택한다.
            candidates=[]
            cached={}
            for name in ('best_success','best_progress'):
                path=folder/(name+'.zip')
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                candidate=MilestonePPO.load(path,device='cpu')
                ev=cached.get(digest)
                if ev is None:
                    ev=evaluate(candidate,stage,settings['repeat'],50,3510000+seed)
                    cached[digest]=ev
                candidates.append(dict(name=name,checkpoint=str(path.resolve()),
                                       sha256=digest,validation=ev,score=success_score(ev)))
            chosen=max(candidates,key=lambda r:r['score'])
            write(folder/'selection.json',dict(selected=chosen,candidates=candidates,
                  final_test_used_for_selection=False))
            selected=MilestonePPO.load(chosen['checkpoint'],device='cpu')
            final=evaluate(selected,stage,settings['repeat'],test_count,910000+seed)
            greedy=evaluate(selected,stage,settings['repeat'],1,1910000+seed,greedy=True)
            assert sources()==config['sources']
            result=dict(config=config,decisions=model.num_timesteps,physics_ticks=monitor.ticks,
                simulated_seconds=monitor.ticks/120,episodes=monitor.episodes,
                training_successes=monitor.wins,selection=chosen,final=final,greedy=greedy,
                elapsed_seconds=time.perf_counter()-monitor.started,sources_after=sources())
            write(folder/'summary.json',result)
        finally:
            if monitor is not None:monitor.episode_log.close()
            if model is not None:model.logger.close()
            vec.close()
    return str(folder/'summary.json')


def run_phase(root,variant,stages,args,completed):
    tasks=[(str(root),variant,stage,seed,args.tick_budget,args.eval_every,args.eval_count,
            args.test_count,VARIANTS[variant]) for stage in stages for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_job,p) for p in tasks]
        for future in as_completed(futures):
            path=future.result()
            completed.append(path)
            write(root/'progress.json',dict(phase=variant,completed=completed,
                  active_stages=stages,total_this_phase=len(tasks)))
            print('FINISHED '+path,flush=True)
    failed=[]
    gates=[]
    for stage in stages:
        rows=[read(root/variant/f'stage_{stage:02d}'/f'seed_{s}'/'summary.json') for s in args.seeds]
        # 후속 조건 실행 여부는 시험 점수가 아닌 별도 검증 결과만 사용한다.
        rates=[r['selection']['validation']['success_rate'] for r in rows]
        passed=np.mean(rates)>=.8 and min(rates)>=.5
        if not passed:failed.append(stage)
        gates.append(dict(stage=stage,validation_rates=rates,passed=bool(passed),
                          gate='mean >=80% and every seed >=50%',final_test_used=False))
    write(root/(variant+'_gate.json'),gates)
    return failed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--tick-budget',type=int,default=2400000)
    p.add_argument('--eval-every',type=int,default=240000)
    p.add_argument('--eval-count',type=int,default=20)
    p.add_argument('--test-count',type=int,default=100)
    p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    args.root=args.root.resolve()
    args.seeds=[int(s) for s in args.seeds.split(',')]
    args.root.mkdir(parents=True,exist_ok=True)
    plan=dict(stages=[2,4],seeds=args.seeds,variants=VARIANTS,
        phase_order=['stable','long','hold'],tick_budget=args.tick_budget,
        eval_every=args.eval_every,eval_count=args.eval_count,test_count=args.test_count,
        settings='PPO lr=1e-4, epochs=3, clip=.1, target_kl=.01, gamma=1, entropy .03 adaptive',
        sources=sources(),reward_unchanged=True,original_spawn=True,
        authored_actions=False,previous_weights_loaded=False,git_actions=False,
        adaptive_stage_gate='validation mean>=.8 and minimum seed>=.5; otherwise next condition',
        budget_note='same physical-tick target; at most one complete rollout overshoot; actual ticks reported')
    if (args.root/'plan.json').exists():assert read(args.root/'plan.json')==plan
    else:write(args.root/'plan.json',plan)
    completed=[]
    stages=[2,4]
    for variant in ('stable','long','hold'):
        if not stages:
            write(args.root/(variant+'_skipped.json'),dict(reason='Previous validation gate passed'))
            continue
        stages=run_phase(args.root,variant,stages,args,completed)
    write(args.root/'completion.json',dict(complete=True,completed=completed,
          unresolved_stages=stages,sources=sources(),shared_policy_trained=False))
    print('TRAINING COMPLETE',flush=True)


if __name__=='__main__':main()
