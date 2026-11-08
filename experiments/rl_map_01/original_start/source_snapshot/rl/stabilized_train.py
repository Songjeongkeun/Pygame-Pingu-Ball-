"""2·4번의 행동·관측, 보상 축소, PPO 안정화를 시드 3개로 비교한다."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from .combined_train import sources as original_sources
from .five_train import read, write
from .five_models import action_probs
from .stabilized_env import StabilizedEnv, OBS_SIZE, EXTRA_FIELDS, ACTIONS, audit_reward
from .stabilized_models import StabilizedPPO, StabilizedExtractor, FixedRate

PROTOCOL = dict(success=100., failure=-25., timeout=-25., milestone=10., reward_scale=.01,
    limit_seconds=180., physics_hz=120, durations_ticks=[12,30,72], action_count=9,
    observation_size=OBS_SIZE, extra_fields=EXTRA_FIELDS, gamma=1., gae_lambda=.98,
    environments=4, n_steps=128, batch_size=256,
    reduced_stage4=dict(lift_first=2., lift_high=2., upper_first=3., approach_cap=2.),
    stages=[2,4], previous_weights_loaded=False, demonstrations=False,
    original_spawn=True, git_actions=False,
    comparison='A=행동·관측 변경, B=A+4번 보조 보상 축소, C=B+갱신·탐색 안정화',
    stage2_B_reuses_A=True, primary_evaluation='categorical_policy_sampling',
    validation_seed=11000, test_seed=910000,
    final_tests_used_for_selection=False, validation_count=20,
    stable_entropy_trigger=.5, stable_drop_threshold=.2, stable_drop_patience=3,
    stable_drop_minimum_best=.5)


def sources():
    result = original_sources()
    for name in ('stabilized_env.py','stabilized_models.py','stabilized_train.py'):
        p = 'rl/'+name
        result[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return result


def settings(variant):
    return (dict(learning_rate=.0001, final_learning_rate=.00003, clip_range=.1,
                 target_kl=.01, n_epochs=3) if variant == 'C' else
            dict(learning_rate=.0003, final_learning_rate=.0003, clip_range=.2,
                 target_kl=.02, n_epochs=5))


def make_model(vec, seed, variant):
    s = settings(variant)
    model = StabilizedPPO('MlpPolicy', vec, seed=seed, device='cpu', verbose=0,
        gamma=1., gae_lambda=.98, n_steps=128, batch_size=256,
        learning_rate=s['learning_rate'], n_epochs=s['n_epochs'],
        clip_range=s['clip_range'], target_kl=s['target_kl'],
        ent_coef=.03 if variant=='C' else .02, vf_coef=.5, max_grad_norm=.5,
        policy_kwargs=dict(features_extractor_class=StabilizedExtractor,
                           net_arch=dict(pi=[128,128], vf=[128,128])))
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    return model


def evaluate(model, stage, variant, count=20, seed=11000, greedy=False, native=False):
    """학습 난수에 영향을 주지 않고 원래 시작점에서 별도 행동 난수로 평가한다."""
    from pingu_ball.engine import World
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    count_env = min(4,count)
    envs = [StabilizedEnv(stage=stage,variant=variant) for _ in range(count_env)]
    observations = [env.reset()[0] for env in envs]
    rngs = [np.random.default_rng(seed+stage*100003+i*104729) for i in range(count_env)]
    remaining = [count//count_env+int(i<count%count_env) for i in range(count_env)]
    worlds = [World(env.unwrapped.level,auto_reset=False) for env in envs] if native else []
    rows, error, usage = [], 0., np.zeros(9,dtype=np.int64)
    try:
        while any(remaining):
            active = [i for i,n in enumerate(remaining) if n]
            probs = action_probs(model,np.stack([observations[i] for i in active]))
            for i,p in zip(active,probs):
                p = p/p.sum()
                action = int(p.argmax()) if greedy else int(rngs[i].choice(9,p=p))
                usage[action] += 1
                raw = envs[i].unwrapped
                def tick(world, number, direction):
                    nonlocal error
                    other = worlds[i]
                    other.step(direction,min(other.physics.dt,180.-other.time))
                    for name in ('x','y','vx','vy'):
                        error = max(error,abs(getattr(world.ball,name)-getattr(other.ball,name)))
                    assert error < 1e-8
                    assert world.cleared==other.cleared and world.failed==other.failed
                _,_,done,_,info = raw.step(action,on_tick=tick if native else None)
                observations[i] = envs[i].observation(raw.observe())
                if done:
                    audit_reward(info)
                    rows.append(info)
                    remaining[i] -= 1
                    if remaining[i]:
                        observations[i] = envs[i].reset()[0]
                        if native: worlds[i].reset()
    finally:
        for env in envs: env.close()
        model.policy.set_training_mode(was_training)
    wins = [r['seconds'] for r in rows if r['success']]
    result = dict(episodes=len(rows),successes=len(wins),success_rate=len(wins)/len(rows),
        mean_success_seconds=float(np.mean(wins)) if wins else None,
        mean_return=float(np.mean([r['episode_return'] for r in rows])),
        mean_common_return=float(np.mean([sum(r['reward_parts'].get(k,0) for k in ('success','failure','milestone')) for r in rows])),
        mean_milestones=float(np.mean([len(r['milestones']) for r in rows])),
        source_rate=float(np.mean([bool(r['armed']) for r in rows])),
        failures=dict(Counter(r['reason'] for r in rows if not r['success'])),
        actions=usage.tolist(),raw=rows,seed=seed,
        protocol='greedy' if greedy else 'categorical_policy_sampling',
        native_comparison=native,max_native_error=error if native else None)
    if stage==4:
        result['phase_rates']={k:float(np.mean([r['auxiliary_state'][k] for r in rows]))
                               for k in ('lift_first','lift_high','upper_first')}
    else:
        result['milestone_rates']={key:float(np.mean([key in r['milestones'] for r in rows]))
                                  for key in ('spring-to-middle','spring-to-upper','spring-to-finish')}
    return result


class TrainingStats(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ticks = self.episodes = self.successes = 0
        self.parts = Counter()
        self.actions = np.zeros(9,dtype=np.int64)

    def _on_step(self):
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            self.ticks += info['tick_delta']
            self.actions[info['action_index']] += 1
            if done:
                audit_reward(info)
                self.episodes += 1
                self.successes += int(info['success'])
                self.parts.update(info['reward_parts'])
        return True


def score(evaluation):
    return (evaluation['success_rate'],evaluation['mean_milestones'],
            evaluation['source_rate'],-(evaluation['mean_success_seconds'] or 180.))


def coefficients(variant, ticks, budget, entropy_trigger=None):
    progress = min(1.,ticks/budget)
    if variant=='C':
        rate = .0001+(.00003-.0001)*progress
        if entropy_trigger is None:
            entropy = .03
        else:
            fraction = min(1.,max(0.,(ticks-entropy_trigger)/max(budget-entropy_trigger,1000000.)))
            entropy = .01+(.003-.01)*fraction
    else:
        rate = .0003
        if progress<=.2: entropy=.02
        elif progress<=.8: entropy=.02+(.005-.02)*(progress-.2)/.6
        else: entropy=.005+(.001-.005)*(progress-.8)/.2
    return rate,entropy


def train_job(payload):
    root,variant,stage,seed,budget,interval,validation_count,test_count = payload
    torch.set_num_threads(1)
    folder=Path(root)/variant/f'stage_{stage:02d}'/f'seed_{seed}'
    if (folder/'summary.json').exists():
        assert read(folder/'config.json')['sources']==sources()
        return str(folder/'summary.json')
    if (folder/'config.json').exists():
        raise RuntimeError(f'미완료 결과는 덮어쓰지 않는다: {folder}')
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'console.log').open('w',encoding='utf-8',buffering=1) as log, \
            contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        vec=DummyVecEnv([lambda:StabilizedEnv(stage=stage,variant=variant) for _ in range(4)])
        try:
            model=make_model(vec,seed,variant)
            model.set_logger(configure(str(folder),['csv']))
            config=dict(stage=stage,variant=variant,seed=seed,from_scratch=True,
                previous_models_loaded=[],physics_budget=budget,validation_interval_ticks=interval,
                validation_count=validation_count,test_count=test_count,protocol=PROTOCOL,
                settings=settings(variant),sources=sources())
            write(folder/'config.json',config)
            model.save(folder/'initial')
            collector=TrainingStats()
            started=time.perf_counter()
            history=[]
            best_score=(-1.,-1.,-1.,-1e9)
            best_step=0
            next_eval=interval
            entropy_trigger=None
            best_rate=0.
            drops=0
            stopped=False

            def checkpoint():
                nonlocal best_score,best_step,entropy_trigger,best_rate,drops,stopped
                ev=evaluate(model,stage,variant,validation_count,11000)
                candidate=score(ev)
                if candidate>=best_score:
                    best_score,best_step=candidate,model.num_timesteps
                    model.save(folder/'best')
                    write(folder/'best_validation.json',dict(step=best_step,physics_ticks=collector.ticks,evaluation=ev))
                if variant=='C':
                    if entropy_trigger is None and ev['success_rate']>=.5:
                        entropy_trigger=collector.ticks
                    drops=drops+1 if best_rate>=.5 and ev['success_rate']<=best_rate-.2+1e-9 else 0
                    best_rate=max(best_rate,ev['success_rate'])
                    stopped=drops>=3
                row=dict(decisions=model.num_timesteps,physics_ticks=collector.ticks,
                    episodes=collector.episodes,training_successes=collector.successes,
                    validation=ev,entropy=model.ent_coef,learning_rate=model.lr_schedule(0.),
                    entropy_trigger_ticks=entropy_trigger,consecutive_drops=drops,
                    training_reward_parts=dict(collector.parts),
                    diagnostics={k:float(v) for k,v in model.logger.name_to_value.items()
                        if k.startswith('train/') and np.isscalar(v) and np.isfinite(v)},
                    elapsed_seconds=time.perf_counter()-started)
                history.append(row)
                write(folder/'metrics.json',history)
                model.save(folder/'latest')
                write(folder/'live.json',dict(stage=stage,variant=variant,seed=seed,
                      decisions=model.num_timesteps,physics_ticks=collector.ticks,
                      validation_success_rate=ev['success_rate'],best_validation_success_rate=best_score[0],
                      training_episodes=collector.episodes,training_successes=collector.successes,
                      elapsed_seconds=row['elapsed_seconds']))
                model.logger.dump(step=model.num_timesteps)
                print(f'물리틱={collector.ticks} 행동={model.num_timesteps} 검증={ev["success_rate"]:.2f} 성공경험={collector.successes}',flush=True)

            checkpoint()
            while collector.ticks<budget and not stopped:
                rate,entropy=coefficients(variant,collector.ticks,budget,entropy_trigger)
                model.learning_rate=rate
                model.lr_schedule=FixedRate(rate)
                model.ent_coef=entropy
                # 각 호출은 완전한 새 온폴리시 롤아웃 1개와 PPO 갱신을 수행한다.
                # reset_num_timesteps=False이므로 게임 상태와 학습 난수 흐름을 이어간다.
                model.learn(total_timesteps=model.n_steps*4,reset_num_timesteps=False,
                            callback=collector,log_interval=None)
                if collector.ticks>=next_eval or collector.ticks>=budget:
                    checkpoint()
                    next_eval+=interval
            if history[-1]['decisions']!=model.num_timesteps:
                checkpoint()
            training_seconds=time.perf_counter()-started
            best=StabilizedPPO.load(folder/'best.zip',device='cpu')
            final=evaluate(best,stage,variant,test_count,910000+seed)
            same=all(torch.equal(v,best.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
            last=final if same else evaluate(model,stage,variant,test_count,910000+seed)
            selected_greedy=evaluate(best,stage,variant,1,123,greedy=True,native=True)
            latest_greedy=selected_greedy if same else evaluate(model,stage,variant,1,123,greedy=True,native=True)
            assert config['sources']==sources()
            write(folder/'summary.json',dict(config=config,final=final,latest_test=last,
                selected_greedy=selected_greedy,latest_greedy=latest_greedy,
                latest_matches_selected_weights=same,best_step=best_step,
                best_validation=read(folder/'best_validation.json'),
                decisions=model.num_timesteps,physics_ticks=collector.ticks,
                training_episodes=collector.episodes,training_successes=collector.successes,
                training_reward_parts=dict(collector.parts),
                stop_reason='validation_drop' if stopped else 'physics_budget',
                training_seconds=training_seconds,elapsed_seconds=time.perf_counter()-started,
                sources_after=sources(),test_used_for_selection=False))
            print(f'완료 선택본={final["success_rate"]:.2f} 마지막={last["success_rate"]:.2f}',flush=True)
        finally:
            vec.close()
    return str(folder/'summary.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--physics-ticks',type=int,default=6000000)
    parser.add_argument('--eval-every',type=int,default=500000)
    parser.add_argument('--validation-count',type=int,default=20)
    parser.add_argument('--test-count',type=int,default=100)
    parser.add_argument('--seeds',default='0,1,2')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if min(args.physics_ticks,args.eval_every,args.validation_count,args.test_count,args.workers)<=0:
        parser.error('예산과 평가 횟수는 양수여야 한다')
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    seeds=[int(s) for s in args.seeds.split(',')]
    plan=dict(protocol=PROTOCOL,physics_budget=args.physics_ticks,eval_every=args.eval_every,
        seeds=seeds,validation_count=args.validation_count,test_count=args.test_count,
        stages=[2,4],variants=['A','B','C'],expected_models=5*len(seeds),sources=sources(),
        actual_jobs=dict(A=[2,4],B=[4],C=[2,4]),stage2_B_same_as_A=True,
        paired_initialization=True,early_stop_only_C=True,
        execution_order='독립 조건을 공통 작업 대기열에서 병렬 실행',
        budget_overshoot='완전한 128×4행동 롤아웃 1개 범위 이내',
        historical_comparison='output/rl_combined_reward_20261101; 경험 예산·행동·관측이 함께 달라짐')
    if (root/'plan.json').exists():
        assert read(root/'plan.json')==plan
    else:
        write(root/'plan.json',plan)
        for p in plan['sources']:
            target=root/'source_snapshot'/p
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
        write(root/'versions.json',{name:importlib.metadata.version(name)
              for name in ('torch','stable-baselines3','numpy','gymnasium')})
    started=time.perf_counter();completed=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # 같은 시드의 A·B·C도 가중치를 공유하지 않으므로 실행 간 의존성이 없다.
        tasks=[(str(root),variant,stage,seed,args.physics_ticks,args.eval_every,
                args.validation_count,args.test_count) for variant,stages in [('A',[2,4]),('B',[4]),('C',[2,4])]
               for seed in seeds for stage in stages]
        write(root/'progress.json',dict(phase='parallel_conditions',completed=[],total=plan['expected_models']))
        futures=[pool.submit(train_job,p) for p in tasks]
        for future in as_completed(futures):
            completed.append(future.result())
            assert plan['sources']==sources()
            write(root/'progress.json',dict(phase='parallel_conditions',completed=completed,
                total=plan['expected_models'],elapsed_seconds=time.perf_counter()-started))
            print(f'완료 {len(completed)}/{plan["expected_models"]}: {completed[-1]}',flush=True)
        for variant in ('A','B','C'):
            write(root/f'phase_{variant}_complete.json',dict(complete=True,
                  summaries=[p for p in completed if Path(p).parts[-4]==variant]))
    write(root/'completion.json',dict(complete=True,models=len(completed),sources=sources(),
        elapsed_seconds=time.perf_counter()-started,summaries=completed))
    print('모든 단계별 학습·평가 완료',flush=True)


if __name__=='__main__':
    main()
