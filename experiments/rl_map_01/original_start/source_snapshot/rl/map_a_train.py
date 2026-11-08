"""강화학습 탭 1번을 A 설정의 PPO로 처음부터 학습·검증·시험한다."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from pingu_ball.engine import World
from .five_models import action_probs
from .five_train import read, write
from .map_a_env import MapAEnv, SPECS, LABELS, OBS_SIZE, CAPACITY, SPRINGS, LIFTS, audit_reward
from .map_a_models import make_model, load_policy
from .stabilized_models import FixedRate
from .stabilized_train import sources as base_sources, coefficients


def sources():
    result = base_sources()
    for name in ('rl/map_a_env.py','rl/map_a_models.py','rl/map_a_train.py',
                 'pingu_ball/rl_maps.py','assets/rl_maps/layouts.json'):
        result[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    return result


def evaluate(model, count=20, seed=11000, greedy=False, native=False):
    """학습 난수와 분리된 확률 선택으로 원래 시작점에서 평가한다."""
    training = model.policy.training
    model.policy.set_training_mode(False)
    envs = [MapAEnv() for _ in range(min(4,count))]
    observations = [env.reset()[0] for env in envs]
    rngs = [np.random.default_rng(seed+i*104729) for i in range(len(envs))]
    remaining = [count//len(envs)+int(i<count%len(envs)) for i in range(len(envs))]
    worlds = [World(env.unwrapped.level,auto_reset=False) for env in envs] if native else []
    rows,error,usage = [],0.,np.zeros(9,dtype=np.int64)
    try:
        while any(remaining):
            active = [i for i,n in enumerate(remaining) if n]
            probabilities = action_probs(model,np.stack([observations[i] for i in active]))
            for i,p in zip(active,probabilities):
                action = int(p.argmax()) if greedy else int(rngs[i].choice(9,p=p/p.sum()))
                usage[action] += 1
                def tick(world,number,direction):
                    nonlocal error
                    other = worlds[i]
                    other.step(direction,min(other.physics.dt,180.-other.time))
                    for field in ('x','y','vx','vy'):
                        error = max(error,abs(getattr(world.ball,field)-getattr(other.ball,field)))
                    assert error<1e-8
                    assert world.cleared==other.cleared and world.failed==other.failed
                raw = envs[i].unwrapped
                _,_,done,_,info = raw.step(action,on_tick=tick if native else None)
                observations[i] = envs[i].observation(raw.observe())
                if done:
                    audit_reward(info);rows.append(info);remaining[i] -= 1
                    if remaining[i]:
                        observations[i] = envs[i].reset()[0]
                        if native:worlds[i].reset()
    finally:
        for env in envs:env.close()
        model.policy.set_training_mode(training)
    wins = [r['seconds'] for r in rows if r['success']]
    return dict(episodes=len(rows),successes=len(wins),success_rate=len(wins)/len(rows),
        mean_return=float(np.mean([r['episode_return'] for r in rows])),
        mean_common_return=float(np.mean([sum(r['reward_parts'].get(k,0.) for k in ('success','failure','milestone')) for r in rows])),
        mean_milestones=float(np.mean([len(r['milestones']) for r in rows])),
        mean_success_seconds=float(np.mean(wins)) if wins else None,
        mean_episode_seconds=float(np.mean([r['seconds'] for r in rows])),
        source_rate=float(np.mean([bool(r['armed']) for r in rows])),
        milestone_rates={s.id:float(np.mean([s.id in r['milestones'] for r in rows])) for s in SPECS},
        failures=dict(Counter(r['reason'] for r in rows if not r['success'])),
        actions=usage.tolist(),seed=seed,protocol='greedy' if greedy else 'categorical_policy_sampling',
        native_comparison=native,max_native_error=error if native else None,raw=rows)


class TrainingStats(BaseCallback):
    def __init__(self, folder):
        super().__init__()
        self.ticks = self.episodes = self.successes = 0
        self.parts = Counter()
        self.folder = folder

    def _on_step(self):
        records = []
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            self.ticks += info['tick_delta']
            if done:
                audit_reward(info)
                self.episodes += 1;self.successes += int(info['success'])
                self.parts.update(info['reward_parts'])
                keys = ('success','reason','seconds','decisions','milestones','armed',
                        'episode_return','learning_return','reward_parts')
                records.append(dict(episode=self.episodes,physics_ticks=self.ticks,
                                    **{k:info[k] for k in keys}))
        if records:
            with (self.folder/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                for record in records:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        return True


def selection_score(ev):
    return ev['success_rate'],ev['mean_milestones'],ev['source_rate'],-(ev['mean_success_seconds'] or 180.)


def train_job(payload):
    root,seed,budget,interval,validation_count,test_count = payload
    folder = Path(root)/f'seed_{seed}'
    if folder.exists():
        raise RuntimeError(f'기존 학습을 덮어쓰지 않는다: {folder}')
    folder.mkdir()
    torch.set_num_threads(1)
    with (folder/'console.log').open('w',encoding='utf-8',buffering=1) as stream, \
            contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        vec = DummyVecEnv([MapAEnv for _ in range(4)])
        try:
            model = make_model(vec,seed)
            model.set_logger(configure(str(folder),['csv']))
            config = dict(seed=seed,variant='A-map1',map_kind='reinforcement',map_number=1,
                observation_size=OBS_SIZE,from_scratch=True,previous_models_loaded=[],
                physics_budget=budget,validation_count=validation_count,test_count=test_count,
                sources=sources(),learning_rate=.0003,gamma=1.,gae_lambda=.98,n_envs=4,
                n_steps=128,batch_size=256,n_epochs=5,clip_range=.2,target_kl=.02,
                vf_coef=.5,max_grad_norm=.5,policy_layers=[128,128],reward_scale=.01,
                limit_seconds=180.,demonstrations=False,original_spawn=True)
            write(folder/'config.json',config);model.save(folder/'initial')
            observation = vec.reset()
            initial = action_probs(model,observation)[0]
            assert np.allclose(initial,np.full(9,1/9)) and model.num_timesteps==0
            # 초기 상태의 정책 확률과 가중치 해시를 학습 전에 기록한다.
            digest = hashlib.sha256()
            for name,value in sorted(model.policy.state_dict().items()):
                digest.update(name.encode());digest.update(value.cpu().numpy().tobytes())
            write(folder/'initialization.json',dict(steps=0,probabilities=initial.tolist(),
                  parameter_sha256=digest.hexdigest(),previous_models_loaded=[]))
            collector = TrainingStats(folder)
            history = []
            best_score = (-1.,-1.,-1.,-1e9)
            best_step = 0
            next_eval = interval
            started = time.perf_counter()

            def checkpoint():
                nonlocal best_score,best_step
                ev = evaluate(model,validation_count,11000)
                candidate = selection_score(ev)
                if candidate>=best_score:
                    best_score,best_step = candidate,model.num_timesteps
                    model.save(folder/'best')
                    write(folder/'best_validation.json',dict(step=best_step,physics_ticks=collector.ticks,evaluation=ev))
                row = dict(decisions=model.num_timesteps,physics_ticks=collector.ticks,
                    episodes=collector.episodes,training_successes=collector.successes,
                    validation=ev,entropy=model.ent_coef,learning_rate=.0003,
                    training_reward_parts=dict(collector.parts),elapsed_seconds=time.perf_counter()-started,
                    diagnostics={k:float(v) for k,v in model.logger.name_to_value.items()
                                 if k.startswith('train/') and np.isscalar(v) and np.isfinite(v)})
                history.append(row);write(folder/'metrics.json',history);model.save(folder/'latest')
                write(folder/'live.json',dict(seed=seed,physics_ticks=collector.ticks,decisions=model.num_timesteps,
                    training_episodes=collector.episodes,training_successes=collector.successes,
                    validation_success_rate=ev['success_rate'],mean_milestones=ev['mean_milestones'],
                    elapsed_seconds=row['elapsed_seconds']))
                model.logger.dump(step=model.num_timesteps)
                print(f'물리틱={collector.ticks} 검증={ev["success_rate"]:.2f} 구간={ev["mean_milestones"]:.2f}/16',flush=True)

            checkpoint()
            while collector.ticks<budget:
                rate,entropy = coefficients('A',collector.ticks,budget)
                model.learning_rate = rate;model.lr_schedule = FixedRate(rate);model.ent_coef = entropy
                model.learn(total_timesteps=512,reset_num_timesteps=False,callback=collector,log_interval=None)
                if collector.ticks>=next_eval or collector.ticks>=budget:
                    checkpoint();next_eval += interval
            best = load_policy(folder/'best.zip')
            final = evaluate(best,test_count,910000+seed)
            same = all(torch.equal(v,best.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
            last = final if same else evaluate(model,test_count,910000+seed)
            selected_greedy = evaluate(best,1,123,greedy=True,native=True)
            latest_greedy = selected_greedy if same else evaluate(model,1,123,greedy=True,native=True)
            assert config['sources']==sources()
            result = dict(config=config,final=final,latest_test=last,selected_greedy=selected_greedy,
                latest_greedy=latest_greedy,latest_matches_selected_weights=same,
                best_step=best_step,best_validation=read(folder/'best_validation.json'),
                decisions=model.num_timesteps,physics_ticks=collector.ticks,training_episodes=collector.episodes,
                training_successes=collector.successes,training_reward_parts=dict(collector.parts),
                elapsed_seconds=time.perf_counter()-started,sources_after=sources(),test_used_for_selection=False)
            write(folder/'summary.json',result)
            print(f'완료 선택본={final["success_rate"]:.2f} 마지막={last["success_rate"]:.2f}',flush=True)
        finally:
            vec.close()
    return str(folder/'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--physics-ticks',type=int,default=6000000)
    parser.add_argument('--eval-every',type=int,default=500000)
    parser.add_argument('--validation-count',type=int,default=20)
    parser.add_argument('--test-count',type=int,default=100)
    parser.add_argument('--seeds',default='0,1,2')
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if min(args.physics_ticks,args.eval_every,args.validation_count,args.test_count,args.workers)<=0:
        parser.error('학습·검증·시험 횟수와 작업 수는 양수여야 한다')
    seeds = [int(s) for s in args.seeds.split(',')]
    if len(seeds)!=len(set(seeds)):parser.error('시드는 중복할 수 없다')
    root = args.root.resolve()
    if root.exists():raise RuntimeError('새 결과 폴더를 사용해야 한다')
    root.mkdir(parents=True)
    env = MapAEnv()
    plan = dict(map_kind='reinforcement',map_number=1,map_title=env.unwrapped.level.title,
        variant='A-map1',seeds=seeds,physics_budget=args.physics_ticks,eval_every=args.eval_every,
        validation_count=args.validation_count,test_count=args.test_count,sources=sources(),
        observation_size=OBS_SIZE,object_capacity=CAPACITY,milestones=[asdict(s) for s in SPECS],
        milestone_labels=list(LABELS),spring_count=len(SPRINGS),lift_count=len(LIFTS),
        level=asdict(env.unwrapped.level),reward_adaptations=[
            '16개 시작 특수 발판→목표 실제 착지 구간, 구간별 최초 +10',
            '점프 6개마다 높이/가로 최초 +0.5씩, 반동 후 최대3초',
            '리프트 2개마다 최초착지12/높은위치12/윗층착지18/접근최대12.96',
            '접근 보상은 해당 리프트 앞 발판 착지로 활성화해 다른 층의 동일 x좌표를 제외'],
        original_A_sources_unchanged=True,previous_weights_loaded=False,
        original_spawn=True,curriculum=False,demonstrations=False,git_actions=False)
    env.close();write(root/'plan.json',plan)
    for name in plan['sources']:
        target = root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(name,target)
    write(root/'versions.json',{name:importlib.metadata.version(name) for name in ('torch','stable-baselines3','numpy','gymnasium')})
    started = time.perf_counter();completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(train_job,(str(root),seed,args.physics_ticks,args.eval_every,args.validation_count,args.test_count)) for seed in seeds]
        for future in as_completed(tasks):
            completed.append(future.result())
            write(root/'progress.json',dict(completed=completed,total=len(seeds),elapsed_seconds=time.perf_counter()-started))
            print(f'완료 {len(completed)}/{len(seeds)}: {completed[-1]}',flush=True)
    assert plan['sources']==sources()
    write(root/'completion.json',dict(complete=True,models=len(completed),summaries=completed,
                                    sources=sources(),elapsed_seconds=time.perf_counter()-started))
    print('강화학습 맵1의 A 실험 완료',flush=True)


if __name__=='__main__':main()
