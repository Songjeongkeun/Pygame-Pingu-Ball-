"""Frozen local experiment for user milestone rewards; no old weights loaded."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
from pathlib import Path
import shutil
import time
import torch
from stable_baselines3.common.logger import configure
from . import five_train as old
from .milestone_env import MilestoneEnv, MilestoneRawEnv, OBS_SIZE
from .milestone_models import ALGORITHMS, MilestoneExtractor

ORIGINAL_SOURCES = old.sources
ORIGINAL_MAKE = old.make_model
ORIGINAL_MONITOR = old.Monitor
write, read = old.write, old.read

PROTOCOL = dict(success=100., failure=-25., timeout=-25., milestone=10.,
                reward_scale=.01, deadline_seconds=180., action_seconds=.05, physics_hz=120,
                action_repeat=6, gamma=1., learning_rate=.0003, joint_learning_rate=.0001,
                milestone_condition='native source landing followed by native target-group landing',
                milestone_once_per_episode=True, failure_timeout_penalty_once=True,
                no_distance_or_contact_or_survival_reward=True, authored_intermediate_targets=True,
                raw_points_in_logs=True)


def sources():
    result = ORIGINAL_SOURCES()
    for name in ('milestone_env.py','milestone_models.py','milestone_train.py'):
        p = 'rl/'+name
        result[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result['rl/five_report.py'] = hashlib.sha256(Path('rl/five_report.py').read_bytes()).hexdigest()
    return result


def entropy(progress, joint=False):
    p = min(1., max(0., progress))
    if joint:
        return .005+(.001-.005)*p
    if p <= .2:
        return .02
    if p <= .8:
        return .02+(.005-.02)*(p-.2)/.6
    return .005+(.001-.005)*(p-.8)/.2


class MilestoneMonitor(ORIGINAL_MONITOR):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('eval_every',40000)
        super().__init__(*args, **kwargs)

    def after_update(self, model):
        model.ent_coef = entropy(model.num_timesteps/self.budget, len(self.stages)>1)
        if model.num_timesteps >= self.next_eval or model.num_timesteps >= self.budget:
            self.checkpoint(model)
            self.next_eval += self.eval_every


def make_model(algorithm, vec, seed, joint=False):
    model = ORIGINAL_MAKE(algorithm, vec, seed, joint)
    model.ent_coef = .005 if joint else .02
    return model


def install():
    # Reuse tested balanced distillation/rollout machinery. Every process installs
    # the new environment and extractor before any model construction or load.
    old.FiveEnv, old.OBS_SIZE = MilestoneEnv, OBS_SIZE
    old.FiveExtractor, old.ALGORITHMS = MilestoneExtractor, ALGORITHMS
    old.sources, old.make_model, old.Monitor = sources, make_model, MilestoneMonitor


def export_milestones(root):
    result = {}
    for stage in range(1,6):
        env = MilestoneRawEnv(stage)
        result[str(stage)] = [dict(id=s.id, source_ids=s.source, target_group=s.target,
            source_world_rect_t0=env.bounds(s.source), target_world_rect_t0=env.bounds(s.target),
            source_document_rect_t0=tuple(v-(env.level.spawn[0]-__import__('pingu_ball.special_lab',fromlist=['make_lab']).make_lab(stage).spawn[0])
                                           if i==0 else v for i,v in enumerate(env.bounds(s.source))),
            moving_supports_follow_ids=True) for s in env.specs]
        env.close()
    write(Path(root)/'milestones.json',result)


def expert_job(payload):
    install()
    root, algorithm, stage, seed, budget, test_count = payload
    torch.set_num_threads(1)
    folder = Path(root)/'experts'/algorithm/f'seed_{seed}'/f'stage_{stage:02d}'
    if (folder/'summary.json').exists():
        assert read(folder/'summary.json')['config']['sources']==sources()
        return str(folder/'summary.json')
    if (folder/'config.json').exists():
        raise RuntimeError(f'Incomplete model retained; do not overwrite: {folder}')
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'console.log').open('w',encoding='utf8',buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        vec = old.vector([stage]*4,'milestone')
        try:
            model = make_model(algorithm,vec,seed)
            model.set_logger(configure(str(folder),['csv']))
            config = dict(algorithm=algorithm,stage=stage,seed=seed,decisions_budget=budget,
                reward_mode='milestone',from_scratch=True,original_spawn=True,
                action_count=3,observation_size=OBS_SIZE,sources=sources(),protocol=PROTOCOL,
                test_count=test_count,validation_count=20,validation_seed=11000,
                primary_evaluation='sampled_policy',previous_models_loaded=[],
                reward='success +100, death-or-180s-timeout -25 once; first source-to-target landing +10; learning /100')
            write(folder/'config.json',config)
            model.save(folder/'initial')
            monitor = MilestoneMonitor(folder,[stage],budget,'milestone')
            model.after_update = monitor.after_update
            monitor.checkpoint(model)
            model.learn(total_timesteps=budget,callback=monitor,log_interval=None)
            old.finish(model,monitor,folder,config,test_count)
        finally:
            vec.close()
    return str(folder/'summary.json')


def teachers(root, seed, gate):
    result = {}
    for stage in range(1,6):
        candidates = []
        for algorithm in ('a2c','ppo'):
            folder = Path(root)/'experts'/algorithm/f'seed_{seed}'/f'stage_{stage:02d}'
            model = ALGORITHMS[algorithm].load(folder/'best.zip',device='cpu')
            ev = old.evaluate(model,[stage],50,3510000,'milestone')
            write(folder/'teacher_validation_50.json',ev)
            candidates.append((old.validation_score(ev),algorithm,folder,ev))
        _, algorithm, folder, ev = max(candidates,key=lambda r:r[0])
        result[str(stage)] = dict(algorithm=algorithm,training_seed=seed,checkpoint=str((folder/'best.zip').resolve()),
            validation=ev,passes_gate=ev['min_success_rate']>=gate,selection_used_final_test=False)
    return result


def consolidate(payload):
    install()
    torch.set_num_threads(1)
    old.select_teachers = teachers
    return old.consolidation_job(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--expert-steps',type=int,default=400000)
    parser.add_argument('--joint-steps',type=int,default=200000)
    parser.add_argument('--seeds',default='0')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--test-count',type=int,default=100)
    args = parser.parse_args()
    install()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True,exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    plan = dict(stages=[1,2,3,4,5],algorithms=['a2c','ppo'],seeds=seeds,
        expert_steps=args.expert_steps,joint_steps=args.joint_steps,test_count=args.test_count,
        gate=.8,reward_mode='milestone',protocol=PROTOCOL,sources=sources(),
        previous_weights_loaded=False,human_actions=False,authored_intermediate_targets=True,
        git_actions=False,general_maps=False,validation_count=20,teacher_validation_count=50,
        action_probability_sampling=True,prior_reward_comparison_is_confounded=True,
        note='New reward, action period, horizon and milestone inputs all change; not a reward-only ablation.')
    if (args.root/'plan.json').exists():
        assert read(args.root/'plan.json')==plan
    else:
        write(args.root/'plan.json',plan)
        for p in plan['sources']:
            dest = args.root/'source_snapshot'/p
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest)
        export_milestones(args.root)
    started = time.perf_counter()
    tasks = [(str(args.root),a,s,seed,args.expert_steps,args.test_count)
             for seed in seeds for s in range(1,6) for a in ('a2c','ppo')]
    completed, consolidated = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(expert_job,p) for p in tasks]
        for future in as_completed(futures):
            completed.append(future.result())
            assert sources()==plan['sources']
            write(args.root/'progress.json',dict(phase='experts',completed=completed,total=len(tasks),
                  elapsed_seconds=time.perf_counter()-started))
            print(f'EXPERT {len(completed)}/{len(tasks)} {completed[-1]}',flush=True)
        futures = [pool.submit(consolidate,(str(args.root),seed,args.joint_steps,.8,'milestone',args.test_count,False))
                   for seed in seeds]
        for future in as_completed(futures):
            consolidated.append(future.result())
            write(args.root/'progress.json',dict(phase='consolidation',completed=completed,
                 consolidation=consolidated,total=len(tasks),elapsed_seconds=time.perf_counter()-started))
            print('CONSOLIDATED '+consolidated[-1],flush=True)
    write(args.root/'completion.json',dict(complete=True,experts=len(completed),consolidation=consolidated,
        consolidation_status=[read(p)['status'] for p in consolidated],sources=sources(),
        elapsed_seconds=time.perf_counter()-started))


if __name__=='__main__':
    main()
