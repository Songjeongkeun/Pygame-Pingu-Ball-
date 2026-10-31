from __future__ import annotations
import argparse
import csv
import random
import time
from pathlib import Path
import sys
import numpy as np
import torch
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tutorial2_ppo_setting.environment import Stage2Env, RewardConfig, ACTIONS, OBS_VERSION
from tutorial2_ppo_setting.network import ActorCritic, action_sample
from tutorial2_ppo_setting.learn import ppo_update

RECIPE_ID='tutorial2-team-spring-10-0p5-0p5-100k-ppo-v1'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def csv_append(path, item):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fresh=not path.exists()
    with path.open('a',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(item))
        if fresh:w.writeheader()
        w.writerow(item)


def read_checkpoint(path):

    return torch.load(path,map_location='cpu',weights_only=False)


def entropy_at(decisions, max_decisions, start=.02, end=.001):
    progress = min(1., max(0., decisions / max(1, max_decisions)))
    return start + (end-start) * progress


def save_checkpoint(path, net, optimizer, args, decisions, frames, episodes, best_metric):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        'model':net.state_dict(), 'optimizer':optimizer.state_dict(),
        'recipe_id':RECIPE_ID, 'algo':'ppo', 'width':args.width,
        'obs_dim':net.body[0].in_features, 'action_dim':len(ACTIONS), 'version':OBS_VERSION,
        'stage2_rewards':vars(RewardConfig()).copy(), 'train_greedy_mix':0., 'eval_greedy_mix':args.greedy_mix,
        'config':vars(args).copy(), 'steps':decisions,'physical_frames':frames,
        'episodes':episodes,'best_metric':best_metric,
        'random_state':random.getstate(), 'numpy_state':np.random.get_state(),
        'torch_state':torch.get_rng_state(),
    },path)


def load_model(path):
    ck=read_checkpoint(path)
    env=Stage2Env()
    if ck.get('algo')!='ppo' or ck.get('version')!=OBS_VERSION or ck.get('obs_dim')!=env.obs_dim or ck.get('action_dim')!=len(ACTIONS):
        raise RuntimeError('Incompatible model observation/action architecture')
    net=ActorCritic(env.obs_dim,len(ACTIONS),ck['width'])
    net.load_state_dict(ck['model'])
    net.eval()
    return ck,net


def evaluate(net, *, trials=100, mode='mixed', seed=60000, greedy_mix=.05,
             live=False, render_every=10):
    if trials<1:raise ValueError('trials must be >=1')
    if mode not in ('mixed','stochastic','greedy'):
        raise ValueError('Invalid mode')
    py_state=random.getstate();np_state=np.random.get_state();pt_state=torch.get_rng_state()
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    env=Stage2Env()
    display=None
    if live:
        from tutorial2_ppo_setting.display import LiveDisplay
        display=LiveDisplay(env, fps=0)
    results=[]
    try:
        for ep in range(trials):
            obs=env.reset();done=False;count=0
            while not done:
                if display and not display.events():
                    return None
                action,_,_=action_sample(net, obs, mode=mode, greedy_mix=greedy_mix)
                obs,reward,done,info=env.step(action)
                count+=1
                if display and (count%render_every==0 or done):
                    display.draw('eval',ep+1,count,action,reward,info)
            results.append(info)
    finally:
        if display:display.close()
        random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(pt_state)
    n=len(results)
    stats=('success','spring1','spring2','spring3','middle','upper','finish','timeout')
    counts={name:sum(int(row[name]) for row in results) for name in stats}
    actions=np.sum([np.asarray(row['actions'],dtype=np.int64) for row in results],axis=0)
    directions=[int(actions[3*i:3*i+3].sum()) for i in range(3)]
    total=max(1,sum(directions))
    outcome={'wins':counts['success'],'trials':n,'win_pct':round(100.*counts['success']/n,2)}
    outcome.update({name+'_pct':round(100.*counts[name]/n,2) for name in stats if name!='success'})
    outcome.update({'mean_max_x':round(float(np.mean([row['max_x'] for row in results])),1),
                    'mean_seconds':round(float(np.mean([row['seconds'] for row in results])),2),
                    'mean_reward':round(float(np.mean([row['reward_total'] for row in results])),2),
                    'direction_pct':[round(100.*d/total,1) for d in directions],
                    'failures':{}})
    for row in results:
        key=row['reason']
        outcome['failures'][key]=outcome['failures'].get(key,0)+1
    return outcome


def ensure_same_resume_config(previous, args):


    keys=('width','lr','gamma_second','gae_lambda','entropy_start','entropy_end',
          'entropy_total_steps','n_envs','n_steps','batch_size','ppo_epochs',
          'value_coef','max_grad_norm','clip','target_kl','greedy_mix')
    for key in keys:
        if previous.get(key)!=getattr(args,key):
            raise RuntimeError(f'Resume hyperparameter mismatch: {key} ({previous.get(key)} != {getattr(args,key)})')


def training(args):
    if args.steps < 1:raise ValueError('steps must be >=1')
    seed_all(args.seed)
    envs=[Stage2Env() for _ in range(args.n_envs)]
    states=[env.reset() for env in envs]
    net=ActorCritic(envs[0].obs_dim,len(ACTIONS),args.width)
    optimizer=torch.optim.Adam(net.parameters(),lr=args.lr,eps=1e-5)
    out=Path(args.save_dir)
    out.mkdir(parents=True,exist_ok=True)
    decisions=frames=episodes=0
    best=(-1.,)*7
    if args.resume:
        ck=read_checkpoint(args.resume)
        if ck.get('recipe_id') != RECIPE_ID:
            raise RuntimeError('Only stage-2 PPO checkpoints may be resumed; stage-4 is incompatible.')
        ensure_same_resume_config(ck.get('config',{}),args)
        net.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        decisions,frames,episodes=ck['steps'],ck['physical_frames'],ck['episodes']
        best=tuple(ck.get('best_metric',best))
        random.setstate(ck['random_state']);np.random.set_state(ck['numpy_state'])
        torch.set_rng_state(ck['torch_state'])
        print(f'[RESUME] decisions={decisions} -- four worlds begin new episodes')
    display=None
    if args.live:
        from tutorial2_ppo_setting.display import LiveDisplay
        display=LiveDisplay(envs[0],fps=args.fps)
    print(f'[START PPO TUTORIAL2] decisions={decisions}/{args.steps} obs={envs[0].obs_dim} actions={len(ACTIONS)} '
          f'n_envs={args.n_envs} n_steps={args.n_steps} batch={args.n_envs*args.n_steps} '
          f'lr={args.lr} gamma_second={args.gamma_second} gae={args.gae_lambda} '
          f'entropy={args.entropy_start}->{args.entropy_end} lambda_schedule_steps={args.entropy_total_steps} '
          f'epochs={args.ppo_epochs} minibatch={args.batch_size} train=stochastic eval_mix={args.greedy_mix:.0%}')
    print('[TUTORIAL2 REWARDS] common win=+100 fail/timeout=-25; per spring: '
          'first VALID target landing=+10, height alignment=+0.5 (±16px), '
          'horizontal overlap=+0.5; no progress/first-spring/clear bonus; movement/wait=0')
    next_eval=(decisions//args.eval_every+1)*args.eval_every
    next_save=(decisions//args.save_every+1)*args.save_every
    start=time.perf_counter();last_logs={}
    try:
        while decisions < args.steps:
            streams=[[] for _ in range(args.n_envs)]
            stop=False
            for tick in range(args.n_steps):
                for i,env in enumerate(envs):
                    if decisions>=args.steps or stop:
                        break
                    if display and i==0 and not display.events():
                        stop=True;break
                    old_obs=states[i]
                    action,lp,_=action_sample(net,old_obs,mode='stochastic')
                    next_obs,reward,done,info=env.step(action)
                    streams[i].append({'obs':old_obs,'next_obs':next_obs,'action':action,
                                       'logp':lp,'reward':reward,'done':done,'frames':info['frames']})
                    decisions+=1;frames+=info['frames']
                    if display and i==0 and (decisions%args.render_every==0 or done):
                        display.draw('ppo stage2',episodes+1,decisions,action,reward,info)
                    states[i]=next_obs
                    if done:
                        episodes+=1
                        csv_append(out/'episodes.csv',{'episode':episodes,'decisions':decisions,'frames':frames,
                          'env_idx':i,'success':int(info['success']),'spring1':info['spring1'], 'spring2':info['spring2'], 'spring3':info['spring3'],
                          'middle':info['middle'], 'upper':info['upper'], 'finish':info['finish'],
                          'milestone':info['milestone'],'max_x':round(info['max_x'],1),
                          'seconds':round(info['seconds'],2),'reward':round(info['reward_total'],2),
                          'reason':info['reason']})
                        if info['success'] or episodes%args.log_every==0:
                            print(f'[PPO TUTORIAL2] ep={episodes} decisions={decisions}/{args.steps} '
                                  f'win={int(info["success"])} springs={info["spring1"]}{info["spring2"]}{info["spring3"]} decks={info["milestone"]}/3 '
                                  f'x={info["max_x"]:.1f} reward={info["reward_total"]:+.1f} '
                                  f'elapsed={time.perf_counter()-start:.0f}s')
                        states[i]=env.reset()
                if decisions>=args.steps or stop:break
            if any(streams):
                ent=entropy_at(decisions,args.entropy_total_steps,args.entropy_start,args.entropy_end)
                last_logs=ppo_update(net,optimizer,streams,gamma_per_second=args.gamma_second,
                   gae_lambda=args.gae_lambda,epochs=args.ppo_epochs,batch_size=args.batch_size,
                   clip=args.clip,entropy_coef=ent,value_coef=args.value_coef,
                   max_grad_norm=args.max_grad_norm,target_kl=args.target_kl,
                   greedy_mix=0.,normalize_advantage=True)
            if args.print_updates:
                print('[UPDATE]',decisions,last_logs)
            if decisions>=next_eval and not stop:
                result=evaluate(net,trials=args.eval_trials,mode='mixed',seed=args.eval_seed,
                                greedy_mix=args.greedy_mix)
                print(f'[VAL PPO TUTORIAL2] decisions={decisions} episodes={episodes} {result} optimizer={last_logs}')
                csv_append(out/'evaluations.csv',{'decisions':decisions,'episodes':episodes,
                    **{key:value for key,value in result.items() if isinstance(value,(int,float))}})
                metric=(result['win_pct'],result['finish_pct'],result['upper_pct'],result['middle_pct'],
                        result['spring3_pct'],result['spring2_pct'],result['spring1_pct'])
                if metric>best:
                    best=metric
                    save_checkpoint(out/'best.pt',net,optimizer,args,decisions,frames,episodes,best)
                next_eval+=args.eval_every
            if decisions>=next_save or stop:
                save_checkpoint(out/'last.pt',net,optimizer,args,decisions,frames,episodes,best)
                next_save+=args.save_every
            if stop:
                print('[STOP] ESC or window close; saved last.pt');return
        save_checkpoint(out/'last.pt',net,optimizer,args,decisions,frames,episodes,best)
        final=evaluate(net,trials=args.eval_trials,mode='mixed',seed=args.eval_seed,
                       greedy_mix=args.greedy_mix)
        print('[FINAL MIXED 95/5]',final)
        print(f'[DONE] algo=ppo decisions={decisions} physics_frames={frames} episodes={episodes} files={out}')
    except KeyboardInterrupt:
        save_checkpoint(out/'last.pt',net,optimizer,args,decisions,frames,episodes,best)
        print('[INTERRUPTED] Saved last.pt')
    finally:
        if display:display.close()


def cli():
    parser=argparse.ArgumentParser(description='튜토리얼 2단계 PPO 학습·평가')
    sub=parser.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train')
    tr.add_argument('--steps',type=int,default=1000000,help='cumulative decisions')
    tr.add_argument('--save-dir',default=str(Path(__file__).resolve().parent / 'result_tutorial2_new'))
    tr.add_argument('--resume',default=None)
    tr.add_argument('--seed',type=int,default=42)
    tr.add_argument('--width',type=int,default=128)
    tr.add_argument('--lr',type=float,default=.0003)
    tr.add_argument('--gamma-second',type=float,default=1.0)
    tr.add_argument('--gae-lambda',type=float,default=.98)
    tr.add_argument('--entropy-start',type=float,default=.02)
    tr.add_argument('--entropy-end',type=float,default=.001)
    tr.add_argument('--entropy-total-steps',type=int,default=300000)
    tr.add_argument('--value-coef',type=float,default=.5)
    tr.add_argument('--max-grad-norm',type=float,default=.5)
    tr.add_argument('--n-envs',type=int,default=4)
    tr.add_argument('--n-steps',type=int,default=128)
    tr.add_argument('--batch-size',type=int,default=256)
    tr.add_argument('--ppo-epochs',type=int,default=5)
    tr.add_argument('--clip',type=float,default=.15,help='old PPO config unchanged')
    tr.add_argument('--target-kl',type=float,default=.03,help='old PPO safety limit unchanged; max 5 epochs')
    tr.add_argument('--greedy-mix',type=float,default=.05,help='evaluation mixture only; training remains stochastic as original PPO')
    tr.add_argument('--eval-every',type=int,default=20000)
    tr.add_argument('--eval-trials',type=int,default=100)
    tr.add_argument('--eval-seed',type=int,default=50000)
    tr.add_argument('--save-every',type=int,default=5000)
    tr.add_argument('--log-every',type=int,default=100)
    tr.add_argument('--print-updates',action='store_true')
    tr.add_argument('--live',action='store_true',help='render first of 4 games; slower')
    tr.add_argument('--render-every',type=int,default=30)
    tr.add_argument('--fps',type=int,default=0)
    ev=sub.add_parser('evaluate')
    ev.add_argument('--model',default=str(Path(__file__).resolve().parent / 'result_tutorial2' / 'last.pt'))
    ev.add_argument('--mode',choices=('mixed','stochastic','greedy','both'),default='mixed')
    ev.add_argument('--trials',type=int,default=500)
    ev.add_argument('--greedy-trials',type=int,default=1)
    ev.add_argument('--greedy-mix',type=float,default=.05)
    ev.add_argument('--seed',type=int,default=60000)
    ev.add_argument('--live',action='store_true')
    ev.add_argument('--render-every',type=int,default=10)
    sm=sub.add_parser('smoke')
    sm.add_argument('--seed',type=int,default=42)
    sm.add_argument('--episodes',type=int,default=5)
    args=parser.parse_args()
    if args.command=='train':
        if (args.steps<=0 or args.n_envs<1 or args.n_steps<1 or args.batch_size<1
                or args.ppo_epochs<1 or args.eval_every<1 or args.save_every<1
                or args.log_every<1 or args.render_every<1 or args.entropy_total_steps<1
                or args.lr<=0 or not 0 < args.gamma_second <=1
                or not 0<=args.greedy_mix<=1 or args.max_grad_norm<=0):
            parser.error('Invalid training hyperparameter')
        training(args)
    elif args.command=='smoke':
        seed_all(args.seed)
        env=Stage2Env()
        print(f'[SMOKE] original world ready obs={env.obs_dim} actions={len(ACTIONS)}')
        for i in range(args.episodes):
            obs=env.reset();done=False
            while not done:
                obs, r, done, info=env.step(random.randrange(len(ACTIONS)))
            print(f'ep={i+1} score={info["reward_total"]:+.2f} springs={info["spring1"]}{info["spring2"]}{info["spring3"]} decks={info["milestone"]}/3')
        print('[SMOKE DONE] no learning')
    else:
        seed_all(args.seed)
        ck,net=load_model(args.model)
        print(f'[CHECKPOINT] decisions={ck["steps"]} episodes={ck["episodes"]} recipe={ck.get("recipe_id","original") }')
        modes=('mixed','stochastic','greedy') if args.mode=='both' else (args.mode,)
        for mode in modes:
            n=args.greedy_trials if mode=='greedy' and args.mode=='both' else args.trials
            result=evaluate(net,trials=n,mode=mode,seed=args.seed,greedy_mix=args.greedy_mix,
                            live=args.live,render_every=args.render_every)
            print(f'[EVAL {mode.upper()}]',result)

if __name__=='__main__':
    cli()
