"""Stage 2: warm-start a 6-physics-frame 3-action PPO from the user's 12-frame PPO.

NO BC, DAgger, action labels, demonstrations or teacher imitation are used.
The source is an already trained pure PPO with the *same* common observation
and three actions. Only the action repeat changes from 12 to 6; optional PPO
fine-tuning reduces entropy and learning rate to avoid destroying good weights.
The user's 12-frame checkpoint and the 1/3/5 team models are NEVER modified.

From repository root: python -m tutorial_2.reward_transfer_2_6.train --help
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from rl.milestone_train import install
from rl.milestone_env import OBS_SIZE
from rl.milestone_models import MilestonePPO
from tutorial_2.reward_repeat_2_4 import train as base
from .config_logic import (equivalent_saved_config, SOURCE_RELATIVE, OUT_RELATIVE, make_config,
                           read_source, update_learning_schedule, sha256_file,
                           make_monitor_args, prepare_fresh_output)

ROOT = Path(__file__).resolve().parents[1]  # tutorial_2/


def json_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def model_check(model):
    if model.action_space.n != 3 or model.observation_space.shape != (OBS_SIZE,):
        raise ValueError('Not the 3-action 1~5 common observation PPO model')
    fields = dict(n_steps=128, batch_size=256, n_epochs=5, gamma=1., gae_lambda=.98)
    for key, want in fields.items():
        if getattr(model, key) != want:
            raise ValueError(f'Source model PPO setting mismatch: {key}')
    if model.clip_range(1.) != .2:
        raise ValueError('Source model PPO clip mismatch')
    if model.target_kl != .02:
        raise ValueError('Source model target KL mismatch')


def compare_hold_physics(macro_steps=300, seed=42):
    """Same 12 physics ticks = 1x repeat12 or 2x repeat6, identical directions.

    Assert unchanged physics & personal reward accounting. Deliberately compares
    trajectories, not the number of RL decisions (which SHOULD differ).
    """
    env12 = base.make_env(2, 'personal', 12)
    env6 = base.make_env(2, 'personal', 6)
    rng = np.random.default_rng(seed)
    total_ticks = 0
    episodes = 0
    try:
        a, _ = env12.reset(seed=seed)
        b, _ = env6.reset(seed=seed)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)
        for _ in range(macro_steps):
            act = int(rng.integers(0, 3))
            oa, ra, da, ta, ia = env12.step(act)
            rb = 0.
            for _half in range(2):
                if env6.team.unwrapped.done:
                    break
                ob, r, db, tb, ib = env6.step(act)
                rb += r
            wa, wb = env12.team.unwrapped.world, env6.team.unwrapped.world
            xyz = ('x','y','vx','vy')
            xa = [float(getattr(wa.ball, k)) for k in xyz] + [float(wa.time)]
            xb = [float(getattr(wb.ball, k)) for k in xyz] + [float(wb.time)]
            np.testing.assert_allclose(xa, xb, rtol=0, atol=2e-6,
                                       err_msg=f'Native physical trajectory mismatch, action={act}')
            if (da,ta,wa.cleared,wa.failed) != (db,tb,wb.cleared,wb.failed):
                raise AssertionError('Terminal/done mismatch: repeat12 vs repeat6')
            if abs(ra-rb)>1e-6:
                raise AssertionError(f'Personal reward difference: repeat12={ra}, repeat6sum={rb}')
            total_ticks += int(ia['tick_delta'])
            if da or ta:
                episodes += 1
                oa, _ = env12.reset(seed=seed+episodes)
                ob, _ = env6.reset(seed=seed+episodes)
                np.testing.assert_allclose(oa, ob, rtol=0, atol=1e-7)
    finally:
        env12.close()
        env6.close()
    print(f'[PHYSICS PASS] stage=2 12x1 == 6x2 actions={macro_steps} '
          f'episodes={episodes} ticks={total_ticks}: same native physics and personal rewards', flush=True)


def source_and_model(args, *, require_source=True):
    install()
    torch.set_num_threads(1)
    source = read_source(args.source, OBS_SIZE) if require_source else None
    vec = base.make_vec(2, 'personal', 6)
    try:
        model = MilestonePPO.load(str(args.source), env=vec, device='cpu')
        model_check(model)
        return source, vec, model
    except BaseException:
        vec.close()
        raise


def baseline(args):
    source, vec, model = source_and_model(args)
    try:
        # Identical checkpoint weights and evaluation seeds: only repeat changes.
        six = base.evaluate(model, 2, 'personal', 6, trials=args.trials, seed=args.eval_seed)
        twelve = base.evaluate(model, 2, 'personal', 12, trials=args.trials, seed=args.eval_seed)
    finally:
        vec.close()
    result = {'source_checkpoint':source['checkpoint'], 'source_sha256':source['sha256'],
              'eval_seed':args.eval_seed, 'trials':args.trials,
              'repeat_12':twelve, 'repeat_6_without_training':six,
              'important':'These two tests use EXACTLY the same PPO weights.'}
    out = args.out/'before_transfer.json'
    json_save(out,result)
    print(f"[BEFORE] repeat=12 win={twelve['wins']}/{args.trials} "
          f"({twelve['success_rate']:.1%}); repeat=6 win={six['wins']}/{args.trials} "
          f"({six['success_rate']:.1%}) on eval seed={args.eval_seed}",flush=True)
    print(f'[SAVED] {out}',flush=True)


def run_train(args):
    if args.physics_budget < 1 or args.validation_trials < 1 or args.eval_every_frames < 1:
        raise ValueError('Bad physics-budget, validation or checkpoint interval')
    if args.seed < 0 or not (0 <= args.entropy <= .2) or not (0 < args.learning_rate <= .01):
        raise ValueError('Bad adaptation seed, entropy or learning rate')
    install()
    torch.set_num_threads(1)
    source = read_source(args.source, OBS_SIZE)
    cfg = make_config(source, seed=args.seed, lr=args.learning_rate, entropy=args.entropy,
                      validation_trials=args.validation_trials,
                      validation_seed=args.validation_seed,
                      eval_every_frames=args.eval_every_frames, obs_size=OBS_SIZE)
    folder = args.out
    settings = folder/'config.json'
    if args.resume:
        if not settings.is_file() or not (folder/'last.zip').is_file() or not (folder/'latest.json').is_file():
            raise FileNotFoundError('No previous 6-frame transfer to resume: '+str(folder))
        old_cfg = json.loads(settings.read_text(encoding='utf-8'))
        if not equivalent_saved_config(old_cfg, cfg):
            raise ValueError('6-frame transfer resume settings mismatch')
        latest = json.loads((folder/'latest.json').read_text(encoding='utf-8'))
        start_physics = int(latest['physical_frames'])
        start_episodes = int(latest['episodes'])
        start_wins = int(latest['training_wins'])
        if start_physics >= args.physics_budget:
            raise ValueError('Already reached budget; increase --physics-budget')
    else:
        # The prior package could crash on the *first* validation callback, leaving
        # config.json but no checkpoint. It is safe to retry ONLY this empty run.
        # Existing model files/history are never deleted or overwritten silently.
        recovered = prepare_fresh_output(folder, cfg)
        if recovered:
            print(f'[RECOVER] Incomplete run with config only: {folder} '
                  '(no model checkpoint existed; starting 6-frame PPO again)', flush=True)
        start_physics = start_episodes = start_wins = 0

    vec = base.make_vec(2, 'personal', 6)
    try:
        if args.resume:
            model = MilestonePPO.load(str(folder/'last.zip'), env=vec, device='cpu')
            print(f'[RESUME 6-FRAME PPO] physics={start_physics}->{args.physics_budget} '
                  f'decisions={model.num_timesteps}', flush=True)
        else:
            model = MilestonePPO.load(str(args.source), env=vec, device='cpu')
            # Keep policy weights/optimizer, but begin a NEW on-policy PPO run
            # under 6-frame dynamics. Reset time-step counter, not policy.
            model.set_random_seed(args.seed)
            print(f'[BRANCH] PPO policy 12-frame best -> 6-frame physics, '
                  f'source SHA256={source["sha256"][:12]} seed={args.seed}', flush=True)
        model_check(model)
        update_learning_schedule(model,args.learning_rate)
        model.ent_coef=float(args.entropy)
        model.verbose=0
        # PhysicsMonitor comes from tutorial_2.reward_repeat_2_4. Its checkpoint() reads
        # args.stage/reward/repeat, absent from this transfer CLI. Supply the
        # complete fixed environment contract, not the raw argparse namespace.
        monitor_args=make_monitor_args(args)
        monitor=base.PhysicsMonitor(folder,monitor_args,start_frames=start_physics,
                                    start_episodes=start_episodes,start_wins=start_wins)
        model.after_update=monitor.after_update
        # Equivalent source training weights; NO teacher actions, BC or DAgger.
        model.learn(total_timesteps=args.physics_budget + 1024,
                    callback=monitor,reset_num_timesteps=not args.resume)
        if monitor.last_saved_frames != monitor.physical_frames:
            monitor.checkpoint()
        if not (folder/'best.zip').exists():
            raise RuntimeError('Failed to save a 6-frame PPO model')
    finally:
        vec.close()
    result=dict(config=cfg,completed_decisions=int(model.num_timesteps),
                completed_physics_frames=int(monitor.physical_frames),
                best_validation=json.loads((folder/'best_validation.json').read_text(encoding='utf-8')),
                stopped_for_budget=monitor.stopped_for_budget,
                stopped_for_collapse=monitor.stopped_for_collapse)
    json_save(folder/'summary.json',result)
    print(f'[SAVED] {folder}',flush=True)


def evaluate(args):
    install()
    torch.set_num_threads(1)
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    saved_cfg = model_path.with_name('config.json')
    if saved_cfg.exists():
        cfg = json.loads(saved_cfg.read_text(encoding='utf-8'))
        if cfg.get('target_repeat') != 6 or cfg.get('stage') != 2:
            raise ValueError('Evaluation model is not a 6-frame stage-2 transfer')
    else:
        raise FileNotFoundError(f'Missing transfer settings {saved_cfg}')
    vec = base.make_vec(2,'personal',6)
    try:
        model=MilestonePPO.load(str(model_path),env=vec,device='cpu')
        model_check(model)
        ev=base.evaluate(model,2,'personal',6,trials=args.trials,seed=args.eval_seed)
    finally:
        vec.close()
    print(json.dumps(ev,ensure_ascii=False,indent=2),flush=True)
    if args.output:
        json_save(args.output,ev)


def cli():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    sm=sub.add_parser('smoke',help='Check repeat12 once vs repeat6 twice physical trajectories and rewards')
    sm.add_argument('--macro-steps',type=int,default=300)
    sm.add_argument('--seed',type=int,default=42)
    bl=sub.add_parser('baseline',help='Same source PPO without retraining: repeat12 vs repeat6')
    bl.add_argument('--source',type=Path,default=ROOT/SOURCE_RELATIVE)
    bl.add_argument('--trials',type=int,default=200)
    bl.add_argument('--eval-seed',type=int,default=120000)
    bl.add_argument('--out',type=Path,default=ROOT/OUT_RELATIVE)
    tr=sub.add_parser('train',help='Continue the same 3-action PPO in repeat=6 world; no BC/DAgger')
    tr.add_argument('--source',type=Path,default=ROOT/SOURCE_RELATIVE)
    tr.add_argument('--out',type=Path,default=ROOT/OUT_RELATIVE)
    tr.add_argument('--physics-budget',type=int,default=240000)
    tr.add_argument('--seed',type=int,default=42)
    tr.add_argument('--learning-rate',type=float,default=.0001,
                    help='Adaptation PPO LR; original 12-frame training used 0.0003')
    tr.add_argument('--entropy',type=float,default=.01,
                    help='6-frame exploration; original 12-frame training used 0.05')
    tr.add_argument('--validation-seed',type=int,default=50000)
    tr.add_argument('--validation-trials',type=int,default=20)
    tr.add_argument('--eval-every-frames',type=int,default=60000)
    tr.add_argument('--resume',action='store_true')
    tr.add_argument('--no-early-stop',action='store_true')
    tr.add_argument('--collapse-entropy',type=float,default=.1)
    ev=sub.add_parser('evaluate',help='Independent 6-frame student PPO evaluation')
    ev.add_argument('--model',type=Path,default=ROOT/OUT_RELATIVE/'best.zip')
    ev.add_argument('--trials',type=int,default=200)
    ev.add_argument('--eval-seed',type=int,default=120000)
    ev.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.command=='smoke':compare_hold_physics(args.macro_steps,args.seed)
    elif args.command=='baseline':baseline(args)
    elif args.command=='train':run_train(args)
    else:evaluate(args)


if __name__=='__main__':
    cli()
