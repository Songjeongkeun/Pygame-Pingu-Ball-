from __future__ import annotations
import copy
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutorial2_ppo_setting.environment import Stage2Env, SPRINGS, RewardConfig, ACTIONS, OBS_VERSION
from tutorial2_ppo_setting.spring_rewards import SpringRewardTracker, SpringRewards
from tutorial2_ppo_setting.network import ActorCritic, action_sample
from tutorial2_ppo_setting.learn import build_batch, ppo_update
import tutorial2_ppo_train as runner


class Platform:
    def __init__(self, pid, x, y, kind='normal'):
        self.id, self.x, self.y, self.w, self.h, self.kind = pid, float(x), float(y), 64., 64., kind
    def rect_at(self, time, physics=None):
        return (self.x,self.y,self.w,self.h)


class Physics:
    radius=14.
    bounce_speed=370.
    max_speed=213.3


class FakeLevel:
    def __init__(self):
        self.spawn=(288.,594.)
        self.goal=(1280.,256.,64.,64.)
        self.platforms=(
            [Platform(f'start-{i}',192+i*64,704) for i in range(4)] +
            [Platform('spring-1-0',448,704,'spring')] +
            [Platform(f'middle-{i}',576+i*64,576) for i in range(3)] +
            [Platform('spring-2-0',768,576,'spring')] +
            [Platform(f'upper-{i}',896+i*64,448) for i in range(2)] +
            [Platform('spring-3-0',1024,448,'spring')] +
            [Platform(f'finish-{i}',1152+i*64,320) for i in range(4)]
        )


class Ball:
    def __init__(self,x,y):
        self.x=x;self.y=y;self.vx=0.;self.vy=0.


class World:
    def __init__(self,level,auto_reset=False):
        self.level=level
        self.platforms=copy.deepcopy(level.platforms)
        self.physics=Physics()
        self.auto_reset=auto_reset
        self.reset()

    def reset(self):
        self.ball=Ball(*self.level.spawn)
        self.time=0.;self.failed=False;self.cleared=False
        self.last_failure='';self.events=[]

    def step(self,direction):
        self.events=[]
        self.ball.x += direction * 1.2
        self.ball.vx = direction * 120.
        self.time += 1. / 120.
        if self.time >= 1.:
            self.failed=True
            self.last_failure='fake_fall'
            self.events.append(('fail',self.last_failure))


def game_modules():
    pkg=types.ModuleType('pingu_ball');pkg.__path__=[]
    engine=types.ModuleType('pingu_ball.engine');engine.World=World
    special=types.ModuleType('pingu_ball.special_lab');special.make_lab=lambda n: FakeLevel() if n==2 else (_ for _ in ()).throw(AssertionError(n))
    play=types.ModuleType('pingu_ball.playfield');play.expand_playfield=lambda level: level
    return {'pingu_ball':pkg,'pingu_ball.engine':engine,
            'pingu_ball.special_lab':special,'pingu_ball.playfield':play}


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.mock=patch.dict(sys.modules,game_modules())
        self.mock.start()

    def tearDown(self):
        self.mock.stop()

    def test_observation_actions_and_geometry(self):
        env=Stage2Env()
        self.assertEqual(env.obs_dim,40)
        self.assertEqual(len(ACTIONS),9)
        self.assertEqual(tuple(env.springs),SPRINGS)
        self.assertTrue(np.isfinite(env.observe()).all())
        self.assertEqual(env.observe().dtype,np.float32)

    def _simulate_landing(self, env, pid, time):
        env.world.time=time
        return env._landing(pid)

    def test_target_landing_only_and_not_arbitrary_pad(self):
        env=Stage2Env()
        self.assertEqual(self._simulate_landing(env,'spring-1-0',1.),0.)
        self.assertEqual(self._simulate_landing(env,'start-2',2.),0.)
        self.assertEqual(self._simulate_landing(env,'spring-1-0',3.),0.)
        self.assertEqual(self._simulate_landing(env,'middle-0',4.),10.)
        self.assertEqual(self._simulate_landing(env,'spring-1-0',5.),0.)
        self.assertEqual(self._simulate_landing(env,'middle-1',6.),0.)
        self.assertEqual(env.spring_tracker.parts['target_landing'],10.)
        self.assertEqual(env.spring_tracker.parts['target_height'],0.)
        self.assertEqual(env.spring_tracker.parts['target_horizontal'],0.)

    def test_spring_reward_is_direction_agnostic(self):
        from types import SimpleNamespace
        world=World(FakeLevel())
        world.platforms.extend([
            Platform('left_upper',100,460),
            Platform('right_upper',780,460),
            Platform('below',450,810),
        ])
        t=SpringRewardTracker(world,{'spring-1-0':['left_upper','right_upper','below']})
        for dest in ('left_upper','right_upper','below'):
            t.reset()
            world.time=0.
            world.ball.x=470.;world.ball.y=620.
            self.assertEqual(t.on_landing(world,'spring-1-0'),0.)
            world.time=1.
            self.assertEqual(t.on_landing(world,dest),10.)

    def test_height_and_horizontal_one_time_only_with_new_entry(self):
        world=World(FakeLevel())
        tr=SpringRewardTracker(world,{'spring-1-0':['middle-0']})
        world.time=0.
        world.ball.x=480.;world.ball.y=640.
        tr.on_landing(world,'spring-1-0')

        world.time=.2;world.ball.x=570.;world.ball.y=650.
        self.assertEqual(tr.on_motion(world),.5)
        world.time=.4;world.ball.y=562.
        self.assertEqual(tr.on_motion(world),.5)
        world.time=.5
        self.assertEqual(tr.on_motion(world),0.)
        world.time=.8
        self.assertEqual(tr.on_landing(world,'middle-0'),10.)
        tr.on_landing(world,'spring-1-0')
        world.time=1.;world.ball.x=480.;world.ball.y=650.
        tr.on_motion(world)
        world.time=1.1;world.ball.x=580.;world.ball.y=562.
        self.assertEqual(tr.on_motion(world),0.)
        self.assertEqual(sum(tr.parts.values()),11.)

    def test_target_bonus_requires_spring_and_valid_flight(self):
        world=World(FakeLevel())
        tr=SpringRewardTracker(world,{'spring-1-0':['middle-0']})
        world.time=1
        self.assertEqual(tr.on_landing(world,'middle-0'),0.)
        world.time=2
        tr.on_landing(world,'spring-1-0')
        world.time=6
        self.assertEqual(tr.on_landing(world,'middle-0'),0.)

    def test_reward_config_is_only_common_and_team_spring(self):
        cfg=RewardConfig()
        self.assertEqual(cfg.success,100.)
        self.assertEqual(cfg.fail_or_timeout,-25.)
        self.assertEqual(cfg.target_landing,10.)
        self.assertEqual(cfg.target_height,.5)
        self.assertEqual(cfg.target_horizontal,.5)
        self.assertFalse(hasattr(cfg,'tutorial_progress_per_pixel'))

    def test_failing_terminal_once(self):
        env=Stage2Env()
        done=False
        while not done:
            _,r,done,info=env.step(4)
        self.assertFalse(info['success'])
        self.assertEqual(info['reward_parts']['terminal'],-25.)
        with self.assertRaises(RuntimeError):env.step(4)

    def test_success_terminal(self):
        env=Stage2Env()
        env.world.cleared=True
        _,r,done,info=env.step(4)
        self.assertTrue(done)
        self.assertEqual(r,100.)
        self.assertTrue(info['success'])

    def test_ppo_multienv_training_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            import argparse
            args=argparse.Namespace(steps=32,save_dir=tmp,resume=None,seed=42,width=32,lr=.0003,
                gamma_second=1.,gae_lambda=.98,entropy_start=.02,entropy_end=.001,entropy_total_steps=300000,
                value_coef=.5,max_grad_norm=.5,n_envs=4,n_steps=4,batch_size=8,ppo_epochs=2,
                clip=.15,target_kl=.03,greedy_mix=.05,eval_every=16,eval_trials=2,eval_seed=50000,
                save_every=16,log_every=1000,print_updates=False,live=False,render_every=30,fps=0)
            runner.training(args)
            model=Path(tmp)/'last.pt'
            self.assertTrue(model.exists())
            ck,net=runner.load_model(model)
            self.assertEqual(ck['version'],OBS_VERSION)
            self.assertEqual(ck['steps'],32)
            self.assertEqual(ck['config']['n_envs'],4)
            out=runner.evaluate(net,trials=3,mode='mixed',seed=60000)
            self.assertEqual(out['trials'],3)
            self.assertIn('spring1_pct',out)


if __name__=='__main__':unittest.main()
