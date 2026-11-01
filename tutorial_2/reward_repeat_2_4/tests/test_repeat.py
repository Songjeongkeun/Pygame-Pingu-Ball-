"""Dependency-free unit tests. Full PinguBall game must be tested by user smoke."""
from __future__ import annotations
import ast
import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from tutorial_2.reward_repeat_2_4.reward_components import Stage4Reward, Stage2Reward


ROOT = Path(__file__).resolve().parents[1]


def test_unmodified_personal_stage4_reward_once():
    class Lift:
        def rect_at(self, *args): return 380., 300., 64., 64.
    world = SimpleNamespace(level=SimpleNamespace(spawn=(96.,448.)),
          ball=SimpleNamespace(x=96.,y=450.), time=0., events=[],
          physics=SimpleNamespace(radius=14.),
          special=SimpleNamespace(platforms={'lift':Lift()}))
    s=Stage4Reward(world)
    assert s.on_tick(world)==0.
    world.ball.x=384.
    assert s.on_tick(world)==pytest.approx(12.96)
    assert s.on_tick(world)==0.
    world.events=[('land','lift')]; world.ball.x=400.; world.ball.y=200.;world.time=1.
    assert s.on_tick(world)==pytest.approx(24.)
    world.events=[('land','finish-0')];world.time=2.
    assert s.on_tick(world)==pytest.approx(28.)
    assert s.on_tick(world)==0.
    assert sum(s.detail()['reward_parts'].values())==pytest.approx(64.96)


def test_new_stage2_diagnostic_flags_are_information_only():
    s=Stage2Reward.__new__(Stage2Reward)
    s.tracker=SimpleNamespace(target_paid={'spring-1-0'},height_paid=set(),
                              horizontal_paid=set(),parts={'target_landing':10.})
    s.spring_seen={'spring-1-0','spring-2-0'}
    s.deck_seen={'middle-','upper-'}
    s.first_spring_paid=True
    s.spring1_contacts=2
    s.max_x=810.
    assert s.detail()['spring2']==1
    assert s.detail()['spring3']==0
    assert s.detail()['middle']==1
    assert s.detail()['upper']==1
    assert s.detail()['finish']==0
    assert s.detail()['reward_parts']=={'target_landing':10.}


def test_reward_env_exactly_one_repeat_change():
    source=(ROOT/'reward_env.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    assert 'MilestoneEnv(stage=self.stage, limit=180., repeat=repeat)' in source
    assert 'first_spring_bonus=0.' in source
    assert 'self.team.unwrapped.step(' in source
    assert 'self.team.observation(raw_obs)' in source
    assert 'float(step_points/100.)' in source
    assert "repeat not in (6, 12)" in source
    assert 'stage not in (2, 4)' in source


def test_no_demonstrations_and_same_team_ppo():
    source=(ROOT/'train.py').read_text(encoding='utf-8')
    ast.parse(source)
    assert "team_train.make_model('ppo', vec, seed, joint=False)" in source
    assert 'ENTROPY_COEF = .05' in source
    assert 'BC=False, DAgger=False, pretrained=False, teacher=False' in source
    assert 'm.ent_coef = ENTROPY_COEF' in source
    assert "m.n_steps == 128" in source
    assert "m.batch_size == 256" in source
    assert "m.n_epochs == 5" in source
    assert "m.clip_range(1.) == .2" in source
    assert "m.gamma == 1." in source
    assert "m.gae_lambda == .98" in source
    assert "m.learning_rate == .0003" in source
    assert 'self.physical_frames += int(info[\'tick_delta\'])' in source
    assert 'repeat=args.repeat' in source or 'args.repeat' in source


def load_train_with_fake_app(monkeypatch):
    # No SB3/Gym/pingu_ball wheel is available in tool runtime.
    # Fake ONLY the imports to test the callback and selection logic.
    for name in ('stable_baselines3','stable_baselines3.common',
                 'stable_baselines3.common.callbacks','stable_baselines3.common.vec_env',
                 'rl','rl.five_train','rl.milestone_train','rl.milestone_env',
                 'rl.milestone_models','rl.five_models','tutorial_2.reward_repeat_2_4.reward_env'):
        mod=ModuleType(name)
        monkeypatch.setitem(sys.modules,name,mod)
    class DummyCallback:
        def __init__(self,verbose=0):self.verbose=verbose
    sys.modules['stable_baselines3.common.callbacks'].BaseCallback=DummyCallback
    sys.modules['stable_baselines3.common.vec_env'].DummyVecEnv=lambda f:f
    sys.modules['rl.milestone_train'].install=lambda:None
    sys.modules['rl.milestone_env'].MilestoneEnv=object
    sys.modules['rl.milestone_env'].OBS_SIZE=99
    sys.modules['rl.milestone_models'].MilestonePPO=object
    sys.modules['rl.five_models'].action_probs=lambda model,obs:np.array([[1/3]*3]*4)
    sys.modules['tutorial_2.reward_repeat_2_4.reward_env'].RewardOnlyEnv=object
    sys.modules['rl'].five_train=sys.modules['rl.five_train']
    sys.modules['rl'].milestone_train=sys.modules['rl.milestone_train']
    sys.modules['rl'].milestone_env=sys.modules['rl.milestone_env']
    sys.modules['rl'].milestone_models=sys.modules['rl.milestone_models']
    if 'tutorial_2.reward_repeat_2_4.train' in sys.modules:
        monkeypatch.delitem(sys.modules,'tutorial_2.reward_repeat_2_4.train')
    mod=importlib.import_module('tutorial_2.reward_repeat_2_4.train')
    return mod


def test_physics_budget_counter_and_resume_directory(monkeypatch,tmp_path):
    m=load_train_with_fake_app(monkeypatch)
    cfg=SimpleNamespace(out=tmp_path,stage=2,reward='personal',repeat=12,seed=42,
                        eval_every_frames=60000,physics_budget=100,
                        no_early_stop=False,collapse_entropy=.1)
    monitor=m.PhysicsMonitor(m.folder_for(cfg),cfg)
    monitor.locals={'infos':[{'tick_delta':12,'success':False},{'tick_delta':3,'success':True},
                             {'tick_delta':12,'success':False},{'tick_delta':0,'success':False}],
                    'dones':[False,True,False,False]}
    assert monitor._on_step() is True
    assert monitor.physical_frames==27 and monitor.episodes==1 and monitor.training_wins==1
    monitor.locals={'infos':[{'tick_delta':12}]*4, 'dones':[False]*4}
    assert monitor._on_step() is True and monitor.physical_frames==75
    assert monitor._on_step() is False and monitor.physical_frames==123
    assert monitor.stopped_for_budget
    assert str(m.folder_for(cfg)).endswith('stage_02/personal/repeat_12/seed_42')


def test_checkpoint_scoring_rewards_second_spring(monkeypatch):
    m=load_train_with_fake_app(monkeypatch)
    first=dict(stage=2,reward_mode='personal',success_rate=0.,spring3_rate=0.,
               upper_rate=0.,spring2_rate=0.1,mean_target_landings=.5,
               first_spring_rate=.9,mean_max_x=700.)
    second=dict(first,spring2_rate=.4,mean_target_landings=.6)
    assert m.score(second)>m.score(first)
    assert m.ENTROPY_COEF==.05



def test_evaluation_separately_reports_second_spring(monkeypatch):
    m=load_train_with_fake_app(monkeypatch)
    class FakePolicy:
        training=True
        def set_training_mode(self,v):self.training=v
    class FakeVec:
        def reset(self):return np.zeros((4,99),dtype=np.float32)
        def step(self,actions):
            infos=[dict(success=False,reason='fail',episode_return=-15.,bonus_return=10.,
                        seconds=10.,max_x=780.,first_spring=1,middle=1,spring2=1,
                        upper=0,spring3=0,finish=0,target_landing_count=1,
                        springs_seen=2,reward_parts={'target_landing':10.}) for _ in range(4)]
            return self.reset(),np.zeros(4),np.ones(4,dtype=bool),infos
        def close(self):pass
    monkeypatch.setattr(m,'make_vec',lambda *a,**k: FakeVec())
    policy=FakePolicy()
    model=SimpleNamespace(policy=policy)
    ev=m.evaluate(model,2,'personal',12,trials=10,seed=70000)
    assert ev['trials']==10 and ev['wins']==0
    assert ev['first_spring_rate']==1. and ev['middle_rate']==1.
    assert ev['spring2_rate']==1. and ev['spring3_rate']==0.
    assert ev['mean_target_landings']==1. and ev['mean_max_x']==780.
    assert ev['repeat']==12 and ev['mean_policy_entropy']==pytest.approx(np.log(3))
    assert policy.training is True


def test_identical_common_contract_6_or_12(monkeypatch):
    m=load_train_with_fake_app(monkeypatch)
    class FakeWorldEnv:
        def __init__(self,repeat):
            self.unwrapped=SimpleNamespace(repeat=repeat)
            self.team=SimpleNamespace(unwrapped=self.unwrapped)
            self.action_space=SimpleNamespace(n=3)
            self.observation_space=SimpleNamespace(shape=(99,))
    monkeypatch.setattr(m,'RewardOnlyEnv',lambda stage,repeat:FakeWorldEnv(repeat))
    monkeypatch.setattr(m,'MilestoneEnv',lambda stage,limit,repeat:FakeWorldEnv(repeat))
    assert m.make_env(2,'personal',6).unwrapped.repeat==6
    assert m.make_env(2,'personal',12).unwrapped.repeat==12
    assert m.make_env(2,'original',6).unwrapped.repeat==6
    assert m.make_env(2,'original',12).unwrapped.repeat==12
    with pytest.raises(ValueError):m.make_env(1,'personal',12)
    with pytest.raises(ValueError):m.make_env(2,'personal',9)
