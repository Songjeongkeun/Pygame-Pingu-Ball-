"""외부 패키지가 없는 CI 환경에서도 env-wrapper의 보상 합산/리셋 회귀를 검사한다.
실제 게임이나 PPO 성능 시험은 아니며, 별도 통합 테스트에서 확인해야 한다.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace as S
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

class StubBase:
    def __init__(self,stage=2,reward_mode='milestone',limit=180.,repeat=6):
        self.stage=stage
        self.action_space=S(n=3)
        if stage==4:
            self.specs=(S(id='lift-to-finish',source=('lift',),target=('finish-0',)),)
            self.platform_by_id={'lift':S(kind='normal'),'finish-0':S(kind='normal')}
            devices={'lift':S(type_id=508,y=256.,options={'travel':192.})}
            goal=(960.,192.,64.,64.)
        else:
            self.specs=(S(id='spring-target',source=('spring-1-0',),target=('middle-0',)),)
            self.platform_by_id={'spring-1-0':S(kind='spring'),'middle-0':S(kind='normal')}
            devices={}
            goal=(1280.,256.,64.,64.)
        self.world=S(ball=S(x=288.,y=680.),physics=S(radius=14.,bounce_height=96.),
                     level=S(goal=goal),
                     special=S(devices=devices),failed=False,cleared=False,events=[],time=0.)
        self.reset()
    def reset(self,*,seed=None,options=None):
        self.world.ball.x,self.world.ball.y=288.,680.
        self.world.time=0.
        self.world.failed=self.world.cleared=False
        self.bonus_return=0.;self.episode_return=0.
        self.reward_events=[];self.ticks=0
        return self.observe(),self.info()
    def bounds(self,ids):
        pads={'spring-1-0':(448.,704.,64.,64.),'middle-0':(576.,576.,64.,64.),
              'lift':(384.,330. if self.ticks>=2 else 448.,64.,64.),
              'finish-0':(512.,256.,64.,64.)}
        return pads[ids[0]]
    def observe(self):
        return {'global':np.zeros(1,np.float32),'entities':np.zeros((1,1),np.float32),
                'local':np.zeros((1,1),np.float32)}
    def info(self,tick_delta=0):
        return dict(success=False,milestones=[],episode_return=self.episode_return)
    def step(self,action,on_tick=None):
        self.ticks+=1
        self.world.time=self.ticks*.05
        if self.stage==4:
            if self.ticks==1:
                self.world.ball.x,self.world.ball.y=400.,430.
                self.world.events=[('land','lift')]
            else:
                self.world.ball.x,self.world.ball.y=400.,315.
                self.world.events=[]
        elif self.ticks==1:
            self.world.ball.x,self.world.ball.y=470.,650.
            self.world.events=[('land','spring-1-0')]
        else:
            self.world.ball.x,self.world.ball.y=600.,562.
            self.world.events=[]
        if on_tick:on_tick(self.world,self.ticks,int(action)-1)
        self.episode_return+=10.
        return self.observe(),.1,False,False,self.info()

def load_stub_bridge():
    gym=ModuleType('gymnasium')
    spaces=ModuleType('gymnasium.spaces')
    spaces.Box=lambda *a,**k:object()
    gym.ObservationWrapper=object
    gym.spaces=spaces
    milestone=ModuleType('rl.milestone_env')
    milestone.MilestoneRawEnv=StubBase
    milestone.OBS_SIZE=3
    name='tutorial_2.tutorial_test._test_environment_v4'
    spec=importlib.util.spec_from_file_location(name,ROOT/'tutorial_2/tutorial_test/environment_v4.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'gymnasium':gym,'gymnasium.spaces':spaces,'rl.milestone_env':milestone}):
        spec.loader.exec_module(module)
    return module.RewardRawEnv

class TestRewardBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.Env=load_stub_bridge()

    def test_common_has_no_additional_reward(self):
        env=self.Env(stage=2,reward='common')
        for _ in range(2):obs,reward,done,trunc,info=env.step(2)
        self.assertAlmostEqual(reward,.1)
        self.assertAlmostEqual(info['personal_total'],0.)

    def test_combined_adds_only_spring_bonus(self):
        env=self.Env(stage=2,reward='combined')
        env.step(2)
        _,reward,_,_,info=env.step(2)
        self.assertAlmostEqual(reward,.11)
        self.assertAlmostEqual(info['personal_total'],1.)
        self.assertAlmostEqual(info['episode_return'],21.)

    def test_guided_adds_progress_without_duplication(self):
        env=self.Env(stage=2,reward='guided')
        _,reward1,_,_,info1=env.step(2)
        _,reward2,_,_,info2=env.step(2)
        self.assertGreater(reward1,.1)
        self.assertGreater(reward2,.1)
        self.assertEqual(info2['personal_counts']['spring_height'],1)
        self.assertEqual(info2['personal_counts']['spring_horizontal'],1)
        self.assertGreater(info2['personal_total'],1.)
        self.assertAlmostEqual(info2['episode_return'],20.+info2['personal_total'])

    def test_stage4_combined_lift_and_rise(self):
        env=self.Env(stage=4,reward='combined')
        _,first,_,_,info1=env.step(2)
        _,second,_,_,info2=env.step(2)
        self.assertAlmostEqual(first,.22)
        self.assertAlmostEqual(second,.22)
        self.assertAlmostEqual(info2['personal_total'],24.)
        self.assertEqual(info2['personal_counts']['moving_high'],1)

    def test_stage4_guided_compatible_with_lift(self):
        env=self.Env(stage=4,reward='guided')
        env.step(2)
        _,r,_,_,info=env.step(2)
        self.assertGreaterEqual(r,.22)
        self.assertEqual(info['personal_counts']['moving_first'],1)
        self.assertEqual(info['personal_counts']['moving_high'],1)
        self.assertGreater(info['personal_total'],24.)

    def test_reset_prevents_reward_leak(self):
        env=self.Env(stage=2,reward='guided')
        env.step(2);env.step(2)
        env.reset()
        self.assertEqual(env.bonus_tracker.total,0.)
        _,reward,_,_,info=env.step(2)
        self.assertAlmostEqual(info['episode_return'],10.+info['personal_total'])

if __name__=='__main__':unittest.main()
