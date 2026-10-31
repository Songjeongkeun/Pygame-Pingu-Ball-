import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tutorial4_ppo_setting.environment import Stage4Env, RewardConfig, ACTIONS
from tutorial4_ppo_setting.network import ActorCritic, action_distribution, action_sample
from tutorial4_ppo_setting.learn import build_batch, ppo_update
import tutorial4_ppo_train as train


from mock_world import mocked_engine


class FakeEnv:
    def __init__(self):
        self.obs_dim=36
        self.reset()
    def reset(self):
        self.turn=0
        self.actions=np.zeros(9,dtype=int)
        return np.zeros(36,dtype=np.float32)
    def step(self,action):
        self.turn+=1
        self.actions[action]+=1
        terminal=self.turn>=3
        obs=np.zeros(36,dtype=np.float32)
        obs[0]=self.turn/5
        info={'success':False,'lift':int(self.turn>=2),'lift_high':0,'upper':0,
              'milestone':0,'timeout':False,'max_x':150.,'seconds':self.turn*.1,
              'reward_total':-25.,'reason':'fake fall','actions':self.actions.tolist(),
              'frames':12}
        return obs, (-25 if terminal else .5), terminal,info


class Tests(unittest.TestCase):
    def test_original_reward_unchanged(self):
        rc=RewardConfig()
        self.assertEqual(rc.success,100)
        self.assertEqual(rc.fail_or_timeout,-25)
        self.assertEqual(rc.milestone,10)
        self.assertEqual(rc.lift_first,12)
        self.assertEqual(rc.lift_high,12)
        self.assertEqual(rc.upper_first,18)
        self.assertEqual(rc.approach_per_pixel,.045)
    def test_original_bonus_paid(self):
        with mocked_engine():
            env=Stage4Env()
            self.assertEqual(env._landing('lift'),12)
            self.assertEqual(env._landing('finish-0'),28)
            self.assertEqual(env._landing('finish-7'),0)
            self.assertEqual(env.rewards_paid['upper'],18)
    def test_obs_and_actions(self):
        with mocked_engine():
            env=Stage4Env()
            self.assertEqual(env.obs_dim,36)
            self.assertEqual(len(ACTIONS),9)
    def test_mixture_probability(self):
        l=torch.tensor([[0.,0.,1.]],requires_grad=True)
        dist=action_distribution(l,greedy_mix=.05)
        self.assertTrue(torch.allclose(dist.probs,.95*torch.softmax(l,-1)+.05*torch.tensor([[0.,0.,1.]])))
        dist.log_prob(torch.tensor([2])).backward()
        self.assertTrue(torch.isfinite(l.grad).all())
    def test_entropy_linear(self):
        self.assertAlmostEqual(train.entropy_at(0,300000),.02)
        self.assertAlmostEqual(train.entropy_at(150000,300000),.0105)
        self.assertAlmostEqual(train.entropy_at(300000,300000),.001)
        self.assertAlmostEqual(train.entropy_at(999999,300000),.001)
    def test_gamma_one_and_no_gae_leak(self):
        class ZeroValue(torch.nn.Module):
            def forward(self,obs):
                return torch.zeros(len(obs),9),torch.zeros(len(obs))
        def tr(rew,done=False):
            return {'obs':np.zeros(36,dtype=np.float32),'next_obs':np.zeros(36,dtype=np.float32),
                    'reward':rew,'done':done,'frames':72,'action':1,'logp':-2.197}

        _,_,_,adv,returns=build_batch(ZeroValue(),[[tr(0),tr(25)],[tr(50,True)]],
                                       gamma_per_second=1.,gae_lambda=.98)
        self.assertAlmostEqual(adv[0].item(),.98,places=5)
        self.assertAlmostEqual(adv[1].item(),1.,places=5)
        self.assertAlmostEqual(adv[2].item(),2.,places=5)
        self.assertTrue(torch.allclose(adv,returns))
    def test_policy_updates_with_four_streams(self):
        torch.manual_seed(12)
        model=ActorCritic(36,9,32)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        streams=[]
        for e in range(4):
            rows=[]
            for k in range(8):
                obs=np.random.default_rng(e*100+k).standard_normal(36).astype(np.float32)
                ns=np.random.default_rng(10000+e*100+k).standard_normal(36).astype(np.float32)
                a=(e+k)%9
                with torch.no_grad():
                    logits,_=model(torch.tensor(obs).unsqueeze(0))
                    lp=action_distribution(logits,greedy_mix=.05).log_prob(torch.tensor([a])).item()
                rows.append(dict(obs=obs,next_obs=ns,action=a,logp=lp,reward=float(k%3-1)*25,
                                 done=(k==7),frames=[12,30,72][k%3]))
            streams.append(rows)
        prev=model.actor.weight.detach().clone()
        logs=ppo_update(model,optimizer,streams,epochs=2,batch_size=16)
        self.assertEqual(logs['batch_transitions'],32)
        self.assertFalse(torch.equal(prev,model.actor.weight))
        self.assertGreaterEqual(logs['epochs_used'],1)
    def test_training_and_resume_mock_world(self):
        args=dict(steps=32,save_dir='',resume=None,seed=42,width=16,lr=.0003,
                  gamma_second=1.,gae_lambda=.98,entropy_start=.02,entropy_end=.001,
                  entropy_total_steps=300000,value_coef=.5,max_grad_norm=.5,
                  n_envs=4,n_steps=4,batch_size=8,ppo_epochs=2,clip=.15,
                  target_kl=.03,greedy_mix=.05,eval_every=16,eval_trials=3,
                  eval_seed=50000,save_every=16,log_every=10000,
                  print_updates=False,live=False,render_every=30,fps=0)
        with tempfile.TemporaryDirectory() as tmp:
            args['save_dir']=tmp
            with patch.object(train,'Stage4Env',FakeEnv):
                train.training(SimpleNamespace(**args))
                last=Path(tmp)/'last.pt'
                best=Path(tmp)/'best.pt'
                self.assertTrue(last.exists())
                self.assertTrue(best.exists())
                ck=train.read_checkpoint(last)
                self.assertEqual(ck['steps'],32)
                self.assertEqual(ck['upper_landing_extra'],18)
                self.assertEqual(ck['config']['n_envs'],4)
                args['steps']=48
                args['resume']=str(last)
                train.training(SimpleNamespace(**args))
                self.assertEqual(train.read_checkpoint(last)['steps'],48)
    def test_no_old_checkpoint_resume(self):
        args=SimpleNamespace(width=128,lr=.0003, gamma_second=1., gae_lambda=.98,entropy_start=.02,
             entropy_end=.001,entropy_total_steps=300000,n_envs=4,n_steps=128,batch_size=256,
             ppo_epochs=5,value_coef=.5,max_grad_norm=.5,clip=.15,target_kl=.03,greedy_mix=.05)
        old=vars(args).copy();old['n_envs']=1
        with self.assertRaisesRegex(RuntimeError,'n_envs'):
            train.ensure_same_resume_config(old,args)

if __name__=='__main__':unittest.main()
