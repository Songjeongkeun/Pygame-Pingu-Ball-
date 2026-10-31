"""Verify user reward semantics, real landings, original physics, and model IO."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from pingu_ball.engine import World
from rl.milestone_env import (MilestoneEnv, MilestoneRawEnv, MilestoneTracker,
                              Milestone, definitions, terminal_points, OBS_SIZE, GLOBALS)
from rl.milestone_train import install, make_model, entropy
from rl import five_train as training
from rl.milestone_models import ALGORITHMS
from rl.five_models import action_probs

torch.set_num_threads(1)


class MilestoneTests(unittest.TestCase):
    def test_contract_reset_and_observable_milestones(self):
        for stage in range(1,6):
            env = MilestoneEnv(stage=stage)
            check_env(env,warn=True,skip_render_check=True)
            x,_ = env.reset()
            self.assertEqual(x.shape,(OBS_SIZE,))
            self.assertEqual(env.unwrapped.limit,180.)
            self.assertEqual(env.unwrapped.repeat,6)
            self.assertEqual(env.action_space.n,3)
            np.testing.assert_array_equal(x[24:29],np.eye(5)[stage-1])
            self.assertEqual(x[29+8],1.)
            env.unwrapped.tracker.land(env.unwrapped.specs[0].source[0],1)
            armed = env.observation(env.unwrapped.observe())
            self.assertEqual(armed[29+9],1.)
            np.testing.assert_array_equal(env.reset()[0],x)
            env.close()

    def test_key_requires_source_and_actual_target_event(self):
        t = MilestoneTracker(definitions(2))
        self.assertEqual(t.land('middle-0',1),[])
        self.assertEqual(t.land('spring-1-0',2),[])
        self.assertEqual(t.land('start-0',3),[])
        self.assertEqual(t.land('middle-0',4),['spring-to-middle'])
        self.assertEqual(t.land('middle-1',5),[])
        self.assertEqual(t.land('spring-1-0',6),[])
        self.assertEqual(t.land('middle-2',7),[])

    def test_connected_moving_group_never_pays_for_internal_motion(self):
        t = MilestoneTracker(definitions(3))
        for tick in range(50):
            self.assertEqual(t.land(f'ferry-{tick%3}',tick),[])
        self.assertEqual(t.land('finish-0',51),['ferry-to-finish'])
        self.assertEqual(t.land('finish-1',52),[])
        with self.assertRaises(ValueError):
            MilestoneTracker((Milestone('bad',('ferry-0',),('ferry-0',)),))

    def test_terminal_once_and_deadline(self):
        self.assertEqual(terminal_points(False,True,True),-25.)
        self.assertEqual(terminal_points(True,False,True),100.)
        self.assertEqual(terminal_points(False,False,False),0.)
        env = MilestoneEnv(stage=1,limit=.05)
        _,r,done,truncated,info = env.step(1)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(r,-.25)
        self.assertEqual(info['episode_return'],-25.)
        self.assertEqual(len(info['reward_events']),1)
        with self.assertRaises(RuntimeError): env.step(1)
        env.close()

    def test_real_landings_pay_ten_only_after_special(self):
        raw = MilestoneRawEnv(2)
        def land(name):
            q = raw.platform_by_id[name]
            x,y,w,h = q.rect_at(raw.world.time,raw.world.physics)
            b = raw.world.ball
            b.x,b.y,b.vx,b.vy = x+w/2,y-raw.world.physics.radius-.1,0.,150.
            return raw.step(1)
        self.assertEqual(land('middle-0')[1],0.)
        self.assertEqual(land('spring-1-0')[1],0.)
        _,reward,_,_,info = land('middle-0')
        self.assertEqual(reward,.1)
        self.assertEqual(info['episode_return'],10.)
        self.assertEqual(info['milestones'],['spring-to-middle'])
        self.assertEqual(land('middle-1')[1],0.)
        raw.reset()
        self.assertEqual(raw.tracker.completed,set())
        self.assertEqual(raw.tracker.armed,set())
        raw.close()

    def test_same_step_milestone_and_goal_add_without_double_credit(self):
        raw = MilestoneRawEnv(2)
        raw.tracker.land('spring-1-0',0)
        def finish(*args):
            raw.world.events = [('land','middle-0')]
            raw.world.cleared = True
        with patch.object(raw.world,'step',finish):
            _,reward,done,_,info = raw.step(1)
        self.assertTrue(done)
        self.assertEqual(reward,1.1)
        self.assertEqual(info['episode_return'],110.)
        self.assertEqual(info['base_return'],100.)
        self.assertEqual(info['bonus_return'],10.)
        raw.close()

    def test_everyday_movement_bounce_and_survival_are_zero(self):
        raw = MilestoneRawEnv(1)
        for _ in range(20):
            _,reward,done,_,info = raw.step(1)
            self.assertFalse(done)
            self.assertEqual(reward,0.)
        self.assertEqual(info['milestones'],[])
        raw.close()

    def test_native_physics_unmodified(self):
        for stage in range(1,6):
            env = MilestoneEnv(stage=stage)
            raw = env.unwrapped
            native = World(raw.level,auto_reset=False)
            rng = np.random.default_rng(stage)
            for _ in range(160):
                a = int(rng.integers(3))
                _,_,done,_,info = env.step(a)
                for _ in range(6):
                    if native.failed or native.cleared: break
                    native.step(a-1,native.physics.dt)
                for name in ('x','y','vx','vy'):
                    self.assertAlmostEqual(getattr(native.ball,name),getattr(raw.world.ball,name),places=10)
                self.assertEqual(native.cleared,info['success'])
                if done: break
            env.close()

    def test_entropy_schedule_and_model_training_roundtrip(self):
        for p,wanted in [(0,.02),(.2,.02),(.8,.005),(1,.001)]:
            self.assertAlmostEqual(entropy(p),wanted)
        self.assertEqual(entropy(0,True),.005)
        install()
        vec = training.vector([1]*4,'milestone')
        try:
            models = [make_model(a,vec,123) for a in ('a2c','ppo')]
            for a,b in zip(models[0].policy.parameters(),models[1].policy.parameters()):
                self.assertTrue(torch.equal(a,b))
            for algorithm,model in zip(('a2c','ppo'),models):
                self.assertEqual(model.gamma,1.)
                self.assertEqual(model.learning_rate,.0003)
                np.testing.assert_allclose(action_probs(model,vec.reset()),1/3,atol=1e-7)
                model.learn(1024)
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp)/'policy.zip'
                    model.save(p)
                    loaded = ALGORITHMS[algorithm].load(p,device='cpu')
                    obs = vec.reset()
                    np.testing.assert_allclose(action_probs(model,obs),action_probs(loaded,obs))
        finally:
            vec.close()


if __name__=='__main__':
    unittest.main()
