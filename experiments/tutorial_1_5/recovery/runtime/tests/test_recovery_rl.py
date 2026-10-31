"""재학습 조건의 공정한 비교, 보상 보존과 체크포인트 선택을 검사한다."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from pingu_ball.engine import World
from rl.milestone_env import MilestoneEnv
from rl.recovery_train import (VARIANTS, make_model, vector, parameter_hash, success_score,
    progress_score, evaluate, run_job, sources)

torch.set_num_threads(1)


class RecoveryTests(unittest.TestCase):
    def test_same_initial_weights_and_uniform_policy_for_all_conditions(self):
        hashes=[]
        for settings in VARIANTS.values():
            vec=vector(2,settings['repeat'])
            model=make_model(vec,11,settings)
            hashes.append(parameter_hash(model))
            self.assertEqual(model.gamma,1.)
            self.assertEqual(model.learning_rate,.0001)
            self.assertEqual(model.n_epochs,3)
            with torch.no_grad():
                p=model.policy.get_distribution(torch.as_tensor(vec.reset())).distribution.probs
                np.testing.assert_allclose(p.numpy(),1/3,atol=1e-7)
            vec.close()
        self.assertEqual(len(set(hashes)),1)

    def test_progress_cannot_replace_actual_success(self):
        def ev(rate,milestones,source):
            return dict(success_rate=rate,mean_milestones=milestones,milestone_rate=bool(milestones),
                        source_landing_rate=source,mean_sources=source,mean_success_seconds=None)
        self.assertGreater(success_score(ev(.1,0,0)),success_score(ev(0,3,1)))
        self.assertGreater(progress_score(ev(0,1,1)),progress_score(ev(0,0,1)))
        self.assertGreater(progress_score(ev(0,0,1)),progress_score(ev(0,0,0)))

    def test_native_physics_and_zero_movement_reward_at_both_action_periods(self):
        for stage in (2,4):
            for repeat in (6,12):
                env=MilestoneEnv(stage=stage,repeat=repeat)
                raw=env.unwrapped
                native=World(raw.level,auto_reset=False)
                def tick(world,count,direction):
                    native.step(direction,min(native.physics.dt,180.-native.time))
                    for name in ('x','y','vx','vy'):
                        self.assertAlmostEqual(getattr(world.ball,name),getattr(native.ball,name),places=7)
                for i in range(15):
                    raw.step(1,on_tick=tick)
                    self.assertEqual(raw.episode_return,0.)
                env.close()

    def test_evaluation_does_not_consume_training_torch_rng(self):
        vec=vector(4,6);model=make_model(vec,5,dict(n_steps=16,gae_lambda=.98,repeat=6))
        before=torch.random.get_rng_state().clone()
        a=evaluate(model,4,6,2,123)
        b=evaluate(model,4,6,2,123)
        self.assertEqual(a,b)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        vec.close()

    def test_complete_small_run_saves_both_candidates_and_keeps_reward(self):
        from rl.five_train import read
        with tempfile.TemporaryDirectory() as tmp:
            result=run_job((tmp,'smoke',4,0,384,10000,2,2,
                            dict(n_steps=16,gae_lambda=.98,repeat=6)))
            row=read(result);folder=Path(result).parent
            self.assertEqual(row['config']['sources'],sources())
            self.assertGreaterEqual(row['physics_ticks'],384)
            self.assertTrue((folder/'best_success.zip').exists())
            self.assertTrue((folder/'best_progress.zip').exists())
            self.assertFalse(read(folder/'selection.json')['final_test_used_for_selection'])
            self.assertEqual(row['final']['episodes'],2)
            self.assertTrue(row['config']['from_scratch'])


if __name__=='__main__':unittest.main()
