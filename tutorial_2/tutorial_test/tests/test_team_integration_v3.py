"""실제 프로젝트 의존성을 설치한 후 원본 게임 물리/관측 동등성 시험."""
from __future__ import annotations
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

DEPS=all(importlib.util.find_spec(x) is not None for x in ('gymnasium','stable_baselines3'))

@unittest.skipUnless(DEPS,'현 실행환경에 gymnasium 또는 stable_baselines3가 없음')
class TestTeamIntegration(unittest.TestCase):
    def test_same_actions_same_physics_stage2(self):
        self._parity(2)

    def test_same_actions_same_physics_stage4(self):
        self._parity(4)

    def _parity(self,stage):
        import numpy as np
        from tutorial_2.tutorial_test.environment_v3 import RewardEnv
        a=RewardEnv(stage=stage,reward='common')
        b=RewardEnv(stage=stage,reward='combined')
        rng=np.random.default_rng(318)
        try:
            obs1,_=a.reset(seed=318)
            obs2,_=b.reset(seed=318)
            np.testing.assert_array_equal(obs1,obs2)
            for _ in range(200):
                action=int(rng.integers(3))
                obs1,r1,d1,t1,i1=a.step(action)
                obs2,r2,d2,t2,i2=b.step(action)
                np.testing.assert_allclose(obs1,obs2,rtol=0,atol=1e-5)
                self.assertEqual((d1,t1,i1['milestones'],i1['success']),
                                 (d2,t2,i2['milestones'],i2['success']))
                self.assertAlmostEqual(r2-r1,i2['personal_reward']/100.,places=5)
                if d1 or t1:
                    obs1,_=a.reset();obs2,_=b.reset()
                    np.testing.assert_allclose(obs1,obs2,rtol=0,atol=1e-5)
        finally:
            a.close();b.close()

if __name__=='__main__':unittest.main()
