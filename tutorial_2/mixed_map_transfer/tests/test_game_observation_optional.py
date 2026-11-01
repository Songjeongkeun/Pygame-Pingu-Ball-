"""Run native map schema smoke tests when this package sits in the game repository."""
import unittest

try:
    import gymnasium  # noqa: F401
    import pingu_ball.rl_maps  # noqa: F401
    import rl.milestone_env  # noqa: F401
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False


@unittest.skipUnless(NATIVE_AVAILABLE, 'Real game / Gymnasium unavailable in this interpreter')
class NativeMapSchemaSmoke(unittest.TestCase):
    def test_ten_native_maps_keep_trained_observation_shape(self):
        import numpy as np
        from rl.milestone_env import OBS_SIZE
        from tutorial_2.mixed_map_transfer.env import MixedMapTransferEnv
        for number in range(1, 11):
            with self.subTest(map=number):
                env = MixedMapTransferEnv(number, limit=60.)
                try:
                    obs, info = env.reset(seed=42)
                    self.assertEqual((OBS_SIZE,), obs.shape)
                    self.assertEqual(np.float32, obs.dtype)
                    self.assertTrue(np.isfinite(obs).all())
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertEqual(3, env.action_space.n)
                    self.assertEqual(6, env.repeat)
                    for _ in range(3):
                        obs, rew, done, truncated, step_info = env.step(1)
                        self.assertTrue(np.isfinite(obs).all())
                        if done or truncated:
                            break
                finally:
                    env.close()
