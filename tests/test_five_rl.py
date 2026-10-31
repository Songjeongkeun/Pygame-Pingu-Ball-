"""원래 물리 궤적과의 일치, 보상 상태의 관측 가능성, 균등한 정책 통합을 검사한다."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from pingu_ball.engine import World
from rl.five_env import FiveEnv, FiveRawEnv, CAPACITY, FIELDS, GLOBALS
from rl.five_models import ALGORITHMS, action_probs
from rl.five_train import vector, make_model, distill, validation_score, evaluate, select_teachers, write

torch.set_num_threads(1)


class FiveTests(unittest.TestCase):
    def test_contract_and_original_reset_all_maps(self):
        for s in range(1, 6):
            env = FiveEnv(stage=s)
            check_env(env, warn=True, skip_render_check=True)
            a, _ = env.reset(seed=7)
            env.step(2)
            b, _ = env.reset(seed=7)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(env.action_space.n, 3)
            np.testing.assert_array_equal(a[24:29], np.eye(5)[s-1])
        for s in (0, 6, 10):
            with self.assertRaises(ValueError):
                FiveEnv(stage=s)

    def test_live_actions_match_unmodified_native_world(self):
        for stage in range(1, 6):
            env = FiveEnv(stage=stage)
            native = World(env.unwrapped.level, auto_reset=False)
            rng = np.random.default_rng(stage)
            for step in range(200):
                action = int(rng.integers(3))
                _, _, done, _, info = env.step(action)
                for _ in range(12):
                    dt = min(native.physics.dt, 40-native.time)
                    if native.failed or native.cleared or dt < 1e-10:
                        break
                    native.step(action-1, dt)
                for name in ('x', 'y', 'vx', 'vy'):
                    self.assertAlmostEqual(getattr(native.ball, name), getattr(env.unwrapped.world.ball, name), places=10)
                self.assertEqual(native.cleared, info['success'])
                self.assertEqual(native.failed, env.unwrapped.world.failed)
                if done:
                    break

    def test_contact_reward_once_capped_and_observable(self):
        env = FiveRawEnv(1)
        i = len(env.terrain)
        self.assertEqual(env.contact_credit([i]), .01)
        env.bonus_credit += .01
        self.assertEqual(env.observe()['entities'][i, -2], 1)
        self.assertEqual(env.contact_credit([i]), 0)
        self.assertEqual(env.contact_credit([i+1], failed=True), 0)
        env.bonus_credit = .095
        self.assertAlmostEqual(env.contact_credit([i+2, i+3]), .005)
        env.bonus_credit = .1
        self.assertEqual(env.contact_credit([i+4]), 0)
        env.reset()
        self.assertEqual(env.bonus_credit, 0)
        self.assertEqual(env.observe()['entities'][i, -2], 0)
        terminal = FiveRawEnv(1, reward_mode='terminal')
        self.assertEqual(terminal.contact_credit([len(terminal.terrain)]), 0)

    def test_finite_horizon_failure_has_exact_negative_reward(self):
        env = FiveEnv(stage=1, limit=.1)
        _, reward, done, truncated, info = env.step(1)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(reward, -1.)
        self.assertEqual(info['reason'], 'training_time_limit')

    def test_actual_special_landings_reach_reward_and_observation(self):
        for stage in (1, 3, 4, 5):
            env = FiveRawEnv(stage)
            q = next(iter(env.world.special.platforms.values()))
            env.world.special.sync(0.)
            x, y, width, height = q.rect_at(0., env.world.physics)
            b = env.world.ball
            b.x, b.y, b.vx, b.vy = x+width/2, y-env.world.physics.radius-.1, 0., 150.
            _, reward, _, _, _ = env.step(1)
            index = env.support_lookup[q.id]
            self.assertEqual(env.landing_index, index)
            self.assertAlmostEqual(reward, .01)
            self.assertEqual(env.observe()['entities'][index, -1], 1.)
            self.assertEqual(env.observe()['entities'][index, -2], 1.)

    def test_distillation_learns_map_conditioned_teachers_and_saves(self):
        vec = vector([s for s in range(1, 6) for _ in range(2)])
        try:
            student = make_model('ppo', vec, 123)
            data = {}
            for s in range(1, 6):
                x, _ = FiveEnv(stage=s).reset()
                data[s] = (np.repeat(x[None], 32, axis=0),
                           np.repeat(np.eye(3, dtype=np.float32)[(s-1)%3][None], 32, axis=0))
            history = distill(student, data, 40, 12)
            self.assertLess(history[-1]['kl'], history[0]['kl']*.5)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'student'
                student.save(path)
                loaded = ALGORITHMS['ppo'].load(path, device='cpu')
                x = vec.reset()
                np.testing.assert_allclose(action_probs(student, x), action_probs(loaded, x))
        finally:
            vec.close()

    def test_joint_auxiliary_update_has_every_map(self):
        vec = vector([s for s in range(1, 6) for _ in range(2)])
        try:
            model = make_model('ppo', vec, 8, joint=True)
            teachers = {s: make_model('a2c', vector([s]), s) for s in range(1, 6)}
            model.teachers = teachers
            model.joint_budget = 1280
            with tempfile.TemporaryDirectory() as tmp:
                model.set_logger(configure(tmp, ['csv']))
                model.learn(1280)
                self.assertTrue(np.isfinite(model.logger.name_to_value['train/teacher_kl']))
                model.save(Path(tmp)/'joint')
                loaded = ALGORITHMS['ppo'].load(Path(tmp)/'joint', device='cpu')
                self.assertFalse(getattr(loaded, 'teachers', None))
                model.logger.close()
            for teacher in teachers.values():
                teacher.get_env().close()
        finally:
            vec.close()

    def test_selection_protects_weakest_map(self):
        def ev(rates):
            return dict(min_success_rate=min(rates), mean_success_rate=np.mean(rates),
                        maps=[dict(greedy=dict(success=False), mean_success_seconds=None) for _ in rates])
        self.assertGreater(validation_score(ev([.8]*5)), validation_score(ev([1., 1., 1., 1., .2])))

    def test_sampled_evaluation_is_reproducible_and_does_not_consume_training_rng(self):
        vec = vector([1])
        try:
            model = make_model('a2c', vec, 55)
            before = torch.random.get_rng_state().clone()
            a = evaluate(model, [1], count=4, seed=1234)
            b = evaluate(model, [1], count=4, seed=1234)
            self.assertEqual(a, b)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertEqual(a['maps'][0]['episodes'], 4)
            self.assertTrue(model.policy.training)
        finally:
            vec.close()

    def test_teacher_selection_uses_validation_not_final_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stage in range(1, 6):
                for algorithm, validation, final in [('a2c', .8, 0.), ('ppo', .2, 1.)]:
                    ev = dict(min_success_rate=validation, mean_success_rate=validation,
                              maps=[dict(greedy=dict(success=False), mean_success_seconds=5.)])
                    write(root/'experts'/algorithm/'seed_0'/f'stage_{stage:02d}'/'summary.json',
                          dict(best_validation=dict(evaluation=ev), final=dict(mean_success_rate=final)))
            chosen = select_teachers(root, 0, .8)
            self.assertTrue(all(row['algorithm']=='a2c' and row['passes_gate'] for row in chosen.values()))


if __name__ == '__main__':
    unittest.main()
