"""Single-use pickups, continuous collisions and the learning interface."""
import json
import unittest

from pingu_ball.config import PHYSICS as P
from pingu_ball.engine import Level, Platform, PowerUp, Spike, World
from pingu_ball.environment import Environment
from pingu_ball.geometry import circle_rect_overlap
from pingu_ball.levels import make_levels


def world_with(items=(), blocks=(), spikes=()):
    level = Level(1, 'Power-up test', '', list(blocks), [], list(spikes),
                  (128., 400.), (1500., 700., 64., 64.), powerups=list(items))
    return World(level, auto_reset=False)


class PowerUpTests(unittest.TestCase):
    def test_pickup_is_one_charge_and_does_not_change_score(self):
        w = world_with([PowerUp(128., 400., 'forward'), PowerUp(160., 400., 'up')])
        w.step(0)
        self.assertEqual(w.ball.power_up, 'forward')
        self.assertEqual(w.collected_powerups, {0})
        self.assertEqual(w.score, 7)
        w.ball.x = 160.
        w.step(0)
        self.assertEqual(w.collected_powerups, {0})
        self.assertTrue(w.use_power_up(-1))
        self.assertIsNone(w.ball.power_up)
        self.assertFalse(w.use_power_up(1))
        w.step(0)
        self.assertEqual(w.ball.power_up, 'up')
        self.assertEqual(w.collected_powerups, {0, 1})

    def test_fast_ball_collects_sensor_without_tunnelling(self):
        w = world_with([PowerUp(160., 400., 'forward', size=2.)])
        w.ball.vx = 1000.
        w.step(1, .05)
        self.assertEqual(w.ball.power_up, 'forward')

    def test_up_jump_resets_fall_velocity_and_reaches_192px(self):
        w = world_with()
        w.ball.power_up = 'up'
        w.ball.vx, w.ball.vy = 90., 500.
        w.step(0, P.power_up_speed/P.gravity, use_power=True, power_direction=-1)
        self.assertAlmostEqual(w.ball.y, 400.-192., places=5)
        self.assertAlmostEqual(w.ball.vy, 0., places=5)
        self.assertAlmostEqual(w.ball.x, 128., places=5)
        self.assertIsNone(w.ball.power_up)

    def test_forward_jump_in_both_directions_then_returns_to_normal(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                w = world_with()
                w.ball.x = 800.
                w.ball.power_up = 'forward'
                w.step(0, P.forward_jump_duration, use_power=True, power_direction=direction)
                self.assertAlmostEqual(w.ball.x, 800.+direction*128., places=5)
                self.assertAlmostEqual(w.ball.vy, -P.bounce_speed+P.gravity*.3, places=5)
                w.step(0, .1)
                self.assertAlmostEqual(w.ball.vx, 0., places=5)
                self.assertFalse(w.use_power_up(direction))

    def test_forward_jump_wall_keeps_gravity_and_remaining_time(self):
        w = world_with(blocks=[Platform('wall', 256., 0., 64., 700.)])
        w.ball.power_up = 'forward'
        w.step(0, .3, use_power=True, power_direction=1)
        self.assertAlmostEqual(w.ball.x, 242., places=4)
        self.assertAlmostEqual(w.time, .3, places=5)
        self.assertAlmostEqual(w.ball.y, 400.-P.bounce_speed*.3+.5*P.gravity*.3**2, places=4)
        self.assertIsNone(w.last_landing)

    def test_up_jump_ceiling_stops_ascent(self):
        w = world_with(blocks=[Platform('ceiling', 0., 200., 1600.)])
        w.ball.power_up = 'up'
        w.step(0, .4, use_power=True)
        self.assertGreater(w.ball.vy, 0.)
        self.assertGreaterEqual(w.ball.y, 278.)
        self.assertIsNone(w.last_landing)

    def test_no_item_pause_and_terminal_states_do_not_consume(self):
        w = world_with()
        self.assertFalse(w.use_power_up(1))
        for flag in ('paused', 'failed', 'cleared'):
            w.reset()
            w.ball.power_up = 'up'
            setattr(w, flag, True)
            w.step(0, use_power=True)
            self.assertEqual(w.ball.power_up, 'up')

    def test_reset_restores_pickups_and_removes_held_charge_and_boost(self):
        w = world_with([PowerUp(128., 400., 'forward')])
        w.step(0)
        w.step(0, use_power=True, power_direction=1)
        w.reset()
        self.assertFalse(w.collected_powerups)
        self.assertIsNone(w.ball.power_up)
        self.assertEqual(w.ball.boost_until, 0.)
        w.step(0)
        self.assertEqual(w.ball.power_up, 'forward')

    def test_spikes_take_priority_over_pickup(self):
        w = world_with([PowerUp(96., 400., 'up')], spikes=[Spike(64., 384.)])
        w.ball.x = 96.
        w.step(0)
        self.assertTrue(w.failed)
        self.assertFalse(w.collected_powerups)

    def test_each_map_has_both_items_outside_solid_tiles(self):
        for level in make_levels():
            self.assertEqual({q.kind for q in level.powerups}, {'forward', 'up'})
            for item in level.powerups:
                self.assertFalse(any(circle_rect_overlap(item.x, item.y, item.size/2, q.rect_at(0))
                                     for q in level.platforms))

    def test_learning_observation_and_explicit_activation(self):
        env = Environment(1)
        env.world = world_with([PowerUp(128., 400., 'up')])
        observation, reward, terminated, truncated, _ = env.step(0)
        self.assertEqual(observation['power_up'], 'up')
        self.assertTrue(observation['powerups'][0]['collected'])
        self.assertEqual(reward, 0)
        observation, reward, _, _, _ = env.step(0, use_power=True, power_direction=-1)
        self.assertIsNone(observation['power_up'])
        self.assertLess(observation['ball'][3], 0.)
        self.assertEqual(reward, 0)
        json.dumps(observation, allow_nan=False)
        self.assertFalse(terminated or truncated)
        with self.assertRaises(ValueError):
            env.step(0, use_power=True, power_direction=2)
