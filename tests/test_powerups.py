"""Single-use pickups, continuous collisions and the learning interface."""
import json
import unittest

from pingu_ball.config import PHYSICS as P
from pingu_ball.engine import Level, Platform, PowerUp, Spike, World
from pingu_ball.environment import Environment
from pingu_ball.geometry import circle_rect_overlap
from pingu_ball.levels import make_levels
from pingu_ball.specials import Device


def world_with(items=(), blocks=(), spikes=(), devices=()):
    level = Level(1, 'Power-up test', '', list(blocks), [], list(spikes),
                  (128., 400.), (1500., 700., 64., 64.), powerups=list(items),specials=list(devices))
    return World(level, auto_reset=False)


class PowerUpTests(unittest.TestCase):
    def test_two_distinct_items_are_held_together_and_dash_is_used_first(self):
        w = world_with([PowerUp(128., 400., 'forward'), PowerUp(160., 400., 'up')])
        w.step(0)
        self.assertEqual(w.ball.power_up, 'forward')
        self.assertEqual(w.collected_powerups, {0})
        self.assertEqual(w.score, 7)
        w.ball.x = 160.
        w.step(0)
        self.assertEqual(w.collected_powerups, {0,1})
        self.assertEqual(w.ball.held_powerups,('forward','up'))
        self.assertTrue(w.use_power_up(-1))
        self.assertEqual(w.ball.power_up, 'up')
        self.assertTrue(w.use_power_up(1))
        self.assertIsNone(w.ball.power_up)
        self.assertFalse(w.use_power_up(1))
        self.assertEqual(w.collected_powerups, {0, 1})

    def test_dash_has_priority_even_when_jump_is_picked_up_first(self):
        w=world_with([PowerUp(128.,400.,'up'),PowerUp(128.,400.,'forward')])
        w.step(0)
        self.assertEqual(w.ball.held_powerups,('forward','up'))
        w.step(0,use_power=True,power_direction=-1)
        self.assertLess(w.ball.vx,0.)
        self.assertEqual(w.ball.held_powerups,('up',))
        w.step(0,use_power=True)
        self.assertEqual(w.ball.held_powerups,())
        self.assertIsNone(w.ball.forward_target_x)
        self.assertAlmostEqual(w.ball.vx,0.)

    def test_duplicate_items_do_not_stack_extra_charges(self):
        w=world_with([PowerUp(128.,400.,'forward'),PowerUp(128.,400.,'forward')])
        w.step(0)
        self.assertEqual(w.collected_powerups,{0,1})
        self.assertEqual(w.ball.held_powerups,('forward',))
        self.assertTrue(w.use_power_up(1))
        self.assertFalse(w.use_power_up(1))

    def test_failure_clears_both_held_items_and_restores_both_pickups(self):
        w=world_with([PowerUp(128.,400.,'up'),PowerUp(128.,400.,'forward')])
        w.auto_reset=True
        w.step(0)
        self.assertEqual(w.ball.held_powerups,('forward','up'))
        w.ball.y=w.physics.height+P.radius+1
        w.step(0)
        self.assertEqual(w.failures,1)
        self.assertEqual(w.ball.held_powerups,())
        self.assertFalse(w.collected_powerups)

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

    def test_forward_jump_is_one_shallow_four_tile_arc_in_both_directions(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                w = world_with()
                w.ball.x = 800.
                w.ball.power_up = 'forward'
                apex=P.forward_jump_up_speed/P.gravity
                w.ball.vy=500. # Reset falling motion into a fresh forward leap.
                w.step(0,apex,use_power=True,power_direction=direction)
                self.assertAlmostEqual(w.ball.x,800.+direction*P.forward_jump_distance/2,places=5)
                self.assertAlmostEqual(w.ball.y,400.-P.forward_jump_height,places=5)
                self.assertAlmostEqual(w.ball.vy,0.,places=5)
                self.assertIsNotNone(w.ball.forward_target_x)
                self.assertEqual(w.ball.boost_until,0.)
                w.step(0,apex)
                self.assertAlmostEqual(w.ball.x,800.+direction*P.forward_jump_distance,places=5)
                self.assertAlmostEqual(w.ball.y,400.,places=5)
                self.assertIsNone(w.ball.forward_target_x)
                w.step(0, .1)
                self.assertAlmostEqual(w.ball.vx, 0., places=5)
                self.assertFalse(w.use_power_up(direction))

    def test_forward_jump_wall_keeps_gravity_and_remaining_time(self):
        w = world_with(blocks=[Platform('wall', 256., 0., 64., 700.)])
        w.ball.power_up = 'forward'
        w.step(0, .3, use_power=True, power_direction=1)
        self.assertAlmostEqual(w.ball.x, 242., places=4)
        self.assertAlmostEqual(w.time, .3, places=5)
        self.assertAlmostEqual(w.ball.y, 400.-P.forward_jump_up_speed*.3+.5*P.gravity*.3**2, places=4)
        self.assertIsNone(w.last_landing)
        self.assertIsNone(w.ball.forward_target_x)
        self.assertEqual(w.ball.held_powerups,())

    def test_forward_jump_lands_across_a_gap_and_restores_normal_bounce(self):
        w=world_with(blocks=[Platform('landing',352.,414.)])
        w.ball.power_up='forward'
        w.step(0,use_power=True,power_direction=1)
        for _ in range(120):
            w.step(0)
            if w.last_landing=='landing':break
        self.assertEqual(w.last_landing,'landing')
        self.assertLess(w.ball.vy,0.)
        self.assertIsNone(w.ball.forward_target_x)
        self.assertFalse(w.failed)

    def test_forward_jump_ceiling_stops_the_arc_without_tunnelling(self):
        w=world_with(blocks=[Platform('ceiling',0.,350.,1600.,20.)])
        w.ball.power_up='forward'
        w.step(0,.4,use_power=True,power_direction=1)
        self.assertIsNone(w.ball.forward_target_x)
        self.assertGreater(w.ball.vy,0.)
        self.assertGreaterEqual(w.ball.y,384.)

    def test_portal_keeps_the_remaining_forward_jump_distance(self):
        w=world_with(devices=[Device('in',407,200.,352.,target='out'),
                             Device('out',408,700.,352.,target='in',options={'exit_angle':0.})])
        w.ball.power_up='forward'
        w.step(0,.2,use_power=True,power_direction=1)
        self.assertTrue(any(k=='teleport' for k,p in w.events))
        self.assertGreater(w.ball.x,700.)
        self.assertAlmostEqual(w.ball.forward_target_x-w.ball.x,
                               P.forward_jump_distance-P.forward_jump_speed*.2,places=5)

    def test_blue_laser_replaces_forward_jump_with_its_own_rebound(self):
        w=world_with(devices=[Device('blue',502,240.,300.,options={'color':'blue','angle':90.,'length':300.})])
        w.ball.power_up='forward'
        w.step(0,.3,use_power=True,power_direction=1)
        self.assertTrue(any(k=='repel' for k,p in w.events))
        self.assertIsNone(w.ball.forward_target_x)
        self.assertLess(w.ball.vx,0.)
        self.assertFalse(w.failed)

    def test_forward_jump_trajectory_matches_at_30_60_and_120_hz(self):
        states=[]
        for hz in (30,60,120):
            w=world_with();w.ball.power_up='forward'
            for tick in range(int(.6*hz)):
                w.step(0,1/hz,use_power=tick==0,power_direction=1)
            states.append((w.ball.x,w.ball.y,w.ball.vx,w.ball.vy))
        for state in states[1:]:
            for actual,expected in zip(state,states[0]):self.assertAlmostEqual(actual,expected,places=5)

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
        self.assertIsNone(w.ball.forward_target_x)
        w.step(0)
        self.assertEqual(w.ball.power_up, 'forward')

    def test_spikes_take_priority_over_pickup(self):
        w = world_with([PowerUp(96., 400., 'up')], spikes=[Spike(64., 384.)])
        w.ball.x = 96.
        w.step(0)
        self.assertTrue(w.failed)
        self.assertFalse(w.collected_powerups)

    def test_document_items_are_outside_solid_tiles(self):
        for level in make_levels()[1:]:
            world = World(level)
            for item in level.powerups:
                self.assertFalse(any(circle_rect_overlap(item.x, item.y, item.size/2, q.rect_at(0))
                                     for q in world.platforms if q.active(0, {})))

    def test_learning_observation_and_explicit_activation(self):
        env = Environment(1)
        env.world = world_with([PowerUp(128., 400., 'up')])
        observation, reward, terminated, truncated, _ = env.step(0)
        self.assertEqual(observation['power_up'], 'up')
        self.assertEqual(observation['held_powerups'],['up'])
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
