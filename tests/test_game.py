import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from pingu_ball.config import PHYSICS as P
from pingu_ball.engine import World, Level, Platform, Collectible, Spike, advance_x
from pingu_ball.environment import Environment
from pingu_ball.geometry import sweep_circle_rect, circle_rect_overlap
from pingu_ball.levels import make_levels
from pingu_ball.storage import Progress
from pingu_ball.validate import run_route


def empty_world(platforms=None, fish=None, spikes=None, goal=(1500, 700, 64, 64), auto_reset=True):
    level = Level(1, 'Test', '', platforms or [], fish or [], spikes or [], (128., 400.), goal)
    return World(level, auto_reset=auto_reset)


class PhysicsTests(unittest.TestCase):
    def test_spec_dimensions_and_derived_values(self):
        self.assertEqual((P.width, P.height, P.tile, P.radius), (1600, 832, 64, 14))
        self.assertAlmostEqual(P.gravity, 948.148148148)
        self.assertAlmostEqual(P.bounce_speed, 426.6666666667)
        self.assertAlmostEqual(P.max_speed, 213.3333333333)
        self.assertAlmostEqual(P.platform_speed, P.max_speed/3)
        self.assertEqual(P.bounce_height+2*P.radius, 124)
        self.assertEqual(P.pad_height+2*P.radius, 172)

    def test_normal_free_flight_apex_and_return(self):
        world = empty_world()
        world.ball.vx = P.max_speed
        world.ball.vy = -P.bounce_speed
        for _ in range(54):
            world.step(1)
        self.assertAlmostEqual(world.ball.y, 304., places=5)
        self.assertAlmostEqual(world.ball.vy, 0., places=5)
        self.assertAlmostEqual(world.ball.x-128, 96., places=5)
        for _ in range(54):
            world.step(1)
        self.assertAlmostEqual(world.ball.x-128, 192., places=5)
        self.assertAlmostEqual(world.ball.y, 400., places=5)
        self.assertAlmostEqual(world.time, .9, places=7)

    def test_spring_height_flight_time_and_range(self):
        world = empty_world()
        world.ball.vx = P.max_speed
        world.ball.vy = -P.pad_speed
        apex = P.pad_speed/P.gravity
        world.step(1, apex)
        self.assertAlmostEqual(world.ball.y, 256., places=5)
        world.step(1, apex)
        self.assertAlmostEqual(world.ball.x-128., 235.151015, places=4)
        self.assertAlmostEqual(world.ball.y, 400., places=5)
        self.assertAlmostEqual(world.time, 1.102270384, places=6)

    def test_acceleration_from_rest_and_reverse(self):
        world = empty_world()
        world.ball.vy = -P.bounce_speed
        for _ in range(108):
            world.step(1)
        self.assertAlmostEqual(world.ball.x-128., 189.62962963, places=5)
        world.step(-1)
        self.assertLess(world.ball.vx, P.max_speed)

    def test_release_stops_at_zero_with_exact_distance(self):
        stopping_time = P.max_speed/P.deceleration
        self.assertAlmostEqual(stopping_time, .0148148148)
        distance, velocity = advance_x(P.max_speed, 0, P.deceleration, stopping_time+.1)
        self.assertAlmostEqual(distance, 1.58024691358)
        self.assertEqual(velocity, 0.)

    def test_entry_restart_and_failure_start_without_input(self):
        world = World(make_levels()[5])
        initial = world.ball.y
        self.assertEqual(initial, 704-110)
        for _ in range(54):
            world.step(0)
        self.assertAlmostEqual(world.ball.y, 690., places=4)
        self.assertAlmostEqual(world.time, .45)
        ferry = next(q for q in world.platforms if q.kind == 'moving')
        self.assertNotEqual(ferry.rect_at(world.time), ferry.rect_at(0))
        world.reset()
        world.step(0)
        self.assertGreater(world.ball.y, initial)
        world.fail('test')
        world.step(0)
        self.assertGreater(world.time, 0)

    def test_pause_freezes_ball_and_dynamic_platforms(self):
        world = World(make_levels()[6])
        world.step(0)
        world.paused = True
        before = (world.time, world.ball.x, world.ball.y, [q.rect_at(world.time) for q in world.platforms])
        for _ in range(120):
            world.step(1)
        self.assertEqual(before, (world.time, world.ball.x, world.ball.y, [q.rect_at(world.time) for q in world.platforms]))

    def test_landing_replaces_fall_speed(self):
        world = empty_world([Platform('floor', 0, 640, 1600)])
        world.ball.y = 620.
        world.ball.vy = 700.
        world.step(0, P.dt*2)
        self.assertEqual(world.last_landing, 'floor')
        self.assertLess(world.ball.vy, -418.)

    def test_spring_landing(self):
        world = empty_world([Platform('pad', 0, 640, 1600, kind='spring')])
        world.ball.y = 625.
        world.ball.vy = 100.
        for _ in range(3):
            world.step(0)
        self.assertEqual(world.last_landing, 'pad')
        self.assertLess(world.ball.vy, -500.)

    def test_moving_platform_never_carries_ball(self):
        world = empty_world([Platform('ferry', 100, 640, kind='moving', travel=128)])
        world.ball.y = 626.
        world.ball.vy = 1.
        for _ in range(108):
            world.step(0)
        self.assertAlmostEqual(world.ball.x, 128.)
        self.assertEqual(world.ball.vx, 0.)

    def test_moving_platform_speed_and_turnaround(self):
        block = Platform('ferry', 128, 640, kind='moving', travel=128)
        self.assertAlmostEqual(block.rect_at(1)[0]-128, P.platform_speed)
        self.assertAlmostEqual(block.rect_at(128/P.platform_speed)[0], 256)
        self.assertAlmostEqual(block.rect_at(256/P.platform_speed)[0], 128)

    def test_render_rates_preserve_fixed_tick_state(self):
        states = []
        for fps in (30, 60, 120):
            world = empty_world([Platform('floor', 0, 640, 1600)])
            accumulator = 0.
            ticks = 0
            for _ in range(fps*2):
                accumulator += 1/fps
                while accumulator+1e-12 >= P.dt:
                    world.step(1 if ticks < 60 else 0)
                    accumulator -= P.dt
                    ticks += 1
            states.append((world.ball.x, world.ball.y, world.ball.vx, world.ball.vy, world.time))
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[1], states[2])


class CollisionTests(unittest.TestCase):
    def test_holding_either_wall_preserves_free_vertical_motion(self):
        for direction, x in ((1, 242.), (-1, 334.)):
            for vy in (-120., 0., 120.):
                with self.subTest(direction=direction, vy=vy):
                    world = empty_world([Platform('wall', 256, 0, 64, 700)], auto_reset=False)
                    world.ball.x, world.ball.y, world.ball.vy = x, 200., vy
                    for _ in range(24):
                        world.step(direction)
                    self.assertAlmostEqual(world.time, .2, places=8)
                    self.assertAlmostEqual(world.ball.y, 200.+vy*.2+.5*P.gravity*.2**2, places=5)
                    self.assertAlmostEqual(world.ball.vy, vy+P.gravity*.2, places=5)
                    self.assertAlmostEqual(world.ball.x, x, places=4)
                    self.assertIsNone(world.last_landing)

    def test_wall_contact_releases_below_block(self):
        world = empty_world([Platform('wall', 256, 200)], auto_reset=False)
        world.ball.x, world.ball.y, world.ball.vy = 242., 260., 160.
        for _ in range(24):
            world.step(1)
        self.assertGreater(world.ball.x, 256.)
        self.assertAlmostEqual(world.ball.vy, 160.+P.gravity*.2, places=5)
        self.assertIsNone(world.last_landing)

    def test_wall_input_can_reverse_away_from_contact(self):
        world = empty_world([Platform('wall', 256, 0, 64, 700)], auto_reset=False)
        world.ball.x, world.ball.y = 242., 200.
        for _ in range(12):
            world.step(1)
        world.step(-1, .1)
        self.assertLess(world.ball.x, 230.)
        self.assertAlmostEqual(world.ball.vy, P.gravity*.2, places=5)

    def test_stacked_block_seams_do_not_interrupt_falling(self):
        world = empty_world([Platform(str(i), 256, 128+64*i) for i in range(8)], auto_reset=False)
        world.ball.x, world.ball.y, world.ball.vy = 242., 150., 120.
        for _ in range(48):
            world.step(1)
        self.assertFalse(world.failed)
        self.assertIsNone(world.last_landing)
        self.assertAlmostEqual(world.ball.y, 150.+120.*.4+.5*P.gravity*.4**2, places=5)
        self.assertAlmostEqual(world.ball.vy, 120.+P.gravity*.4, places=5)

    def test_moving_wall_and_turnaround_preserve_fall_speed(self):
        for phase in (0., 63.6, 64., 127.6):
            for direction in (-1, 1):
                with self.subTest(phase=phase, direction=direction):
                    wall = Platform('moving', 400, 0, 64, 700, kind='moving', travel=64., phase=phase)
                    world = empty_world([wall], auto_reset=False)
                    left = wall.rect_at(0)[0]
                    world.ball.x = left-P.radius if direction > 0 else left+64+P.radius
                    world.ball.y = 200.
                    for _ in range(24):
                        world.step(direction)
                    self.assertFalse(world.failed)
                    self.assertIsNone(world.last_landing)
                    self.assertAlmostEqual(world.ball.y, 200.+.5*P.gravity*.2**2, places=5)
                    self.assertAlmostEqual(world.ball.vy, P.gravity*.2, places=5)

    def test_upper_corners_do_not_bounce_or_activate_blocks(self):
        for kind in ('normal', 'spring', 'breakable', 'switch', 'launcher'):
            for x in (250., 326.):
                with self.subTest(kind=kind, x=x):
                    world = empty_world([Platform('corner', 256, 300, kind=kind, group='A')], auto_reset=False)
                    world.ball.x, world.ball.y, world.ball.vy = x, 280., 120.
                    for _ in range(24):
                        world.step(0)
                    self.assertIsNone(world.last_landing)
                    self.assertFalse(world.ball.launch)
                    self.assertFalse(world.switches)
                    self.assertIsNone(world.platforms[0].break_at)
                    self.assertAlmostEqual(world.ball.vy, 120.+P.gravity*.2, places=5)
                    self.assertAlmostEqual(world.ball.y, 280.+120.*.2+.5*P.gravity*.2**2, places=4)

    def test_ceiling_impact_keeps_remaining_gravity(self):
        world = empty_world([Platform('ceiling', 0, 300, 1600)], auto_reset=False)
        world.ball.y, world.ball.vy = 380., -300.
        impact = (300.-(300.**2-4*P.gravity)**.5)/P.gravity
        world.step(0, .1)
        self.assertIsNone(world.last_landing)
        self.assertAlmostEqual(world.ball.vy, P.gravity*(.1-impact), places=3)
        self.assertAlmostEqual(world.ball.y, 378.+.5*P.gravity*(.1-impact)**2, places=3)

    def test_lower_corner_does_not_cancel_upward_motion(self):
        world = empty_world([Platform('corner', 256, 300)], auto_reset=False)
        world.ball.x, world.ball.y, world.ball.vy = 250., 384., -200.
        for _ in range(12):
            world.step(0)
        self.assertIsNone(world.last_landing)
        self.assertAlmostEqual(world.ball.vy, -200.+P.gravity*.1, places=5)

    def test_corner_penetration_is_separated_without_a_jump(self):
        for x in (250., 326.):
            for y, vy in ((289., 20.), (375., -100.)):
                with self.subTest(x=x, y=y):
                    block = Platform('corner', 256, 300)
                    world = empty_world([block], auto_reset=False)
                    world.ball.x, world.ball.y, world.ball.vy = x, y, vy
                    world.step(0)
                    self.assertFalse(world.failed)
                    self.assertIsNone(world.last_landing)
                    self.assertFalse(circle_rect_overlap(world.ball.x, world.ball.y, P.radius, block.rect_at(world.time)))
                    self.assertAlmostEqual(world.ball.vy, vy+P.gravity*P.dt, places=5)

    def test_floor_and_held_wall_resolve_together(self):
        wall = Platform('wall', 256, 0, 64, 640)
        floor = Platform('floor', 0, 640, 256)
        world = empty_world([wall, floor], auto_reset=False)
        reference = empty_world([floor], auto_reset=False)
        for w in (world, reference):
            w.ball.x, w.ball.y, w.ball.vy = 242., 626., 5.
        for _ in range(108):
            world.step(1)
            reference.step(0)
        self.assertFalse(world.failed)
        self.assertEqual(world.last_landing, 'floor')
        self.assertAlmostEqual(world.ball.y, reference.ball.y, places=4)
        self.assertAlmostEqual(world.ball.vy, reference.ball.vy, places=4)

    def test_overlap_correction_is_not_a_landing(self):
        world = empty_world([Platform('timed', 96, 640, kind='timed', period=1., active_for=.5, phase=.6)], auto_reset=False)
        world.time = .399
        world.ball.y, world.ball.vy = 630., -20.
        world.step(0)
        self.assertFalse(world.failed)
        self.assertIsNone(world.last_landing)
        self.assertGreater(world.ball.vy, -20.)

    def test_wall_fall_still_checks_fish_and_goal(self):
        world = empty_world([Platform('wall', 256, 0, 64, 700)],
                            fish=[Collectible(242., 218., size=2.)],
                            goal=(240., 220., 4., 4.), auto_reset=False)
        world.ball.x, world.ball.y = 242., 200.
        for _ in range(30):
            world.step(1)
        self.assertTrue(world.cleared)
        self.assertEqual(world.plus, 1)

    def test_many_events_do_not_discard_remaining_motion(self):
        level = Level(1, '', '', [], [Collectible(130.+i, 400., size=.1) for i in range(30)],
                      [], (128., 400.), (1500., 700., 64., 64.))
        world = World(level, physics=replace(P, bounce_height=0.), auto_reset=False)
        world.ball.vx = P.max_speed
        world._piece(1, .2)
        self.assertEqual(world.plus, 30)
        self.assertAlmostEqual(world.ball.x, 128.+P.max_speed*.2, places=6)
        self.assertAlmostEqual(world.time, .2, places=8)

    def test_continuous_side_collision(self):
        hit = sweep_circle_rect((0, 50), (200, 0), (90, 20, 64, 80), 14)
        self.assertAlmostEqual(hit[0], 76/200)
        self.assertEqual(hit[1], (-1., 0.))

    def test_circle_corner_is_not_square(self):
        self.assertFalse(circle_rect_overlap(6, 6, 4.5, (10, 10, 64, 64)))
        self.assertTrue(circle_rect_overlap(7, 7, 4.5, (10, 10, 64, 64)))

    def test_fast_ball_does_not_pass_through_wall(self):
        world = empty_world([Platform('wall', 220, 360, 64, 200)])
        world.ball.x = 200.
        world.ball.vx = 5000.
        world.step(0)
        self.assertLessEqual(world.ball.x, 206.0001)
        self.assertEqual(world.ball.vx, 0.)
        self.assertGreater(world.ball.vy, 0.)
        self.assertIsNone(world.last_landing)

    def test_ceiling_stops_ascent(self):
        world = empty_world([Platform('ceiling', 0, 300, 1600)])
        world.ball.vy = -P.bounce_speed
        for _ in range(30):
            world.step(0)
        self.assertGreaterEqual(world.ball.y, 378.-1e-4)
        self.assertGreater(world.ball.vy, 0.)
        self.assertIsNone(world.last_landing)

    def test_spike_failure_precedes_fish_and_goal(self):
        world = empty_world(fish=[Collectible(96, 400)], spikes=[Spike(64, 384)], goal=(64, 384, 64, 64))
        world.ball.x = 96.
        world.step(0)
        self.assertEqual(world.failures, 1)
        self.assertFalse(world.cleared)
        self.assertEqual(world.plus, 0)

    def test_fast_ball_does_not_pass_through_spikes(self):
        world = empty_world(spikes=[Spike(256, 384)])
        world.ball.vx = 8000.
        world.step(1, .03)
        self.assertEqual(world.failures, 1)

    def test_collectibles_use_center_and_are_unique(self):
        world = empty_world(fish=[Collectible(150, 400, size=10)])
        world.ball.x = 140.
        world.step(0)
        self.assertEqual(world.plus, 0)
        world.ball.x = 150.
        world.step(0)
        self.assertEqual(world.plus, 1)
        world.step(0)
        self.assertEqual(world.plus, 1)

    def test_simultaneous_fish_before_goal_score(self):
        world = empty_world(fish=[Collectible(128, 400)], goal=(120, 390, 64, 64))
        world.step(0)
        self.assertTrue(world.cleared)
        self.assertEqual(world.score, 8)
        self.assertEqual((world.ball.vx, world.ball.vy), (0., 0.))
        elapsed = world.time
        world.step(1)
        self.assertEqual(elapsed, world.time)

    def test_lower_boundary_has_no_floor(self):
        world = empty_world()
        world.ball.y = P.height+P.radius+1
        world.step(0)
        self.assertEqual(world.failures, 1)
        self.assertEqual(world.time, 0.)


class SpecialBlockTests(unittest.TestCase):
    def test_breakable_timer_and_reset(self):
        world = empty_world([Platform('break', 0, 640, 1600, kind='breakable')])
        world.ball.y = 626.
        world.ball.vy = 1.
        world.step(0)
        block = world.platforms[0]
        self.assertAlmostEqual(block.break_at, .5, places=5)
        for _ in range(58):
            world.step(0)
        self.assertFalse(block.destroyed)
        for _ in range(3):
            world.step(0)
        self.assertTrue(block.destroyed)
        world.reset()
        self.assertFalse(world.platforms[0].destroyed)
        self.assertIsNone(world.platforms[0].break_at)

    def test_switch_requires_full_horizontal_exit_to_rearm(self):
        world = empty_world([Platform('switch', 96, 640, kind='switch', group='A')])
        world.ball.y = 626.
        world.ball.vy = 1.
        world.step(0)
        self.assertTrue(world.switches['A'])
        for _ in range(110):
            world.step(0)
        self.assertTrue(world.switches['A'])
        world.ball.x = 180.
        world.step(0)
        world.ball.x, world.ball.y, world.ball.vy = 128., 626., 1.
        world.step(0)
        self.assertFalse(world.switches['A'])

    def test_linked_blocks_reset_to_inactive(self):
        world = empty_world([Platform('linked', 128, 640, kind='linked', group='A')])
        self.assertFalse(world.platforms[0].active(world.time, world.switches))
        world.switches['A'] = True
        self.assertTrue(world.platforms[0].active(world.time, world.switches))
        world.reset()
        self.assertFalse(world.platforms[0].active(world.time, world.switches))

    def test_timed_period_starts_immediately(self):
        block = Platform('timed', 0, 640, kind='timed', period=2., active_for=1.)
        self.assertTrue(block.active(0., {}))
        self.assertFalse(block.active(1.5, {}))
        self.assertTrue(block.active(2., {}))
        world = empty_world([block])
        world.step(0, .1)
        self.assertAlmostEqual(world.time, .1)

    def test_timed_spawn_unresolvable_crush(self):
        world = empty_world([Platform('left', 64, 0, 64, 832),
                             Platform('timed', 140, 0, 64, 832, 'timed', period=1., active_for=.5, phase=.6)])
        world.time = .399
        world.ball.x = 134.
        world.step(0)
        self.assertEqual(world.failures, 1)
        self.assertEqual(world.time, 0.)

    def test_stopping_at_wall_is_not_crush(self):
        world = empty_world([Platform('wall', 256, 0, 64, 640), Platform('floor', 0, 640, 256)])
        world.ball.x = 240.
        world.ball.vx = P.max_speed
        for _ in range(120):
            world.step(1)
        self.assertEqual(world.failures, 0)

    def test_launcher_and_release_preserve_speed(self):
        world = empty_world([Platform('launch', 0, 640, 1600, kind='launcher')])
        world.ball.y, world.ball.vy = 626., 1.
        world.step(0)
        self.assertTrue(world.ball.launch)
        self.assertEqual(world.ball.y, 672.)
        self.assertEqual(world.ball.vy, 0.)
        x, y = world.ball.x, world.ball.y
        world.step(0)
        self.assertEqual(world.ball.y, y)
        self.assertGreater(world.ball.x, x)
        world.step(1)
        self.assertFalse(world.ball.launch)
        self.assertAlmostEqual(world.ball.vx, P.max_speed)
        self.assertGreater(world.ball.vy, 0.)

    def test_both_launcher_directions_keep_block_center_height(self):
        for arrow in (-1, 1):
            with self.subTest(arrow=arrow):
                world = empty_world([Platform('launch', 288, 512, kind='launcher', arrow=arrow)])
                world.ball.x, world.ball.y, world.ball.vy = 320., 498., 1.
                world.step(0)
                self.assertEqual(world.ball.y, 544.)
                self.assertEqual(world.ball.vx, arrow*P.max_speed)
                self.assertEqual(world.ball.launch_source, 'launch')
                for _ in range(40):
                    world.step(0)
                    self.assertEqual(world.ball.y, 544.)
                    self.assertTrue(world.ball.launch)
                    self.assertEqual(world.failures, 0)
                self.assertGreater((world.ball.x-320.)*arrow, 64.)
                self.assertIsNone(world.ball.launch_source)

    def test_center_launch_still_collides_with_other_blocks(self):
        world = empty_world([Platform('launch', 288, 512, kind='launcher'),
                             Platform('wall', 416, 512)])
        world.ball.x, world.ball.y, world.ball.vy = 320., 498., 1.
        world.step(0)
        for _ in range(60):
            world.step(0)
        self.assertFalse(world.ball.launch)
        self.assertEqual(world.ball.vx, 0.)
        self.assertGreater(world.ball.vy, 0.)
        self.assertLessEqual(world.ball.x, 416-P.radius)

    def test_launcher_wall_reenables_gravity(self):
        world = empty_world([Platform('wall', 256, 350, 64, 200)])
        world.ball.x = 240.
        world.ball.launch = True
        world.ball.vx = P.max_speed
        for _ in range(8):
            world.step(0)
        self.assertFalse(world.ball.launch)
        self.assertEqual(world.ball.vx, 0.)
        self.assertGreater(world.ball.vy, 0.)

    def test_failure_resets_every_transient_state(self):
        world = World(make_levels()[7])
        world.time = 4.
        world.collected = {0, 3}
        world.switches = {'A': True}
        world.ball.launch = True
        world.platforms[0].destroyed = True
        world.platforms[0].break_at = 5.
        world.fail('test')
        self.assertEqual(world.time, 0.)
        self.assertFalse(world.collected)
        self.assertFalse(world.switches)
        self.assertFalse(world.ball.launch)
        self.assertFalse(world.platforms[0].destroyed)
        self.assertIsNone(world.platforms[0].break_at)
        self.assertEqual((world.ball.vx, world.ball.vy), (0., 0.))


class ProgressTests(unittest.TestCase):
    def test_best_score_then_time_and_unlock(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'progress.json'
            progress = Progress(path)
            self.assertTrue(progress.complete(1, 10, 8.))
            self.assertEqual(progress.unlocked, 2)
            self.assertFalse(progress.complete(1, 7, 4.))
            self.assertFalse(progress.complete(1, 10, 9.))
            self.assertTrue(progress.complete(1, 10, 7.))
            saved = Progress(path)
            self.assertEqual(saved.records['1'], {'score': 10, 'time': 7.})
            self.assertEqual(saved.unlocked, 2)

    def test_tutorial_records_only_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            progress = Progress(Path(temp)/'record.json')
            self.assertTrue(progress.complete(0, None, 5.))
            self.assertFalse(progress.complete(0, None, 4.))
            self.assertEqual(progress.records['0'], {'cleared': True})
            self.assertEqual(progress.unlocked, 1)

    def test_legacy_save_migration(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'record.json'
            path.write_text(json.dumps({'version': 1, 'unlocked': 2, 'records':
                                        {'0': {'score': None, 'time': 8.}, '1': {'score': 10, 'time': 12.}}}))
            progress = Progress(path)
            self.assertEqual(progress.records['0'], {'cleared': True})
            self.assertEqual(progress.records['1']['score'], 10)

    def test_corrupt_and_nonfinite_save(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'record.json'
            path.write_text('{broken')
            progress = Progress(path)
            self.assertFalse(progress.records)
            self.assertTrue(progress.error)
            path.write_text('{"version":2,"records":{"1":{"score":8,"time":NaN}}}')
            self.assertFalse(Progress(path).records)


class EnvironmentTests(unittest.TestCase):
    def test_observation_exposes_targets_hazards_and_block_rules(self):
        env = Environment(7)
        observation, _ = env.reset()
        self.assertEqual(observation['stage'], 7)
        self.assertEqual(observation['goal'], env.world.level.goal)
        self.assertEqual(len(observation['collectibles']), 4)
        self.assertEqual([item['value'] for item in observation['collectibles']], [1, 1, 1, -1])
        self.assertTrue(all(not item['collected'] for item in observation['collectibles']))
        self.assertEqual(observation['spikes'][0]['triangles'], env.world.level.spikes[0].triangles)
        linked = next(block for block in observation['platforms'] if block['kind'] == 'linked')
        self.assertEqual(linked['group'], 'A')
        self.assertFalse(linked['active'])
        # A plain data observation can be stored or sent to a separate learner.
        json.dumps(observation, allow_nan=False)

    def test_item_rewards_and_collection_flags(self):
        env = Environment(1)
        env.world.ball.x, env.world.ball.y = 672., 366.
        observation, reward, terminated, truncated, _ = env.step(0)
        self.assertEqual(reward, 1)
        self.assertTrue(observation['collectibles'][0]['collected'])
        self.assertFalse(terminated or truncated)
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(reward, 0)
        env.world.ball.x, env.world.ball.y = 64., 650.
        observation, reward, _, _, _ = env.step(0)
        self.assertEqual(reward, -1)
        self.assertTrue(observation['collectibles'][3]['collected'])

    def test_repeated_episodes_reset_all_learning_state(self):
        env = Environment(7)
        initial, _ = env.reset()
        for _ in range(20):
            env.world.collected = {0, 3}
            env.world.switches['A'] = True
            env.world.platforms[0].destroyed = True
            env.world.ball.y = 900.
            env.step(1)
            observation, _ = env.reset()
            self.assertEqual(observation, initial)

    def test_observation_contains_input_and_dynamic_state(self):
        env = Environment(5)
        observation, info = env.reset()
        self.assertEqual(observation['previous_action'], 0)
        observation, _, terminated, truncated, _ = env.step(1)
        self.assertEqual(observation['previous_action'], 1)
        self.assertGreater(observation['time'], 0)
        moving = next(q for q in observation['platforms'] if q['kind'] == 'moving')
        self.assertEqual(moving['direction'], 1)
        self.assertFalse(terminated or truncated)

    def test_failure_returns_terminal_state_before_reset(self):
        env = Environment(0)
        env.world.ball.y = 900.
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -7)
        self.assertTrue(observation['failed'])
        self.assertEqual(observation['ball'][1], 900.)
        with self.assertRaises(RuntimeError):
            env.step(0)
        observation, _ = env.reset()
        self.assertEqual(observation['ball'][1], 594.)
        self.assertFalse(observation['failed'])

    def test_success_and_time_limit(self):
        env = Environment(1)
        env.world.ball.x, env.world.ball.y = 1504., 670.
        _, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual((reward, info['score']), (7, 7))
        self.assertEqual(info['reason'], 'success')
        env = Environment(0, time_limit=P.dt*1.5)
        env.step(0)
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 0)
        self.assertEqual(info['reason'], 'time_limit')
        with self.assertRaises(RuntimeError):
            env.step(0)
        self.assertAlmostEqual(observation['time'], P.dt*1.5)


class LevelTests(unittest.TestCase):
    def test_scores_and_grades(self):
        world = World(make_levels()[1])
        for collected, score, grade in [({3}, 6, 'B'), (set(), 7, 'A'), ({0}, 8, 'S'), ({0, 1}, 9, 'SS'), ({0, 1, 2}, 10, 'SSS')]:
            world.collected = collected
            self.assertEqual((world.score, world.grade), (score, grade))
        self.assertIsNone(World(make_levels()[0]).grade)

    def test_eight_stages_grid_counts_and_safe_spawns(self):
        levels = make_levels()
        self.assertEqual(len(levels), 8)
        for level in levels:
            self.assertEqual(sum(f.value > 0 for f in level.collectibles), 1 if level.index == 0 else 3)
            self.assertEqual(sum(f.value < 0 for f in level.collectibles), 0 if level.index == 0 else 1)
            self.assertEqual(len({p.id for p in level.platforms}), len(level.platforms))
            for block in level.platforms:
                self.assertEqual((block.x % 64, block.y % 64, block.w, block.h), (0, 0, 64, 64))
                self.assertTrue(0 <= block.x < 1600 and 64 <= block.y < 832)
                self.assertFalse(circle_rect_overlap(*level.spawn, P.radius, block.rect_at(0)))
            for fish in level.collectibles:
                self.assertFalse(any(circle_rect_overlap(fish.x, fish.y, fish.size/2, q.rect_at(0)) for q in level.platforms))
            for spike in level.spikes:
                self.assertEqual((spike.w, spike.h), (128, 64))

    def test_all_sixteen_real_input_routes(self):
        for level in make_levels():
            for collect_all in (False, True):
                with self.subTest(stage=level.index, all_fish=collect_all):
                    result = run_route(level, collect_all)
                    self.assertTrue(result['passed'], str(result))


if __name__ == '__main__':
    unittest.main()
