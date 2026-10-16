import json
import math
from pathlib import Path
import tempfile
import unittest

from pingu_ball.config import PHYSICS
from pingu_ball.engine import World, Level, Platform, Collectible, Spike, advance_x
from pingu_ball.geometry import sweep_circle_rect, sweep_circle_polygon, circle_rect_overlap
from pingu_ball.levels import make_levels
from pingu_ball.storage import Progress
from pingu_ball.validate import run_route


def empty_world(platforms=None,fish=None,spikes=None,goal=(1100,550,20,20)):
    level=Level(1,'Test','',platforms or [],fish or [],spikes or [],(100.,200.),goal)
    w=World(level);w.started=True
    return w


class PhysicsTests(unittest.TestCase):
    def test_normal_free_flight(self):
        w=empty_world();w.ball.vx=300.;w.ball.vy=-300.
        heights=[]
        for _ in range(96):w.step(1);heights.append(w.ball.y)
        self.assertAlmostEqual(min(heights),140.,places=5)
        self.assertAlmostEqual(w.ball.x,340.,places=5)
        self.assertAlmostEqual(w.ball.y,200.,places=5)
        self.assertAlmostEqual(w.time,.8,places=7)

    def test_pad_free_flight(self):
        w=empty_world();w.ball.vx=300.;w.ball.vy=-PHYSICS.pad_speed
        t=PHYSICS.pad_speed/PHYSICS.gravity
        w.step(1,t)
        self.assertAlmostEqual(w.ball.y,110.,places=5)
        w.step(1,t)
        self.assertAlmostEqual(w.ball.x-100.,293.938769,places=4)
        self.assertAlmostEqual(w.ball.y,200.,places=5)

    def test_acceleration_from_rest(self):
        w=empty_world();w.ball.vy=-300.
        for _ in range(96):w.step(1)
        self.assertAlmostEqual(w.ball.x-100.,225.,places=5)

    def test_release_deceleration(self):
        w=empty_world();w.ball.vx=300.
        for _ in range(8):w.step(0)
        self.assertAlmostEqual(w.ball.vx,0.,places=5)
        self.assertAlmostEqual(w.ball.x-100.,10.,places=5)

    def test_first_input_and_pause_freeze_everything(self):
        w=World(make_levels()[6]);initial=(w.ball.x,w.ball.y)
        for _ in range(120):w.step(0)
        self.assertEqual(w.time,0.);self.assertEqual(initial,(w.ball.x,w.ball.y))
        w.step(1);w.paused=True
        state=(w.time,w.ball.x,w.ball.y,[q.rect_at(w.time) for q in w.platforms])
        for _ in range(120):w.step(1)
        self.assertEqual(state,(w.time,w.ball.x,w.ball.y,[q.rect_at(w.time) for q in w.platforms]))

    def test_landing_replaces_fall_speed(self):
        w=empty_world([Platform('floor',0,300,1200)])
        w.ball.y=280.;w.ball.vy=700.
        for _ in range(8):
            w.step(0)
            if w.last_landing:break
        self.assertEqual(w.last_landing,'floor')
        self.assertLess(w.ball.vy,-295.)

    def test_spring_landing(self):
        w=empty_world([Platform('pad',0,300,1200,kind='spring')])
        w.ball.y=290.;w.ball.vy=100.
        for _ in range(15):
            w.step(0)
            if w.last_landing:break
        self.assertEqual(w.last_landing,'pad')
        self.assertLess(w.ball.vy,-360.)

    def test_moving_block_does_not_carry_airborne_ball(self):
        w=empty_world([Platform('move',50,300,20,kind='moving',travel=150)])
        for _ in range(30):w.step(0)
        self.assertAlmostEqual(w.ball.x,100.)

    def test_render_fps_does_not_change_fixed_tick_simulation(self):
        states=[]
        for fps in (30,60,144):
            w=empty_world();w.ball.vy=-300.
            accumulator=0.;ticks=0
            for _ in range(fps):
                accumulator+=1/fps
                while accumulator+1e-12>=PHYSICS.dt:
                    w.step(1 if ticks<60 else 0)
                    accumulator-=PHYSICS.dt;ticks+=1
            states.append((w.ball.x,w.ball.y,w.ball.vx,w.ball.vy,w.time))
        self.assertEqual(states[0],states[1]);self.assertEqual(states[1],states[2])


class CollisionTests(unittest.TestCase):
    def test_continuous_side_collision(self):
        hit=sweep_circle_rect((0,50),(200,0),(90,20,20,80),4.5)
        self.assertAlmostEqual(hit[0],85.5/200)
        self.assertEqual(hit[1],(-1.,0.))

    def test_circle_corner_is_not_square(self):
        self.assertFalse(circle_rect_overlap(6,6,4.5,(10,10,20,20)))
        self.assertTrue(circle_rect_overlap(7,7,4.5,(10,10,20,20)))

    def test_fast_ball_does_not_pass_through_wall(self):
        w=empty_world([Platform('wall',220,180,20,100)])
        w.ball.x=200.;w.ball.vx=5000.
        w.step(0)
        self.assertLessEqual(w.ball.x,215.5001)
        self.assertEqual(w.ball.vx,0.)

    def test_ceiling_stops_ascent_without_bouncing(self):
        w=empty_world([Platform('ceiling',0,140,1200)])
        w.ball.vy=-300.
        for _ in range(30):w.step(0)
        self.assertGreaterEqual(w.ball.y,164.5-1e-4)
        self.assertGreater(w.ball.vy,0.)

    def test_spike_contact_wins_over_goal_and_fish(self):
        w=empty_world(fish=[Collectible(100,100)],spikes=[Spike(80,80)],goal=(90,90,20,20))
        w.ball.x=w.ball.y=100.;w.step(0)
        self.assertEqual(w.failures,1);self.assertFalse(w.cleared);self.assertEqual(w.plus,0)

    def test_collectibles_use_center_not_ball_edge(self):
        w=empty_world(fish=[Collectible(110,200,size=10)])
        w.ball.x=102.;w.step(0)
        self.assertEqual(w.plus,0)
        w.ball.x=110.;w.ball.y=200.;w.step(0)
        self.assertEqual(w.plus,1)
        w.step(0);self.assertEqual(w.plus,1)

    def test_simultaneous_fish_before_goal_score(self):
        w=empty_world(fish=[Collectible(100,200)],goal=(95,195,10,10))
        w.step(0)
        self.assertTrue(w.cleared);self.assertEqual(w.score,8)


class SpecialBlockTests(unittest.TestCase):
    def test_breakable_delay_and_reset(self):
        w=empty_world([Platform('break',0,300,1200,kind='breakable')])
        w.ball.y=295.5;w.ball.vy=1.;w.step(0)
        first=w.platforms[0].break_at
        self.assertIsNotNone(first)
        for _ in range(65):w.step(0)
        self.assertTrue(w.platforms[0].destroyed)
        w.reset();self.assertFalse(w.platforms[0].destroyed)
        self.assertIsNone(w.platforms[0].break_at)

    def test_switch_does_not_toggle_on_every_bounce(self):
        w=empty_world([Platform('switch',80,300,40,kind='switch',group='A')])
        w.ball.y=295.5;w.ball.vy=1.;w.step(0)
        self.assertTrue(w.switches['A'])
        for _ in range(100):w.step(0)
        self.assertTrue(w.switches['A'])
        w.ball.x=140.;w.step(0)
        w.ball.x=100.;w.ball.y=295.5;w.ball.vy=1.;w.step(0)
        self.assertFalse(w.switches['A'])

    def test_linked_block_and_reset(self):
        w=empty_world([Platform('linked',50,300,20,kind='linked',group='A')])
        self.assertFalse(w.platforms[0].active(w.time,w.switches))
        w.switches['A']=True
        self.assertTrue(w.platforms[0].active(w.time,w.switches))
        w.reset();self.assertFalse(w.platforms[0].active(w.time,w.switches))

    def test_timed_period(self):
        q=Platform('timed',0,300,20,kind='timed',period=2.,active_for=1.)
        self.assertTrue(q.active(0.,{}));self.assertFalse(q.active(1.5,{}));self.assertTrue(q.active(2.,{}))

    def test_timed_spawn_unresolvable_crush_fails(self):
        w=empty_world([Platform('left',90,180,20,40),
                       Platform('timed',118,180,20,40,'timed',period=1.,active_for=.5,phase=.6)])
        w.time=.399;w.ball.x=114.;w.ball.y=200.;w.ball.vy=0.
        w.step(0)
        self.assertEqual(w.failures,1)
        self.assertEqual(w.time,0.)

    def test_stopping_against_wall_is_not_crush(self):
        w=empty_world([Platform('wall',200,0,20,500),Platform('floor',0,500,200)])
        w.ball.x=190.;w.ball.y=300.;w.ball.vx=300.
        for _ in range(120):w.step(1)
        self.assertEqual(w.failures,0)

    def test_launcher_and_input_release_preserve_horizontal_speed(self):
        w=empty_world([Platform('launch',0,300,1200,kind='launcher')])
        w.ball.y=295.5;w.ball.vy=1.;w.step(0)
        self.assertTrue(w.ball.launch);self.assertEqual(w.ball.vy,0.)
        x,y=w.ball.x,w.ball.y
        w.step(0)
        self.assertAlmostEqual(w.ball.y,y);self.assertGreater(w.ball.x,x)
        w.step(1)
        self.assertFalse(w.ball.launch);self.assertEqual(w.ball.vx,300.);self.assertGreater(w.ball.vy,0.)

    def test_launcher_wall_collision_reenables_gravity(self):
        w=empty_world([Platform('wall',200,150,20,100)])
        w.ball.x=190.;w.ball.launch=True;w.ball.vx=300.
        for _ in range(8):w.step(0)
        self.assertFalse(w.ball.launch);self.assertEqual(w.ball.vx,0.);self.assertGreater(w.ball.vy,0.)

    def test_fail_resets_all_transient_state(self):
        w=World(make_levels()[7]);w.started=True;w.time=4.;w.collected={0,3};w.switches={'A':True}
        w.ball.launch=True;w.platforms[0].destroyed=True
        w.fail('test')
        self.assertFalse(w.started);self.assertEqual(w.time,0.);self.assertEqual(w.collected,set())
        self.assertEqual(w.switches,{});self.assertFalse(w.ball.launch);self.assertFalse(w.platforms[0].destroyed)


class ProgressTests(unittest.TestCase):
    def test_best_score_then_time_and_unlock(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'progress.json';p=Progress(path)
            self.assertTrue(p.complete(1,10,8.));self.assertEqual(p.unlocked,2)
            self.assertFalse(p.complete(1,7,4.))
            self.assertFalse(p.complete(1,10,9.))
            self.assertTrue(p.complete(1,10,7.))
            saved=Progress(path)
            self.assertEqual(saved.records['1'],{'score':10,'time':7.})
            self.assertEqual(saved.unlocked,2)

    def test_tutorial_has_no_grade_or_score(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Progress(Path(temp)/'record.json')
            p.complete(0,None,5.);self.assertEqual(p.unlocked,1)
            self.assertFalse(p.complete(0,None,6.));self.assertTrue(p.complete(0,None,4.))
            self.assertIsNone(p.records['0']['score'])

    def test_corrupt_save_does_not_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'record.json';path.write_text('{broken')
            p=Progress(path);self.assertEqual(p.records,{});self.assertTrue(p.error)


class LevelTests(unittest.TestCase):
    def test_score_grades(self):
        w=World(make_levels()[1])
        for collected,score,grade in [({3},6,'B'),(set(),7,'A'),({0},8,'S'),({0,1},9,'SS'),({0,1,2},10,'SSS')]:
            w.collected=collected
            self.assertEqual(w.score,score);self.assertEqual(w.grade,grade)

    def test_eight_stages_and_collectible_counts(self):
        levels=make_levels();self.assertEqual(len(levels),8)
        for level in levels:
            self.assertEqual(sum(f.value>0 for f in level.collectibles),1 if level.index==0 else 3)
            self.assertEqual(sum(f.value<0 for f in level.collectibles),0 if level.index==0 else 1)
            self.assertEqual(len({p.id for p in level.platforms}),len(level.platforms))

    def test_every_stage_has_real_no_fish_and_all_fish_routes(self):
        for level in make_levels():
            for collect_all in (False,True):
                with self.subTest(stage=level.index,all_fish=collect_all):
                    result=run_route(level,collect_all)
                    self.assertTrue(result['passed'],str(result))


if __name__=='__main__':unittest.main()
