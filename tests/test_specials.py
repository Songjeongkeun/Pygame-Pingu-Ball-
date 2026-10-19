"""Behavior and collision regression tests for all 37 specification IDs."""
import json
import unittest

from pingu_ball.config import PHYSICS as P
from pingu_ball.engine import Collectible, Level, Platform, World
from pingu_ball.environment import Environment
from pingu_ball.geometry import circle_rect_overlap
from pingu_ball.specials import CATALOG, Device


def make_world(*devices, blocks=(), fish=()):
    level = Level(0,'Special test','',list(blocks),list(fish),[],(128.,200.),(1500.,700.,64.,64.),specials=list(devices))
    return World(level,auto_reset=False)


def land(world,name):
    q = next(q for q in world.platforms if q.id == name)
    x,y,w,_ = q.rect_at(world.time,world.physics)
    world.ball.x,world.ball.y = x+w/2,y-P.radius-1.
    world.ball.vx,world.ball.vy = 0.,300.
    world.step(0,.01)
    return q


class PlatformAndMovementTests(unittest.TestCase):
    def test_catalog_has_all_37_and_rejects_unknown_ids(self):
        self.assertEqual(len(CATALOG),37)
        with self.assertRaises(ValueError): Device('bad',999,0,0)
        with self.assertRaises(ValueError): Device('bad',302,0,0,options={'period':0})
        with self.assertRaises(ValueError): make_world(Device('same',401,0,0),Device('same',403,64,0))

    def test_301_breaks_after_landing_even_after_player_leaves(self):
        w = make_world(Device('block',301,64,500))
        q = land(w,'block')
        self.assertIsNotNone(q.break_at)
        w.ball.x,w.ball.y = 300.,100.
        w.step(0,.51)
        self.assertTrue(q.destroyed)

    def test_401_provides_actual_144px_spring(self):
        w = make_world(Device('block',401,64,500))
        land(w,'block')
        self.assertLess(w.ball.vy,-500.)

    def test_302_disappears_and_reappears_without_support_while_off(self):
        w = make_world(Device('block',302,64,500,options={'period':2.,'active_for':1.}))
        q = w.platforms[0]
        self.assertTrue(q.active(.5,{})); self.assertFalse(q.active(1.5,{})); self.assertTrue(q.active(2.5,{}))
        w.time=1.2; w.ball.x,w.ball.y=96.,470.; w.ball.vy=300.
        w.step(0,.15)
        self.assertIsNone(w.last_landing)
        self.assertGreater(w.ball.y,500.)

    def test_403_falls_and_resets_to_original_position(self):
        w = make_world(Device('block',403,64,500,options={'delay':.01}))
        q=land(w,'block'); w.ball.x,w.ball.y=300.,100.
        w.step(0,.25)
        self.assertGreater(q.rect_at(w.time)[1],510.)
        w.reset(); self.assertEqual(w.platforms[0].rect_at(0)[1],500.)

    def test_307_acceleration_changes_speed_then_expires(self):
        w = make_world(Device('block',307,64,500,options={'factor':2.,'duration':.2}))
        land(w,'block'); w.ball.x,w.ball.y=400.,200.
        w.step(1,.1)
        self.assertAlmostEqual(w.ball.vx,P.max_speed*2)
        w.step(1,.2)
        self.assertAlmostEqual(w.ball.vx,P.max_speed)

    def test_501_pingpongs_and_stays_a_real_landing_platform(self):
        w=make_world(Device('block',501,64,500,options={'travel':128.}))
        q=w.platforms[0]
        self.assertAlmostEqual(q.rect_at(128/P.platform_speed)[0],192.)
        self.assertAlmostEqual(q.rect_at(256/P.platform_speed)[0],64.)
        land(w,'block'); self.assertEqual(w.last_landing,'block')

    def test_508_vertical_motion_and_actual_top_contact(self):
        w=make_world(Device('block',508,64,400,options={'travel':128.,'speed':64.}))
        q=w.platforms[0]
        self.assertEqual(q.rect_at(1)[1],464.)
        self.assertEqual(q.rect_at(4)[1],400.)
        land(w,'block')
        self.assertEqual(w.last_landing,'block')
        self.assertLess(w.ball.vy,0.)
        self.assertLess(abs(w.ball.y-(q.rect_at(w.time)[1]-14)),8.)

    def test_507_wall_sliding_limits_fall_speed_without_auto_bounce(self):
        w=make_world(Device('wall',507,256,100,h=500))
        w.ball.x,w.ball.y,w.ball.vy=242.,300.,300.
        w.step(1,.15)
        self.assertLess(w.ball.vy,52.)
        self.assertGreater(w.ball.vy,0.)
        self.assertIsNone(w.last_landing)

    def test_511_magnet_attracts_then_repels_target_with_real_geometry(self):
        d=Device('magnet',511,64,500,target='target',options={'radius':300.,'speed':64.})
        w=make_world(d,blocks=[Platform('target',256,500)])
        land(w,'magnet'); w.ball.x,w.ball.y=500.,100.
        before=w.platforms[0].x; w.step(0,.2)
        self.assertLess(w.platforms[0].x,before)
        land(w,'magnet'); w.ball.x,w.ball.y=500.,100.
        before=w.platforms[0].x; w.step(0,.2)
        self.assertGreater(w.platforms[0].x,before)

    def test_512_shift_moves_target_and_switches_back_after_reentry(self):
        w=make_world(Device('shift',512,64,500,target='target',options={'offset':(64.,-64.)}),blocks=[Platform('target',300,500)])
        land(w,'shift'); self.assertEqual((w.platforms[0].x,w.platforms[0].y),(364.,436.))
        w.ball.x,w.ball.y=500.,100.; w.step(0)
        land(w,'shift'); self.assertEqual((w.platforms[0].x,w.platforms[0].y),(300.,500.))


class SwitchTests(unittest.TestCase):
    def test_304_requires_collection_before_activating_group(self):
        w=make_world(Device('star',304,64,500,group='A',options={'required':1}),fish=[Collectible(500,100)])
        land(w,'star'); self.assertFalse(w.switches.get('A',False))
        w.ball.x,w.ball.y=500.,100.; w.step(0)
        self.assertEqual(w.plus,1)
        land(w,'star'); self.assertTrue(w.switches['A'])

    def test_404_power_enables_connected_cannon_and_turns_off(self):
        w=make_world(Device('power',404,64,500,group='A'),Device('cannon',603,700,100,group='A'))
        w.step(); self.assertFalse(w.special.projectiles)
        land(w,'power'); w.ball.x,w.ball.y=400.,100.; w.step(0,.03)
        self.assertTrue(w.special.projectiles)
        land(w,'power'); self.assertFalse(w.switches['power:A'])

    def test_305_barrier_opens_for_switch_group(self):
        w=make_world(Device('switch',415,64,500,group='A'),Device('barrier',305,256,300,group='A'))
        barrier=w.special.platforms['barrier']
        self.assertTrue(barrier.active(0,w.switches))
        land(w,'switch'); self.assertFalse(barrier.active(w.time,w.switches))

    def test_415_416_linked_platform_becomes_collidable_and_toggles_once(self):
        w=make_world(Device('switch',415,64,500,group='A'),Device('linked',416,256,500,group='A'))
        q=w.special.platforms['linked']; self.assertFalse(q.active(0,w.switches))
        land(w,'switch'); self.assertTrue(q.active(w.time,w.switches))
        w.step(0,.03); self.assertTrue(w.switches['A'])
        land(w,'linked'); self.assertEqual(w.last_landing,'linked')

    def test_inactive_platform_lab_switch_is_reachable_with_ordinary_input(self):
        from pingu_ball.special_lab import make_lab
        from pingu_ball.route_driver import RouteDriver
        w=World(make_lab(416),auto_reset=False)
        driver=RouteDriver(w,[(352.,'land:step-2'),(224.,'land:switch')])
        for _ in range(1200):
            w.step(driver.direction())
            if w.switches.get('demo',False) or w.failed: break
        self.assertTrue(w.switches.get('demo',False))
        self.assertFalse(w.failed)

    def test_510_electric_contact_is_not_fatal_with_power_on_or_off(self):
        for on in (False,True):
            w=make_world(Device('power',404,500,600,group='A',options={'on':on}),Device('wire',510,64,500,group='A'))
            land(w,'wire'); self.assertFalse(w.failed); self.assertEqual(w.last_landing,'wire')

    def test_306_red_only_becomes_lethal_after_085_seconds(self):
        w=make_world(Device('red',306,64,500))
        land(w,'red'); self.assertFalse(w.failed)
        w.ball.x,w.ball.y=400.,100.; w.step(0,.86)
        self.assertFalse(w.failed)
        land(w,'red'); self.assertTrue(w.failed)

    def test_306_blue_repel_after_05_seconds_does_not_kill(self):
        w=make_world(Device('blue',306,64,500,options={'color':'blue'}))
        land(w,'blue'); w.ball.x,w.ball.y=400.,100.; w.step(0,.51)
        land(w,'blue'); self.assertFalse(w.failed); self.assertLess(w.ball.vy,-500.)


class TransportTests(unittest.TestCase):
    def test_402_supports_both_horizontal_directions(self):
        for direction in (-1,1):
            w=make_world(Device('shooter',402,500,500,options={'direction':direction}))
            land(w,'shooter'); self.assertTrue(w.ball.launch)
            self.assertAlmostEqual(w.ball.vx,direction*P.launch_speed)

    def test_405_all_four_directions_and_406_all_eight(self):
        for type_id,count in ((405,4),(406,8)):
            for direction in range(count):
                w=make_world(Device('shooter',type_id,700,500,options={'direction':direction}))
                land(w,'shooter')
                self.assertTrue(w.ball.launch)
                self.assertAlmostEqual((w.ball.vx**2+w.ball.vy**2)**.5,P.launch_speed)
                expected=-90+direction*360/count
                import math
                self.assertAlmostEqual(w.ball.vx,P.launch_speed*math.cos(math.radians(expected)),places=6)
                self.assertAlmostEqual(w.ball.vy,P.launch_speed*math.sin(math.radians(expected)),places=6)

    def test_vertical_shooter_releases_source_and_releases_on_input(self):
        w=make_world(Device('shooter',405,700,500,options={'direction':0}))
        land(w,'shooter'); w.step(0,.15)
        self.assertIsNone(w.ball.launch_source)
        w.step(1); self.assertFalse(w.ball.launch)

    def test_black_white_holes_and_portals_transport_with_cooldown(self):
        for a,b in ((407,408),(408,407),(410,410)):
            w=make_world(Device('in',a,100,200,group='A'),Device('out',b,600,200,group='A'))
            w.ball.x,w.ball.y=132.,232.; w.step(0)
            self.assertAlmostEqual(w.ball.x,632.)
            self.assertLess(w.ball.y,200.)
            self.assertFalse(w.failed)
            count=len([e for e in w.events if e[0]=='teleport'])
            self.assertEqual(count,1)

    def test_unpaired_or_disabled_portal_cannot_trap_simulation(self):
        w=make_world(Device('in',410,100,200,group='A'))
        w.ball.x,w.ball.y=132.,232.; w.step(0)
        self.assertGreater(w.time,0); self.assertFalse(w.failed)
        w=make_world(Device('in',410,100,200,target='out'),Device('out',410,600,200,options={'power_group':'off'}))
        w.ball.x,w.ball.y=132.,232.; w.step(0)
        self.assertFalse(w.failed); self.assertGreater(w.time,0)

    def test_blocked_destination_does_not_teleport_inside_solid(self):
        w=make_world(Device('in',410,100,200,target='out'),Device('out',410,600,200),blocks=[Platform('wall',580,100,150,100)])
        w.ball.x,w.ball.y=132.,232.; w.step(0)
        self.assertAlmostEqual(w.ball.x,132.)
        self.assertTrue(any(e[0]=='teleport_blocked' for e in w.events))

    def test_diagonal_exit_clears_the_portal_sensor(self):
        w=make_world(Device('in',410,100,200,target='out'),Device('out',410,600,200,options={'exit_angle':45.}))
        w.ball.x,w.ball.y=132.,232.; w.step(0)
        self.assertFalse(circle_rect_overlap(w.ball.x,w.ball.y,P.radius,(600,200,64,64)))
        self.assertGreater(w.ball.x,664.)


class HazardTests(unittest.TestCase):
    def test_201_202_211_221_spike_geometry_is_lethal(self):
        for type_id in (201,202,211,221):
            w=make_world(Device('hazard',type_id,256,400))
            poly=w.special.polygons(w.special.devices['hazard'],0)[0]
            w.ball.x=sum(v[0] for v in poly)/3; w.ball.y=sum(v[1] for v in poly)/3
            w.step(0); self.assertTrue(w.failed,msg=str(type_id))

    def test_202_small_spike_does_not_use_full_tile_hitbox(self):
        w=make_world(Device('small',202,256,400))
        w.ball.x,w.ball.y=260.,405.; w.step(0)
        self.assertFalse(w.failed)

    def test_303_spark_active_kills_inactive_does_not(self):
        for time,dead in ((.1,True),(1.5,False)):
            w=make_world(Device('spark',303,256,400,options={'period':2.,'active_for':1.}))
            w.time=time; w.ball.x,w.ball.y=288.,432.; w.step(0)
            self.assertEqual(w.failed,dead)

    def test_502_red_kills_blue_repels_and_fast_ball_cannot_tunnel(self):
        for color in ('red','blue'):
            w=make_world(Device('laser',502,200,400,options={'color':color,'length':300.}))
            w.ball.x,w.ball.y,w.ball.vy=350.,300.,3000.
            w.step(0,.08)
            self.assertEqual(w.failed,color=='red')
            if color=='blue': self.assertLess(w.ball.vy,0.)

    def test_503_504_have_distinct_initial_axis_and_rotate(self):
        for type_id,point in ((503,(350.,432.)),(504,(232.,550.))):
            w=make_world(Device('laser',type_id,200,400))
            w.ball.x,w.ball.y=point; w.step(0); self.assertTrue(w.failed)
            d=w.special.devices['laser']
            self.assertNotEqual(w.special.polygons(d,0),w.special.polygons(d,.25))

    def test_rotating_blue_laser_repels_without_death(self):
        w=make_world(Device('laser',503,200,400,options={'color':'blue'}))
        w.ball.x,w.ball.y=350.,432.; w.step(0)
        self.assertFalse(w.failed); self.assertLess(w.ball.vy,0)

    def test_blue_laser_repels_from_underside_in_correct_direction(self):
        w=make_world(Device('laser',502,200,400,options={'color':'blue'}))
        w.ball.x,w.ball.y=350.,440.; w.step(0)
        self.assertFalse(w.failed); self.assertGreater(w.ball.vy,0.)

    def test_rotating_laser_sweeps_into_stationary_player(self):
        w=make_world(Device('laser',503,200,400,options={'angular_speed':1000.}))
        w.ball.x,w.ball.y=232.,560.; w.ball.launch=True
        w.step(0,.1)
        self.assertTrue(w.failed)
        self.assertLess(w.time,.1)

    def test_electric_platform_remains_solid_when_its_explicit_power_is_off(self):
        w=make_world(Device('wire',510,64,500,options={'power_group':'off'}))
        land(w,'wire')
        self.assertEqual(w.last_landing,'wire')
        self.assertFalse(w.failed)

    def test_601_asteroid_uses_moving_collision_not_only_spawn_hitbox(self):
        w=make_world(Device('asteroid',601,500,300,w=24,h=24,options={'velocity':(-400.,0.)}))
        w.ball.launch=True  # Hold a stationary target to isolate relative hazard CCD.
        w.ball.x,w.ball.y=400.,312.; w.step(0,.3)
        self.assertTrue(w.failed)

    def test_603_cannon_spawns_real_projectiles_that_hit(self):
        w=make_world(Device('cannon',603,500,300,options={'speed':400.,'angle':180.,'period':1.}))
        w.ball.launch=True
        w.ball.x,w.ball.y=400.,332.; w.step(0,.3)
        self.assertTrue(w.special.projectiles); self.assertTrue(w.failed)

    def test_602_604_605_move_and_are_hazards_never_support(self):
        for type_id in (602,604,605):
            w=make_world(Device('hazard',type_id,300,300))
            d=w.special.devices['hazard']
            self.assertNotEqual(w.special.rect(d,0),w.special.rect(d,.25))
            self.assertFalse(w.platforms)
            rect=w.special.rect(d,0)
            w.ball.x,w.ball.y=rect[0]+32,rect[1]+32; w.step(0)
            self.assertTrue(w.failed,msg=str(type_id))

    def test_606_top_stomp_destroys_enemy_and_bounces(self):
        w=make_world(Device('enemy',606,256,400))
        w.ball.x,w.ball.y,w.ball.vy=288.,370.,300.
        w.step(0,.08)
        self.assertFalse(w.failed); self.assertTrue(w.special.state['enemy']['destroyed'])
        self.assertLess(w.ball.vy,0)

    def test_606_side_contact_kills(self):
        w=make_world(Device('enemy',606,256,400))
        w.ball.x,w.ball.y,w.ball.vx=200.,430.,P.max_speed
        w.step(1,.3); self.assertTrue(w.failed)

    def test_hazard_wins_over_goal_at_same_position(self):
        w=make_world(Device('spark',303,256,400))
        w.level.goal=(256.,400.,64.,64.); w.ball.x,w.ball.y=288.,432.
        w.step(0); self.assertTrue(w.failed); self.assertFalse(w.cleared)

    def test_pause_and_reset_cover_timers_projectiles_and_device_state(self):
        w=make_world(Device('fall',403,64,500),Device('cannon',603,900,100))
        land(w,'fall'); before=w.time
        w.paused=True; w.step(1,.5); self.assertEqual(w.time,before)
        w.reset()
        self.assertFalse(w.special.projectiles); self.assertIsNone(w.special.platforms['fall'].fall_at)
        self.assertIsNone(w.special.state['fall']['triggered'])

    def test_learning_observations_include_json_serializable_device_state(self):
        env=Environment(0)
        env.world=make_world(Device('cannon',603,900,100),Device('slide',507,256,300))
        env.world.step()
        observation=env.observe()
        self.assertEqual(len(observation['specials']),2)
        self.assertTrue(observation['projectiles'])
        json.dumps(observation,allow_nan=False)


if __name__=='__main__': unittest.main()
