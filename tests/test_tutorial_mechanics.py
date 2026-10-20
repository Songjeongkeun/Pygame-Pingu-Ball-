"""Gameplay contracts of the newly imported tutorial hazards and dash gate."""
import unittest
from pingu_ball.engine import Level, World, Platform
from pingu_ball.specials import Device
from pingu_ball.special_lab import make_lab
from tools.verify_tutorials import route


def world_with(device,blocks=()):
    return World(Level(0,'hazard','',list(blocks),[],[],(32,32),
                       (1400,700,64,64),specials=[device]),auto_reset=False)


class TutorialMechanicsTests(unittest.TestCase):
    def test_supported_spike_does_not_hit_above_its_visible_tip(self):
        d=Device('spike',201,0,0)
        w=world_with(d,[Platform('base',0,64)])
        self.assertFalse(w.special.candidates((32,5),(0,0),0))
        self.assertTrue(any(e[2]=='fail' for e in w.special.candidates((32,32),(0,0),0)))

    def test_sun_is_non_solid_and_even_square_corners_are_lethal(self):
        w=world_with(Device('sun',701,100,100))
        self.assertFalse(w.platforms)
        w.ball.x,w.ball.y=101,101
        w.step(0)
        self.assertTrue(w.failed)
        self.assertIn('태양',w.last_failure)

    def test_icicle_forming_half_second_is_safe_then_contact_is_lethal(self):
        for time,expected in ((.49,False),(.501,True)):
            w=world_with(Device('ice',702,100,100))
            w.time=time
            x,y,_,_=w.special.rect(w.level.specials[0],time)
            w.ball.x,w.ball.y=x+32,y+32
            w.step(0,.001)
            self.assertEqual(w.failed,expected)

    def test_icicle_falls_and_restarts_from_its_original_spawner(self):
        d=Device('ice',702,100,100)
        w=world_with(d)
        self.assertEqual(w.special.rect(d,.5)[1],100)
        self.assertGreater(w.special.rect(d,1.)[1],200)
        self.assertEqual(w.special.rect(d,4.)[1],100)
        w.time=1.;w.reset()
        self.assertEqual(w.special.rect(d,w.time)[1],100)

    def test_sea_lion_rotates_through_all_four_edges_of_the_joined_blocks(self):
        d=Device('lion',703,100,200,options={'path':(100,200,192,64),'speed':64})
        w=world_with(d)
        for time,angle in ((1,0),(3.5,90),(5,180),(7.5,270)):
            self.assertEqual(w.special.patrol_pose(d,time)[2],angle)
        self.assertEqual(w.special.patrol_pose(d,0),w.special.patrol_pose(d,8))
        self.assertEqual(w.special.rect(d,0)[2:],(56,56))

    def test_hidden_half_of_patrol_does_not_kill_through_terrain(self):
        d=Device('lion',703,100,200,options={'path':(100,200,192,64),'phase':32.})
        w=world_with(d,[Platform('base',100,200)])
        hits=w.special.candidates((132,216),(0,0),0)
        self.assertFalse(any(hit[2]=='fail' for hit in hits))
        hits=w.special.candidates((132,185),(0,0),0)
        self.assertTrue(any(hit[2]=='fail' for hit in hits))

    def test_final_gap_is_cleared_with_dash_but_not_with_the_dash_press_omitted(self):
        witness=route(10)
        self.assertTrue(witness.world.cleared)
        w=World(make_lab(10),auto_reset=False)
        omitted=0
        for direction,use in witness.inputs:
            if use and w.ball.power_up=='forward':
                use=False;omitted+=1
            w.step(direction,use_power=bool(use),power_direction=direction if use else 0)
        for _ in range(600):
            if w.failed or w.cleared: break
            w.step(1)
        self.assertEqual(omitted,1)
        self.assertFalse(w.cleared)
        self.assertTrue(w.failed)


if __name__=='__main__': unittest.main()
