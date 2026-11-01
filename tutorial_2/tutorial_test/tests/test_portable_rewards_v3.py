"""외부 PPO 패키지가 없어도 실행되는 독립 보상 단위 테스트."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as S

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tutorial_2.tutorial_test.portable_rewards_v3 import PortableBonusTracker, BonusConfig


class WorldFixture:
    def __init__(self, source='spring', vertical=False):
        self.spec = S(id='source_to_goal',source=(source,),target=('target-0',))
        device = S(type_id=508, y=256., options={'travel':192.})
        self.world = S(time=0.,failed=False,cleared=False,
                       ball=S(x=0.,y=0.),physics=S(radius=14.,bounce_height=96.),
                       events=[],special=S(devices={source:device} if vertical else {}))
        self.rects = {source:(384.,448.,64.,64.) if vertical else (448.,704.,64.,64.),
                      'target-0':(576.,576.,64.,64.)}
        self.env=S(specs=(self.spec,),world=self.world,
                   platform_by_id={source:S(kind='normal' if vertical else 'spring')},
                   bounds=lambda ids:self.rects[ids[0]])
        self.tracker=PortableBonusTracker(self.env)
    def tick(self, t, x, y, event=None):
        self.world.time=t
        self.world.ball.x=x
        self.world.ball.y=y
        self.world.events=[] if event is None else [('land',event)]
        return self.tracker.on_tick(self.world)


class TestSpring(unittest.TestCase):
    def test_defaults_match_final_settings(self):
        c=BonusConfig()
        self.assertEqual((c.spring_height,c.spring_horizontal,c.vertical_tolerance,c.spring_window_seconds),(.5,.5,16.,3.))

    def test_no_source_no_bonus(self):
        f=WorldFixture()
        self.assertEqual(f.tick(.1,590,562),[])

    def test_both_align_once_and_no_double(self):
        f=WorldFixture()
        f.tick(.0,470,650,'spring')
        ev=f.tick(.1,600,562)
        self.assertEqual(sorted([e['kind'] for e in ev]),['spring_height','spring_horizontal'])
        self.assertAlmostEqual(sum(e['points'] for e in ev),1.)
        self.assertEqual(f.tick(.2,600,562),[])
        self.assertEqual(f.tracker.counts['spring_height'],1)
        self.assertEqual(f.tracker.counts['spring_horizontal'],1)

    def test_horizontal_is_independent_as_original(self):
        f=WorldFixture();f.tick(.0,470,650,'spring')
        ev=f.tick(.1,590,150)
        self.assertEqual([e['kind'] for e in ev],['spring_horizontal'])
        self.assertEqual(f.tracker.total,.5)

    def test_expired_flight_has_no_bonus(self):
        f=WorldFixture();f.tick(.0,470,650,'spring')
        self.assertEqual(f.tick(3.01,590,562),[])

    def test_unrelated_landing_deactivates(self):
        f=WorldFixture();f.tick(.0,470,650,'spring')
        f.tick(.1,500,640,'not-related')
        self.assertEqual(f.tick(.2,590,562),[])

    def test_bonuses_do_not_include_common_ten(self):
        f=WorldFixture();f.tick(0.,470,650,'spring')
        ev=f.tick(.1,600,562)
        self.assertTrue(all(v['points']!=10 for v in ev))
        self.assertEqual(f.tracker.total,1.)

    def test_failed_world_no_reward(self):
        f=WorldFixture();f.tick(0.,470,650,'spring')
        f.world.failed=True
        self.assertEqual(f.tick(.1,600,562),[])

    def test_reset_clears_counters(self):
        f=WorldFixture();f.tick(0.,470,650,'spring');f.tick(.1,600,562)
        f.tracker.reset()
        self.assertEqual(f.tracker.total,0.)
        self.assertEqual(sum(f.tracker.counts.values()),0)


class TestMoving(unittest.TestCase):
    def test_first_contact_12_once(self):
        f=WorldFixture('lift',vertical=True)
        ev=f.tick(0.,400,430,'lift')
        self.assertEqual([(e['kind'],e['points']) for e in ev],[('moving_first',12.)])
        self.assertEqual(f.tick(.1,400,430,'lift'),[])

    def test_ascent_12_once_and_relative_height(self):
        f=WorldFixture('lift',vertical=True)
        f.tick(0.,400,430,'lift')
        f.rects['lift']=(384.,330.,64.,64.)
        ev=f.tick(.5,400,315)
        self.assertEqual([(e['kind'],e['points']) for e in ev],[('moving_high',12.)])
        self.assertEqual(f.tick(.6,400,315),[])
        self.assertEqual(f.tracker.total,24.)

    def test_no_high_without_first_contact(self):
        f=WorldFixture('lift',vertical=True)
        f.rects['lift']=(384.,256.,64.,64.)
        self.assertEqual(f.tick(.5,400,245),[])

    def test_no_high_after_contact_expired(self):
        f=WorldFixture('lift',vertical=True)
        f.tick(0.,400,430,'lift')
        f.rects['lift']=(384.,330.,64.,64.)
        self.assertEqual(f.tick(1.06,400,315),[])

    def test_no_high_outside_platform_x(self):
        f=WorldFixture('lift',vertical=True)
        f.tick(0.,400,430,'lift')
        f.rects['lift']=(384.,330.,64.,64.)
        self.assertEqual(f.tick(.5,1000,315),[])

    def test_generic_no_stage_condition(self):
        f=WorldFixture('vertical-platform-id',vertical=True)
        self.assertEqual(f.tick(0.,400,430,'vertical-platform-id')[0]['points'],12.)


if __name__=='__main__': unittest.main()
