"""Campaign replacement, native devices, progression and tutorial preservation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pingu_ball.campaign import TILE_ADDITIONS, campaign_layouts
from pingu_ball.config import CAMPAIGN_ID, REGULAR_STAGE_COUNT
from pingu_ball.engine import World
from pingu_ball.levels import make_level, make_levels
from pingu_ball.levels import waypoint_direction
from pingu_ball.special_lab import LAB_TYPES, make_lab
from pingu_ball.storage import Progress

ROOT = Path(__file__).resolve().parents[1]


class CampaignTests(unittest.TestCase):
    def test_document_order_and_fixed_requested_goal_supports(self):
        self.assertEqual(REGULAR_STAGE_COUNT, 12)
        self.assertEqual([stage['number'] for stage in campaign_layouts()], list(range(1, 13)))
        self.assertEqual([level.index for level in make_levels()], list(range(13)))
        for number, position in ((5,(768,640)), (8,(64,448)), (12,(1664,928))):
            level = make_level(number)
            self.assertTrue(any((q.x,q.y)==position and q.kind=='normal' for q in level.platforms))
            self.assertEqual(level.goal[:2], (position[0], position[1]-64))
        self.assertFalse((ROOT/'pingu_ball/reference_levels.py').exists())
        self.assertFalse(list((ROOT/'assets/maps').glob('stage_*.png')))
        with self.assertRaises(ValueError): make_level(13)

    def test_breakable_tiles_disappear_on_contact_and_restore_on_reset(self):
        w = World(make_level(2), auto_reset=False)
        q = next(q for q in w.platforms if q.kind=='breakable')
        w.ball.x,w.ball.y,w.ball.vy = q.x+32,q.y-15,30
        for _ in range(8): w.step(0)
        self.assertTrue(q.destroyed)
        w.reset()
        self.assertFalse(w.special.platforms[q.id].destroyed)

    def test_timed_ghosts_repeat_and_reset_from_the_same_stage_clock(self):
        w = World(make_level(5))
        q = next(q for q in w.platforms if q.id.startswith('timed_off'))
        self.assertFalse(q.active(0, {}))
        self.assertTrue(q.active(1.2, {}))
        for t in (0,1.2,2.4): self.assertEqual(q.active(t,{}),q.active(t+q.period,{}))
        before = q.rect_at(0), q.active(0,{})
        w.step(1); w.reset()
        q = w.special.platforms[q.id]
        self.assertEqual((q.rect_at(0),q.active(0,{})),before)

    def test_perimeter_animals_are_in_both_final_maps_and_rotate(self):
        for n in (11,12):
            w = World(make_level(n))
            d = next(d for d in w.level.specials if d.type_id==703)
            self.assertEqual(d.options['size'],56)
            left,top,width,height=d.options['path']
            poses=[w.special.patrol_pose(d,t/64) for t in range(0,round(2*(width+height)),16)]
            self.assertEqual({angle for _,_,angle in poses},{0,90,180,270})
            for x,y,_ in poses:
                self.assertTrue(x in (left,left+width) or y in (top,top+height))

    def test_tutorial_layouts_and_assets_are_unchanged(self):
        self.assertEqual(len(LAB_TYPES),10)
        for n in range(1,11): self.assertEqual(make_lab(n).index,0)
        baseline = ROOT/'tests/fixtures/tutorial_baseline.json'
        for name,digest in json.loads(baseline.read_text(encoding='utf-8')).items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)

    def test_layout_counts_include_only_requested_additions_and_removals(self):
        for source in campaign_layouts():
            level=make_level(source['number'])
            self.assertEqual(len(level.platforms)+len(level.specials),
                             len(source['tiles'])+len(source.get('patrols',[]))
                             + len(TILE_ADDITIONS.get(source['number'], []))
                             - (1 if source['number']==4 else 0))
            self.assertEqual(len(level.powerups),len(source['items']))
            self.assertFalse(any(q.id.startswith('approach') for q in level.platforms+level.specials))
            w=World(level)
            self.assertEqual((w.physics.bounce_height,w.physics.pad_height),(96,144))

    def land(self,w,x,y):
        since=w.time
        for _ in range(1200):
            w.step(waypoint_direction(w,x))
            if w.failed or w.cleared: break
            if w.last_landing_time>since and any(q.id==w.last_landing and q.x<=x<=q.x+q.w
                                                 and abs(q.y-y)<.01 for q in w.platforms):
                return
        self.fail((w.level.index,x,y,w.time,w.last_failure,w.last_landing))

    def test_seventh_map_original_springs_reach_the_repositioned_pickup(self):
        w=World(make_level(7),auto_reset=False)
        for x,y in ((224,896),(352,768),(480,640),(608,512),(736,384),(864,256)):
            self.land(w,x,y)
        self.assertFalse(w.failed)
        self.assertEqual(w.ball.power_up,'up')
        self.assertFalse(any(d.type_id==508 for d in w.level.specials))

    def test_third_map_repositioned_springs_reach_the_original_goal(self):
        w=World(make_level(3),auto_reset=False)
        # Replay from the original lower deck's right edge.
        w.ball.x,w.ball.y,w.ball.vy=950,882-1e-6,-w.physics.bounce_speed
        for x,y in ((1120,832),(1248,704),(1376,576)):
            self.land(w,x,y)
        since=w.last_landing_time
        while w.time-since<.37: w.step(0)
        for _ in range(600):
            w.step(waypoint_direction(w,w.level.goal[0]+32))
            if w.cleared or w.failed: break
        self.assertTrue(w.cleared,w.last_failure)

    def test_third_map_ferry_reaches_the_upper_ledge_exit_and_turns(self):
        w=World(make_level(3),auto_reset=False)
        ferry=[q for q in w.platforms if q.kind=='moving']
        self.assertEqual(len(ferry),2)
        for q in ferry: self.assertEqual(q.travel,448)
        turn=(448-ferry[0].phase)/96
        self.assertAlmostEqual(max(q.rect_at(turn)[0]+q.w for q in ferry),832)
        self.assertGreater(ferry[0].rect_at(turn)[0],ferry[0].rect_at(turn+.2)[0])
        self.assertEqual(w.level.movement_markers,[(256,512),(768,512)])
        for t in (0,turn,turn+.2,8):
            self.assertAlmostEqual(ferry[1].rect_at(t)[0]-ferry[0].rect_at(t)[0],64)

    def test_third_map_full_spawn_to_goal_inputs_and_old_range_regression(self):
        from tools.verify_stage3 import route
        replay=route()
        self.assertTrue(replay.world.cleared)
        self.assertEqual(replay.world.failures,0)
        for old_range in (False,True):
            level=make_level(3)
            if old_range:
                for d in level.specials:
                    if d.type_id==501: d.options['travel']=292.8
            w=World(level,auto_reset=False)
            for direction in replay.inputs:
                self.assertIn(direction,(-1,0,1))
                w.step(direction)
                if w.failed or w.cleared: break
            self.assertEqual(w.cleared,not old_range)
            self.assertEqual(w.failed,old_range)

    def test_sixth_map_ferry_returns_to_required_launcher_before_goal(self):
        from tools.verify_campaign_routes import route
        replay=route(6)
        self.assertTrue(replay.world.cleared)
        self.assertEqual(replay.world.failures,0)
        self.assertIn('spring-to-ferry',[m['label'] for m in replay.milestones])
        labels=[m['label'] for m in replay.milestones]
        expected=['ride-left','spring-27','normal-20','upper-launcher','goal']
        self.assertEqual([label for label in labels if label in expected],expected)
        level=make_level(6)
        ferry=[d for d in level.specials if d.type_id==501]
        self.assertEqual([(d.y,d.options['travel'],d.options['speed']) for d in ferry],
                         [(704,896,96),(704,896,96)])
        self.assertEqual(level.movement_markers,[(576,704),(1472,704)])
        self.assertFalse(level.show_movement_rails)
        self.assertEqual(level.goal[:2],(1408,448))
        self.assertTrue(any((q.x,q.y)==(1408,512) for q in level.platforms))
        # The ferry and right spring cannot reach even the bottom of the goal.
        for q in replay.world.platforms:
            if q.kind=='moving' or q.id in ('spring-28','normal-22','normal-23','normal-24'):
                height=replay.world.physics.pad_height if q.kind=='spring' else replay.world.physics.bounce_height
                self.assertGreater(q.y-height-replay.world.physics.radius,level.goal[1]+level.goal[3])
        for regression in ('fixed','old_ferry_height','missing_upper_launcher'):
            level=make_level(6)
            if regression=='old_ferry_height':
                for d in level.specials:
                    if d.type_id==501: d.y=640
            if regression=='missing_upper_launcher':
                level.specials=[d for d in level.specials if d.id!='launcher_right-25']
            world=World(level,auto_reset=False)
            boarded=False
            for direction,use,power_direction in replay.inputs:
                world.step(direction,use_power=use,power_direction=power_direction)
                boarded |= (world.last_landing or '').startswith('ferry')
                if world.failed or world.cleared: break
            self.assertEqual(boarded,regression!='old_ferry_height')
            self.assertEqual(world.cleared,regression=='fixed')
            if regression=='fixed': self.assertEqual(world.failures,0)

    def test_fourth_map_full_route_requires_fixed_spring_heights(self):
        from tools.verify_stage4 import route
        replay=route()
        self.assertTrue(replay.world.cleared)
        self.assertEqual(replay.world.failures,0)
        for regression in ('fixed','old_spring_height'):
            with self.subTest(regression=regression):
                level=make_level(4)
                if regression=='old_spring_height':
                    next(d for d in level.specials if d.id=='spring-35').y=576
                world=World(level,auto_reset=False)
                for direction in replay.inputs:
                    self.assertIn(direction,(-1,0,1))
                    world.step(direction)
                    if world.failed or world.cleared: break
                self.assertEqual(world.cleared,regression=='fixed')
                self.assertEqual(world.failed,regression!='fixed')

    def test_fourth_map_breakable_has_no_spike_inside_its_tile(self):
        level=make_level(4)
        tile=next(d for d in level.specials if d.type_id==301)
        self.assertEqual((tile.x,tile.y),(192,768))
        spikes=[d for d in level.specials if d.type_id==201]
        self.assertEqual(len(spikes),11)
        for spike in spikes:
            self.assertFalse(tile.x<spike.x+spike.w and spike.x<tile.x+tile.w
                             and tile.y<spike.y+spike.h and spike.y<tile.y+tile.h)

    def test_fifth_map_full_route_and_original_spring_height_regression(self):
        from tools.verify_campaign_routes import route
        replay=route(5)
        self.assertTrue(replay.world.cleared)
        self.assertEqual(replay.world.failures,0)
        labels=[m['label'] for m in replay.milestones]
        expected=['spring-33','spring-32','spring-31','spring-72','spring-73','launcher_left-47']
        self.assertEqual([label for label in labels if label in expected],expected)
        springs=[d for d in replay.world.level.specials if d.type_id==401]
        self.assertEqual(sorted(d.y for d in springs),[320,448,576,704,832])
        self.assertTrue(all((d.w,d.h)==(64,64) for d in springs))
        for regression in ('fixed','old_height','missing_steps'):
            level=make_level(5)
            if regression=='old_height':
                next(d for d in level.specials if d.id=='spring-32').y=640
            if regression=='missing_steps':
                level.specials=[d for d in level.specials if d.id not in ('spring-72','spring-73')]
            world=World(level,auto_reset=False)
            reached_launcher=False
            for direction,use,power_direction in replay.inputs:
                world.step(direction,use_power=use,power_direction=power_direction)
                reached_launcher |= world.last_landing=='launcher_left-47'
                if world.failed or world.cleared: break
            self.assertEqual(world.cleared,regression=='fixed')
            self.assertEqual(reached_launcher,regression=='fixed')
            if regression=='fixed': self.assertEqual(world.failures,0)

    def test_ninth_map_repositioned_tiles_reach_the_launcher_and_goal(self):
        w=World(make_level(9),auto_reset=False)
        # The icicles have already passed; begin at the original first spring.
        w.time=1.
        w.ball.x,w.ball.y,w.ball.vy=736,754-1e-6,-w.physics.pad_speed
        for x,y in ((864,640),(992,640),(1120,640),(1248,640),(1376,576)):
            self.land(w,x,y)
        since=w.last_landing_time
        while w.time-since<.37: w.step(0)
        for _ in range(1200):
            w.step(0 if w.ball.launch else waypoint_direction(w,1312))
            if w.cleared or w.failed: break
        self.assertTrue(w.cleared,w.last_failure)
        self.assertEqual(w.level.goal[:2],(64,448))

    def test_old_campaign_records_reset_but_tutorial_times_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'progress.json'
            path.write_text(json.dumps({'version':3,'unlocked':7,
                'records':{'0':{'cleared':True},'7':{'score':10,'time':4}},
                'tutorial_records':{'1':5.25,'10':18.5}}),encoding='utf-8')
            p=Progress(path)
            self.assertEqual(p.unlocked,1)
            self.assertEqual(p.records,{'0':{'cleared':True}})
            self.assertEqual(p.tutorial_records,{'1':5.25,'10':18.5})
            for n in range(1,13): p.complete(n,7,float(n))
            saved=Progress(path)
            self.assertEqual(saved.unlocked,12)
            self.assertEqual(saved.tutorial_records,p.tutorial_records)
            self.assertEqual(set(saved.records),{str(i) for i in range(13)})
            data=json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual((data['version'],data['campaign']),(4,CAMPAIGN_ID))


if __name__=='__main__': unittest.main()
