"""Guide footprints must agree with the simulated reversal positions."""
from dataclasses import replace
import unittest
from pingu_ball.engine import World
from pingu_ball.levels import make_level
from pingu_ball.special_lab import make_lab
from pingu_ball.playfield import expand_playfield


class MovementMarkerTests(unittest.TestCase):
    def test_all_native_and_display_maps_match_real_platform_reversal_rectangles(self):
        levels=[make_level(n) for n in range(1,13)]+[make_lab(n) for n in range(1,11)]
        counts={}
        for source in levels:
            for level in (source,expand_playfield(source)):
                with self.subTest(title=level.title,width=level.width):
                    world=World(level);expected=[]
                    for d in level.specials:
                        if d.type_id not in (501,508):continue
                        travel=d.options.get('travel',128.);speed=d.options.get('speed',world.physics.platform_speed)
                        phase=d.options.get('phase',0.)
                        block=world.special.platforms[d.id]
                        for distance in (0.,travel):
                            time=((distance-phase)%(2*travel))/speed
                            expected.append(block.rect_at(time))
                    self.assertEqual(len(world.movement_endpoints),len(expected))
                    for rect in expected:
                        self.assertTrue(any(all(abs(a-b)<1e-6 for a,b in zip(rect,marker))
                                            for marker in world.movement_endpoints))
                    for marker in world.movement_endpoints:self.assertEqual(marker[2:],(64,64))
            counts[source.title]=len(World(source).movement_endpoints)
        self.assertEqual(sum(bool(n) for n in counts.values()),8)
        self.assertEqual(sum(counts.values()),32)

    def test_manual_markers_cannot_leave_stale_or_one_tile_guides(self):
        level=make_level(7)
        world=World(replace(level,movement_markers=[(9999,9999)]))
        self.assertEqual(world.movement_endpoints,World(level).movement_endpoints)
        self.assertEqual(len(world.movement_endpoints),4)
        self.assertFalse(any(abs(x-1408)<1e-5 and y==448 for x,y,_,_ in world.movement_endpoints))
        before=world.movement_endpoints
        for _ in range(30):world.step(0)
        world.reset()
        self.assertEqual(world.movement_endpoints,before)


if __name__=='__main__':unittest.main()
