"""Full-width worlds preserve document distances and move the actual walls."""
from dataclasses import asdict
import unittest
from pingu_ball.engine import World
from pingu_ball.levels import make_level
from pingu_ball.special_lab import make_lab
from pingu_ball.playfield import expand_playfield, VIEW_WIDTH, VIEW_HEIGHT


class PlayfieldTests(unittest.TestCase):
    def test_all_regular_and_tutorial_objects_translate_together(self):
        for source in [make_level(n) for n in range(1,13)]+[make_lab(n) for n in range(1,11)]:
            with self.subTest(title=source.title):
                original=asdict(source)
                fitted=expand_playfield(source)
                dx=(fitted.width-source.width)/2
                self.assertGreaterEqual(fitted.width/fitted.height,VIEW_WIDTH/VIEW_HEIGHT)
                self.assertEqual(fitted.spawn,(source.spawn[0]+dx,source.spawn[1]))
                self.assertEqual(fitted.goal,(source.goal[0]+dx,*source.goal[1:]))
                for field in ('platforms','collectibles','powerups','specials','spikes'):
                    old,new=getattr(source,field),getattr(fitted,field)
                    self.assertEqual(len(old),len(new))
                    for a,b in zip(old,new):
                        self.assertEqual((b.x,b.y),(a.x+dx,a.y))
                        if field=='specials' and 'path' in a.options:
                            path=a.options['path']
                            self.assertEqual(b.options['path'],(path[0]+dx,*path[1:]))
                self.assertEqual(fitted.movement_markers,[(x+dx,y) for x,y in source.movement_markers])
                self.assertEqual(asdict(source),original)

    def test_neutral_spawn_trajectory_matches_in_all_22_maps(self):
        for source in [make_level(n) for n in range(1,13)]+[make_lab(n) for n in range(1,11)]:
            with self.subTest(title=source.title):
                fitted=expand_playfield(source);dx=(fitted.width-source.width)/2
                old,new=World(source),World(fitted)
                for _ in range(184):
                    old.step(0);new.step(0)
                    self.assertAlmostEqual(new.ball.x-dx,old.ball.x,places=5)
                    self.assertAlmostEqual(new.ball.y,old.ball.y,places=5)
                    self.assertEqual(new.last_landing,old.last_landing)
                    self.assertEqual(new.failures,old.failures)

    def test_ball_crosses_former_walls_and_stops_at_the_new_walls(self):
        source=make_level(2);level=expand_playfield(source);dx=(level.width-source.width)/2
        for direction,start,limit in [(-1,dx+20,dx+14),(1,dx+source.width-20,dx+source.width-14)]:
            w=World(level,auto_reset=False);w.ball.x,w.ball.y=start,80
            for _ in range(30):w.step(direction)
            self.assertFalse(w.failed)
            self.assertGreater((w.ball.x-limit)*direction,0)
        for direction,x in [(-1,14.1),(1,level.width-14.1)]:
            w=World(level,auto_reset=False);w.ball.x,w.ball.y=x,80
            for _ in range(20):w.step(direction)
            self.assertFalse(w.failed)
            self.assertGreaterEqual(w.ball.x,14-1e-5)
            self.assertLessEqual(w.ball.x,level.width-14+1e-5)


if __name__=='__main__':unittest.main()
