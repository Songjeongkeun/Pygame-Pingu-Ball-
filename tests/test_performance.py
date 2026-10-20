"""Optimizations must retain continuous collision results, including edge hits."""
from dataclasses import asdict
import random
import unittest
from unittest.mock import patch

from pingu_ball import geometry
from pingu_ball.engine import World
from pingu_ball.levels import make_level


class BroadPhaseTests(unittest.TestCase):
    def test_rectangle_filter_matches_the_unfiltered_continuous_solver(self):
        rng=random.Random(915)
        cases=[((0,50),(200,0),(90,20,64,80),14),
               ((0,50),(76-1e-7,0),(90,20,64,80),14),
               ((100,100),(-1e5,1e5),(-500,-500,64,64),14)]
        for _ in range(600):
            cases.append(((rng.uniform(-500,500),rng.uniform(-500,500)),
                          (rng.uniform(-2000,2000),rng.uniform(-2000,2000)),
                          (rng.uniform(-200,200),rng.uniform(-200,200),
                           rng.uniform(1,160),rng.uniform(1,160)),rng.uniform(.1,32)))
        expected=[]
        with patch.object(geometry,'swept_bounds_overlap',return_value=True):
            for args in cases: expected.append(geometry.sweep_circle_rect(*args))
        self.assertEqual([geometry.sweep_circle_rect(*args) for args in cases],expected)

    def test_far_rectangles_skip_corner_tests_but_fast_sweeps_still_hit(self):
        with patch.object(geometry,'ray_circle',wraps=geometry.ray_circle) as corners:
            self.assertIsNone(geometry.sweep_circle_rect((20,20),(1,1),(900,700,64,64),14))
            corners.assert_not_called()
        self.assertIsNotNone(geometry.sweep_circle_rect((20,720),(1200,0),(900,700,64,64),14))

    def test_campaign_ticks_match_with_broad_phase_filters_disabled(self):
        for number in (5,8,11,12):
            rng=random.Random(number)
            directions=[rng.choice((-1,0,1)) for _ in range(120)]
            def record():
                world=World(make_level(number));rows=[]
                for direction in directions:
                    world.step(direction)
                    rows.append((world.time,asdict(world.ball),world.failures,
                                 world.last_landing,world.last_failure,list(world.events)))
                return rows
            filtered=record()
            with patch.object(geometry,'swept_bounds_overlap',return_value=True), \
                 patch('pingu_ball.specials.swept_bounds_overlap',return_value=True):
                reference=record()
            self.assertEqual(filtered,reference,number)


if __name__=='__main__': unittest.main()
