"""강화학습 맵1의 오른쪽 위 발판 세 개 삭제와 나머지 맵 보존을 검사한다."""
import unittest

from pingu_ball.engine import World
from pingu_ball.playfield import expand_playfield
from pingu_ball.rl_maps import make_rl_level, rl_map_layouts


class RLMap1LayoutTests(unittest.TestCase):
    def test_only_map1_has_the_three_upper_right_tiles_removed(self):
        removed = {f'turn0landing-{i}' for i in range(3)}
        maps = rl_map_layouts()
        first = maps[0]
        self.assertEqual(len(first['platforms']), 52)
        self.assertTrue(removed.isdisjoint(p['id'] for p in first['platforms']))
        self.assertEqual((first['width'], first['height']), (3776, 1152))
        self.assertEqual(first['spawn'], [160.0, 786.0])
        self.assertEqual(first['goal'], [512.0, 320, 64, 64])
        for data in maps[1:]:
            with self.subTest(map=data['number']):
                self.assertTrue(removed.issubset(p['id'] for p in data['platforms']))

    def test_left_buffer_and_jump_staircase_are_preserved(self):
        level = make_rl_level(1)
        lookup = {p.id: p for p in level.platforms}
        for i in range(3):
            p = lookup[f'turn0buffer-{i}']
            self.assertEqual((p.x, p.y, p.w, p.h, p.kind), (3136+i*64, 384, 64, 64, 'normal'))
        for i in range(4):
            self.assertEqual(lookup[f'turn0-spring{i}-0'].kind, 'spring')
        self.assertEqual(len(level.specials), 12)
        self.assertEqual({d.type_id for d in level.specials}, {301, 302, 501, 508})

    def test_removed_region_has_no_remaining_collision(self):
        world = World(expand_playfield(make_rl_level(1)), auto_reset=False)
        world.ball.x, world.ball.y = 3552., 340.
        world.ball.vx = world.ball.vy = 0.
        landings = []
        for _ in range(60):
            world.step(0)
            landings.extend(value for event, value in world.events if event == 'land')
        self.assertGreater(world.ball.y, 448.)
        self.assertFalse(landings)
        self.assertFalse(world.failed)


if __name__ == '__main__':
    unittest.main()
