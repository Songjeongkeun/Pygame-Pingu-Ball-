"""Dependency-free adapter tests; run: python -m unittest discover -s tutorial_2/mixed_map_transfer/tests -v"""
import unittest
from tutorial_2.mixed_map_transfer.adapter_utils import ranked_indices, rect_separation, parse_map_selection


class TestPureAdapter(unittest.TestCase):
    def test_distance_inside(self):
        self.assertEqual(0, rect_separation((0, 0, 64, 64), 32, 24))

    def test_distance_near_edge_not_center(self):
        self.assertEqual(2, rect_separation((0, 0, 64, 64), 66, 40))

    def test_local_sort_nearest_and_tie_break(self):
        rec = [{'rect': (600, 100, 64, 64)},
               {'rect': (0, 0, 64, 64)},
               {'rect': (80, 0, 64, 64)}]
        self.assertEqual([1, 2, 0], ranked_indices(rec, 30, 30))
        rec = [{'rect': (0, 0, 10, 10)}, {'rect': (0, 0, 10, 10)}]
        self.assertEqual([0, 1], ranked_indices(rec, 1, 1))

    def test_all_maps(self):
        self.assertEqual(list(range(1, 11)), parse_map_selection('all'))
        self.assertEqual(list(range(1, 11)), parse_map_selection('1-10'))

    def test_subset_and_duplicates(self):
        self.assertEqual([1, 2, 4], parse_map_selection('1,2,2,4'))

    def test_bad_map(self):
        for bad in ('0', '11', '3-2', 'x', '1,,3', ''):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_map_selection(bad)


if __name__ == '__main__':
    unittest.main()
