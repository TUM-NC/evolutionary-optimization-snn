import unittest

from utility.list_operation import (
    count_occurrences,
    flat_list,
    get_depths,
    remove_multiple_indices,
)


class TestList(unittest.TestCase):
    def test_remove_single(self):
        start = [1, 2, 3]
        remove = {2}

        out = remove_multiple_indices(start, remove)

        self.assertEqual([1, 2], out)

    def test_remove_none(self):
        start = [1, 2, 3]
        remove = set()

        out = remove_multiple_indices(start, remove)

        self.assertEqual(start, out)

    def test_remove_all(self):
        start = [1, 2, 3]
        remove = {0, 1, 2}

        out = remove_multiple_indices(start, remove)

        self.assertEqual([], out)

    def test_list_flat(self):
        self.assertEqual([1, 2, 3, 4], flat_list([[1, 2, [3, [4]]]]))
        self.assertEqual(
            [1, 2, 3, 4, 5, 6], flat_list([[1, [2, [3, [4]]], [5, [6]]]])
        )
        self.assertEqual([], flat_list([]))
        self.assertEqual([], flat_list([[]]))

    def test_count_occurrences(self):
        self.assertEqual({1: 2, 2: 3}, count_occurrences([1, 2, 2, 2, 1]))
        self.assertEqual({}, count_occurrences([]))
        self.assertEqual({"a": 1, 2: 2}, count_occurrences(["a", 2, 2]))

    def test_depth_levels(self):
        self.assertEqual({0: 1, 1: 2}, get_depths([[1, 2]]))
        self.assertEqual({}, get_depths([]))
        self.assertEqual({0: 1, 1: 1}, get_depths([[[]]]))
        self.assertEqual({0: 3, 1: 2}, get_depths([[[]], [[]], []]))
        self.assertEqual(
            {0: 3, 1: 2, 2: 4}, get_depths([[[1, 2, 3]], [[4]], []])
        )
