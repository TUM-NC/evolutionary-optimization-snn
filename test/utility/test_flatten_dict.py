import unittest

from utility.flatten_dict import flatten_dict


class TestFlattenDict(unittest.TestCase):
    def test_simple_dict(self):
        d = {"a": 1, "b": {"c": 2}, "d": [1, 2, 3]}
        flat = flatten_dict(d)
        self.assertEqual({"a": 1, "b.c": 2, "d": [1, 2, 3]}, flat)

    def test_empty(self):
        flat = flatten_dict({})
        self.assertEqual({}, flat)

    def test_nested(self):
        d = {"a": {"b": {"c": {"d": {"e": 5}}}}}
        flat = flatten_dict(d)
        self.assertEqual({"a.b.c.d.e": 5}, flat)

    def test_boolean_key(self):
        d = {True: "a", "a": {False: "b"}}
        flat = flatten_dict(d)
        self.assertEqual({"True": "a", "a.False": "b"}, flat)

    def test_int_key(self):
        d = {1: "a", "a": {1: "b"}}
        flat = flatten_dict(d)
        self.assertEqual({"1": "a", "a.1": "b"}, flat)

    def test_boolean_duplicate(self):
        d = {True: 1, "True": 2}
        flat = flatten_dict(d)
        self.assertEqual({"True": 2}, flat)  # last value wins
