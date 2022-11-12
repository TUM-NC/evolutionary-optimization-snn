import random
import unittest

from utility.random import get_int_with_exclude, random_rates


class TestRandom(unittest.TestCase):
    def test_random_value_single_value(self):
        rates = {"test": 1}
        value = random_rates(rates)
        self.assertEqual("test", value)

    def test_random_value_multiple_value(self):
        rates = {"test": 0, "test2": 1}
        value = random_rates(rates)
        self.assertEqual("test2", value)

    def test_random_in_exclude(self):
        n = get_int_with_exclude([0], 2)
        self.assertEqual(n, 1)

    def test_random_in_exclude_in_range(self):
        for _ in range(50):
            n = get_int_with_exclude(max=20)
            self.assertTrue(0 <= n <= 19)

    def test_random_rates_reproducible(self):
        rates = {"test": 0.5, "test2": 0.5, "test3": 0.5}
        random.seed(1)
        v1 = random_rates(rates)
        random.seed(1)
        v2 = random_rates(rates)
        random.seed(2)
        v3 = random_rates(rates)

        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, v3)

    def test_rate_roughly(self):
        rates = {"b": 0.1, "a": 0.8, "c": 0.1, "d": 0}
        count = {"a": 0, "b": 0, "c": 0, "d": 0}

        random.seed(1)

        for i in range(1000):
            v = random_rates(rates)
            count[v] += 1

        self.assertTrue(700 < count["a"] < 900)
        self.assertTrue(80 < count["b"] < 120)
        self.assertTrue(80 < count["c"] < 120)
        self.assertTrue(0 == count["d"])

    def test_random_rate_should_not_depend_on_order(self):
        rates1 = {"test2": 0.5, "test": 0.5, "test3": 0.5}
        rates2 = {"test": 0.5, "test2": 0.5, "test3": 0.5}

        random.seed(1)
        v1 = random_rates(rates1)
        random.seed(1)
        v2 = random_rates(rates2)

        self.assertEqual(v1, v2)

    def test_get_int_reproducible(self):
        random.seed(1)
        n1 = get_int_with_exclude(max=1000)
        random.seed(1)
        n2 = get_int_with_exclude(max=1000)
        random.seed(2)
        n3 = get_int_with_exclude(max=1000)

        self.assertEqual(n1, n2)
        self.assertNotEqual(n1, n3)
