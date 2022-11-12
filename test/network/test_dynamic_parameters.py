import unittest

from network.dynamic_parameter import DynamicParameter


class TestDynamicParameter(unittest.TestCase):
    def test_initialization(self):
        a = DynamicParameter(weight=5)
        self.assertEqual(5, a.parameters["weight"])
        self.assertEqual(1, len(a.parameters))

    def test_getter(self):
        a = DynamicParameter(weight=5)
        self.assertEqual(5, a.weight)

    def test_getter_not_set(self):
        a = DynamicParameter()
        self.assertRaises(AttributeError, a.__getattr__, "weight")
        self.assertEqual(0, len(a.parameters))

    def test_setter(self):
        a = DynamicParameter(weight=5)
        a.weight = 6
        self.assertEqual(6, a.weight)
        self.assertEqual(6, a.parameters["weight"])

    def test_setter_new(self):
        a = DynamicParameter(weight=5)
        a.x = 5
        self.assertTrue("x" in a.parameters)
        self.assertEqual(5, a.x)
        self.assertEqual(5, a.parameters["x"])
