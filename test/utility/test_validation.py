import unittest

from utility.validation import (
    any_check,
    chain_checks,
    contains_only_given_keys,
    greater_than_zero,
    is_bool,
    is_float,
    is_int,
    is_none,
    is_number,
    is_percent,
    is_positive,
    is_valid_on_all_dict_values,
    valid_values,
)


class TestValidation(unittest.TestCase):
    def test_is_number(self):
        self.assertTrue(is_number(4))
        self.assertTrue(is_number(4.5))
        self.assertTrue(is_number(0))
        self.assertTrue(is_number(1))
        self.assertTrue(is_number(-2))
        self.assertFalse(is_number("test"))
        self.assertFalse(is_number(True))
        self.assertFalse(is_number(False))
        self.assertFalse(is_number([]))

    def test_is_float(self):
        self.assertTrue(is_float(4.5))
        self.assertTrue(is_float(0.1))
        self.assertTrue(is_float(-1.1))
        self.assertTrue(is_float(-9999.0))
        self.assertFalse(is_float(True))
        self.assertFalse(is_float(False))
        self.assertFalse(is_float(5), "int is not float")
        self.assertFalse(is_float("test"))
        self.assertFalse(is_float([]))
        self.assertFalse(is_float({}))

    def test_is_int(self):
        self.assertTrue(is_int(0))
        self.assertTrue(is_int(1))
        self.assertTrue(is_int(1000))
        self.assertTrue(is_int(-1000))
        self.assertFalse(is_int(0.1))
        self.assertFalse(is_int(0.0))
        self.assertFalse(is_int("1"))
        self.assertFalse(is_int([]))
        self.assertFalse(is_int({}))

    def test_is_positive(self):
        self.assertTrue(is_positive(0), "should include 0")
        self.assertTrue(is_positive(1))
        self.assertTrue(is_positive(1.5))
        self.assertTrue(is_positive(9999))
        self.assertFalse(is_positive(-1))
        self.assertFalse(is_positive(-1.1))
        self.assertFalse(is_positive(-9999))
        self.assertFalse(greater_than_zero(None))

    def test_greater_than_zero(self):
        self.assertTrue(greater_than_zero(1))
        self.assertTrue(greater_than_zero(1.1))
        self.assertTrue(greater_than_zero(999))
        self.assertTrue(greater_than_zero(0.00001))
        self.assertFalse(greater_than_zero(-1))
        self.assertFalse(greater_than_zero(-1.5))
        self.assertFalse(greater_than_zero(0))
        self.assertFalse(greater_than_zero(0.0000))
        self.assertFalse(greater_than_zero(None))

    def test_is_percent(self):
        self.assertTrue(is_percent(0.1))
        self.assertTrue(is_percent(0.5))
        self.assertTrue(is_percent(1))
        self.assertTrue(is_percent(0))
        self.assertFalse(is_percent(2))
        self.assertFalse(is_percent(-0.001))
        self.assertFalse(is_percent("1"))
        self.assertFalse(is_percent("test"))

    def test_is_bool(self):
        # we only want real boolean values here
        self.assertTrue(is_bool(True))
        self.assertTrue(is_bool(False))
        self.assertFalse(is_bool("False"))
        self.assertFalse(is_bool("True"))
        self.assertFalse(is_bool("0"))
        self.assertFalse(is_bool("1"))
        self.assertFalse(is_bool(0))
        self.assertFalse(is_bool(1))

    def test_is_optional(self):
        self.assertTrue(is_none(None))
        self.assertFalse(is_none(1))
        self.assertFalse(is_none(0))
        self.assertFalse(is_none(False))

    def test_is_valid_on_all_dict_values(self):
        all_numbers = {"a": 2, "b": 1.2, "c": -3}
        also_bool = {"a": 2, "b": False}
        only_bool = {"b": False}
        test_all_numbers = is_valid_on_all_dict_values(is_number)

        self.assertTrue(test_all_numbers(all_numbers))
        self.assertFalse(test_all_numbers(also_bool))
        self.assertFalse(test_all_numbers(only_bool))

    def test_is_valid_on_all_dict_values_empty(self):
        always_false = is_valid_on_all_dict_values(lambda x: False)

        self.assertTrue(always_false({}))
        self.assertFalse(always_false({"test": "abc"}))

    def test_contains_only_given_keys(self):
        no_keys = contains_only_given_keys([])
        only_abc = contains_only_given_keys(["a", "b", "c"])

        self.assertTrue(only_abc({}), "empty can only contain given")
        self.assertTrue(no_keys({}), "empty can only contain given")
        self.assertTrue(only_abc({"a": 1, "b": 2, "c": 3}))
        self.assertTrue(only_abc({"b": 2}), "can contain parts of given keys")
        self.assertFalse(only_abc({"a": 1, "b": 2, "c": 3, "d": 4}))
        self.assertFalse(no_keys({"a": 1, "b": 2, "c": 3, "d": 4}))
        self.assertFalse(no_keys({"a": 1}))

    def test_valid_values(self):
        a_b_c = valid_values(["a", "b", "c"])
        self.assertTrue(a_b_c("a"))
        self.assertTrue(a_b_c("b"))
        self.assertTrue(a_b_c("c"))
        self.assertFalse(a_b_c(""))
        self.assertFalse(a_b_c(5))

    def test_valid_values_mixed_type(self):
        mixed_types = valid_values(["a", True])
        self.assertTrue(mixed_types("a"))
        self.assertTrue(mixed_types(True))
        self.assertFalse(mixed_types("c"))
        self.assertFalse(mixed_types(""))
        self.assertFalse(mixed_types(5))

    def test_chain_checks(self):
        v = 3.3

        self.assertTrue(is_positive(v))
        self.assertTrue(is_float(v))
        self.assertFalse(is_int(v))

        self.assertTrue(chain_checks(is_positive, is_float)(v))
        self.assertTrue(chain_checks(is_float, is_positive)(v))
        self.assertTrue(chain_checks(is_float, is_positive, is_float)(v))
        self.assertFalse(chain_checks(is_positive, is_int)(v))
        self.assertFalse(chain_checks(is_int, is_positive)(v))
        self.assertFalse(chain_checks(is_float, is_positive, is_int)(v))

    def test_any_check(self):
        def raise_exception(_):
            raise RuntimeError()

        v = 3.3

        self.assertTrue(any_check(is_int, is_float)(v))
        self.assertFalse(any_check(is_int, is_none)(v))
        self.assertTrue(any_check(is_float, raise_exception)(v))
        with self.assertRaises(RuntimeError):
            any_check(raise_exception, is_float)(v)
        with self.assertRaises(RuntimeError):
            any_check(is_int, raise_exception)(v)
