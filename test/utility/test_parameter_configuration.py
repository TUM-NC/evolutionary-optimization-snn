import random
import unittest
import warnings

from utility.parameter_configuration import (
    check_parameter_values_on_specification,
    get_mutable_parameters,
    init_parameter_values,
    is_valid_parameter_configuration,
    is_valid_parameter_value,
    parameter_to_value,
)
from utility.validation import any_check, is_int, is_none, is_positive

mutable_sample_config = {
    "a": {"type": "random_int", "min": 0, "max": 5},
    "b": {"type": "random_bool"},
    "c": {"type": "fixed", "value": 5},
}


class TestParameterConfiguration(unittest.TestCase):
    def test_mutable_parameters(self):
        p = get_mutable_parameters(mutable_sample_config)
        self.assertEqual(2, len(p))
        self.assertTrue(("a", {"type": "random_int", "min": 0, "max": 5}) in p)
        self.assertTrue(("b", {"type": "random_bool"}) in p)

    def test_mutable_for_fixed(self):
        c = {"a": 5, "b": {"type": "random_bool"}}
        p = get_mutable_parameters(c)

        self.assertEqual(1, len(p))
        self.assertTrue(("b", {"type": "random_bool"}) in p)

    def test_parameter_to_value(self):
        for _ in range(100):
            rand_int = parameter_to_value(mutable_sample_config["a"])
            self.assertTrue(0 <= rand_int <= 5)

            rand_bool = parameter_to_value(mutable_sample_config["b"])
            self.assertTrue(isinstance(rand_bool, bool))

            fixed = parameter_to_value(mutable_sample_config["c"])
            self.assertEqual(5, fixed)

    def test_init_values(self):
        values = init_parameter_values(mutable_sample_config)
        self.assertEqual(3, len(values))
        self.assertTrue("a" in values)
        self.assertTrue(0 <= values["a"] <= 5)
        self.assertTrue("b" in values)
        self.assertTrue("c" in values)
        self.assertEqual(5, values["c"])
        # test for values should be sufficient through other tests

    def test_int_distribution(self):
        random.seed(1)

        values = [
            parameter_to_value(mutable_sample_config["a"]) for _ in range(600)
        ]

        self.assertTrue(80 < values.count(0) < 120)
        self.assertTrue(80 < values.count(1) < 120)
        self.assertTrue(80 < values.count(2) < 120)
        self.assertTrue(80 < values.count(3) < 120)
        self.assertTrue(80 < values.count(4) < 120)

    def test_random_rates(self):
        random_rates = {"type": "random_rates", "rates": {"a": 1}}
        value = parameter_to_value(random_rates)
        self.assertEqual("a", value)

    def test_rates_distribution(self):
        random.seed(1)
        random_rates = {"type": "random_rates", "rates": {0: 0.25, 1: 0.75}}
        values = [parameter_to_value(random_rates) for _ in range(1000)]

        self.assertEqual(1000, values.count(0) + values.count(1))
        self.assertTrue(220 < values.count(0) < 270)
        self.assertTrue(700 < values.count(1) < 800)

    def test_bool_distribution(self):
        random.seed(1)

        values = [
            parameter_to_value(mutable_sample_config["b"]) for _ in range(1000)
        ]

        self.assertTrue(480 < values.count(True) < 520)
        self.assertTrue(480 < values.count(False) < 520)

    def test_random_choice(self):
        config = {"type": "random_choice", "values": [1, 2, 10]}

        random.seed(1)
        values = []
        for _ in range(1000):
            v = parameter_to_value(config)
            self.assertTrue(v in [1, 2, 10])
            values.append(v)

        # 1000 iterations should be enough, to have reached each value once
        self.assertTrue(280 < values.count(1) < 370)
        self.assertTrue(280 < values.count(2) < 370)
        self.assertTrue(280 < values.count(10) < 370)

    def test_valid_parameter_configuration(self):
        self.assertTrue(
            is_valid_parameter_configuration(mutable_sample_config["a"])
        )
        self.assertTrue(
            is_valid_parameter_configuration(mutable_sample_config["b"])
        )
        self.assertTrue(
            is_valid_parameter_configuration(mutable_sample_config["c"])
        )

    def test_valid_parameter_random_int(self):
        max_missing = {"type": "random_int", "min": 0}
        min_missing = {"type": "random_int", "max": 5}
        both_missing = {"type": "random_int"}
        min_bigger_max = {"type": "random_int", "min": 6, "max": 5}
        allows_negative = {"type": "random_int", "min": -5, "max": -4}

        with warnings.catch_warnings(record=True):
            self.assertFalse(is_valid_parameter_configuration(max_missing))
            self.assertFalse(is_valid_parameter_configuration(min_missing))
            self.assertFalse(is_valid_parameter_configuration(both_missing))
            self.assertFalse(is_valid_parameter_configuration(min_bigger_max))
            self.assertTrue(is_valid_parameter_configuration(allows_negative))

    def test_valid_parameter_choice(self):
        valid_choice = {"type": "random_choice", "values": [1, 2, 3]}
        no_values = {"type": "random_choice"}
        string_value = {"type": "random_choice", "values": "abc"}

        self.assertTrue(is_valid_parameter_configuration(valid_choice))
        with warnings.catch_warnings(record=True):
            self.assertFalse(is_valid_parameter_configuration(no_values))
            self.assertFalse(is_valid_parameter_configuration(string_value))

    def test_valid_parameter_rates(self):
        valid_choice = {"type": "random_rates", "rates": {1: 0.5, 2: 0.5}}
        no_values = {"type": "random_rates"}
        string_value = {"type": "random_rates", "rates": "abc"}

        self.assertTrue(is_valid_parameter_configuration(valid_choice))
        with warnings.catch_warnings(record=True):
            self.assertFalse(is_valid_parameter_configuration(no_values))
            self.assertFalse(is_valid_parameter_configuration(string_value))

    def test_shortcut_fixed(self):
        configuration = 7
        self.assertEqual(7, parameter_to_value(configuration))

    def test_valid_parameter_value(self):
        self.assertTrue(is_valid_parameter_value(is_positive)(7))
        self.assertFalse(is_valid_parameter_value(is_positive)(-5))
        self.assertFalse(
            is_valid_parameter_value(is_positive)(
                {"type": "random_choice", "values": [-1, 2, 3]}
            ),
            "check all values of choice",
        )
        self.assertTrue(
            is_valid_parameter_value(is_positive)(
                {"type": "random_choice", "values": [1, 2, 3]}
            ),
            "all values should match",
        )
        self.assertTrue(
            is_valid_parameter_value(is_positive)(
                {"type": "random_int", "min": 0, "max": 10}
            )
        )
        self.assertFalse(
            is_valid_parameter_value(is_positive)(
                {"type": "random_int", "min": -5, "max": 10}
            )
        )
        self.assertFalse(
            is_valid_parameter_value(is_positive)(
                {"type": "random_int", "min": -5, "max": -10}
            )
        )
        self.assertFalse(
            is_valid_parameter_value(is_int)({"type": "random_bool"})
        )
        self.assertTrue(
            is_valid_parameter_value(is_int)(
                {"type": "random_int", "min": -5, "max": 10}
            )
        )
        self.assertTrue(
            is_valid_parameter_value(is_int)({"type": "fixed", "value": 10})
        )
        self.assertFalse(
            is_valid_parameter_value(is_int)({"type": "fixed", "value": True})
        )

    def test_random_rates_valid_values(self):
        self.assertFalse(
            is_valid_parameter_value(is_positive)(
                {"type": "random_rates", "rates": {-1: 0.3, 2: 0.3, 3: 0.1}}
            ),
            "check all values of choice",
        )
        self.assertFalse(
            is_valid_parameter_value(is_positive)(
                {"type": "random_rates", "rates": {-1: "a", 2: 0.3, 3: 0.1}}
            ),
            "rates should be float",
        )
        self.assertTrue(
            is_valid_parameter_value(is_positive)(
                {"type": "random_rates", "rates": {1: 0.3, 2: 0.3, 3: 0.1}}
            ),
            "all values should match",
        )

    def test_specification(self):
        specification = {"a": is_int}

        self.assertFalse(
            check_parameter_values_on_specification(specification, {})
        )
        self.assertTrue(
            check_parameter_values_on_specification(specification, {"a": 1})
        )
        self.assertFalse(
            check_parameter_values_on_specification(specification, {"a": 5.5})
        )

    def test_optional_specification(self):
        specification = {"a": any_check(is_none, is_int)}

        self.assertTrue(
            check_parameter_values_on_specification(specification, {})
        )
        self.assertTrue(
            check_parameter_values_on_specification(specification, {"a": 1})
        )
        self.assertFalse(
            check_parameter_values_on_specification(specification, {"a": 5.5})
        )

    def test_specification_multiple(self):
        specification = {"a": any_check(is_none, is_int), "b": is_positive}

        self.assertFalse(
            check_parameter_values_on_specification(specification, {})
        )
        self.assertTrue(
            check_parameter_values_on_specification(specification, {"b": 4})
        )
        self.assertTrue(
            check_parameter_values_on_specification(
                specification, {"a": 1, "b": 4}
            )
        )
        self.assertFalse(
            check_parameter_values_on_specification(
                specification, {"a": 1, "b": -4}
            )
        )
        self.assertFalse(
            check_parameter_values_on_specification(
                specification, {"a": 5.5, "b": 4}
            )
        )

    def test_specification_too_many_values(self):
        with warnings.catch_warnings(record=True) as w:
            check_parameter_values_on_specification({"a": is_none}, {"b": 1})
            self.assertEqual(1, len(w))
