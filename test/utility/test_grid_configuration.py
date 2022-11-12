import unittest

from utility.grid_configuration import GridConfiguration


class TestGridConfiguration(unittest.TestCase):
    def test_reformat_grid_config(self):
        grid_config = [[{"a": 1}, {"a": 2}], [{"b": 3}]]
        grid_config_dict = {"options": [[{"a": 1}, {"a": 2}], [{"b": 3}]]}
        grid_full = {
            "options": [
                {"alternatives": [{"a": 1}, {"a": 2}]},
                {"alternatives": [{"b": 3}]},
            ]
        }
        grid_shortcut_alternative = [[{"a": 1}, {"a": 2}], {"b": 3}]

        a = GridConfiguration._reformat_grid_config(grid_config)
        b = GridConfiguration._reformat_grid_config(grid_config_dict)
        c = GridConfiguration._reformat_grid_config(grid_full)
        d = GridConfiguration._reformat_grid_config(grid_shortcut_alternative)

        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a, d)

    def test_empty_grid(self):
        self.assertRaises(RuntimeError, GridConfiguration, None, [])

    def test_amount_alternatives_simple(self):
        grid_config = [[{"a": 1}, {"a": 2}], [{"b": 3}]]
        gc = GridConfiguration(grid_config=grid_config)
        self.assertEqual(2, gc.get_amount_alternatives())

    def test_amount_alternatives(self):
        grid_config = [
            [{"a": 1}, {"a": 2}],
            [{"b": 3}, {"b": 4}, {"b": 5}],
            [{"c": 1}, {"c": 2}],
        ]
        gc = GridConfiguration(grid_config=grid_config)
        self.assertEqual(2 * 3 * 2, gc.get_amount_alternatives())

    def test_amount_alternatives_single(self):
        gc = GridConfiguration(grid_config=[{"a": 1}])
        self.assertEqual(1, gc.get_amount_alternatives())

    def test_get_option_complex(self):
        grid_config = [
            [{"a": 1}, {"a": 2}],
            [{"b": 3}, {"b": 4}, {"b": 5}],
            [{"c": 1}, {"c": 2}],
        ]
        option = 0 * 2 + 2 * 2 + 0 * 5  # right to left format
        expected = {"a": 1, "b": 5, "c": 1}

        gc = GridConfiguration(grid_config=grid_config)
        configuration = gc.get_option(option)
        config = configuration._configuration

        self.assertEqual(expected, config)

    def test_get_option_overwrite(self):
        base_config = {"a": 1, "b": 2}
        grid = [{"a": 2}]
        expected = {"a": 2, "b": 2}

        gc = GridConfiguration(base_config=base_config, grid_config=grid)
        configuration = gc.get_option(0)
        config = configuration._configuration

        self.assertEqual(config, expected)

    def test_get_option_different(self):
        base_config = {"a": 1, "b": 1}
        grid = [[{"a": 2}, {"b": 2}]]

        gc = GridConfiguration(base_config=base_config, grid_config=grid)

        config_0 = gc.get_option(0)._configuration
        config_1 = gc.get_option(1)._configuration

        self.assertEqual({"a": 2, "b": 1}, config_0)
        self.assertEqual({"a": 1, "b": 2}, config_1)

    def test_poolsize(self):
        gc = GridConfiguration(
            grid_config={"pool_size": 2, "options": [{"a": 1}]}
        )
        self.assertEqual(2, gc.pool_size)

    def test_overwrite_only_first_dimension(self):
        base = {"a": {"b": 1, "c": 2}, "b": 5}
        overwrite = {"a": {"d": 3}}

        out = GridConfiguration.overwrite_dict_first_dimension(base, overwrite)

        self.assertEqual({"a": {"d": 3}, "b": 5}, out)

    def test_option_headers(self):
        base = {"x": 1}
        grid = [
            [{"b": 1, "c": 1}, {"b": 2}, {"d": {"f": 5}}],
            [{"a": 1}],
            [{"d": {"e": 5}}],
        ]

        gc = GridConfiguration(base_config=base, grid_config=grid)

        headers = gc.get_option_headers()
        self.assertEqual(["a", "b", "c", "d.e", "d.f"], headers)
