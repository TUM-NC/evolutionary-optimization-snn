import unittest

from simulator.grid import GridSimulator
from utility.grid_configuration import GridConfiguration


class TestGrid(unittest.TestCase):
    def test_table_creation(self):
        gc = GridConfiguration(
            grid_config=[[{"a": 1}, {"a": 2}], [{"b": 2}, {"b": 3}]]
        )
        computations = [
            {"index": 0, "x": 1},
            {"index": 1, "x": 0},
            {"index": 2, "x": 4},
            {"index": 3, "x": 3},
        ]

        sim = GridSimulator(grid_configuration=gc)
        sim._computations = computations

        headers, values = sim.get_table_values()

        self.assertEqual(["a", "b", "x"], headers)
        self.assertEqual([[1, 2, 1], [1, 3, 0], [2, 2, 4], [2, 3, 3]], values)
