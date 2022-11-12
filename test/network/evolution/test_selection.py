import random
import unittest
from unittest.mock import Mock

from network.evolution.selection import best, tournament_selection


class TestSelection(unittest.TestCase):
    def test_tournament_selection_total(self):
        population = [1, 2, 3, 4]
        fitness = [4, 3, 2, 1]
        selection = tournament_selection(population, fitness, k=4, p=1)
        self.assertEqual(1, selection)

    def test_tournament_selection_too_large_k(self):
        """
        should use k=length
        :return:
        """
        population = [1, 2, 3, 4]
        fitness = [4, 3, 2, 1]
        selection = tournament_selection(population, fitness, k=10, p=1)
        self.assertEqual(1, selection)

    def test_tournament_selection_reproducible(self):
        population = [1, 2, 3, 4]
        fitness = [4, 4, 2, 1]
        random.seed(1)
        selection1 = tournament_selection(population, fitness, k=2, p=0.5)
        random.seed(1)
        selection2 = tournament_selection(population, fitness, k=2, p=0.5)
        random.seed(2)
        selection3 = tournament_selection(population, fitness, k=2, p=0.5)

        self.assertEqual(selection1, selection2)
        self.assertNotEqual(selection1, selection3)

    def test_tournament_reproducible_objects(self):
        count = 10
        p1 = [Mock(id=i) for i in range(count)]
        p2 = [Mock(id=i) for i in range(count)]
        fitness = [1 for _ in range(count)]

        random.seed(2)
        selection1 = tournament_selection(p1, fitness, k=2, p=0.5)
        selection1_2 = tournament_selection(p1, fitness, k=2, p=0.5)
        random.seed(2)
        selection2 = tournament_selection(p2, fitness, k=2, p=0.5)
        selection2_2 = tournament_selection(p2, fitness, k=2, p=0.5)
        random.seed(3)
        selection3 = tournament_selection(p1, fitness, k=2, p=0.5)

        self.assertEqual(selection1.id, selection2.id)
        self.assertNotEqual(selection1, selection2)
        self.assertEqual(selection1_2.id, selection2_2.id)
        self.assertNotEqual(selection1_2, selection2_2)
        self.assertNotEqual(selection1.id, selection3.id)

    def test_best_pre_sorted(self):
        population = [1, 2, 3, 4]
        fitness = [4, 3, 2, 1]
        selection = best(population, fitness, n=4)
        self.assertEqual(population, selection)

    def test_best_single(self):
        population = [1, 2, 3, 4]
        fitness = [4, 3, 2, 10]
        selection = best(population, fitness, n=1)
        self.assertEqual([4], selection)

    def test_best_reverse(self):
        population = [1, 2, 3, 4]
        fitness = [1, 2, 3, 4]
        selection = best(population, fitness, n=4)
        self.assertEqual([4, 3, 2, 1], selection)
