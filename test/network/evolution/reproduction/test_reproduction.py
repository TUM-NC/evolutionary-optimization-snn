import random
import unittest

from network.evolution.generator import Generator
from network.evolution.reproduction.reproduction import Reproduction
from utility.configuration import Configuration


class TestReproduction(unittest.TestCase):
    def test_selection_reproducible(self):
        population = [0, 1, 2, 3, 4]
        fitness = [3, 3, 3, 3, 3]

        random.seed(1)
        r1 = Reproduction()
        s1 = r1.selection(population, fitness)

        random.seed(1)
        r2 = Reproduction()
        s2 = r2.selection(population, fitness)

        random.seed(2)
        r3 = Reproduction()
        s3 = r3.selection(population, fitness)

        self.assertEqual(s1, s2)
        self.assertNotEqual(s2, s3)

    def test_different_selection_reproducible(self):
        population = [0, 1, 2, 3, 4]
        fitness = [3, 3, 3, 3, 3]

        random.seed(1)
        r1 = Reproduction()
        s1 = r1.get_different_selection(population, fitness, 1)

        random.seed(1)
        r2 = Reproduction()
        s2 = r2.get_different_selection(population, fitness, 1)

        random.seed(2)
        r3 = Reproduction()
        s3 = r3.get_different_selection(population, fitness, 1)

        random.seed(1)
        r4 = Reproduction()
        s4 = r4.get_different_selection(population, fitness, 2)

        random.seed(1)
        s5 = r1.get_different_selection(population, fitness, 1)

        self.assertEqual(s1, s2)
        self.assertNotEqual(s2, s3)
        self.assertNotEqual(s4, s2)
        self.assertEqual(s1, s5)

    def test_create_networks_reproducible(self):
        generator = Generator(2, 2)

        random.seed(1)
        p1 = generator.generate_networks(10)
        random.seed(1)
        p2 = generator.generate_networks(10)

        fitness = [1 for _ in range(2)]

        c = Configuration({})

        random.seed(6)
        r1 = Reproduction(c)
        n1, o1 = r1.create_networks(p1, fitness, 10)

        random.seed(6)
        r2 = Reproduction(c)
        n2, o2 = r2.create_networks(p2, fitness, 10)

        self.assertEqual(o1, o2)

        for i, n in enumerate(n1):
            self.assertEqual(0, n.distance(n2[i]))
