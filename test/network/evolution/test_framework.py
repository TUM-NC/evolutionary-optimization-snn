import re
import unittest
from io import StringIO
from unittest.mock import patch

from experiment.dummy import Dummy
from network.evolution.framework import Framework
from network.evolution.stats import Stats
from utility.configuration import Configuration


def get_dummy_framework(parameters: dict = {}):
    """
    Return the framework class with dummy experiment
    :param parameters:
    :return:
    """
    e = Dummy()
    c = Configuration(config=parameters)
    return Framework(experiment=e, configuration=c)


def dummy_experiment_epoch(parameters: dict):
    framework = get_dummy_framework(parameters)

    population = framework.generator.generate_networks(
        framework.population_size
    )
    fitness = [0 for i in range(framework.population_size)]

    networks, operations = framework.do_epoch(population, fitness)
    return networks


class MockException(RuntimeError):
    pass


class TestFramework(unittest.TestCase):
    def test_size_population_after_epoch(self):
        parameters = {
            "population_size": 50,
            "num_best": 5,
            "random_factor": 0,
            "reproduction_rates": {
                "mutation": 1,
            },
        }
        new_population = dummy_experiment_epoch(parameters)
        self.assertEqual(50, len(new_population))

    def test_size_population_after_epoch_only_random(self):
        parameters = {
            "population_size": 50,
            "num_best": 0,
            "random_factor": 1,
            "reproduction_rates": {
                "mutation": 1,
            },
        }
        new_population = dummy_experiment_epoch(parameters)
        self.assertEqual(50, len(new_population))

    def test_size_population_after_epoch_odd_numbers(self):
        parameters = {
            "population_size": 474,
            "num_best": 3,
            "random_factor": 0.8,
            "reproduction_rates": {
                "mutation": 1,
            },
        }
        new_population = dummy_experiment_epoch(parameters)
        self.assertEqual(474, len(new_population))

    def test_size_population_crossover(self):
        parameters = {
            "population_size": 473,
            "num_best": 3,
            "random_factor": 0.8,
            "reproduction_rates": {"mutation": 0.5, "crossover": 0.5},
        }
        new_population = dummy_experiment_epoch(parameters)
        self.assertEqual(473, len(new_population))

    def test_simple_evolution(self):
        # test the correct functioning of the evolution
        parameters = {
            "population_size": 20,
            "num_generations": 10,
            "print_status": False,
        }

        f = get_dummy_framework(parameters)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            stats = f.evolution()

            # should not have any print, when print_status is false
            self.assertEqual("", fake_out.getvalue())

        # index of epochs starts at 0
        self.assertEqual(stats.get_latest_epoch(), 9)

    def test_evolution_finish_on_target(self):
        parameters = {
            "population_size": 20,
            "num_generations": 10,
            "print_status": False,
            "fitness_target": 25,
        }

        f = get_dummy_framework(parameters)
        stats = f.evolution()

        _, fitness = stats.get_best_network_alltime()

        self.assertGreaterEqual(fitness, 25)
        # for dummy should never need 10 epochs
        self.assertLess(stats.get_latest_epoch(), 10)

    def test_evolution_print(self):
        parameters = {
            "population_size": 10,
            "num_generations": 1,
            "print_status": True,
        }

        f = get_dummy_framework(parameters)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            f.evolution()
            regex = (
                r"^Epoch 1\/1 - "
                r"Best fitness: [\d.]* - "
                r"Average fitness: [\d.]* - "
                r"Took [\d.]*s$"
            )
            match = re.match(
                regex,
                fake_out.getvalue(),
            )
            self.assertIsNotNone(match)

    def test_evolution_continue(self):
        parameters = {
            "population_size": 20,
            "num_generations": 5,
            "print_status": False,
        }

        f = get_dummy_framework(parameters)

        stats = f.evolution()
        self.assertEqual(4, stats.get_latest_epoch())
        _, best_fitness_so_far = stats.get_best_network_alltime()

        f.num_generations = 6
        f.evolution(stats)

        self.assertEqual(5, stats.get_latest_epoch())
        _, next_best_fitness = stats.get_best_network_alltime()
        self.assertGreaterEqual(next_best_fitness, best_fitness_so_far)

    def test_save_stat_regularly(self):
        parameters = {
            "population_size": 20,
            "num_generations": 5,
            "print_status": False,
            "save_stat_regularly": True,
        }

        f = get_dummy_framework(parameters)
        temporary_stats_file = f.get_temporary_file()

        original_stats = f.evolution()
        saved_stats = Stats.from_file(temporary_stats_file)

        self.assertEqual(True, saved_stats.compare(original_stats))

    def test_save_stat_in_case_of_error(self):
        executions_left = 3
        parameters = {
            "population_size": 20,
            "num_generations": 5,
            "print_status": False,
            "save_stat_regularly": True,
        }

        f = get_dummy_framework(parameters)
        do_epoch = f.do_epoch  # do not mock this function

        def epoch(population, fitness_score):
            nonlocal executions_left, do_epoch
            executions_left -= 1
            if executions_left == 0:
                raise MockException("Abort after two epochs")
            # call original function, when not exceeded the limit
            return do_epoch(population, fitness_score)

        temporary_stats = f.get_temporary_file()

        with patch.object(Framework, "do_epoch", side_effect=epoch):
            try:
                f.evolution()
            except MockException:
                # expected to happen, since defined above after two epochs
                pass

        saved_stats = Stats.from_file(temporary_stats)
        self.assertEqual(2, saved_stats.get_latest_epoch())

    def test_cache_order_all_uncached(self):
        parameters1 = {
            "population_size": 5,
            "num_generations": 1,
            "print_status": False,
            "cache_evolution": False,
        }
        parameters2 = {
            "population_size": 5,
            "num_generations": 1,
            "print_status": False,
            "cache_evolution": True,
        }

        f1 = get_dummy_framework(parameters1)
        f2 = get_dummy_framework(parameters2)

        population = f1.generator.generate_networks(f1.population_size)

        fitness1 = f1.evaluate(population)
        fitness2 = f2.evaluate(population)

        self.assertEqual(fitness1, fitness2)

    @patch.object(Dummy, "fitness", return_value=[1, 3, 4])
    def test_cache_order(self, mock):
        parameters = {
            "population_size": 5,
            "num_generations": 1,
            "print_status": False,
            "cache_evolution": True,
        }

        f = get_dummy_framework(parameters)
        population = f.generator.generate_networks(f.population_size)

        f.fitness_cache[population[0]] = 1000
        f.fitness_cache[population[2]] = 2000

        fitness = f.evaluate(population)

        uncached = [population[1], population[3], population[4]]
        mock.assert_called_once_with(uncached)
        self.assertEqual([1000, 1, 2000, 3, 4], fitness)

    def test_seed_should_reproduce(self):
        parameters = {
            "seed": 123,
        }
        f1 = get_dummy_framework(parameters)
        population1 = f1.generator.generate_networks(10)

        f2 = get_dummy_framework(parameters)
        population2 = f2.generator.generate_networks(10)

        for i in range(10):
            n1 = population1[i]
            n2 = population2[i]

            self.assertEqual(0, n1.distance(n2))

    def test_different_no_seed_should_be_different(self):
        f1 = get_dummy_framework()
        population1 = f1.generator.generate_networks(2)

        f2 = get_dummy_framework()
        population2 = f2.generator.generate_networks(2)

        self.assertNotEqual(0, population1[0].distance(population2[0]))

    def test_different_seed_different(self):
        p1 = {"seed": 1}
        f1 = get_dummy_framework(p1)
        population1 = f1.generator.generate_networks(1)

        p2 = {"seed": 2}
        f2 = get_dummy_framework(p2)
        population2 = f2.generator.generate_networks(1)

        self.assertNotEqual(0, population1[0].distance(population2[0]))

    def test_seed_epoch(self):
        p = {
            "seed": 7,
            "print_status": False,
            "population_size": 100,
            "reproduction_rates": {"mutation": 1},
        }
        f1 = get_dummy_framework(p)

        fitness = [1 for _ in range(10)]

        population1 = f1.generator.generate_networks(20)
        n1, o1 = f1.do_epoch(population1, fitness)

        f2 = get_dummy_framework(p)
        population2 = f2.generator.generate_networks(20)
        n2, o2 = f2.do_epoch(population2, fitness)

        self.assertEqual(o1, o2)

        for a1, a2 in zip(n1, n2):
            self.assertEqual(0, a1.distance(a2))

    def test_seed_evolution(self):
        p = {
            "seed": 1,
            "print_status": False,
            "population_size": 50,
            "num_generations": 5,
        }
        f1 = get_dummy_framework(p)
        s1 = f1.evolution()

        f2 = get_dummy_framework(p)
        s2 = f2.evolution()

        self.assertTrue(s1.is_same_populations(s2))
