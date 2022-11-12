import tempfile
import unittest
from unittest import mock

from network.evolution.origin import Origin, ReproductionType
from network.evolution.stats import EpochStats, Stats
from network.network import Network
from network.neuron import Neuron


def get_network():
    """
    Get an arbitrary network
    """
    return Network([Neuron(0), Neuron(1)], [Neuron(2), Neuron(3)])


class TestStats(unittest.TestCase):
    def test_add_epoch(self):
        stats = Stats()
        stats.add_epoch([get_network()], [0])
        stats.add_epoch([get_network()], [2])
        stats.add_epoch([get_network()], [1])

        self.assertEqual(3, len(stats.data))

    def test_latest_epoch_empty(self):
        stats = Stats()

        latest = stats.get_latest_epoch()

        self.assertEqual(None, latest)

    def test_latest_epoch_non_complete(self):
        stats = Stats()
        stats.add_epoch([get_network()], [0])
        stats.add_epoch([get_network()], [2])

        latest = stats.get_latest_epoch()

        self.assertEqual(1, latest)

    def test_latest_epoch_complete(self):
        stats = Stats()
        stats.add_epoch([get_network()], [0])
        stats.add_epoch([get_network()], [2])
        stats.add_epoch([get_network()], [2])

        latest = stats.get_latest_epoch()

        self.assertEqual(2, latest)

    def test_latest_pf_empty(self):
        stats = Stats()

        pop, fit = stats.get_latest_population_fitness()

        self.assertEqual(None, pop)
        self.assertEqual(None, fit)

    def test_latest_pf_two_epochs(self):
        stats = Stats()
        stats.add_epoch([get_network()], [0])
        pop_expected = [get_network()]
        fit_expected = [1]
        stats.add_epoch(pop_expected, fit_expected)

        pop, fit = stats.get_latest_population_fitness()

        self.assertEqual(pop_expected, pop)
        self.assertEqual(fit_expected, fit)

    def test_best_in_middle(self):
        stats = Stats()
        stats.add_epoch([get_network()], [0])
        best_network = get_network()
        stats.add_epoch([best_network], [2])
        stats.add_epoch([get_network()], [1])

        supposed_best, _ = stats.get_best_network_alltime()

        self.assertEqual(best_network, supposed_best)

    def test_best_variations(self):
        stats = Stats()
        n1 = get_network()
        n2 = get_network()
        n3 = get_network()
        stats.add_epoch([n1], [0])
        best_network = get_network()
        stats.add_epoch([n2, best_network], [1, 2])
        stats.add_epoch([n3], [1])

        supposed_best, supposed_best_fitness = stats.get_best_network_alltime()

        self.assertEqual(2, supposed_best_fitness)
        self.assertEqual(best_network.hash(), supposed_best.hash())

    def test_import_export(self):
        stats = Stats()
        best_network = get_network()
        stats.add_epoch(
            [best_network],
            [2],
            operations=[Origin(ReproductionType.Random, [])],
        )
        stats.add_epoch([get_network()], [1])

        with tempfile.NamedTemporaryFile() as tmp_file:
            filename = tmp_file.name
            stats.to_file(filename)
            imported = Stats.from_file(filename)

        assumed_best, _ = imported.get_best_network_alltime()

        self.assertEqual(best_network.hash(), assumed_best.hash())
        epoch: EpochStats = imported.get_epoch(0)
        self.assertEqual(
            Origin(ReproductionType.Random, []), epoch["operations"][0]
        )

    def test_compare_stats_epochs_simple_cases(self):
        s1 = Stats()
        s2 = Stats()

        self.assertTrue(s1.compare(s2))

    # since time is compared, we mock it, to be the same for all epochs
    @mock.patch("time.time", mock.MagicMock(return_value=123))
    def test_compare_stats_same_fitness(self):
        s1 = Stats()
        s1.add_epoch([get_network(), get_network(), get_network()], [1, 2, 3])
        s2 = Stats()
        s2.add_epoch([get_network(), get_network(), get_network()], [1, 2, 3])

        self.assertTrue(s1.compare(s2))
        self.assertTrue(s2.compare(s1))

    def test_epoch_population_should_be_same_as_fitness(self):
        s = Stats()
        self.assertRaises(RuntimeError, s.add_epoch, [], [1])
        self.assertRaises(RuntimeError, s.add_epoch, [get_network()], [1, 2])
        self.assertRaises(
            RuntimeError, s.add_epoch, [get_network(), get_network()], [1]
        )

    @mock.patch("time.time")
    def test_time_took(self, time):
        s = Stats()
        time.return_value = 0
        s.start_epoch()
        time.return_value = 1
        s.add_epoch([get_network()], [0])
        time.return_value = 2
        s.start_epoch()
        time.return_value = 4
        s.add_epoch([get_network()], [0])

        self.assertEqual(3, s.get_total_time_took())

    def test_history_operations(self):
        dummy = [0, 0, 0]
        s = Stats()
        s.add_epoch(
            dummy,
            dummy,
            [Origin("random", []), Origin("random", []), Origin("random", [])],
        )
        s.add_epoch(
            dummy,
            dummy,
            [
                Origin("mutation", [1]),
                Origin("random", []),
                Origin("random", []),
            ],
        )
        s.add_epoch(
            dummy,
            dummy,
            [
                Origin("crossover", [1, 0]),
                Origin("random", []),
                Origin("random", []),
            ],
        )
        s.add_epoch(
            dummy,
            dummy,
            [Origin("same", [0]), Origin("random", []), Origin("random", [])],
        )

        history = s.get_origin(0, 3)
        expected_history = [
            Origin("same", [0]),
            [
                Origin("crossover", [1, 0]),
                [Origin("random", [])],
                [Origin("mutation", [1]), [Origin("random", [])]],
            ],
        ]
        self.assertEqual(expected_history, history)

    def test_get_origin_distribution(self):
        dummy = [0, 0]
        s = Stats()
        s.add_epoch(
            dummy,
            dummy,
            [Origin("random", []), Origin("random", [])],
        )
        s.add_epoch(
            dummy,
            dummy,
            [Origin("mutation", [0]), Origin("mutation", [1])],
        )
        s.add_epoch(
            dummy,
            dummy,
            [Origin("crossover", [0, 1]), Origin("random", [])],
        )
        self.assertEqual(
            {"crossover": 1, "mutation": 2, "random": 2},
            s.get_origin_distribution(0),
        )
        self.assertEqual({"random": 1}, s.get_origin_distribution(1))

        s.add_epoch(
            dummy,
            dummy,
            [Origin("crossover", [0, 1]), Origin("crossover", [0, 1])],
        )
        s.add_epoch(
            dummy,
            dummy,
            [Origin("crossover", [0, 1]), Origin("crossover", [0, 1])],
        )
        self.assertEqual(
            {"crossover": 5, "mutation": 4, "random": 6},
            s.get_origin_distribution(0),
        )
        self.assertEqual(
            {"crossover": 5, "mutation": 4, "random": 6},
            s.get_origin_distribution(1),
        )
