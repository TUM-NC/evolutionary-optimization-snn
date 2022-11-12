import unittest

from network.encoder.brian.float import FloatBrianEncoder


class TestFloatBrianEncoder(unittest.TestCase):
    def test_get_spikes_rates(self):
        encoder = FloatBrianEncoder(number_of_neurons=2)
        data = [[0.1, 0], [1, 0.5]]

        rates = encoder.get_spike_rates(data)

        self.assertEqual([20, 10, 110, 60], rates)

    def test_rate_to_brian_spike_times_exact(self):
        encoder = FloatBrianEncoder(number_of_neurons=2, poisson=False)
        indices, times = encoder._convert_rate_to_brian([1, 2, 3])
        self.assertEqual([0, 1, 1, 2, 2, 2], list(indices))
        self.assertEqual([0.5, 1 / 3, 2 / 3, 0.25, 0.5, 0.75], list(times))

    def test_rate_long_rough(self):
        encoder = FloatBrianEncoder(number_of_neurons=2, poisson=False)
        indices, times = encoder._convert_rate_to_brian([110])
        self.assertEqual(110, len(indices))
        self.assertEqual(110, list(indices).count(0))
        self.assertEqual(110, len(times))
