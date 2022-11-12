from typing import List, Tuple

import numpy as np
from brian2 import Hz, PoissonGroup, SpikeGeneratorGroup, array, second

from network.encoder.brian.encoder import BrianEncoder
from network.encoder.float import FloatEncoder


class FloatBrianEncoder(FloatEncoder, BrianEncoder):
    """
    Encoder for floating point values in brian
    """

    def __init__(self, number_of_neurons: int, poisson=True):
        super().__init__(number_of_neurons=number_of_neurons)
        self.poisson = poisson

    def get_spike_generator(self, spike_data: List[Tuple[float]]):
        """
        Return a spike source, with the rates, calculated from data

        :param spike_data:
        :return:
        """
        if self.poisson:
            return self._get_poisson_spike_generator(spike_data)
        else:
            return self._get_exact_spike_generator(spike_data)

    def _get_exact_spike_generator(self, spike_data: List[Tuple[float]]):
        """
        Get a spike generator with the given rates with exact spike times

        :param spike_data:
        :return:
        """
        rates = self.get_spike_rates(spike_data)
        spike_indices, times = self._convert_rate_to_brian(rates)
        times = times * second

        return SpikeGeneratorGroup(
            N=len(rates), indices=spike_indices, times=times
        )

    def _convert_rate_to_brian(self, rates: List[int]):
        """
        Convert a rate to brian spike indices and spike times (without unit)

        :param rates:
        :return:
        """
        times = []
        spike_indices = []

        for index, rate in enumerate(rates):
            neuron_spike_times = self._value_to_spikes(rate)
            neuron_indices = np.full(len(neuron_spike_times), index)

            spike_indices.extend(neuron_indices)
            times.extend(neuron_spike_times)

        return array(spike_indices), array(times)

    @staticmethod
    def _value_to_spikes(target_rate):
        """
        Convert a single value to a spike (time) pattern
        :param value:
        :return:
        """
        time_until_spike = (1 * second) / (
            target_rate + 1
        )  # off by one, to center spikes
        number_of_spikes = int(
            (1 * second) / time_until_spike
        )  # floor to next lower int
        spike_times = np.array(
            [time_until_spike * i for i in range(1, number_of_spikes)]
        )
        return spike_times

    def _get_poisson_spike_generator(self, spike_data: List[Tuple[float]]):
        """
        Return a spike source, with the rates, calculated from data

        :param spike_data:
        :return:
        """
        rates = self.get_spike_rates(spike_data) * Hz
        return PoissonGroup(len(rates), rates)

    def get_spike_rates(self, spike_data: List[Tuple[float]]):
        """
        Return the rates for the given input data
        Returns a flat array, with concatenated values

        :param spike_data:
        :return:
        """
        rates = []
        for data_set in spike_data:
            for value in data_set:
                rates.append(self.get_rate(value))

        return rates

    @staticmethod
    def get_rate(value):
        """
        Convert a value (from 0 to 1) to a corresponding rate
        :param value:
        :return:
        """
        return int(value * 100) + 10

    def is_deterministic(self):
        """
        When spikes are poisson distributed, it is not deterministic
        :return:
        """
        return not self.poisson
