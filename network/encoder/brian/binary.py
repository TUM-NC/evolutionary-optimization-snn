"""
Implementation of a binary encoder using brian
"""
from typing import List, Tuple

from brian2 import SpikeGeneratorGroup, array, ms, np, second

from network.encoder.binary import BinaryEncoder
from network.encoder.brian.encoder import BrianEncoder


class BinaryBrianEncoder(BinaryEncoder, BrianEncoder):
    """
    Implementation of a binary encoder using brian
    """

    def __init__(
        self,
        number_of_neurons: int,
        true_rate=100,
        false_rate=50,
        simulation_time=1000 * ms,
    ):
        super().__init__(number_of_neurons=number_of_neurons)
        self.true_rate = true_rate
        self.false_rate = false_rate
        self.simulation_time = (
            simulation_time  # TODO: check if can use repeat instead
        )

    def get_spike_generator(self, spike_data: List[Tuple[bool]]):
        """
        Generate a spike generator group, for all of the given input data
        :param spike_data:
        :return:
        """
        input_patterns = len(spike_data)
        spike_indices = []
        times = []

        for i, data in enumerate(spike_data):
            sd_spike_indices, sd_times = self._get_spikes(spike_data=data)

            # add offset, because of multiple inputs
            sd_spike_indices += i * self.number_of_neurons

            spike_indices.extend(sd_spike_indices)
            times.extend(sd_times)

        amount_spike_generators = self.number_of_neurons * input_patterns
        times = times * second
        return SpikeGeneratorGroup(
            N=amount_spike_generators, indices=spike_indices, times=times
        )

    def _get_spikes(self, spike_data: Tuple[bool]):
        """
        Get spike times and indices for the single given value
        :param spike_data:
        :return:
        """
        spike_indices = []
        times = []
        for index, neuron_data in enumerate(spike_data):
            neuron_spike_times = self._value_to_spikes(neuron_data)
            neuron_indices = np.full(len(neuron_spike_times), index)

            spike_indices.extend(neuron_indices)
            times.extend(neuron_spike_times)

        return array(spike_indices), array(times)

    def _value_to_spikes(self, value: bool):
        """
        Convert a single value to a spike (time) pattern
        :param value:
        :return:
        """
        target_rate = self.true_rate if value else self.false_rate
        time_until_spike = (1 * second) / target_rate
        number_of_spikes = int(
            self.simulation_time / time_until_spike
        )  # floor to next lower int
        spike_times = np.array(
            [time_until_spike * i for i in range(number_of_spikes)]
        )
        return spike_times
