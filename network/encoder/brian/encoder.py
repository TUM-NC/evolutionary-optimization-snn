"""
Interface for brian encoders
"""
from brian2 import SpikeGeneratorGroup

from network.encoder.encoder import Encoder


class BrianEncoder(Encoder):
    """
    Abstract class for brian encoders
    """

    def get_spike_generator(self, spike_data) -> SpikeGeneratorGroup:
        """
        Should return a spike generator for the given data

        :param spike_data:
        :return:
        """
        raise NotImplementedError("Please Implement this method")

    def is_deterministic(self):
        """
        whether the spikes produced are deterministic, and can be reused
        :return:
        """
        return True
