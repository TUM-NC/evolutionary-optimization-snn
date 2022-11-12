"""
Interface for a brian decoder
"""
from abc import ABC
from typing import Optional

from brian2 import Quantity

from network.decoder.decoder import Decoder


class BrianDecoder(Decoder, ABC):
    """
    Abstract class for a brian decoder
    """

    spikes: Optional[Quantity] = None

    def set_spikes(self, spikes):
        """
        Set the spike pattern to the decoder,
        to retrieve a value later on, using the get value function
        :param spikes:
        :return:
        """
        self.spikes = spikes
