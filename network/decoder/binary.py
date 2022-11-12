"""
Abstract class for a binary decoder
"""
from typing import Tuple

from network.decoder.decoder import Decoder


class BinaryDecoder(Decoder):
    """
    Interface for a binary decoder
    """

    def __init__(self):
        super().__init__(1)  # for now only one fixed output neuron

    def get_value(self) -> Tuple[bool, ...]:
        """

        :return: classification output as first value in tuple,
        additional information in following entries
        """
        raise NotImplementedError("Please Implement this method")
