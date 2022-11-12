"""
Implementation of a binary decoder in brian
"""
from typing import Tuple

from network.decoder.binary import BinaryDecoder
from network.decoder.brian.decoder import BrianDecoder


class BinaryBrianDecoder(BinaryDecoder, BrianDecoder):
    """
    Actual implementation of a binary decoder in brian
    """

    boundary: float
    ideal_distance: int
    simulated_seconds: float

    def __init__(self, boundary=None, simulated_seconds=1, ideal_distance=25):
        super().__init__()
        # idea: lower frequency than boundary is false, higher is true
        if boundary is None:
            boundary = 75
        self.boundary = boundary
        self.ideal_distance = ideal_distance
        self.simulated_seconds = (
            simulated_seconds  # simulated time, to get rate from raw count
        )

    def get_value(self) -> Tuple[bool, float]:
        """
        :return: classification output as first entry in tuple,
        distance from boundary as second parameter
        """
        if self.spikes is None:
            raise RuntimeError("Spikes have to be set, before getting a value")

        # only one output neuron here -> first output neuron
        spikes = self.spikes[0]

        count = spikes.size
        rate = count / self.simulated_seconds

        self.spikes = None  # reset after value read

        return (rate > self.boundary), rate

    def get_absolute_distance(self, rate: float, expected: bool):
        """
        Return the absolute distance, to the expected value

        :param rate:
        :param expected:
        :return:
        """
        if not expected:
            target = self.boundary - self.ideal_distance
        else:
            target = self.boundary + self.ideal_distance

        distance = target - rate
        return abs(distance)
