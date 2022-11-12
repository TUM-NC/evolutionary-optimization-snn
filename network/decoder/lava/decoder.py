"""
Interface for a brian decoder
"""
from abc import ABC

from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.process import AbstractProcess

from network.decoder.decoder import Decoder


class LavaDecoder(Decoder, ABC):
    """
    Abstract class for a lava decoder
    """

    def set_value(self, output_process: "OutputProcess"):
        """
        Get the value from the lava output process
        :param output_process:
        :return:
        """
        raise NotImplementedError("Please Implement this method")

    def get_output_process(self) -> "OutputProcess":
        raise NotImplementedError("Please Implement this method")


class OutputProcess(AbstractProcess):
    spikes_in: InPort
