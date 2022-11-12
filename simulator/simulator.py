"""
Provide an abstract class for simulators
"""
from typing import Callable, Dict, List

from network.decoder.decoder import Decoder
from network.encoder.encoder import Encoder
from network.network import Network
from utility.parameter_configuration import (
    check_parameter_values_on_specification,
)


class Simulator:
    """
    Abstract class of a simulator
    """

    encoder: Encoder
    decoder: Decoder

    def __init__(
        self, networks: List[Network], encoder: Encoder, decoder: Decoder
    ):
        self.networks = networks
        self.encoder = encoder
        self.decoder = decoder

    def simulate(self):
        """
        Simulation of networks with and returns output from decoder
        :return: Output from decoder
        """
        raise NotImplementedError("Please Implement this method")

    @classmethod
    def validate_parameters(
        cls, neuron_parameters: dict, synapse_parameters: dict
    ):
        """
        Validate the given parameter values

        :param neuron_parameters:
        :param synapse_parameters:
        :return:
        """
        neurons_valid = check_parameter_values_on_specification(
            cls.get_neuron_parameters(), neuron_parameters
        )
        synapses_valid = check_parameter_values_on_specification(
            cls.get_synapse_parameters(), synapse_parameters
        )
        return neurons_valid and synapses_valid

    @staticmethod
    def get_neuron_parameters() -> Dict[str, Callable]:
        """
        Return a specification for supported neuron parameters
        Format should be: key -> callable (which checks for validity)
        :return:
        """
        return {}

    @staticmethod
    def get_synapse_parameters() -> Dict[str, Callable]:
        """
        Return a specification for supported synapse parameters
        Format should be: key -> callable (which checks for validity)
        :return:
        """
        return {}
