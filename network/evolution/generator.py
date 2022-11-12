"""
Provide function for generating a population of networks
"""
from typing import Optional

from experiment.experiment import Experiment
from network.decoder.decoder import Decoder
from network.encoder.encoder import Encoder
from network.evolution.reproduction.mutator import Mutator
from network.network import Network
from utility.configurable import Configurable
from utility.configuration import Configuration
from utility.parameter_configuration import (
    is_valid_parameter_value,
    parameter_to_value,
)
from utility.validation import is_int, is_positive


class Generator(Configurable):
    """
    Provides methods to generate a population of networks
    based on the given configuration
    """

    number_inputs: int
    number_outputs: int

    generate_hidden_neurons = {"type": "random_int", "min": 0, "max": 2}
    generate_synapses = {"type": "random_int", "min": 0, "max": 2}

    mutator: Mutator

    def __init__(
        self,
        number_inputs: int,
        number_outputs: int,
        configuration: Optional[Configuration] = None,
    ):
        super().__init__(configuration=configuration)

        self.number_inputs = number_inputs
        self.number_outputs = number_outputs

        self.mutator = Mutator(configuration)

    @classmethod
    def create_from_experiment(
        cls,
        experiment: Experiment,
        configuration: Optional[Configuration] = None,
    ):
        """
        Create a generator based on an experiment
        :param experiment:
        :param configuration:
        :return:
        """
        return cls.create_from_encoder_decoder(
            encoder=experiment.encoder,
            decoder=experiment.decoder,
            configuration=configuration,
        )

    @classmethod
    def create_from_encoder_decoder(
        cls,
        encoder: Encoder,
        decoder: Decoder,
        configuration: Optional[Configuration] = None,
    ):
        """
        Create a generator from an encoder and a decoder
        :param encoder:
        :param decoder:
        :param configuration:
        :return:
        """
        number_inputs = encoder.number_of_neurons
        number_outputs = decoder.number_of_neurons

        return cls(
            number_inputs=number_inputs,
            number_outputs=number_outputs,
            configuration=configuration,
        )

    def set_configurable(self):
        self.add_configurable_attribute(
            "generate_hidden_neurons",
            validate=[
                is_valid_parameter_value(is_int),
                is_valid_parameter_value(is_positive),
            ],
        )
        self.add_configurable_attribute(
            "generate_synapses",
            validate=[
                is_valid_parameter_value(is_int),
                is_valid_parameter_value(is_positive),
            ],
        )

    def generate_network(self) -> Network:
        """
        Generate a single network based on the given parameters

        :return:
        """
        network = self.create_empty_network()

        # initial hidden neurons
        hidden_neurons_count = parameter_to_value(self.generate_hidden_neurons)
        for _ in range(hidden_neurons_count):
            self.mutator.add_hidden_neuron(network)

        # synapses
        synapses_count = parameter_to_value(self.generate_synapses)
        for _ in range(synapses_count):
            self.mutator.add_random_synapse(network)

        return network

    def generate_networks(self, amount: int):
        """
        Generate a list of networks based on the given parameters

        :param amount: amount of networks to generate
        :return: list of generated networks
        """
        return [self.generate_network() for _ in range(amount)]

    def create_empty_network(self):
        """
        Create a network with the amount of specified input and output neurons
        :return:
        """
        input_list = []
        for i in range(self.number_inputs):
            uid = i  # first uids are reserved for input neurons
            neuron = self.mutator.create_random_neuron(uid=uid)
            input_list.append(neuron)

        output_list = []
        for i in range(self.number_outputs):
            uid = (
                i + self.number_inputs
            )  # output neurons follow after input neurons
            neuron = self.mutator.create_random_neuron(uid=uid)
            output_list.append(neuron)

        return Network(input_neurons=input_list, output_neurons=output_list)
