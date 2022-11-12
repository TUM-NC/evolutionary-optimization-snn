"""
Provides an XOR experiment using the brian simulator
"""
from typing import List, Literal, Union

from experiment.brian.brian_experiment import BrianExperiment
from network.decoder.brian.binary import BinaryBrianDecoder
from network.decoder.brian.classification import ClassificationBrianDecoder
from network.encoder.brian.binary import BinaryBrianEncoder
from network.encoder.brian.float import FloatBrianEncoder
from network.network import Network


class XOR(BrianExperiment):
    """
    Actual experiment implementation using brian for simulation
    """

    encoder: Union[BinaryBrianEncoder, FloatBrianEncoder]
    decoder: Union[BinaryBrianDecoder, ClassificationBrianDecoder]

    float_encoder: bool
    decoder_type: Literal["binary", "classification"]
    rounds: int

    X = [(True, True), (True, False), (False, True), (False, False)]
    X_float = [(1, 1), (1, 0), (0, 1), (0, 0)]
    Y = [False, True, True, False]

    def __init__(
        self,
        decoder_type: Literal["binary", "classification"] = "classification",
        poisson: bool = False,
        rounds: int = 1,
        binary_boundary=None,
    ):
        self.rounds = rounds

        if not poisson:
            self.encoder = BinaryBrianEncoder(number_of_neurons=2)
        else:
            self.encoder = FloatBrianEncoder(number_of_neurons=2)
        self.float_encoder = poisson

        # set the decoder depending on options
        self.decoder_type = decoder_type
        if decoder_type == "classification":
            self.decoder = ClassificationBrianDecoder(classes=2)
        elif decoder_type == "binary":
            self.decoder = BinaryBrianDecoder(boundary=binary_boundary)
        else:
            raise RuntimeError("Given type is not supported")

    def simulate(self, networks: List[Network]):
        """
        Start simulation of all networks on given patterns

        :param networks:
        :return:
        """
        values = self.get_data()

        values = values * self.rounds

        calculated = self._simulate_on_multiple_inputs(networks, values)
        for i, network in enumerate(networks):
            self.set_output_by_network(network, calculated[i])

    def get_data(self):
        """
        Get the data set depending on the encoder

        :return:
        """
        if self.float_encoder:
            return self.X_float
        else:
            return self.X

    def single_fitness(self, network: Network):
        """
        Calculate the fitness values for a single network

        :param network:
        :return:
        """
        return self.performance(network)

    def performance(self, network: Network, simulate=False) -> float:
        """
        Amount of correct classifications

        :param network:
        :param simulate: whether it should simulate the network again
        :return:
        """

        if simulate:
            self.simulate([network])

        calculated = self.get_output_by_network(network)

        correct_classifications = 0
        for index, expected in enumerate(self.Y * self.rounds):
            classification = calculated[index][0]

            # change calculation mode depending on decoder type
            if self.decoder_type == "binary":
                if classification == expected:
                    correct_classifications += 1
            elif self.decoder_type == "classification":
                if (classification == 0 and not expected) or (
                    classification == 1 and expected
                ):
                    correct_classifications += 1

        return correct_classifications / self.rounds
