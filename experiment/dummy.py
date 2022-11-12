"""
Dummy experiment, to test functionality without using a simulator
"""
from typing import Optional

from experiment.experiment import Experiment
from network.decoder.binary import BinaryDecoder
from network.encoder.binary import BinaryEncoder
from network.network import Network


class Dummy(Experiment):
    """
    Dummy experiment, which does no simulation but just network operations
    """

    def __init__(self):
        self.encoder = BinaryEncoder(number_of_neurons=2)
        self.decoder = BinaryDecoder()

    def single_fitness(self, network: Network):
        value = len(network.hidden_neurons) + len(network.synapses)
        return value

    def set_seed(self, seed: Optional[int] = None):
        pass
