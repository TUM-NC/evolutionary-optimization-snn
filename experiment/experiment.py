"""
Provides an abstract experiment class, for other experiments to inherit from
"""

from typing import List, Optional

from network.decoder.decoder import Decoder
from network.encoder.encoder import Encoder
from network.network import Network
from simulator.simulator import Simulator


class Experiment:
    """
    Class, to create experiments from
    """

    encoder: Encoder
    decoder: Decoder

    def fitness(self, networks: List[Network]) -> List[float]:
        """
        Calculate the fitness scores for a list of networks
        By default, just calls the single_fitness score for each network

        :param networks:
        :return:
        """
        return [self.single_fitness(n) for n in networks]

    def single_fitness(self, network: Network) -> float:
        """
        Calculate the fitness for a single network

        :param network:
        :return:
        """
        raise NotImplementedError("Please Implement this method")

    def set_seed(self, seed: Optional[int] = None):
        """
        Set a seed to make simulator behave reproducible

        :param seed:
        :return:
        """

        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def get_simulator_class():
        """
        get the class for the simulator

        :return:
        """
        return Simulator
