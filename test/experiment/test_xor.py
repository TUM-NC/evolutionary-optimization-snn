import os
import unittest

from experiment.brian.xor import XOR
from network.network import Network


class TestXor(unittest.TestCase):
    def test_simulation_of_sample_network(self):
        filename = os.path.join(
            os.path.dirname(__file__), "data/almost-ideal-xor.json"
        )
        net = Network.from_file(filename)

        experiment = XOR(decoder_type="binary")
        experiment.fitness([net])
        self.assertEqual(
            4, experiment.performance(net), "networks classifies all correct"
        )
