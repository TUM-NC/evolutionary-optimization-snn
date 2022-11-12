import random
import unittest

from network.evolution.generator import Generator
from network.evolution.reproduction.crossover import (
    crossover,
    distribute_synapses,
    split_neurons,
)
from network.network import Network
from network.neuron import Neuron
from network.synapse import Synapse


class TestCrossover(unittest.TestCase):
    def test_split_neurons_equally(self):
        n1 = [Neuron(0), Neuron(1), Neuron(2)]
        n2 = [Neuron(0), Neuron(1), Neuron(2)]

        out1, out2 = split_neurons(n1, n2)
        self.assertEqual(3, len(out1))
        self.assertEqual(3, len(out2))

    def test_distribute_synapses(self):
        synapses = [
            Synapse(0, 1),
            Synapse(0, 1),
            Synapse(0, 2),
            Synapse(0, 2),
        ]
        net1 = Network([Neuron(0)], [Neuron(1), Neuron(2, threshold=1)])
        net2 = Network([Neuron(0)], [Neuron(1), Neuron(2, threshold=2)])

        distribute_synapses(synapses, net1, net2)

        self.assertEqual(2, len(net1.synapses))
        self.assertEqual(2, len(net2.synapses))

    def test_crossover(self):
        net1 = Network([Neuron(0)], [Neuron(1)])
        net1.add_neuron(Neuron(5))
        net1.add_synapse(Synapse(5, 1))
        net1.add_synapse(Synapse(0, 5))
        net2 = net1.clone()

        out1, out2 = crossover(net1, net2)

        self.assertEqual(0, out1.distance(out2))
        self.assertEqual(2, len(out1.synapses))
        self.assertEqual(1, len(out1.hidden_neurons))

    def test_crossover_specific_seed(self):
        net1 = Network([Neuron(0, a=1)], [Neuron(1, a=1)])
        net1.add_neuron(Neuron(5, a=1))
        net1.add_synapse(Synapse(5, 1))
        net1.add_synapse(Synapse(0, 5))
        net1.add_synapse(Synapse(1, 0))

        net2 = Network([Neuron(0, a=2)], [Neuron(1, a=2)])
        net2.add_synapse(Synapse(0, 1))

        random.seed(0)
        out1, out2 = crossover(net1, net2)

        # assert exact network
        # out2 has neurons from both input networks
        self.assertEqual(
            (1, 2), (out2.input_neurons[0].a, out2.output_neurons[0].a)
        )
        self.assertEqual(
            (2, 1), (out1.input_neurons[0].a, out1.output_neurons[0].a)
        )
        self.assertEqual(1, len(out2.hidden_neurons))

    def test_crossover_no_unconnected_neurons(self):
        net1 = Network([Neuron(0, a=1)], [Neuron(1, a=1)])
        net1.add_neuron(Neuron(5, a=1))
        net1.add_synapse(Synapse(5, 1))
        net1.add_synapse(Synapse(0, 5))
        net1.add_synapse(Synapse(1, 0))

        net2 = Network([Neuron(0, a=2)], [Neuron(1, a=2)])
        net2.add_synapse(Synapse(0, 1))

    def test_crossover_reproducible(self):
        net1 = Network([Neuron(0)], [Neuron(1)])
        net1.add_neuron(Neuron(5))
        net1.add_synapse(Synapse(5, 1))
        net1.add_synapse(Synapse(0, 5))
        net2 = Network([Neuron(0)], [Neuron(1)])
        net2.add_neuron(Neuron(5))
        net2.add_synapse(Synapse(5, 1))

        random.seed(12)
        out1, out2 = crossover(net1, net2)
        random.seed(12)
        out3, out4 = crossover(net1, net2)
        random.seed(12)
        out5, out6 = crossover(net1.clone(), net2.clone())

        self.assertEqual(0, out1.distance(out3))
        self.assertEqual(0, out2.distance(out4))
        self.assertEqual(0, out1.distance(out5))
        self.assertEqual(0, out2.distance(out6))

    def test_crossover_never_stripable(self):
        g = Generator(2, 2)

        for _ in range(50):
            n1, n2 = g.generate_networks(2)
            out1, out2 = crossover(n1, n2)
            s1, s2 = out1.clone().strip(), out2.clone().strip()

            self.assertEqual(0, s1.distance(out1))
            self.assertEqual(0, s2.distance(out2))
