import unittest

from network.evolution.generator import Generator
from network.evolution.reproduction.merge import merge_two_networks
from network.network import Network
from network.neuron import Neuron
from network.synapse import Synapse


class TestMerge(unittest.TestCase):
    def test_nodes_from_both(self):
        net1 = Network([Neuron(0)], [Neuron(1)])
        net2 = Network([Neuron(0)], [Neuron(1)])

        net1.add_neuron(Neuron(3))
        net2.add_neuron(Neuron(4))

        merged = merge_two_networks(net1, net2)
        self.assertEqual(2, len(merged.hidden_neurons))
        self.assertEqual(1, len(merged.input_neurons))
        self.assertEqual(1, len(merged.output_neurons))

    def test_same_network(self):
        net1 = Network([Neuron(0)], [Neuron(1)])
        net2 = net1.clone()

        merged = merge_two_networks(net1, net2)
        self.assertEqual(net1.hash(), merged.hash())

    def test_super_network(self):
        net1 = Network([Neuron(0)], [Neuron(1)])
        net2 = net1.clone()

        net1.add_synapse(Synapse(0, 1))

        merged = merge_two_networks(net1, net2)
        self.assertEqual(net1.hash(), merged.hash())

    def test_merge_synapses(self):
        net1 = Network([Neuron(0), Neuron(1)], [Neuron(2), Neuron(3)])
        net2 = Network([Neuron(0), Neuron(1)], [Neuron(2), Neuron(3)])

        net1.add_synapse(Synapse(0, 1))
        net2.add_synapse(Synapse(1, 2))

        merged = merge_two_networks(net1, net2)
        self.assertEqual(2, len(merged.synapses))

    def test_merge_synapses_no_duplication(self):
        net1 = Network([Neuron(0), Neuron(1)], [Neuron(2), Neuron(3)])
        net2 = Network([Neuron(0), Neuron(1)], [Neuron(2), Neuron(3)])

        net1.add_synapse(Synapse(0, 1))
        net1.add_synapse(Synapse(1, 2))
        net2.add_synapse(Synapse(1, 2))

        merged = merge_two_networks(net1, net2)
        self.assertEqual(2, len(merged.synapses))

    def test_crossover_never_stripable(self):
        g = Generator(2, 2)

        for _ in range(50):
            n1, n2 = g.generate_networks(2)
            out = merge_two_networks(n1, n2)
            stripped = out.clone().strip()

            self.assertEqual(0, stripped.distance(out))
