import random
import unittest
from unittest.mock import ANY, patch

from network.evolution.reproduction.mutator import Mutator
from network.network import Network
from network.neuron import Neuron
from network.synapse import Synapse
from utility.configuration import Configuration


class TestMutator(unittest.TestCase):
    def test_add_node(self):
        net = Network([Neuron(0)], [Neuron(1)])
        mutator = Mutator()

        mutator.mutate_network(net, "add_node")

        self.assertEqual(1, len(net.hidden_neurons))

        # hidden node, should be connected
        self.assertEqual(2, len(net.synapses))
        stripped = net.clone().strip()
        self.assertEqual(0, stripped.distance(net))

    def test_delete_node_empty(self):
        net = Network([Neuron(0)], [Neuron(1)])
        mutator = Mutator()

        mutator.mutate_network(net, "delete_node")

        self.assertEqual(0, len(net.hidden_neurons))

    def test_delete_node(self):
        net = Network([Neuron(0)], [Neuron(1)])
        mutator = Mutator()

        mutator.mutate_network(net, "add_node")
        mutator.mutate_network(net, "add_node")
        self.assertEqual(2, len(net.hidden_neurons))
        mutator.mutate_network(net, "delete_node")

        self.assertLessEqual(len(net.hidden_neurons), 1)

    def test_add_edge(self):
        net = Network([Neuron(0)], [Neuron(1)])
        mutator = Mutator()

        mutator.mutate_network(net, "add_edge")

        self.assertEqual(1, len(net.synapses))

    def test_delete_edge(self):
        net = Network([Neuron(0), Neuron(1)], [Neuron(2), Neuron(3)])
        net.add_synapse(Synapse(0, 2))
        net.add_synapse(Synapse(1, 3))
        mutator = Mutator()

        mutator.mutate_network(net, "delete_edge")

        self.assertEqual(1, len(net.synapses))

    @patch.object(Mutator, "element_mutation")
    def test_node_param(self, mock):
        net = Network([Neuron(0)], [Neuron(1)])
        mutator = Mutator()

        mutator.mutate_network(net, "node_param")

        mock.assert_called_once()

    def test_node_threshold(self):
        n1 = Neuron(uid=0, threshold=5)
        mutator = Mutator()

        mutator.element_mutation(
            n1, "threshold", {"type": "fixed", "value": 10}
        )

        self.assertEqual(10, n1.threshold)

    @patch.object(Mutator, "element_mutation")
    def test_edge_param(self, mock):
        net = Network([Neuron(0)], [Neuron(1)])
        synapse = Synapse(0, 1, weight=10, delay=5, exciting=True)
        net.add_synapse(synapse)
        mutator = Mutator()

        mutator.mutate_network(net, "edge_param")

        self.assertTrue(mock.called)
        mock.assert_called_once_with(synapse, ANY, ANY)

    def test_synapse_weight(self):
        net = Network([Neuron(0)], [Neuron(1)])
        synapse = Synapse(0, 1, weight=10, delay=5, exciting=True)
        net.add_synapse(synapse)
        mutator = Mutator()

        mutator.element_mutation(
            synapse, "weight", {"type": "fixed", "value": 15}
        )

        self.assertEqual(15, synapse.weight)

    def test_mutations_should_never_be_able_to_strip(self):
        """
        Mutations should never produce a network, with useless nodes
        Random approach, may need to adjust the test amount

        :return:
        """
        net = Network([Neuron(0), Neuron(1)], [Neuron(2), Neuron(3)])
        mutator = Mutator()
        test_amount = 100

        for _ in range(test_amount):
            mutation_type = mutator.get_random_mutation_type()
            mutator.mutate_network(net, mutation_type)

            stripped = net.clone().strip()
            self.assertEqual(0, stripped.distance(net))

    def test_get_random_mutation_type_seed(self):
        mutator = Mutator()
        random.seed(12345)
        t1 = mutator.get_random_mutation_type()
        random.seed(12345)
        t2 = mutator.get_random_mutation_type()
        self.assertEqual(t1, t2)

    def test_mutations_reproducible(self):
        net1 = Network([Neuron(0), Neuron(1)], [Neuron(2), Neuron(3)])
        net2 = Network([Neuron(0), Neuron(1)], [Neuron(2), Neuron(3)])
        mutator = Mutator()

        random.seed(987)
        out1 = mutator.apply_mutations(net1)
        random.seed(987)
        out2 = mutator.apply_mutations(net2)

        self.assertEqual(0, out1.distance(out2))

    def test_add_random_synapse(self):
        net = Network([Neuron(0)], [Neuron(1)])
        mutator = Mutator()
        mutator.add_random_synapse(net)
        self.assertEqual(1, len(net.synapses))

    def test_will_not_add_unavailable_neuron_parameter(self):
        configuration = Configuration(
            {
                "neuron_parameters": {
                    "threshold": {"type": "random_int", "min": 0, "max": 127}
                }
            }
        )

        mutations = ["add_node", "node_param"]
        net = Network([Neuron(0, threshold=5)], [Neuron(1, threshold=5)])
        mutator = Mutator(configuration=configuration)

        for _ in range(100):
            mutation = random.choice(mutations)
            mutator.mutate_network(net, mutation)

            for n in net.get_all_neurons():
                # neurons should only have a threshold parameter
                self.assertIsNotNone(n.threshold)
                # and no leak parameter
                self.assertTrue("leak" not in n.parameters)

    def test_will_not_add_unavailable_synapse_parameter(self):
        configuration = Configuration(
            {
                "synapse_parameters": {
                    "weight": {"type": "random_int", "min": 0, "max": 127},
                }
            }
        )

        mutations = ["add_edge", "edge_param", "add_node"]
        net = Network([Neuron(0)], [Neuron(1)])
        mutator = Mutator(configuration=configuration)

        for _ in range(100):
            mutator.mutate_network(net, random.choice(mutations))

            for s in net.synapses:
                # synapses should only have a weight parameter
                self.assertIsNotNone(s.weight)
                # and no delay/exciting parameter
                self.assertTrue("delay" not in s.parameters)
                self.assertTrue("exciting" not in s.parameters)
