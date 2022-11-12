import unittest

import networkx as nx

from network.network import Network, NeuronType
from network.neuron import Neuron
from network.synapse import Synapse


def create_simple_network():
    net = Network([Neuron(0, threshold=0)], [Neuron(1, threshold=0)])
    hidden_neuron = Neuron(5, threshold=1)
    net.add_neuron(hidden_neuron)
    net.add_synapse(Synapse(0, 5, weight=0))
    net.add_synapse(Synapse(5, 1, weight=0))
    return net


class TestNetwork(unittest.TestCase):
    def test_init_network(self):
        net = Network([Neuron(0), Neuron(1)], [Neuron(2), Neuron(3)])
        self.assertEqual(2, len(net.input_neurons))
        self.assertEqual(2, len(net.output_neurons))
        self.assertEqual(0, len(net.hidden_neurons))
        self.assertEqual(0, len(net.synapses))

    def test_network_convert_hidden_to_set(self):
        net = Network(
            input_neurons=[Neuron(0)],
            output_neurons=[Neuron(1)],
            hidden_neurons=[Neuron(2)],
        )
        self.assertEqual(True, isinstance(net.hidden_neurons, set))

    def test_add_neuron(self):
        net = Network([Neuron(0)], [Neuron(1)])
        neuron = Neuron(5)
        assumed_hidden = set([neuron])
        net.add_neuron(neuron)
        self.assertEqual(assumed_hidden, net.hidden_neurons)

    def test_add_synapse(self):
        net = Network([Neuron(0)], [Neuron(1)])
        synapse = Synapse(0, 1)
        assumed_synapses = set([synapse])
        net.add_synapse(synapse)
        self.assertEqual(assumed_synapses, net.synapses)

    def test_add_synapse_without_neurons(self):
        net = Network([Neuron(0)], [Neuron(1)])
        added1 = net.add_synapse(Synapse(1, 10))
        added2 = net.add_synapse(Synapse(11, 0))
        self.assertEqual(0, len(net.synapses))
        self.assertEqual(False, added1)
        self.assertEqual(False, added2)

    def test_remove_synapse(self):
        net = create_simple_network()
        first_synapse = list(net.synapses)[0]
        net.remove_synapse(first_synapse)
        self.assertEqual(1, len(net.synapses))

    def test_remove_neuron(self):
        net = create_simple_network()
        hidden_neuron = list(net.hidden_neurons)[0]
        net.remove_neuron(hidden_neuron)
        self.assertEqual(set(), net.hidden_neurons)
        # also removes synapses
        self.assertEqual(set(), net.synapses)

    def test_clone(self):
        net = create_simple_network()
        net_clone = net.clone()
        synapses = list(net.synapses)
        net.remove_synapse(synapses[0])
        # does not remove synapse from cloned network
        self.assertEqual(2, len(net_clone.synapses))
        hidden = list(net_clone.hidden_neurons)[0]
        net_clone.remove_neuron(hidden)
        self.assertEqual(
            set(), net_clone.synapses
        )  # should all synapses be removed
        self.assertEqual(set([synapses[1]]), net.synapses)

    def test_add_same_neuron_multiple_times(self):
        net = Network([Neuron(0)], [Neuron(1)])
        neuron = Neuron(5)
        net.add_neuron(neuron)
        self.assertEqual(set([neuron]), net.hidden_neurons)
        added = net.add_neuron(
            neuron
        )  # should not be able to add exactly same neuron
        self.assertEqual(1, len(net.hidden_neurons))
        self.assertEqual(False, added)
        # should also not be able to add other neuron with same id
        neuron_same_uid = Neuron(5)
        added2 = net.add_neuron(neuron_same_uid)
        self.assertEqual(set([neuron]), net.hidden_neurons)
        self.assertEqual(False, added2)

    def test_add_same_synapse(self):
        net = Network([Neuron(0)], [Neuron(1)])
        input_neuron = net.input_neurons[0]
        output_neuron = net.output_neurons[0]
        syn = Synapse(input_neuron.uid, output_neuron.uid)
        net.add_synapse(syn)
        net.add_synapse(syn)
        self.assertEqual(1, len(net.synapses))
        syn_same = Synapse(input_neuron.uid, output_neuron.uid)
        net.add_synapse(syn_same)
        self.assertEqual(set([syn]), net.synapses)

    def test_to_networkx(self):
        input = Neuron(1, threshold=5)
        output = Neuron(2, threshold=10)
        synapse = Synapse(1, 2, weight=10, delay=5, exciting=True)
        net = Network([input], [output])
        net.add_synapse(synapse)

        netx = net.to_networkx()

        self.assertEqual([1, 2], list(netx.nodes))
        self.assertEqual([(1, 2)], list(netx.edges))
        self.assertEqual(
            {"weight": 10, "delay": 5, "exciting": True},
            netx.get_edge_data(1, 2),
        )
        self.assertEqual(
            {
                "neuron_type": NeuronType.Input.value,
                "threshold": 5,
            },
            netx.nodes[1],
        )
        self.assertEqual(
            {
                "neuron_type": NeuronType.Output.value,
                "threshold": 10,
            },
            netx.nodes[2],
        )

    def test_from_networkx(self):
        g = nx.DiGraph()
        g.add_node(
            2, neuron_type=NeuronType.Input.value, delay=4, leak=2, threshold=1
        )
        g.add_node(
            3,
            neuron_type=NeuronType.Hidden.value,
            delay=3,
            leak=3,
            threshold=2,
        )
        g.add_node(
            1,
            neuron_type=NeuronType.Output.value,
            delay=None,
            leak=1,
            threshold=3,
        )
        g.add_edge(2, 3, weight=3, delay=1, exciting=True)

        net = Network.from_networkx(g)

        self.assertEqual(1, len(net.input_neurons))
        self.assertEqual(
            {"uid": 2, "parameters": {"threshold": 1, "delay": 4, "leak": 2}},
            vars(net.input_neurons[0]),
        )
        self.assertEqual(1, len(net.hidden_neurons))
        self.assertEqual(
            {"uid": 3, "parameters": {"threshold": 2, "delay": 3, "leak": 3}},
            vars(list(net.hidden_neurons)[0]),
        )
        self.assertEqual(1, len(net.output_neurons))
        self.assertEqual(
            {
                "uid": 1,
                "parameters": {"threshold": 3, "delay": None, "leak": 1},
            },
            vars(net.output_neurons[0]),
        )

        self.assertEqual(1, len(net.synapses))
        synapse = list(net.synapses)[0]
        self.assertEqual(net.input_neurons[0].uid, synapse.connect_from)
        self.assertEqual(list(net.hidden_neurons)[0].uid, synapse.connect_to)
        self.assertEqual(3, synapse.weight)
        self.assertEqual(1, synapse.delay)
        self.assertEqual(True, synapse.exciting)

    def test_import_export(self):
        net = create_simple_network()
        json = net.to_json_object()
        net2 = Network.from_json_object(json)

        self.assertEqual(0, net.distance(net2))

    def test_distance_neuron_attribute(self):
        net = create_simple_network()
        net2 = net.clone()

        net.input_neurons[0].threshold = 4
        net2.input_neurons[0].threshold = 5

        self.assertEqual(1, net.distance(net2))

    def test_distance_neuron(self):
        net = create_simple_network()
        net2 = net.clone()

        net.add_neuron(Neuron(1234))

        self.assertEqual(1, net.distance(net2))

    def test_distance_synapse_attribute(self):
        net = create_simple_network()
        net2 = net.clone()

        s1 = net.find_synapse_by_neurons_uid(0, 5)
        s1.weight = 4
        s2 = net2.find_synapse_by_neurons_uid(0, 5)
        s2.weight = 5

        self.assertEqual(1, net.distance(net2))

    def test_distance_same(self):
        net = create_simple_network()
        net2 = net.clone()

        self.assertEqual(0, net.distance(net2))
        self.assertEqual(0, net2.distance(net))
        self.assertEqual(0, net.distance(net))

    def test_reachable(self):
        net = create_simple_network()
        reachable = net.reachable_neurons()
        self.assertEqual(set([0, 1, 5]), reachable)

        net.remove_neuron_uid(5)
        reachable = net.reachable_neurons()
        self.assertEqual(set([0]), reachable)

        net.add_neuron(Neuron(5))
        reachable = net.reachable_neurons()
        self.assertEqual(set([0]), reachable)

        net.add_synapse(Synapse(0, 1))
        reachable = net.reachable_neurons()
        self.assertEqual(set([0, 1]), reachable)

    def test_strip_unreachable(self):
        net1 = create_simple_network()
        net2 = net1.clone()
        net2.add_neuron(Neuron(9))
        net2.add_neuron(Neuron(10))
        net2.add_synapse(Synapse(9, 10))

        net1_stripped = net1.clone().strip()
        net2_stripped = net2.clone().strip()
        difference = net1_stripped.distance(net2_stripped)

        self.assertEqual(0, difference)

    def test_strip_useless(self):
        net = create_simple_network()
        net.add_neuron(Neuron(9))
        net.add_synapse(Synapse(1, 9))

        stripped = net.clone().strip()

        self.assertEqual(1, len(stripped.hidden_neurons))

    def test_hash(self):
        net = create_simple_network()
        list(net.hidden_neurons)[0].threshold = 4
        should_be_hash = net.hash()
        list(net.hidden_neurons)[0].uid = 10
        # manually reset uids for neurons in synapse
        for synapse in net.synapses:
            if synapse.connect_to == 5:
                synapse.connect_to = 10
            if synapse.connect_from == 5:
                synapse.connect_from = 10

        new_hash = net.hash()

        # hash should be same, if only uid of hidden neuron changed
        self.assertEqual(should_be_hash, new_hash)

        list(net.hidden_neurons)[0].threshold = 5
        new_hash = net.hash()
        self.assertNotEqual(should_be_hash, new_hash)

    def test_hash_same_input_different_order(self):
        input1 = Neuron(1, threshold=5)
        input2 = Neuron(2, threshold=5)
        output = Neuron(3, threshold=10)
        synapse = Synapse(1, 3, weight=10, delay=5, exciting=True)

        net1 = Network([input1, input2], [output])
        net1.add_synapse(synapse)
        net2 = Network([input2, input1], [output])
        net2.add_synapse(synapse)

        self.assertEqual(
            net1.hash(), net2.hash(), "Order of neurons in list is irrelevant"
        )

        synapse2 = Synapse(2, 3, weight=10, delay=5, exciting=True)
        net3 = Network([input2, input1], [output])
        net3.add_synapse(synapse2)

        self.assertNotEqual(
            net1.hash(),
            net3.hash(),
            "Order of input(uids) should be reflected",
        )

    def test_neurons_should_be_sorted(self):
        net = Network([Neuron(0), Neuron(1)], [Neuron(2), Neuron(3)])
        net.add_neuron(Neuron(10))
        net.add_neuron(Neuron(5))

        self.assertEqual([0, 1, 2, 3, 5, 10], net.get_all_neurons_uid())

    def test_input_neurons_sorted(self):
        input1 = Neuron(2)
        input2 = Neuron(1)
        output = Neuron(3)
        net = Network([input1, input2], [output])

        self.assertEqual([1, 2, 3], net.get_all_neurons_uid())
        self.assertEqual([1, 2], [n.uid for n in net.input_neurons])

    def test_output_neurons_sorted(self):
        input = Neuron(2)
        output1 = Neuron(3)
        output2 = Neuron(1)
        net = Network([input], [output1, output2])

        self.assertEqual([1, 2, 3], net.get_all_neurons_uid())
        self.assertEqual([1, 3], [n.uid for n in net.output_neurons])

    def test_neuron_type(self):
        i1 = Neuron(0)
        i2 = Neuron(1)
        o1 = Neuron(2)
        o2 = Neuron(3)
        h1 = Neuron(50)
        h2 = Neuron(90)

        net = Network([i1, i2], [o1, o2], {h1, h2})

        self.assertEqual(NeuronType.Input, net.get_neuron_type(neuron=i1))
        self.assertEqual(NeuronType.Input, net.get_neuron_type(uid=0))
        self.assertEqual(NeuronType.Input, net.get_neuron_type(1))
        self.assertEqual(NeuronType.Output, net.get_neuron_type(2))
        self.assertEqual(NeuronType.Output, net.get_neuron_type(neuron=o2))
        self.assertEqual(NeuronType.Hidden, net.get_neuron_type(neuron=h1))
        self.assertEqual(NeuronType.Hidden, net.get_neuron_type(uid=90))

        self.assertRaises(RuntimeError, net.get_neuron_type)  # nothing given
        self.assertRaises(
            RuntimeError, net.get_neuron_type, 123
        )  # not in network

    def test_should_not_have_duplicate_neuron_uid_init(self):
        self.assertRaises(
            RuntimeError, Network, [Neuron(0)], [Neuron(1)], {Neuron(0)}
        )
        self.assertRaises(RuntimeError, Network, [Neuron(0)], [Neuron(0)])
        self.assertRaises(
            RuntimeError, Network, [Neuron(1)], [Neuron(0)], {Neuron(0)}
        )

    def test_should_not_have_duplicate_hidden_uid(self):
        net = Network([Neuron(0)], [Neuron(1)])
        net.add_neuron(Neuron(0))
        self.assertEqual(0, len(net.hidden_neurons))
