import random
import unittest

from network.evolution.generator import Generator
from utility.configuration import Configuration


class TestGenerator(unittest.TestCase):
    def test_generator_no_useless(self):
        generator = Generator(2, 2)
        nets = generator.generate_networks(50)

        for net in nets:
            stripped = net.clone().strip()
            self.assertEqual(0, net.distance(stripped))

            self.assertEqual(2, len(net.input_neurons))
            self.assertEqual(2, len(net.output_neurons))

    def test_amount_hidden_neurons(self):
        configuration = Configuration(
            {
                "generate_hidden_neurons": {"type": "fixed", "value": 5},
            }
        )
        generator = Generator(2, 2, configuration=configuration)
        net = generator.generate_networks(1)[0]

        self.assertEqual(5, len(net.hidden_neurons))

    def test_amount_synapses(self, *argv):
        configuration = Configuration(
            {
                "generate_hidden_neurons": {"type": "fixed", "value": 0},
                "generate_synapses": {"type": "fixed", "value": 1},
            }
        )
        generator = Generator(2, 2, configuration=configuration)
        net = generator.generate_networks(1)[0]

        self.assertEqual(1, len(net.synapses))

    def test_amount_neuron_synapse_config(self):
        configuration = Configuration(
            {
                "generate_hidden_neurons": {"type": "fixed", "value": 3},
                "generate_synapses": {"type": "fixed", "value": 0},
            }
        )
        generator = Generator(2, 2, configuration=configuration)

        net = generator.generate_networks(1)[0]

        self.assertEqual(3, len(net.hidden_neurons))
        self.assertEqual(6, len(net.synapses))

    def test_synapse_config(self):
        configuration = Configuration(
            {
                "generate_hidden_neurons": {"type": "fixed", "value": 0},
                "generate_synapses": {"type": "fixed", "value": 1},
            }
        )
        generator = Generator(2, 2, configuration=configuration)

        net = generator.generate_networks(1)[0]

        self.assertEqual(0, len(net.hidden_neurons))
        self.assertEqual(1, len(net.synapses))

    def test_empty_network(self):
        generator = Generator(number_inputs=1, number_outputs=1)
        net = generator.create_empty_network()

        self.assertEqual(1, len(net.input_neurons))
        self.assertEqual(1, len(net.output_neurons))
        self.assertEqual(0, len(net.hidden_neurons))
        self.assertEqual(0, len(net.synapses))

    def test_empty_network_two(self):
        generator = Generator(number_inputs=1, number_outputs=2)
        net = generator.create_empty_network()

        self.assertEqual(1, len(net.input_neurons))
        self.assertEqual(2, len(net.output_neurons))
        self.assertEqual(0, len(net.hidden_neurons))
        self.assertEqual(0, len(net.synapses))

    def test_should_not_have_unavailable_neuron_parameters(self):
        # has no leak parameter
        configuration = Configuration(
            {
                "neuron_parameters": {
                    "threshold": {"type": "random_int", "min": 0, "max": 127}
                }
            }
        )
        generator = Generator(
            number_inputs=20, number_outputs=20, configuration=configuration
        )
        net = generator.generate_network()

        for n in net.get_all_neurons():
            self.assertTrue("leak" not in n.parameters)
            self.assertIsNotNone(n.threshold)

    def test_should_have_neuron_parameters(self):
        configuration = Configuration(
            {
                "neuron_parameters": {
                    "threshold": {"type": "random_int", "min": 0, "max": 127},
                    "leak": {"type": "random_int", "min": 0, "max": 3},
                }
            }
        )
        generator = Generator(
            number_inputs=20, number_outputs=20, configuration=configuration
        )
        net = generator.generate_network()

        for n in net.get_all_neurons():
            self.assertIsNotNone(n.leak)
            self.assertIsNotNone(n.threshold)

    def test_should_only_have_synapse_parameters(self):
        configuration = Configuration(
            {
                "synapse_parameters": {
                    "delay": {"type": "random_int", "min": 0, "max": 15},
                    "exciting": {"type": "random_bool"},
                },
                "generate_synapses": {"type": "fixed", "value": 200},
            }
        )
        generator = Generator(
            number_inputs=20, number_outputs=20, configuration=configuration
        )
        net = generator.generate_network()

        for s in net.synapses:
            self.assertTrue("weight" not in s.parameters)
            self.assertIsNotNone(s.exciting)
            self.assertIsNotNone(s.delay)

    def test_generator_reproducible(self):
        configuration = Configuration(
            {
                "generate_hidden_neurons": {
                    "type": "random_int",
                    "min": 5,
                    "max": 10,
                },
            }
        )
        generator = Generator(2, 2, configuration=configuration)

        random.seed(1234)
        n1 = generator.generate_network()
        n1_2 = generator.generate_network()
        random.seed(1234)
        n2 = generator.generate_network()
        n2_2 = generator.generate_network()
        random.seed(123)
        n3 = generator.generate_network()

        self.assertEqual(0, n1.distance(n2), "first should be same")
        self.assertEqual(0, n1_2.distance(n2_2), "second should be same")
        # easy check, if not same, networks can't be same
        self.assertNotEqual(len(n1.synapses), len(n3.synapses))

    def test_generator_larger_reproducible(self):
        configuration = Configuration(
            {
                "generate_hidden_neurons": {
                    "type": "random_int",
                    "min": 3,
                    "max": 3,
                },
            }
        )

        random.seed(1234)
        g1 = Generator(2, 2, configuration=configuration)
        n1 = g1.generate_networks(30)
        random.seed(1234)
        g2 = Generator(2, 2, configuration=configuration)
        n2 = g2.generate_networks(30)

        for a1, a2 in zip(n1, n2):
            self.assertEqual(0, a1.distance(a2))
