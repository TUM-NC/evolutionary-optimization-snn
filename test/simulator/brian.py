import unittest

from experiment.brian.xor import XOR
from network.decoder.brian.decoder import BrianDecoder
from network.encoder.brian.float import FloatBrianEncoder
from network.evolution.generator import Generator
from network.network import Network
from network.neuron import Neuron
from network.synapse import Synapse
from simulator.brian import BrianSimulator


class DummyDecoder(BrianDecoder):
    def get_value(self):
        return 0


class TestBrian(unittest.TestCase):
    def test_network_export_import(self):
        experiment = XOR()

        generator = Generator.create_from_experiment(experiment=experiment)
        population = generator.generate_networks(1)

        net1 = population[0]
        exported = net1.to_json()
        net2 = Network.from_json(exported)

        f1 = experiment.fitness(network=net1)
        f2 = experiment.fitness(network=net2)
        self.assertEqual(f1, f2)

    def simulator_simple_network(self, network: Network, input_pattern: float):
        """
        simulate a network with one input (float) on given value
        and one output (not evaluated)
        :param network: Network to simulate
        :param input_pattern: float value for input
        :return:
        """
        encoder = FloatBrianEncoder(number_of_neurons=1)
        decoder = DummyDecoder(number_of_neurons=1)

        simulator = BrianSimulator(
            networks=[network],
            encoder=encoder,
            decoder=decoder,
            inputs=[(input_pattern,)],
        )
        simulator.simulate()
        return simulator

    def test_synapse_always_excite(self):
        input_neuron = Neuron(uid=0, threshold=127)
        output_neuron = Neuron(uid=1, threshold=100)
        synapse = Synapse(
            connect_from=0, connect_to=1, weight=120, exciting=True, delay=0
        )

        network = Network(
            input_neurons=[input_neuron], output_neurons=[output_neuron]
        )
        network.add_synapse(synapse)

        simulator = self.simulator_simple_network(network, 1)
        spike_trains = simulator.spikes.spike_trains()

        self.assertTrue(
            60 <= len(spike_trains[0]) <= 140,
            "input should roughly spike 100 times",
        )

        self.assertTrue(
            len(spike_trains[0]) == len(spike_trains[1]),
            "output should roughly spike like input",
        )

    def test_hidden_neuron_always(self):
        input_neuron = Neuron(uid=0, threshold=127)
        output_neuron = Neuron(uid=1, threshold=100)
        hidden_neuron = Neuron(uid=2, threshold=100)

        s1 = Synapse(
            connect_from=0, connect_to=2, exciting=True, weight=125, delay=0
        )
        s2 = Synapse(
            connect_from=2, connect_to=1, exciting=True, weight=125, delay=0
        )

        network = Network(
            input_neurons=[input_neuron],
            output_neurons=[output_neuron],
            hidden_neurons=[hidden_neuron],
        )
        network.add_synapse(s1)
        network.add_synapse(s2)

        simulator = self.simulator_simple_network(network, 1)
        spike_trains = simulator.spikes.spike_trains()

        self.assertTrue(
            len(spike_trains[0])
            == len(spike_trains[1])
            == len(spike_trains[2]),
            "hidden neuron should pass spikes to next neuron",
        )

    def test_inhibitory_synapse(self):
        input_neuron = Neuron(uid=0, threshold=127)
        output_neuron = Neuron(uid=1, threshold=80)
        hidden_neurons = [
            Neuron(uid=2, threshold=100),
            Neuron(
                uid=3, threshold=80
            ),  # should behave like output without inhibitory
        ]

        # always excite hidden neuron
        s1 = Synapse(
            connect_from=0, connect_to=2, exciting=True, weight=125, delay=0
        )
        s2 = Synapse(
            connect_from=2, connect_to=1, exciting=False, weight=125, delay=0
        )
        s3 = Synapse(
            connect_from=0, connect_to=1, weight=80, delay=0, exciting=True
        )
        s4 = Synapse(
            connect_from=0, connect_to=3, weight=80, delay=0, exciting=True
        )

        network = Network(
            input_neurons=[input_neuron],
            output_neurons=[output_neuron],
            hidden_neurons=hidden_neurons,
        )
        network.add_synapse(s1)
        network.add_synapse(s2)
        network.add_synapse(s3)
        network.add_synapse(s4)

        simulator = self.simulator_simple_network(network, 1)
        spike_trains = simulator.spikes.spike_trains()

        self.assertTrue(
            len(spike_trains[0]) == len(spike_trains[2]),
            "hidden and input spike about same",
        )
        self.assertTrue(len(spike_trains[1]) == 0, "output should never spike")
        self.assertTrue(
            len(spike_trains[3]) > 0, "hidden2 (like output) should spike"
        )
