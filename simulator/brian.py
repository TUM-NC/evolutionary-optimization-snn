"""
Actual implementation of a simulator using brian
"""
import os
import platform
from typing import List, Optional, Tuple

import numpy as np
from brian2 import (
    Network,
    NeuronGroup,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    Unit,
    ms,
    seed,
)

from network.decoder.brian.decoder import BrianDecoder
from network.encoder.brian.encoder import BrianEncoder
from network.network import Network as EoNetwork
from simulator.simulator import Simulator
from utility.validation import (
    any_check,
    chain_checks,
    greater_than_zero,
    is_bool,
    is_int,
    is_none,
    is_positive,
)


def set_brian_parameters():
    if platform.system() == "Darwin":
        # set parameters for mac and brian
        os.environ["CC"] = "gcc"
        os.environ["CXX"] = "g++"


set_brian_parameters()


class BrianSimulator(Simulator):
    """
    Simulator using brian as framework for spiking neural networks
    """

    brian_network: Network
    spikes: SpikeMonitor
    _neurons: NeuronGroup
    simulation_time: Unit

    encoder: BrianEncoder
    decoder: BrianDecoder

    def __init__(
        self,
        networks: List[EoNetwork],
        inputs: List[Tuple],
        encoder: BrianEncoder,
        decoder: BrianDecoder,
        simulation_time=1000 * ms,
        brian_seed: Optional[int] = None,
    ):
        super().__init__(networks, encoder, decoder)
        self.inputs = inputs
        self._create_network()
        self.simulation_time = simulation_time

        if brian_seed is not None:
            seed(brian_seed)

    def _create_network(self):
        """
        Initiate the brian network

        :return:
        """
        networks = self.networks
        number_neurons = sum(len(x.get_all_neurons()) for x in networks)

        # create all brian objects that we need
        # leaky integrate and fire neuron
        eqs = """dv/dt = (-v)/(leak*ms) : 1
                 v_th: 1
                 leak: 1"""
        neurons = NeuronGroup(
            N=number_neurons,
            model=eqs,
            threshold="v > v_th",
            reset="v = 0",
            method="euler",
        )
        synapses = Synapses(neurons, neurons, model="w: 1", on_pre="v += w")
        if self.encoder.is_deterministic():
            input_patterns = list(set(self.inputs))
        else:
            input_patterns = self.inputs
        spike_generator = self.encoder.get_spike_generator(input_patterns)
        spike_generator_synapses = Synapses(
            spike_generator, neurons, on_pre="v += 129"
        )
        spikes = SpikeMonitor(neurons)

        # iteration over networks, for neuron variables and
        offset = 0  # offset variable, to get correct neuron ids in networks

        synapse_connections_from = []
        synapse_connections_to = []
        synapse_delay = []
        synapse_weight = []

        spike_generator_synapses_from = []
        spike_generator_synapses_to = []

        neuron_threshold = np.zeros(number_neurons)
        neuron_leak = np.zeros(number_neurons)

        for i, network in enumerate(networks):
            sorted_neurons = network.get_all_neurons()
            sorted_neurons_uid = [n.uid for n in sorted_neurons]

            # set synapse values for all synapses in a network
            for synapse in network.get_all_synapses():
                # get absolute ids for neuron in network
                from_id = (
                    sorted_neurons_uid.index(synapse.connect_from) + offset
                )
                to_id = sorted_neurons_uid.index(synapse.connect_to) + offset

                synapse_connections_from.append(from_id)
                synapse_connections_to.append(to_id)

                synapse_delay.append(synapse.delay * ms)
                weight = synapse.weight
                if not synapse.exciting:
                    weight *= -1
                synapse_weight.append(weight)

            # set neuron thresholds
            for index, neuron in enumerate(sorted_neurons):
                neuron_threshold[index + offset] = neuron.threshold
                neuron_leak[index + offset] = self._get_neuron_leak(neuron)

            # find the correct spike generator for the specified input pattern
            if self.encoder.is_deterministic():
                spike_generator_offset = (
                    input_patterns.index(self.inputs[i])
                    * self.encoder.number_of_neurons
                )
            else:
                spike_generator_offset = i * self.encoder.number_of_neurons

            # add synapse from spike generator to input neurons
            for index, _ in enumerate(network.input_neurons):
                spike_generator_synapses_from.append(
                    index + spike_generator_offset
                )
                spike_generator_synapses_to.append(offset + index)

            offset += len(
                sorted_neurons
            )  # set offset to first neuron id of next network

        # single calls of brian improve performance
        neurons.v_th = neuron_threshold
        neurons.leak = neuron_leak
        if len(synapse_connections_from) != 0:
            synapses.connect(
                i=synapse_connections_from, j=synapse_connections_to
            )
            # set synapse values after connections are established
            synapses.delay = synapse_delay
            synapses.w = synapse_weight
        else:
            # when there are no synpases in all networks, set them to false
            # otherwise, brian will throw an exception
            synapses.active = False
        spike_generator_synapses.connect(
            i=spike_generator_synapses_from, j=spike_generator_synapses_to
        )

        # finally, create the brian network and set class variables
        net = Network(
            neurons,
            synapses,
            spike_generator,
            spike_generator_synapses,
            spikes,
        )
        self.brian_network = net
        self.spikes = spikes
        # add neurons for later access (e.g. for adding state monitor values)
        self._neurons = neurons

    @staticmethod
    def _get_neuron_leak(neuron):
        """
        Use 10 as default value

        :param neuron:
        :return:
        """
        if "leak" not in neuron.parameters:
            return 10  # default value
        return neuron.leak

    def _get_output_spike_trains(self, spike_trains):
        """
        Get spike trains from all networks with output neurons,
        in order of neurons given in network

        :param spike_trains:
        :return:
        """
        network_spike_trains = []

        offset = 0
        for network in self.networks:
            out_spike_trains = []
            sorted_neurons = network.get_all_neurons()
            for output_neuron in network.output_neurons:
                index = sorted_neurons.index(output_neuron) + offset
                out_spike_trains.append(spike_trains[index])
            network_spike_trains.append(out_spike_trains)

            offset += len(sorted_neurons)

        return network_spike_trains

    def _get_decoded_values(self):
        """
        return decoded values for all outputs

        :return:
        """
        spike_trains = self.spikes.spike_trains()
        output_network = self._get_output_spike_trains(spike_trains)

        decoded = []
        for output in output_network:
            self.decoder.set_spikes(output)
            value = self.decoder.get_value()
            decoded.append(value)

        return decoded

    def add_neuron_state_monitor(self):
        """
        Add a neuron state monitor to the simulator
        :return: returns the state monitor for future use
        """
        state_monitor = StateMonitor(self._neurons, "v", record=True)
        self.brian_network.add(state_monitor)
        return state_monitor

    def simulate(self):
        """
        Simulate the network as created before

        :return: values returned by the decoder
        """
        net = self.brian_network
        net.run(self.simulation_time)
        return self._get_decoded_values()

    @staticmethod
    def get_neuron_parameters():
        return {
            "threshold": chain_checks(is_positive, is_int),
            "leak": any_check(
                is_none, chain_checks(greater_than_zero, is_int)
            ),
        }

    @staticmethod
    def get_synapse_parameters():
        return {
            "weight": chain_checks(is_positive, is_int),
            "exciting": is_bool,
            "delay": any_check(is_none, chain_checks(is_positive, is_int)),
        }
