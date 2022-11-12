"""
Actual implementation of a simulator using lava
"""
from typing import List, Tuple

import numpy as np
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF

from network.decoder.lava.decoder import LavaDecoder
from network.encoder.lava.encoder import LavaEncoder
from network.network import Network as EoNetwork
from network.synapse import Synapse
from simulator.simulator import Simulator


class LavaSimulator(Simulator):
    """
    Simulator using brian as framework for spiking neural networks
    """

    encoder: LavaEncoder
    decoder: LavaDecoder

    input_process = None
    output_process = None

    def __init__(
        self,
        networks: List[EoNetwork],
        inputs: List[Tuple],
        encoder: LavaEncoder,
        decoder: LavaDecoder,
        # simulation_time=1000 * ms,
    ):
        super().__init__(networks, encoder, decoder)
        self.inputs = inputs
        self.networks = networks
        self._create_network()
        # self.simulation_time = simulation_time

    def _create_network(self):
        """
        Initiate the network

        :return:
        """
        input_process = self.encoder.get_input_process()

        # TODO: for now only one network
        network = self.networks[0]
        all_neurons = network.get_all_neurons()
        sorted_neurons_uid = [n.uid for n in all_neurons]

        # create neurons
        lif_neurons = []
        for neuron in all_neurons:
            lif_neuron = LIF(shape=(1,), vth=neuron.threshold, dv=0, du=4095)
            lif_neurons.append(lif_neuron)

        # create connections in network
        synapse: Synapse
        for synapse in network.get_all_synapses():
            connect_from_idx = sorted_neurons_uid.index(synapse.connect_from)
            connect_to_idx = sorted_neurons_uid.index(synapse.connect_to)
            dense = Dense(weights=np.array([[synapse.weight]]))
            lif_neurons[connect_from_idx].s_out.connect(dense.s_in)
            dense.a_out.connect(lif_neurons[connect_to_idx].a_in)

        # create input synapses
        for input_index, neuron in enumerate(network.input_neurons):
            index = sorted_neurons_uid.index(neuron.uid)

            connection_matrix = np.zeros((1, len(network.input_neurons)))
            connection_matrix[0][input_index] = 1
            input_dense = Dense(weights=connection_matrix)
            input_process.spikes_out.connect(input_dense.s_in)
            input_dense.a_out.connect(lif_neurons[index].a_in)

        # connect output spikes
        output_process = self.decoder.get_output_process()
        for output_index, neuron in enumerate(network.output_neurons):
            index = sorted_neurons_uid.index(neuron.uid)
            out_lif = lif_neurons[index]

            connection_matrix = np.zeros((len(network.output_neurons), 1))
            connection_matrix[output_index][0] = 1
            dense = Dense(weights=connection_matrix)

            out_lif.s_out.connect(dense.s_in)
            dense.a_out.connect(output_process.spikes_in)

        self.input_process = input_process
        self.output_process = output_process

    def simulate(self):
        """
        Simulate the network as created before

        :return: values returned by the decoder
        """
        self.input_process.run(
            condition=RunSteps(num_steps=10000),
            run_cfg=Loihi1SimCfg(),
        )

        self.decoder.set_value(self.output_process)
        return self.decoder.get_value()
