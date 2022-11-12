"""
Definition for an experiment using brian
"""
from abc import ABC
from typing import List, Optional

from experiment.experiment import Experiment
from network.decoder.brian.decoder import BrianDecoder
from network.encoder.brian.encoder import BrianEncoder
from network.network import Network
from simulator.brian import BrianSimulator


# This class is still abstract
class BrianExperiment(Experiment, ABC):
    """
    Abstract class to create a experiment using the brian simulator
    """

    encoder: BrianEncoder
    decoder: BrianDecoder

    _simulation_result = {}
    _seed: Optional[int] = None

    def _get_simulator(
        self, networks_for_simulation: List[Network], inputs: List[tuple]
    ):
        """
        Start the simulator on multiple networks with each given input pattern

        :param networks_for_simulation:
        :param inputs:
        :return:
        """
        return BrianSimulator(
            networks=networks_for_simulation,
            inputs=inputs,
            encoder=self.encoder,
            decoder=self.decoder,
            brian_seed=self._seed,
        )

    def _simulate_on_multiple_inputs(
        self, networks: List[Network], input_patterns: List[tuple]
    ):
        """
        Simulate each network on each input pattern

        :param networks:
        :param input_patterns:
        :return: list with networks on first level
            and outputs for each input on second level
        """
        # repeat each input pattern for each network
        inputs = input_patterns * len(networks)

        # repeat each network for each pattern
        networks_for_simulation = [n for n in networks for _ in input_patterns]

        simulator = self._get_simulator(networks_for_simulation, inputs)
        outputs = simulator.simulate()

        network_outputs = []
        for i, _ in enumerate(networks):
            network_output = []
            for j, _ in enumerate(input_patterns):
                network_output.append(outputs[i * len(input_patterns) + j])
            network_outputs.append(network_output)
        return network_outputs

    def simulate(self, networks: List[Network]):
        """
        Implement in a subclass, what to simulate exactly

        :param networks:
        :return:
        """
        raise RuntimeError("Implement this function")

    def fitness(self, networks: List[Network]):
        """
        Simulate once before accessing the fitness values

        :param networks:
        :return:
        """
        self.simulate(networks)
        return super().fitness(networks)

    def get_output_by_network(self, network: Network):
        """
        get the output of network after simulation

        :param network:
        :return:
        """
        return self._simulation_result[network]

    def set_output_by_network(self, network: Network, output):
        """
        set the output of a network after simulation

        :param output:
        :param network:
        :return:
        """
        self._simulation_result[network] = output

    def monitor_neurons(self, network: Network, pattern: tuple):
        """
        Return a state monitor for inspecting v values of neurons

        :param network:
        :param pattern:
        :return:
        """
        simulator = self._get_simulator(
            networks_for_simulation=[network], inputs=[pattern]
        )
        state_monitor = simulator.add_neuron_state_monitor()
        simulator.simulate()
        return state_monitor, simulator.spikes

    def set_seed(self, seed: Optional[int] = None):
        """
        Make experiment reproducible, by including a seed

        :param seed:
        :return:
        """
        self._seed = seed

    @staticmethod
    def get_simulator_class():
        """
        get the class for the simulator

        :return:
        """
        return BrianSimulator
