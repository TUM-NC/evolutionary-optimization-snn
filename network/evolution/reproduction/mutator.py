"""
Provide mutator class, to perform mutations on a given network
"""

import random
from typing import Optional

from network.dynamic_parameter import DynamicParameter
from network.network import Network
from network.neuron import Neuron
from network.synapse import Synapse
from utility.configurable import Configurable
from utility.configuration import Configuration
from utility.parameter_configuration import (
    get_mutable_parameters,
    init_parameter_values,
    is_valid_parameter_configuration,
    is_valid_parameter_value,
    parameter_to_value,
)
from utility.random import random_rates
from utility.validation import (
    contains_only_given_keys,
    is_int,
    is_positive,
    is_valid_on_all_dict_values,
)


class Mutator(Configurable):
    """
    This class can handle mutations as specified in the parameters
    """

    mutation_rates = {
        "add_node": 0.08,
        "delete_node": 0.08,
        "add_edge": 0.15,
        "delete_edge": 0.15,
        "node_param": 0.27,
        "edge_param": 0.27,
    }

    number_of_mutations = {"type": "fixed", "value": 7}
    neuron_parameters = {
        "threshold": {"type": "random_int", "min": 0, "max": 127},
        "leak": {"type": "random_choice", "values": [1, 5, 10, 20, 40]},
    }
    synapse_parameters = {
        "weight": {"type": "random_int", "min": 0, "max": 127},
        "delay": {"type": "random_int", "min": 0, "max": 15},
        "exciting": {"type": "random_bool"},
    }

    def __init__(self, configuration: Optional[Configuration] = None):
        super().__init__(configuration=configuration)

    def set_configurable(self):
        self.add_configurable_attribute(
            "number_of_mutations",
            "Amount of mutations to apply, for a single network",
            validate=[
                is_valid_parameter_value(is_int),
                is_valid_parameter_value(is_positive),
            ],
        )
        self.add_configurable_attribute(
            "neuron_parameters",
            "Which parameters a neuron has",
            validate=is_valid_on_all_dict_values(
                is_valid_parameter_configuration
            ),
        )
        self.add_configurable_attribute(
            "synapse_parameters",
            "Which parameters a synapse has",
            validate=is_valid_on_all_dict_values(
                is_valid_parameter_configuration
            ),
        )
        self.add_configurable_attribute(
            "mutation_rates",
            "Probability of mutations to happen",
            validate=[
                contains_only_given_keys(
                    [
                        "add_node",
                        "delete_node",
                        "add_edge",
                        "delete_edge",
                        "node_param",
                        "edge_param",
                    ]
                ),
                is_valid_on_all_dict_values(is_positive),
            ],
        )

    def get_random_mutation_type(self):
        """
        Get a random mutation type, based on given rates

        :return:
        """
        return random_rates(self.mutation_rates)

    def mutate_network(self, network: Network, mutation_type: str):
        """
        Do a single mutation on the given network

        :param network: Network to apply the mutation to (no cloning)
        :param mutation_type: operation to perform on network
        """
        if mutation_type == "add_node":
            self.add_hidden_neuron(network)
        elif mutation_type == "delete_node":
            if len(network.hidden_neurons) == 0:
                return

            random_neuron = random.choice(network.get_hidden_neurons())
            network.remove_neuron(random_neuron)
            network.strip()  # more nodes might need to be removed
        elif mutation_type == "add_edge":
            self.add_random_synapse(network)
        elif mutation_type == "delete_edge":
            if len(network.synapses) == 0:
                return

            random_synapse = random.choice(network.get_all_synapses())
            network.remove_synapse(random_synapse)
            network.strip()  # more nodes might need to be removed
        elif mutation_type == "node_param":
            random_neuron: Neuron = random.choice(network.get_all_neurons())
            mutation_type, mutation_parameter = random.choice(
                get_mutable_parameters(self.neuron_parameters)
            )
            self.element_mutation(
                random_neuron, mutation_type, mutation_parameter
            )
        elif mutation_type == "edge_param":
            if len(network.synapses) == 0:
                return

            random_synapse: Synapse = random.choice(network.get_all_synapses())
            mutation_type, mutation_parameter = random.choice(
                get_mutable_parameters(self.synapse_parameters)
            )
            self.element_mutation(
                random_synapse, mutation_type, mutation_parameter
            )

    def add_hidden_neuron(self, network: Network):
        """
        Add a hidden neuron to the network
        Should be connected to at least one from input (hull)
        Should be connected to at least one from output (hull)
        :param network:
        :return:
        """
        parameters = init_parameter_values(self.neuron_parameters)
        new_neuron = Neuron.with_random_id(
            exclude_ids=network.get_all_neurons_uid(),
            parameters=parameters,
        )
        network.add_neuron(new_neuron)

        # add synapses from input reachable
        reachable = network.reachable_neurons()
        pre_synaptic = random.choice(sorted(reachable))
        s1_parameters = init_parameter_values(self.synapse_parameters)
        s1 = Synapse(
            pre_synaptic,
            new_neuron.uid,
            **s1_parameters,
        )
        network.add_synapse(s1)

        # add synapse to affect output
        influence_out = network.influence_output_neurons()
        post_synaptic = random.choice(sorted(influence_out))
        s2_parameters = init_parameter_values(self.synapse_parameters)
        s2 = Synapse(
            new_neuron.uid,
            post_synaptic,
            **s2_parameters,
        )
        network.add_synapse(s2)

    def create_random_neuron(self, uid):
        """
        create a random neuron with only those parameters as specified
        in configuration

        :param uid:
        :return:
        """
        parameters = init_parameter_values(self.neuron_parameters)
        return Neuron(uid=uid, **parameters)

    def add_random_synapse(self, network: Network):
        """
        Add a random synapse between two neurons
        Synapses should be meaningful: connect reachable node with influential

        :return:
        """
        reachable_neurons = sorted(network.reachable_neurons())
        pre_synaptic = random.choice(reachable_neurons)

        influential_neurons = sorted(network.influence_output_neurons())
        post_synaptic = random.choice(influential_neurons)

        s_parameters = init_parameter_values(self.synapse_parameters)
        synapse = Synapse(
            pre_synaptic,
            post_synaptic,
            **s_parameters,
        )
        network.add_synapse(synapse)

    @staticmethod
    def element_mutation(element: DynamicParameter, key, parameter):
        """
        Do a mutation on a synapse or neuron on given key parameter
        With given parameters for value generation
        :param self:
        :param element:
        :param key:
        :param parameter:
        :return:
        """
        element.parameters[key] = parameter_to_value(parameter)

    def apply_mutations(self, network: Network):
        """
        Clones the network and returns a new network, with applied mutations

        :param network: basic network to start from
        :return: cloned network with mutations applied
        """
        new_network = network.clone()

        number_of_mutations = parameter_to_value(self.number_of_mutations)
        for _ in range(number_of_mutations):
            mutation_type = self.get_random_mutation_type()
            self.mutate_network(new_network, mutation_type=mutation_type)

        return new_network
