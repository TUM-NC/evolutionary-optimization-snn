"""
Provide the merge operation,
to generate on big network from two parent networks
"""

import random
from typing import List

from network.network import Network
from network.neuron import Neuron


def merge_two_networks(net1: Network, net2: Network) -> Network:
    """
    Merge two networks, moving all neurons and synapses into a single network
    If neurons with same uid exist in both networks, a random one is chosen
    If synapse with same start/target are given, a random one is chosen

    :param net1:
    :param net2:
    :return:
    """
    input_neurons = select_neurons_randomly_by_uid(
        net1.input_neurons, net2.input_neurons
    )
    output = select_neurons_randomly_by_uid(
        net1.output_neurons, net2.output_neurons
    )
    hidden = select_neurons_randomly_by_uid(
        net1.get_hidden_neurons(), net2.get_hidden_neurons()
    )

    new_network = Network(
        input_neurons=input_neurons,
        output_neurons=output,
        hidden_neurons=hidden,
    )

    synapses = net1.get_all_synapses() + net2.get_all_synapses()

    # shuffle, to add random synapses for same connections
    random.shuffle(synapses)
    for synapse in synapses:
        new_network.add_synapse(synapse)

    return new_network


def select_neurons_randomly_by_uid(
    neurons1: List[Neuron], neurons2: List[Neuron]
):
    """
    Select random neurons from the two lists
    Keeps only one neuron per uid

    :param neurons1:
    :param neurons2:
    :return:
    """
    output_neurons = []

    all_neurons = list(neurons1) + list(neurons2)
    random.shuffle(all_neurons)
    added_uids = []

    for neuron in all_neurons:
        if neuron.uid not in added_uids:
            output_neurons.append(neuron)
            added_uids.append(neuron.uid)

    return output_neurons
