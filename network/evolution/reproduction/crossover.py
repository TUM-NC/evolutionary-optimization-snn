"""
Module to provide the crossover function,
to get two networks from two parent networks
"""

import random
from typing import List

from network.network import Network
from network.neuron import Neuron
from network.synapse import Synapse


def crossover(net1: Network, net2: Network):
    """
    Perform a crossover operation on the two networks

    :param net1:
    :param net2:
    :return:
    """
    input1, input2 = split_neurons(net1.input_neurons, net2.input_neurons)
    output1, output2 = split_neurons(net1.output_neurons, net2.output_neurons)
    hidden1, hidden2 = split_neurons(
        net1.get_hidden_neurons(), net2.get_hidden_neurons()
    )

    out1 = Network(
        input_neurons=input1, output_neurons=output1, hidden_neurons=hidden1
    )
    out2 = Network(
        input_neurons=input2, output_neurons=output2, hidden_neurons=hidden2
    )

    synapses = net1.get_all_synapses() + net2.get_all_synapses()
    distribute_synapses(synapses, out1, out2)

    return out1.strip(), out2.strip()


def distribute_synapses(synapses: List[Synapse], net1: Network, net2: Network):
    """
    Randomly distribute synapses across the two networks
    :param synapses:
    :param net1:
    :param net2:
    :return:
    """
    networks = [net1, net2]

    for synapse in synapses:
        random.shuffle(networks)
        to_add = networks[0]
        added = to_add.add_synapse(synapse)  # try random first network to add
        if not added:
            # try to add to other network
            # nothing has to be done, when this is also not successful
            networks[1].add_synapse(synapse)


def split_neurons(neurons1: List[Neuron], neurons2: List[Neuron]):
    """
    Split neurons of n1 and n2 randomly into two sets
    Each set will contain at maximum one neuron with the same uid

    :param n1:
    :param neurons2:
    :return:
    """
    out1 = set()
    out2 = set()

    for neuron in neurons1:
        # for first neurons, always choose random list
        to_add = random.choice([out1, out2])
        to_add: set
        to_add.add(neuron)

    ordered_sets = [out1, out2]
    for neuron in neurons2:
        # for next neurons
        random.shuffle(ordered_sets)
        to_add = ordered_sets[0]
        to_add: set

        # check, whether neuron with given id already exists in given set
        already_exists = any(x for x in to_add if x.uid == neuron.uid)
        if already_exists:
            to_add = ordered_sets[1]

        to_add.add(neuron)

    return out1, out2
