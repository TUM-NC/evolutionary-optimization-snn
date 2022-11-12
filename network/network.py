"""
Provide the network class for neural network architecture and parameters
"""
import copy
from enum import Enum
from typing import List, Optional, Set

import matplotlib.pyplot as plt
import networkx as nx

from network.neuron import Neuron
from network.synapse import Synapse
from utility.json_serialize import JsonSerialize


class NeuronType(Enum):
    """
    Enum for different neuron types
    """

    Input = "input"
    Output = "output"
    Hidden = "hidden"


def get_neuron_color(neuron_type: NeuronType):
    """
    Return a color to a neurontype

    :param neuron_type:
    :return:
    """
    if neuron_type == NeuronType.Input:
        return "pink"
    if neuron_type == NeuronType.Output:
        return "lightblue"
    if neuron_type == NeuronType.Hidden:
        return "lightgreen"

    raise RuntimeError("The given neuron type is not specified")


class Network(JsonSerialize):
    """
    Class to represent a spiking neural network
    Contains neurons (input, hidden and output)
    And synapses (at maximum one synapse per pair of neurons)
    """

    input_neurons = List[Neuron]
    hidden_neurons = Set[Neuron]
    output_neurons = List[Neuron]

    synapses = Set[Synapse]

    def __init__(
        self, input_neurons=None, output_neurons=None, hidden_neurons=None
    ):
        if input_neurons is None:
            input_neurons = []
        if output_neurons is None:
            output_neurons = []
        if hidden_neurons is None:
            hidden_neurons = set()

        self.input_neurons = sorted(input_neurons, key=lambda n: n.uid)
        self.output_neurons = sorted(output_neurons, key=lambda n: n.uid)
        self.hidden_neurons = set(hidden_neurons)

        if self._has_duplicate_uid():
            raise RuntimeError("Neuron uid should be unique")

        self.synapses = set()

    def _has_duplicate_uid(self):
        """
        Returns, whether there are duplicate neuron uid given
        :return:
        """
        return len(set(self.get_all_neurons_uid())) != len(
            self.get_all_neurons_uid()
        )

    def clone(self):
        """
        Make a deep copy of the network

        :return:
        """
        return copy.deepcopy(self)

    def get_neuron_type(self, uid: int = None, neuron: Neuron = None):
        """
        Get the type of a neuron
        Raises exception, when neuron not in network

        :param uid:
        :param neuron:
        :return:
        """
        if neuron is None:
            if uid is None:
                raise RuntimeError("Please provide either uid or a neuron")
            neuron = self.find_neuron_by_uid(uid)

        if neuron in self.input_neurons:
            return NeuronType.Input
        if neuron in self.output_neurons:
            return NeuronType.Output
        if neuron in self.hidden_neurons:
            return NeuronType.Hidden

        raise RuntimeError("Given neuron is not in network")

    def get_all_neurons(self) -> List[Neuron]:
        """
        Return a list of all neurons (sorted by uid)

        :return:
        """
        all_neurons = (
            self.input_neurons
            + self.output_neurons
            + list(self.hidden_neurons)
        )
        return sorted(all_neurons, key=lambda n: n.uid)

    def get_hidden_neurons(self) -> List[Neuron]:
        """
        Get all hidden neurons in a sorted ways
        :return:
        """
        return sorted(list(self.hidden_neurons), key=lambda n: n.uid)

    def get_all_synapses(self):
        """
        Returns a list of all synapses (sorted by uid)

        :return:
        """
        synapses = list(self.synapses)
        return sorted(synapses, key=lambda s: (s.connect_from, s.connect_to))

    def get_all_neurons_uid(self) -> List[int]:
        """
        Return a list of all neuron uids

        :return:
        """
        return [n.uid for n in self.get_all_neurons()]

    def add_synapse(self, synapse: Synapse):
        """
        Add a synapse to the network

        Returns false, when a synapse with same neurons already exists
        Also returns false, when specified neurons do not exist

        :param synapse:
        :return: whether the operation was successful
        """
        found = next(
            (
                True
                for syn in self.synapses
                if syn.connect_from == synapse.connect_from
                and syn.connect_to == synapse.connect_to
            ),
            False,
        )
        if found:
            return False

        if (
            self.find_neuron_by_uid(synapse.connect_from) is None
            or self.find_neuron_by_uid(synapse.connect_to) is None
        ):
            return False

        self.synapses.add(synapse)
        return True

    def add_neuron(self, neuron: Neuron):
        """
        Add a neuron to the (hidden) neurons of the network
        Does not do anything, when a neuron with same id already exists

        :param neuron:
        :return: returns, whether the operation was succesful
        """
        if self.find_neuron_by_uid(neuron.uid) is not None:
            return False
        self.hidden_neurons.add(neuron)
        return True

    def remove_neuron(self, neuron: Neuron):
        """
        Remove a (hidden) neuron

        :param neuron:
        :return:
        """
        # removes a neuron and all corresponding synapses
        self.hidden_neurons.remove(neuron)
        self.synapses = set(
            s
            for s in self.synapses
            if neuron.uid not in (s.connect_from, s.connect_to)
        )

    def remove_neuron_uid(self, uid: int):
        """
        Remove a (hidden) neuron by the given uid

        :param uid:
        :return:
        """
        neuron = self.find_hidden_neuron_by_uid(uid=uid)
        self.remove_neuron(neuron=neuron)

    def find_hidden_neuron_by_uid(self, uid: int) -> Optional[Neuron]:
        """
        Find a hidden neuron by the uid

        :param uid:
        :return:
        """
        return next((n for n in self.hidden_neurons if n.uid == uid), None)

    def find_neuron_by_uid(self, uid: int) -> Optional[Neuron]:
        """
        Find any neuron by the uid

        :param uid:
        :return:
        """
        return next((n for n in self.get_all_neurons() if n.uid == uid), None)

    def remove_synapse(self, synapse: Synapse):
        """
        Remove a synapse

        :param synapse:
        :return:
        """
        self.synapses.remove(synapse)

    def find_synapse_by_neurons(
        self, from_neuron: Neuron, to_neuron: Neuron
    ) -> Optional[Synapse]:
        """
        Returns a possible synapse, that connects the two given neurons

        :param from_neuron:
        :param to_neuron:
        :return:
        """
        return self.find_synapse_by_neurons_uid(from_neuron.uid, to_neuron.uid)

    def find_synapse_by_neurons_uid(
        self, connect_from: int, connect_to: int
    ) -> Optional[Synapse]:
        """
        Returns a possible synapse, that connects the two given neurons

        :param connect_from:
        :param connect_to:
        :return:
        """
        return next(
            (
                s
                for s in self.synapses
                if s.connect_from == connect_from
                and s.connect_to == connect_to
            ),
            None,
        )

    def can_reach_output(self):
        """
        Checks, if a spike from any input neuron can reach any output neuron
        If not, simulating does not make sense,
        since there will never be an output spike

        :return:
        """
        output = set(n.uid for n in self.output_neurons)
        reachable = self.reachable_neurons()

        return len(output.intersection(reachable))

    def reachable_neurons(self) -> Set[int]:
        """
        Return a set of all through synapses connected neurons,
        starting with the input neurons

        :return:
        """
        graph = self.to_networkx()
        graph.add_node("start")
        for input_neuron in self.input_neurons:
            graph.add_edge("start", input_neuron.uid)
        return nx.descendants(graph, "start")

    def influence_output_neurons(self) -> Set[int]:
        """
        Returns a list with all neurons, that can influence the output

        :return:
        """
        graph = self.to_networkx().reverse()
        graph.add_node("start")
        for output_neuron in self.output_neurons:
            graph.add_edge("start", output_neuron.uid)
        return nx.descendants(graph, "start")

    def to_networkx(self):
        """
        Return a networkx graph for the current graph
        with all attributes included
        :return:
        """
        graph = nx.DiGraph()

        for neuron in self.get_all_neurons():
            graph.add_node(
                neuron.uid,
                neuron_type=self.get_neuron_type(neuron=neuron).value,
                **neuron.parameters,
            )

        for synapse in self.synapses:
            graph.add_edge(
                synapse.connect_from, synapse.connect_to, **synapse.parameters
            )

        return graph

    @classmethod
    def from_networkx(cls, networkx: nx.DiGraph):
        """
        Create a network from a networkx di graph
        with corresponding attribute values

        :param networkx:
        :return:
        """
        input_neurons = []
        output_neurons = []
        hidden_neurons = set()

        for index in networkx.nodes:
            node_data = networkx.nodes[index]
            neuron_type = NeuronType(node_data["neuron_type"])

            # don't add neuron type to neuron parameters
            del node_data["neuron_type"]
            neuron = Neuron(uid=index, **node_data)

            if neuron_type == NeuronType.Input:
                input_neurons.append(neuron)
            elif neuron_type == NeuronType.Output:
                output_neurons.append(neuron)
            elif neuron_type == NeuronType.Hidden:  # hidden neurons
                hidden_neurons.add(neuron)
            else:
                raise RuntimeError("The given neuron type is not supported")

        net = cls(
            input_neurons=input_neurons,
            output_neurons=output_neurons,
            hidden_neurons=hidden_neurons,
        )

        for (connect_from, connect_to) in networkx.edges:
            edge_data = networkx.get_edge_data(connect_from, connect_to)
            synapse = Synapse(connect_from, connect_to, **edge_data)
            net.add_synapse(synapse)

        return net

    def to_json_object(self) -> dict:
        """
        Return the network as dictionary
        Suitable for json dump
        :return:
        """
        netx = self.to_networkx()
        return nx.readwrite.node_link_data(netx)

    def hash(self):
        """
        Use hashing algorithm, to detect same networks
        Strips away connections, that won't be used (strip)

        :return:
        """
        stripped = self.clone().strip().to_networkx()
        for node in stripped.nodes:
            neuron = stripped.nodes[node]
            neuron_type = neuron["neuron_type"]
            if (
                NeuronType(neuron_type) != NeuronType.Hidden
            ):  # input and output neurons are ordered -> add id to attributes
                neuron["uid"] = node

            neuron["unique"] = "-".join(
                f"{key}:{neuron[key]}" for key in sorted(neuron.keys())
            )

        for _, _, edge in stripped.edges(data=True):
            edge["unique"] = "-".join(
                f"{key}:{edge[key]}" for key in sorted(edge.keys())
            )
        return nx.weisfeiler_lehman_graph_hash(
            stripped, edge_attr="unique", node_attr="unique"
        )

    @classmethod
    def from_json_object(cls, json_object: dict) -> "Network":
        """
        Read a network from a json object
        :param json_object:
        :return:
        """
        networkx = nx.readwrite.node_link_graph(json_object, True, False)
        return cls.from_networkx(networkx)

    def distance(self, compare: "Network") -> float:
        """
        Calculate the edit distance between to networks

        :param compare:
        :return: float
        """

        def match_exact(neuron1, neuron2):
            return neuron1 == neuron2

        net1 = self.to_networkx()
        net2 = compare.to_networkx()
        return nx.graph_edit_distance(
            net1, net2, node_match=match_exact, edge_match=match_exact
        )

    def strip(self) -> "Network":
        """
        Modifies the current network
        All neurons removed, that do not contribute to output spikes
        1. Neurons, that can't be reached via input
        2. Neurons, that can't reach any output neurons

        :return:
        """
        hidden_uids: Set[int] = set(n.uid for n in self.hidden_neurons)

        # 1. find unreachable neurons
        reachable = self.reachable_neurons()
        unreachable: Set[int] = hidden_uids.difference(reachable)

        # 2. neurons, that can't reach any output neurons
        influence = self.influence_output_neurons()
        no_influence: Set[int] = hidden_uids.difference(influence)

        to_remove = unreachable.union(no_influence)

        for remove_uid in to_remove:
            self.remove_neuron_uid(remove_uid)

        # return self, for easier chaining in usage
        return self

    def plot_graph(
        self,
        legend=True,
        spring_layout_seed=None,
        figure_size=None,
        circular_layout=False,
    ):
        """
        Plot the network as a graph
        For more advanced plotting, try an external application

        :param legend: legend for the colors of the neurons
        :param spring_layout_seed: seed for the layout positioning
        :param figure_size: option, to set the size of the output figure
        :param circular_layout: use circular layout instead of spring layout
        :return: figure from matplotlib
        """
        graph = self.to_networkx()
        # draw neurons
        color_map = []

        for node in graph.nodes:
            neuron_type = graph.nodes[node]["neuron_type"]
            color_map.append(get_neuron_color(NeuronType(neuron_type)))

        # remove/rename attribute weight on edges,
        # as it can have unintended side effects during drawing
        for _, _, edge in graph.edges(data=True):
            if "weight" in edge:
                edge["synaptic_weight"] = edge["weight"]
                edge.pop("weight", None)

        fig, ax = plt.subplots()

        ax.set_axis_off()  # no outer axis should be visible
        if figure_size:
            fig.set_size_inches(figure_size)

        if legend:
            # draw sample points in same color as node to generate legend
            for neuron_type in NeuronType:
                color = get_neuron_color(neuron_type)
                ax.plot([0], [0], color=color, label=neuron_type.name)
            plt.legend()

        if circular_layout:
            layout = nx.circular_layout(graph)
        else:
            layout = nx.spring_layout(graph, seed=spring_layout_seed)
        nx.draw_networkx(
            graph,
            pos=layout,
            node_color=color_map,
            with_labels=True,
            ax=ax,
            edgecolors="black",
        )
        return fig
