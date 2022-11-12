import statistics
import time
from typing import List, Optional, Tuple, TypedDict, Union

import networkx as nx
from matplotlib import pyplot as plt

from network.evolution.origin import Origin, ReproductionType
from network.evolution.selection import best
from network.network import Network
from utility.json_serialize import JsonSerialize
from utility.list_operation import count_occurrences, flat_list


class EpochStats(TypedDict):
    """
    Store the stats for a single epoch
    """

    took: float  # time it took for evaluation
    population: List[Network]
    fitness_scores: List[float]
    operations: List[Origin]


class Stats(JsonSerialize):
    """
    Store the stats for one evolution
    """

    data: List[EpochStats]
    last_epoch_start: float

    def __init__(self, data=None):
        if data is None:
            data = []

        self.data = data
        self.last_epoch_start = 0

    def start_epoch(self):
        """
        Set the internal timer, for timing the evaluation
        :return:
        """
        self.last_epoch_start = time.time()

    def add_epoch(
        self,
        population: List[Network],
        fitness_scores: List[float],
        operations: Optional[List[Origin]] = None,
    ):
        """
        Add an epoch for evaluation of statistics

        :param population:
        :param fitness_scores:
        :param operations:
        :return:
        """
        if len(population) != len(fitness_scores) or (
            operations is not None and len(operations) != len(population)
        ):
            raise RuntimeError("Lists should have same amount of values")

        if operations is None:
            operations = []
        epoch_end = time.time()
        took = epoch_end - self.last_epoch_start

        self.data.append(
            EpochStats(
                population=population,
                fitness_scores=fitness_scores,
                took=took,
                operations=operations,
            )
        )

    def get_epoch_information(self, epoch: int, epochs: int):
        """
        Return a string with information about the executed epoch

        :param epoch: index of the epoch
        :param epochs: amount of total epochs to evaluate
        :return:
        """
        e = self.get_epoch(epoch)
        if e is None:
            raise RuntimeError(
                "There is no data available for the given epoch"
            )

        fitness_scores = e["fitness_scores"]

        return (
            "Epoch {epoch:{len_epoch}d}/{total_epoch:d} - "
            "Best fitness: {fitness:3.4f} - "
            "Average fitness: {average:3.4f} - "
            "Took {time:4.4f}s".format(
                epoch=epoch + 1,
                len_epoch=len(str(epochs)),
                total_epoch=epochs,
                fitness=max(fitness_scores),
                average=sum(fitness_scores) / len(fitness_scores),
                time=e["took"],
            )
        )

    def get_epoch(self, epoch: int) -> EpochStats:
        """
        Get statistics for a given epoch
        :param epoch:
        :return:
        """
        return self.data[epoch]

    def get_latest_epoch(self) -> Optional[int]:
        """
        Returns the latest epoch
        Returns None, if no epoch was evaluated so far

        :return:
        """
        if len(self.data) == 0:
            return None
        return len(self.data) - 1

    def get_amount_epochs(self):
        """
        The amount of simulated epochs

        :return:
        """
        return len(self.data)

    def get_latest_population_fitness(
        self,
    ) -> Tuple[Optional[List[Network]], Optional[List[float]]]:
        """
        Return the latest population and fitness from stats
        Returns (None, None) if no population evaluated so far

        :return:
        """
        latest_epoch = self.get_latest_epoch()
        if latest_epoch is None:
            return None, None

        epoch = self.get_epoch(latest_epoch)
        return epoch["population"], epoch["fitness_scores"]

    def get_best_network(self, epoch=None):
        """
        Returns the best performing network from the latest epoch
        :return:
        """
        if epoch is None:
            epoch = self.get_latest_epoch()
        epoch_stats: EpochStats = self.get_epoch(epoch)
        return best(
            epoch_stats["population"], epoch_stats["fitness_scores"], n=1
        )[0]

    def get_best_network_alltime(self):
        """
        Search and find the network with the best performance
        Searches in all epochs
        If multiple with same fitness, returns first occurrence

        :return:
        """
        best_network, best_fitness = None, None

        for epoch in self.data:
            for index, fitness in enumerate(epoch["fitness_scores"]):
                if best_fitness is None or fitness > best_fitness:
                    best_fitness = fitness
                    best_network = epoch["population"][index]

        return best_network, best_fitness

    def compare(self, other_stats: "Stats"):
        """
        Check, whether two stats are the same

        Attention: does not check the networks
        Time(took) should sufficiently indicate, whether two stats are the same

        :param check_population:
        :param other_stats:
        :return:
        """

        def remove_population(json_object):
            for data in json_object["data"]:
                del data["population"]
            return json_object

        s1 = remove_population(self.to_json_object())
        s2 = remove_population(other_stats.to_json_object())

        same_without_population = s1 == s2
        return same_without_population

    def is_same_populations(self, other_stats: "Stats"):
        """
        Check if populations from other stats are the same

        :param other_stats:
        :return:
        """
        if self.get_latest_epoch() != other_stats.get_latest_epoch():
            return False

        for epoch, e in enumerate(self.data):
            for i, net in enumerate(e["population"]):
                net_other_stats = other_stats.data[epoch]["population"][i]
                if net.distance(net_other_stats) != 0:
                    return False

        return True

    def plot_progress(self, show_best=True, show_mean=True, show_median=True):
        """
        Create and show a figure, which shows the progress during evolution

        :param show_best:
        :param show_mean:
        :param show_median:
        :return:
        """
        fig, ax = plt.subplots()

        epoch_label = [i + 1 for i, _ in enumerate(self.data)]

        if show_best:
            best = [max(epoch["fitness_scores"]) for epoch in self.data]
            ax.plot(epoch_label, best, label="Best")
        if show_mean:
            mean = [
                sum(epoch["fitness_scores"]) / len(epoch["fitness_scores"])
                for epoch in self.data
            ]
            ax.plot(epoch_label, mean, linestyle="--", label="Mean")
        if show_median:
            median = [
                statistics.median(epoch["fitness_scores"])
                for epoch in self.data
            ]
            ax.plot(epoch_label, median, label="Median")

        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fitness Score")
        ax.set_xlim(1, len(self.data))

        return fig

    def plot_evolution(self):
        """
        Diagram to plot the evolution
        :return:
        """

        def color_by_reproduction(r: ReproductionType):
            if r == ReproductionType.Random:
                return "lightblue"
            if r == ReproductionType.Same:
                return "green"
            if r == ReproductionType.Mutation:
                return "pink"
            if r == ReproductionType.Crossover:
                return "yellow"
            if r == ReproductionType.Merge:
                return "red"
            raise NotImplementedError("This is an unknown reproduction type")

        graph = nx.Graph()
        color_map = []

        for layer, epoch in enumerate(self.data):
            enumerated_population = list(enumerate(epoch["population"]))
            enumerated_population.sort(
                key=lambda e: epoch["fitness_scores"][e[0]], reverse=True
            )
            for index, network in enumerated_population:
                graph.add_node(f"{index}-{layer}", layer=layer)

                operation = epoch["operations"][index]
                color_map.append(
                    color_by_reproduction(operation.reproduction_type)
                )
                for previous in operation.associated_networks:
                    graph.add_edge(
                        f"{previous}-{layer - 1}", f"{index}-{layer}"
                    )

        fig, ax = plt.subplots()
        pos = nx.multipartite_layout(graph, subset_key="layer")
        nx.draw(graph, pos, with_labels=False, node_color=color_map, ax=ax)

        # draw a legend
        for reproduction_type in ReproductionType:
            color = color_by_reproduction(reproduction_type)
            ax.plot([0], [0], color=color, label=reproduction_type.name)
        ax.legend(loc="lower left")

        # enable axis to be shown
        plt.axis("on")
        ax.tick_params(
            left=True, bottom=True, labelleft=True, labelbottom=True
        )
        # hide tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rank (in population)")

        fig.show()

    def get_total_time_took(self):
        """
        Get the total amount of time took
        :return:
        """
        took = [e["took"] for e in self.data]
        return sum(took)

    def get_origin(self, network: Union[int, Network], epoch=None):
        """
        get full history of a given network

        :param network: can be int or the network
        :param epoch: if none, latest epoch will be searched
        :return: list with lists
        """
        if epoch is None:
            epoch = self.get_latest_epoch()

        epoch_data = self.get_epoch(epoch)
        if isinstance(network, Network):
            index = epoch_data["population"].index(network)
            if index == -1:
                return None
        else:
            index = network

        operation = epoch_data["operations"][index]

        if len(operation.associated_networks) == 0:
            return [operation]
        if len(operation.associated_networks) == 1:
            return [
                operation,
                self.get_origin(
                    operation.associated_networks[0], epoch=epoch - 1
                ),
            ]

        next_level = [operation]
        for n in operation.associated_networks:
            next_level.append(self.get_origin(n, epoch=epoch - 1))
        return next_level

    def get_origin_distribution(
        self, network: Union[int, Network], epoch=None
    ):
        """
        get the distribution of actions across full history
        :param network:
        :param epoch:
        :return:
        """
        history = self.get_origin(network, epoch)
        history_flat: List[Origin] = flat_list(history)
        history_flat_string = [o.reproduction_type for o in history_flat]
        return count_occurrences(history_flat_string)

    def to_json_object(self):
        """
        Convert the object to a dict for json dumps
        :return:
        """
        return {
            "data": [
                {
                    "took": data["took"],
                    "population": [
                        n.to_json_object() for n in data["population"]
                    ],
                    "fitness_scores": data["fitness_scores"],
                    "operations": data["operations"],
                }
                for data in self.data
            ],
        }

    @classmethod
    def from_json_object(cls, json_object: dict) -> "Stats":
        """
        create the class from a json dict

        :param json_object:
        :return:
        """
        data = [
            EpochStats(
                took=d["took"],
                population=[
                    Network.from_json_object(n) for n in d["population"]
                ],
                fitness_scores=d["fitness_scores"],
                operations=[
                    Origin(ReproductionType(o[0]), o[1])
                    for o in d["operations"]
                ],
            )
            for d in json_object["data"]
        ]
        return cls(data)
