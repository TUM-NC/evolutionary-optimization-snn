"""
Provides a class, to perform all reproduction mechanisms on a network
"""
from typing import List, Optional

from network.evolution.origin import Origin, ReproductionType
from network.evolution.reproduction.crossover import crossover
from network.evolution.reproduction.merge import merge_two_networks
from network.evolution.reproduction.mutator import Mutator
from network.evolution.selection import tournament_selection
from network.network import Network
from utility.configurable import Configurable
from utility.configuration import Configuration
from utility.random import random_rates
from utility.validation import (
    contains_only_given_keys,
    is_positive,
    is_valid_on_all_dict_values,
    valid_values,
)


class Reproduction(Configurable):
    """
    Class to chose and provide different reproduction mechanisms
    This includes mutation, crossover and merge
    """

    reproduction_rates = {
        ReproductionType.Mutation.value: 0.85,
        ReproductionType.Crossover.value: 0.1,
        ReproductionType.Merge.value: 0.05,
    }
    selection_type: str = "tournament"
    selection_arguments: dict = {}

    def __init__(self, configuration: Optional[Configuration] = None):
        """
        :param configuration: configuration
        """
        super().__init__(configuration=configuration)
        self.mutator = Mutator(configuration)

    def set_configurable(self):
        self.add_configurable_attribute(
            "reproduction_rates",
            "Probabilities to apply mutation/crossover/merge",
            validate=[
                is_valid_on_all_dict_values(is_positive),
                contains_only_given_keys(
                    [
                        ReproductionType.Mutation.value,
                        ReproductionType.Crossover.value,
                        ReproductionType.Merge.value,
                    ]
                ),
            ],
        )
        self.add_configurable_attribute(
            "selection_type", validate=valid_values("tournament")
        )
        self.add_configurable_attribute(
            "selection_arguments",
            "set custom arguments for the selection, "
            "e.g. k and p for tournament_selection",
        )

    def create_networks(
        self,
        population: List[Network],
        fitness_score: List[float],
        amount: int,
    ):
        """
        Reproduce networks
        based on the given parameters and configuration values

        :param population:
        :param fitness_score:
        :param amount:
        :return:
        """
        new_networks = []
        operations = []
        while len(new_networks) < amount:
            reproduction_type = ReproductionType(
                random_rates(self.reproduction_rates)
            )

            selection = self.selection(population, fitness_score)
            selection_index = population.index(selection)

            if reproduction_type == ReproductionType.Mutation:
                new_network = self.mutator.apply_mutations(selection)
                new_networks.append(new_network)

                operation = Origin(reproduction_type, [selection_index])
                operations.append(operation)
            elif reproduction_type == ReproductionType.Crossover:
                # should use another network for crossover
                selection2 = self.get_different_selection(
                    population, fitness_score, selection_index
                )
                selection2_index = population.index(selection2)

                crossovers = list(crossover(selection, selection2))
                new_networks.extend(crossovers)

                operation = Origin(
                    reproduction_type, [selection_index, selection2_index]
                )
                operations.extend([operation] * 2)
            elif reproduction_type == ReproductionType.Merge:
                selection2 = self.get_different_selection(
                    population, fitness_score, selection_index
                )
                selection2_index = population.index(selection2)

                merge = merge_two_networks(selection, selection2)
                new_networks.append(merge)

                operation = Origin(
                    reproduction_type, [selection_index, selection2_index]
                )
                operations.append(operation)
            else:
                raise NotImplementedError(
                    "This type of reproduction operation is not supported"
                )

        # operations, can extend list by more than 1 -> assure correct size
        return new_networks[:amount], operations[:amount]

    def get_different_selection(
        self,
        population: List[Network],
        fitness_score: List[float],
        exclude: int,
    ):
        """
        Apply selection, exclude one key
        :param population:
        :param fitness_score:
        :param exclude: give index of value to exclude from selection
        :return:
        """
        # make sure, to not include two times the same networks
        population_without_selection = (
            population[:exclude] + population[(exclude + 1) :]
        )
        fitness_score_without_selection = (
            fitness_score[:exclude] + fitness_score[(exclude + 1) :]
        )
        return self.selection(
            population_without_selection,
            fitness_score_without_selection,
        )

    def selection(self, population: List[Network], fitness_score: List[float]):
        """
        Apply the specified selection method

        :param population:
        :param fitness_score:
        :return:
        """
        if self.selection_type == "tournament":
            return tournament_selection(
                population, fitness_score, **self.selection_arguments
            )

        raise NotImplementedError("This selection type is not implemented yet")
