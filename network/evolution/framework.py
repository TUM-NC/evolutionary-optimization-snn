"""
Provide the framework class, for general access to the evolutionary algorithms
"""
import random
import tempfile
from typing import Dict, List, Optional, Tuple

from experiment.experiment import Experiment
from network.evolution.generator import Generator
from network.evolution.origin import Origin, ReproductionType
from network.evolution.reproduction.reproduction import Reproduction
from network.evolution.selection import best
from network.evolution.stats import Stats
from network.network import Network
from utility.configurable import Configurable
from utility.configuration import Configuration
from utility.validation import (
    greater_than_zero,
    is_bool,
    is_int,
    is_number,
    is_percent,
    is_positive,
)


class Framework(Configurable):
    """
    Class to handle the evolutionary optimization
    """

    experiment: Experiment
    generator: Generator
    reproduction: Reproduction

    temporary_file: Optional[str] = None
    fitness_cache: Dict[Network, float] = {}

    # configurable attributes
    random_factor: float = 0.1
    num_best: int = 2
    population_size: int = 500
    num_generations: int = 50
    fitness_target: float = None
    print_status: bool = True
    save_stat_regularly: bool = False
    cache_evolution: bool = True
    cache_evolution_warm_up: bool = True
    seed: Optional[int] = None

    def __init__(
        self,
        experiment: Experiment,
        configuration: Optional[Configuration] = None,
    ):
        """
        :param experiment: Experiment with a fitness function for evaluation
        :param configuration: configuration
        """
        super().__init__(configuration=configuration)
        if self.seed is not None:
            random.seed(self.seed)
            experiment.set_seed(self.seed)

        self.experiment = experiment
        self.generator = Generator.create_from_experiment(
            experiment=experiment, configuration=configuration
        )
        self.reproduction = Reproduction(configuration=configuration)

        if self.save_stat_regularly:
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            self.temporary_file = tmp_file.name

    def set_configurable(self):
        self.add_configurable_attribute(
            "random_factor",
            "introduce new networks into the population in each epoch "
            "in percent",
            is_percent,
        )
        self.add_configurable_attribute(
            "num_best",
            "keep this amount of best networks for new population",
            validate=[is_int, is_positive],
        )
        self.add_configurable_attribute(
            "population_size",
            "amount of networks in each population",
            validate=[is_int, greater_than_zero],
        )
        self.add_configurable_attribute(
            "num_generations",
            "amount of epochs to simulate",
            validate=[is_int, greater_than_zero],
        )
        self.add_configurable_attribute(
            "fitness_target",
            "Finish simulation early, if target is reached",
            validate=is_number,
        )
        self.add_configurable_attribute(
            "print_status", "Print progress of the evolution", validate=is_bool
        )
        self.add_configurable_attribute(
            "save_stat_regularly",
            "Save the stats after each epoch",
            validate=is_bool,
        )
        self.add_configurable_attribute(
            "cache_evolution",
            "Whether to reevalute existing networks during evolution",
            validate=is_bool,
        )
        self.add_configurable_attribute(
            "cache_evolution_warm_up",
            "Whether to include evaluated stats into the cache",
            validate=is_bool,
        )
        self.add_configurable_attribute(
            "seed",
            "Seed for all network related random operations",
            validate=is_int,
        )

    def get_temporary_file(self):
        """
        Returns the temporary file, to save the stats after each epoch to
        :return:
        """
        return self.temporary_file

    def evolution(self, stats: Optional[Stats] = None) -> Stats:
        """
        Start evolution from stats
        Allows to continue evolution or start from empty stats

        param stats:
        """
        if stats is None:
            stats = Stats()
        else:
            # load all old networks values into the cache, to not reevaluate
            if self.cache_evolution and self.cache_evolution_warm_up:
                self.warm_cache(stats)
        epochs = self.num_generations

        latest_so_far = stats.get_latest_epoch()
        if latest_so_far is None:
            latest_so_far = -1
        start_epoch = latest_so_far + 1

        population, fitness_scores = stats.get_latest_population_fitness()

        if self.print_status and self.temporary_file is not None:
            print(f"Saving stats after each epoch to: {self.temporary_file}")

        for i in range(start_epoch, epochs):
            stats.start_epoch()

            if population is None:
                # first time: generate new population
                population = self.generator.generate_networks(
                    self.population_size
                )
                operations = [
                    Origin(ReproductionType.Random, []) for _ in population
                ]
            else:
                # reproduction mechanisms
                population, operations = self.do_epoch(
                    population, fitness_scores
                )

            fitness_scores = self.evaluate(population)

            stats.add_epoch(population, fitness_scores, operations)
            if self.print_status:
                info = stats.get_epoch_information(i, epochs)
                print(info)

            # save stats after each epoch
            if self.temporary_file is not None:
                stats.to_file(self.temporary_file, indent=None)

            # when specified a target, may abort evolution loop
            if self.fitness_target is not None:
                best_network = best(population, fitness_scores, n=1)
                best_index = population.index(best_network[0])
                best_fitness = fitness_scores[best_index]
                if best_fitness >= self.fitness_target:
                    # break evolution, if target reached
                    break

        return stats

    def do_epoch(
        self, population: List[Network], fitness_score: List[float]
    ) -> Tuple[List[Network], List[Origin]]:
        """
        Apply selection and mutation to the given population
         Return the upcoming generation based on the given parameters

        :param population: list of networks
        :param fitness_score: list of fitness scores in order with population
        :return: new population, operations
        """
        # inject random networks for diversity
        random_count = self.get_random_count()
        random_networks = self.generator.generate_networks(random_count)
        random_operations = [
            Origin(ReproductionType.Random, []) for _ in random_networks
        ]

        # fill up available spots with reproduction
        reproduce_amount = self.get_reproduction_amount()
        (
            reproduce_networks,
            reproduce_operations,
        ) = self.reproduction.create_networks(
            population, fitness_score, reproduce_amount
        )

        best_networks = best(population, fitness_score, n=self.num_best)
        best_operations = [
            Origin(ReproductionType.Same, [population.index(n)])
            for n in best_networks
        ]

        # should be in same order
        networks = best_networks + reproduce_networks + random_networks
        operations = best_operations + reproduce_operations + random_operations

        return networks, operations

    def get_random_count(self):
        """
        Get the amount of random networks to add in each epoch

        :return:
        """
        random_networks_count = int(self.population_size * self.random_factor)
        random_networks_count = min(
            random_networks_count, self.population_size - self.num_best
        )  # random should not exceed population
        return random_networks_count

    def get_reproduction_amount(self):
        """
        Get the amount of networks for reproduction
        :return:
        """
        return max(
            0, self.population_size - self.num_best - self.get_random_count()
        )

    def evaluate(self, population: List[Network]):
        """
        Evaluate the given population on the fitness function

        :param population: list of networks to evaluate
        :return: list of fitness scores in same order as the input population
        """
        # when no caching is specified perform fitness function on all elements
        if not self.cache_evolution:
            return self.experiment.fitness(population)

        # possibility: use network hash -> takes more time than it saves
        non_cached = [n for n in population if n not in self.fitness_cache]
        non_cached_fitness = self.experiment.fitness(non_cached)
        for network, fitness in zip(non_cached, non_cached_fitness):
            self.fitness_cache[network] = fitness

        # here, all elements should be now in the cache
        return [self.fitness_cache[n] for n in population]

    def warm_cache(self, stats: Stats):
        """
        prefill cache with elements from stats
        prevent reevaluation, from old run
        :param stats:
        :return:
        """
        for epoch_stats in stats.data:
            population = epoch_stats["population"]
            fitness_scores = epoch_stats["fitness_scores"]
            for network, fitness in zip(population, fitness_scores):
                self.fitness_cache[network] = fitness
