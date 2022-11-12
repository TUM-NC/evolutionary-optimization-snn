"""
Provide function for several selection methods
"""
import random
from typing import List, TypeVar

T = TypeVar("T")  # Typehint for same return type as in parameter


def _zip_population_scores(population: List[T], scores: List[float]):
    return list(zip(scores, population))


def tournament_selection(
    population: List[T], fitness_scores: List[float], k=10, p=1
) -> T:
    """
    Get an element from the population using tournament selection

    :param population: list of elements
    :param fitness_scores: list of fitness scores, in same order as population
    :param k: amount of networks to choose randomly for tournament,
            if larger than population, will be set to population
    :param p: probability, that best network is chosen
    :return: element which was chosen through tournament selection
    """
    zipped = _zip_population_scores(
        population=population, scores=fitness_scores
    )
    # set to population size, if too large
    k = min(k, len(zipped))
    selected_tournament = random.sample(
        zipped, k=k
    )  # use sample, to not get any duplicates
    sorted_tournament = sorted(
        selected_tournament, key=lambda a: a[0], reverse=True
    )

    for agent in sorted_tournament:
        p_choose_best = random.uniform(0, 1)
        if p >= p_choose_best:
            return agent[1]  # returns only the network part

    # return first network, if no agent is chosen through probability
    return sorted_tournament[0][1]


def best(population: List[T], fitness_scores: List[float], n=1) -> List[T]:
    """
    :param population: agents which were evaluated
    :param fitness_scores: list of fitness scores in same order as population
    :param n: number of elements to get
    :return: list of n best elements from population (sorted descending)
    """
    zipped = _zip_population_scores(
        population=population, scores=fitness_scores
    )
    sorted_population = sorted(
        zipped, key=lambda agent: agent[0], reverse=True
    )
    best_zipped = sorted_population[0:n]
    return [
        x[1] for x in best_zipped
    ]  # return only network, without fitness score
