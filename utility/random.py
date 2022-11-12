"""
Provide additional random methods
"""
import random
from typing import Dict, TypeVar

T = TypeVar("T")  # Typehint for same return type as in parameter


def random_rates(rates: Dict[T, float]) -> T:
    """
    Get a key based on the rates, defined as value

    :param rates:
    :return: the key
    """
    keys = sorted(list(rates.keys()))
    weights = [rates[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def get_int_with_exclude(exclude=None, max=1000):
    """
    Get an int between 1 and max without those specified in exclude

    :param exclude:
    :param max:
    :return:
    """
    if exclude is None:
        exclude = set()

    # should use sets, as it's much faster than lists
    return random.choice(sorted(set(range(1, max)) - set(exclude)))
