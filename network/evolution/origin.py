from enum import Enum
from typing import List, NamedTuple


class ReproductionType(str, Enum):
    Mutation = "mutation"
    Crossover = "crossover"
    Merge = "merge"
    Random = "random"
    Same = "Same"


class Origin(NamedTuple):
    """
    Store the stats for a single epoch
    """

    reproduction_type: ReproductionType
    associated_networks: List[int]
