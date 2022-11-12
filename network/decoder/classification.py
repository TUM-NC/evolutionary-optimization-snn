from typing import List, Tuple

import numpy as np

from network.decoder.decoder import Decoder


class ClassificationDecoder(Decoder):
    """
    Interface for a classification decoder
    """

    def __init__(self, classes: int):
        super().__init__(number_of_neurons=classes)

    def get_value(self) -> Tuple[int, ...]:
        """

        :return: classification output as first value in tuple,
        additional information in following entries
        """
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def get_best_class(spikes_per_class: List[int]):
        """
        Get the class, with the most spikes
        Returns -1, if not unique or no spikes

        :param spikes_per_class:
        :return:
        """
        max_spikes = max(spikes_per_class)
        if max_spikes == 0:
            return -1

        classification = int(np.argmax(spikes_per_class))
        # check, whether max value exists multiple times
        remaining_values = (
            spikes_per_class[:classification]
            + spikes_per_class[classification + 1 :]
        )
        if max(remaining_values) == max_spikes:
            # no class, if max class is not unique
            return -1
        return classification
