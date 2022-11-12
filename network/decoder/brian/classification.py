from typing import List, Tuple

from network.decoder.brian.decoder import BrianDecoder
from network.decoder.classification import ClassificationDecoder


class ClassificationBrianDecoder(ClassificationDecoder, BrianDecoder):
    """
    Implementation of a classification decoder in brian
    """

    def get_value(self) -> Tuple[int, List[int]]:
        """
        Get the assumed classification for an output spike pattern
        Return the class with the most spikes in the output neuron
        When there are no spikes, returns -1
        If there are multiple with the same amount, -1 is returned

        :return:
        """
        if self.spikes is None:
            raise RuntimeError("Spikes have to be set, before getting a value")

        spikes = self.spikes
        spikes_per_class = [len(s) for s in spikes]

        classification = self.get_best_class(spikes_per_class)

        self.spikes = None  # reset after value read

        return classification, spikes_per_class
