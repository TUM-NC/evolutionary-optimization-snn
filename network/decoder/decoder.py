"""
Decoder class to convert values into spikes
"""


class Decoder:
    """
    Abstract decoder class
    """

    number_of_neurons: int

    def __init__(self, number_of_neurons):
        self.number_of_neurons = number_of_neurons

    def get_value(self):
        """
        Abstract function, to get a (decoded) value
        :return:
        """
        raise NotImplementedError("Please Implement this method")
