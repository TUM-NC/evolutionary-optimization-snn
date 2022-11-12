"""
Abstract class for encoders
"""


class Encoder:
    """
    Abstract class for encoders
    """

    number_of_neurons: int

    def __init__(self, number_of_neurons: int):
        self.number_of_neurons = number_of_neurons
