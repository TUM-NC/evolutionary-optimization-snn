"""
Provide a synapse class for a network
"""
from network.dynamic_parameter import DynamicParameter


class Synapse(DynamicParameter):
    """
    Synapse class for 1-1 connections between neurons
    Can have a weight, delay and
    either be inhibitory (exciting = 0) or excitatory (exciting = 1)
    """

    # provide default value, to not be written to parameters
    connect_from: int = 0  # uid of pre-synaptic neuron
    connect_to: int = 0  # uid of post-synaptic neuron

    def __init__(self, connect_from: int, connect_to: int, **kwargs):
        super().__init__(**kwargs)
        self.connect_from = connect_from
        self.connect_to = connect_to
