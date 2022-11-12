"""
Provide a neuron class for the network
"""
from network.dynamic_parameter import DynamicParameter
from utility.random import get_int_with_exclude

MAX_UID = 1000


class Neuron(DynamicParameter):
    """
    Configurable neuron
    Has unique uid and variable delay, leak and threshold
    """

    # provide default value, to not be written to parameters
    uid: int = 0

    def __init__(self, uid: int, **kwargs):
        super().__init__(**kwargs)
        self.uid = uid

    @classmethod
    def with_random_id(cls, uid=None, exclude_ids=None, parameters=None):
        """
        Get a neuron with random initialization and the given uid

        :param exclude_ids: exclude these ids, if no uid is given
        :param parameters: parameter values of neuron
        :param uid: fix neuron, if none, generates an uid
        :return:
        """
        if uid is None:
            uid = get_int_with_exclude(exclude=exclude_ids, max=MAX_UID)
        if parameters is None:
            parameters = {}

        return cls(uid=uid, **parameters)
