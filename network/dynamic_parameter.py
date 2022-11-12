"""
Provide a class for neurons and synapses
"""
from typing import Any, Dict


class DynamicParameter:
    parameters: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        self.parameters = kwargs

    def __getattr__(self, name):
        """
        Should get values of defined parameters

        :param name:
        :return:
        """
        if name in self.parameters:
            return self.parameters[name]
        return super().__getattribute__(name)

    def __setattr__(self, key, value):
        # should not be possible, to add new attribute afterwards
        if hasattr(self, key) and key not in self.parameters:
            return super().__setattr__(key, value)
        self.parameters[key] = value

    def get_defined_parameters(self):
        """
        Return all defined parameters

        :return:
        """
        return self.parameters.keys()
