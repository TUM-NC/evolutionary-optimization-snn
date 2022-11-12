"""
Provide an interface for all classes that accept configurations
"""
from typing import Any, Callable, List, Optional, Union

from utility.configuration import Configuration


class Configurable:
    """
    Abstract class for classes which should support a configuration
    """

    _configuration = {}

    def __init__(self, configuration: Optional[Configuration] = None):
        if configuration is None:
            configuration = Configuration()
        self._configuration = configuration

        self.set_configurable()

    def set_configurable(self):
        """
        Set config values for this class

        :return:
        """
        raise NotImplementedError("Please Implement this method")

    def add_configurable_attribute(
        self,
        name: str,
        comment: Optional[str] = None,
        validate: Union[
            None, Callable[[Any], bool], List[Callable[[Any], bool]]
        ] = None,
    ):
        """
        Add a local variable to be configurable#

        :param validate: parameter to validate the value of the config
        :param name: attribute name of the variable
        :param comment: optional comment, to describe value
        """
        default_value = getattr(self, name)
        self._configuration.set_default_config_value(
            name,
            default_value=default_value,
            comment=comment,
            validate=validate,
        )
        value = self._configuration.get_config(name)
        setattr(self, name, value)
