"""
Provide a class for storing and loading configuration
"""
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ruamel.yaml import YAML, CommentedMap


class Configuration:
    """
    Provide a class for storing and loading configuration
    """

    _configuration: Dict[str, Any]
    # contains default configurations
    # first param is default value, second can be a comment
    _configuration_default: Dict[
        str, Tuple[Any, Optional[str], List[Callable[[Any], bool]]]
    ]

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = {}
        self._configuration = config
        self._configuration_default = {}

    @classmethod
    def from_yaml(cls, file):
        """
        Create a configuration from the given yaml file

        :param file:
        :return:
        """
        yaml = YAML(typ="safe")
        with open(file, "r") as f:
            data = yaml.load(f)
        return cls(data)

    def set_default_config_value(
        self,
        name: str,
        default_value,
        comment: Optional[str] = None,
        validate: Union[
            None, Callable[[Any], bool], List[Callable[[Any], bool]]
        ] = None,
    ):
        """

        :param validate:
        :param name: the configuration name
        :param default_value: default value to fallback
        :param comment: optional comment for generation of default config
        """
        # always store functions as list, to make it easier for later access
        if validate is None:
            validate = []
        if not isinstance(validate, List):
            validate = [validate]

        new_value = (default_value, comment, validate)
        if name in self._configuration_default:
            old_value = self._configuration_default[name]
            if old_value[0] != new_value[0]:
                warnings.warn(
                    "Overwrite default config with different values. "
                    "May have unintended behaviour"
                )
        self._configuration_default[name] = new_value

    def get_unused_keys(self):
        """
        Return all keys of the configuration, that are not in use
        :return:
        """
        return set(self._configuration.keys()).difference(
            self._configuration_default.keys()
        )

    def validate_config(self):
        unused_keys = self.get_unused_keys()
        if len(unused_keys) > 0:
            warnings.warn(
                "There are some keys in you config that are not used:"
                + str(unused_keys)
            )

        valid = True
        for key, value in self._configuration.items():
            # skip unused keys, as they have no default config
            if key in unused_keys:
                continue

            _, _, validates = self._configuration_default[key]
            for v in validates:
                passed = v(value)
                if not passed:
                    valid = False
                    warnings.warn(
                        f"The key '{key}' does not match the validation: {v}"
                    )

        return valid

    def get_config(self, name: str):
        """
        Get the config value for a given key
        Cannot get values, for not specified parameters -> raises an exception

        :param name:
        :return: value stored in config or default_config
        """
        if name not in self._configuration_default:
            raise RuntimeError(
                "Should specify default values, before getting custom values"
            )

        if name in self._configuration:
            return self._configuration[name]

        return self._configuration_default[name][0]

    def save_yaml(self, file):
        """
        Save the config as a yaml at the specified destination

        :param file: can be either a file or a filename
        :return:
        """
        insert = CommentedMap()

        for name in self._configuration_default.keys():
            insert[name] = self.get_config(name)
            comment = self._configuration_default[name][1]
            if comment is not None:
                insert.yaml_add_eol_comment(comment, name, column=0)

        yaml = YAML()

        # is string for file is given, interpret as filename
        if isinstance(file, str):
            with open(file, "w") as f:
                yaml.dump(insert, f)
        else:
            yaml.dump(insert, file)

    def get_config_dict(self):
        """
        Return the dict for configurations

        :return:
        """
        return self._configuration
