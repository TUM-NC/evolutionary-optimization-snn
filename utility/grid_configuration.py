from typing import List, Optional

import numpy as np
from ruamel.yaml import YAML

from utility.configuration import Configuration
from utility.flatten_dict import flatten_dict


class GridConfiguration:
    base_config = {}
    grid_config: Optional[List[dict]] = []
    pool_size: Optional[int] = None
    save: Optional[str] = None

    def __init__(self, base_config=None, grid_config: Optional = None):
        if base_config is None:
            base_config = {}
        if grid_config is None:
            grid_config = []

        self.base_config = base_config
        self.grid_config = self._reformat_grid_config(grid_config)

        if isinstance(grid_config, dict):
            if "pool_size" in grid_config:
                self.pool_size = grid_config["pool_size"]
            if "save" in grid_config:
                self.save = grid_config["save"]

    @classmethod
    def from_yaml(cls, base_file, grid_file):
        """
        Initiate a grid configuration from the given yaml files

        :param base_file: base configuration
        :param grid_file: grid configuration
        :return:
        """
        yaml = YAML(typ="safe")

        with open(base_file, "r") as f:
            base = yaml.load(f)
        with open(grid_file, "r") as f:
            grid_config = yaml.load(f)

        return cls(base, grid_config)

    def to_yaml(self, base_path=None, grid_path=None):
        """
        Save whole grid configuration to given paths
        :param base_path:
        :param grid_path:
        :return:
        """
        yaml = YAML(typ="safe")

        if base_path is not None:
            with open(base_path, "w") as f:
                data = self.base_config
                yaml.dump(data, f)

        if grid_path is not None:
            with open(grid_path, "w") as f:
                data = {
                    "pool_size": self.pool_size,
                    "save": self.save,
                    "options": self.grid_config,
                }
                yaml.dump(data, f)

    @staticmethod
    def _reformat_grid_config(grid_config):
        """
        Reformat the grid config to a two level array
        Allows to use some shortcuts, for easier configuration writing

        :param grid_config:
        :return:
        """
        # get options = first level
        if isinstance(grid_config, list):
            options = grid_config
        elif "options" in grid_config:
            options = grid_config["options"]
        else:
            raise RuntimeError("Format of options is not supported")

        # reformat alternatives
        output = []
        for option in options:
            if isinstance(option, list):
                reformatted_option = option
            elif "alternatives" in option and isinstance(
                option["alternatives"], list
            ):
                reformatted_option = option["alternatives"]
            elif isinstance(option, dict):
                reformatted_option = [option]
            else:
                raise RuntimeError("Format of alternatives is not supported")

            output.append(reformatted_option)

        if len(output) == 0:
            raise RuntimeError("Iterating through no options makes no sense")

        return output

    def _get_max_alternatives(self):
        """
        Returns a list with amount of alternatives per option

        :return:
        """
        return [len(option) for option in self.grid_config]

    def get_amount_alternatives(self) -> int:
        """
        Get total count of combinations of grid search
        :return:
        """
        return int(np.prod(self._get_max_alternatives()))

    @staticmethod
    def overwrite_dict_first_dimension(base: dict, overwrite: dict):
        """
        Overwrite values of dict in first dimension

        :param base:
        :param overwrite:
        :return:
        """
        for key, value in overwrite.items():
            base[key] = value

        return base

    def get_option(self, index: int) -> Configuration:
        """
        Get a configuration for the given index

        :param index:
        :return:
        """
        option_selection = self._get_option_selection(index)

        config = self.base_config.copy()

        for i, options in enumerate(self.grid_config):
            selected_option = options[option_selection[i]]
            self.overwrite_dict_first_dimension(config, selected_option)

        return Configuration(config)

    def _get_option_selection(self, index: int) -> List[int]:
        """
        Returns an array corresponding to the configuration
        Counts the options per alternative and uses a custom counting

        raw_values: Index of the configuration to use
        Examples:
            [2 2 2] with index 0 -> [0 0 0], 1 -> [0 0 1], 2 -> [0 1 0]
        """
        max_alternatives = self._get_max_alternatives()

        no_options = len(max_alternatives)  # no of different 1. level options
        option = np.zeros(no_options, dtype=int)

        # go through the options like it is a custom numeric format
        for i in reversed(range(no_options)):
            option[i] = index % max_alternatives[i]
            index /= max_alternatives[i]

        return list(option)

    def get_option_headers(self):
        """
        Get a sorted list of option headers for all options
        Flattens all keys

        :return:
        """
        labels = set()

        for option in self.grid_config:
            for alternative in option:
                alternative_labels = flatten_dict(alternative).keys()
                labels.update(alternative_labels)

        return sorted(list(labels))
