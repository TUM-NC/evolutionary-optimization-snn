"""
Script to generate configuration yaml file with default configuration
"""
import argparse

# hack: include parent directory for imports
import sys

sys.path.append("..")

# required, to make configurable classes available to this script
import experiment.experiment_selection  # noqa: F401,E402
import network.evolution.framework  # noqa: F401,E402
from utility.configurable import Configurable  # noqa: E402
from utility.configuration import Configuration  # noqa: E402


class DummyConfigurable(Configurable):
    """
    Implementation of the configurable abstract class,
    for setting default values to
    """

    def set_configurable(self):
        pass


def init_default_configuration():
    """
    Should initiate the default configuration
    Respects all configurable objects

    :return:
    """
    configuration = Configuration()
    dummy_configurable = DummyConfigurable(configuration)

    configurables: list[Configurable] = get_all_subclasses(Configurable)
    for conf in configurables:
        """to not care for additional constructor parameters:
        get all attributes from the class,
        and apply them to our dummy configurable object"""
        class_attributes = [i for i in conf.__dict__.keys() if i[:1] != "_"]
        for attribute in class_attributes:
            setattr(dummy_configurable, attribute, conf.__dict__[attribute])
        # set configurable should set the default configurations for this class
        conf.set_configurable(dummy_configurable)

    return configuration


def get_all_subclasses(cls):
    """
    Return all subclass from the given class

    :param cls:
    :return:
    """
    return list(cls.__subclasses__())


def main():
    """
    Allows to generate a configuration yaml to the specified file

    :return:
    """
    parser = argparse.ArgumentParser(description="Generate a configuration")
    parser.add_argument(
        "--file",
        type=str,
        nargs="?",
        help="File to save the configuration to",
        default="config.yaml",
    )
    args = parser.parse_args()

    configuration = init_default_configuration()
    with open(args.file, "w", encoding="utf-8") as file:
        configuration.save_yaml(file)


if __name__ == "__main__":
    main()
