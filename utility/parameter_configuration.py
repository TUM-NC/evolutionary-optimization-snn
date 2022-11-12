import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Union

from utility.random import random_rates
from utility.validation import is_int


@dataclass
class ParameterConfiguration:
    type: Literal["random_int", "random_bool", "random_choice", "fixed"]


@dataclass
class RandomInt(ParameterConfiguration):
    min: int
    max: int


@dataclass
class RandomBool(ParameterConfiguration):
    pass


@dataclass
class RandomChoice(ParameterConfiguration):
    values: List[Any]


@dataclass
class Fixed(ParameterConfiguration):
    value: Any


@dataclass
class RandomRates(ParameterConfiguration):
    rates: Dict[Any, float]


def get_mutable_parameters(parameters):
    """
    From a parameter configuration, return all mutable attributes
    :param parameters:
    :return:
    """
    mutable = ["random_int", "random_bool", "random_choice", "random_rates"]
    return [
        (k, v)
        for k, v in sorted(parameters.items())
        if isinstance(v, dict) and "type" in v and v["type"] in mutable
    ]


def init_parameter_values(parameters: Dict[str, ParameterConfiguration]):
    """
    map all given parameters to an initial value

    :param parameters:
    :return:
    """
    return {k: parameter_to_value(v) for k, v in sorted(parameters.items())}


def parameter_to_value(parameter: Union[Dict[str, Any], Any]):
    """
    Convert a single parameter to a value
    :param parameter:
    :return:
    """
    c = convert_dict_to_class(parameter)

    if isinstance(c, RandomInt):
        return random.randint(c.min, c.max)
    if isinstance(c, RandomBool):
        return bool(random.getrandbits(1))
    if isinstance(c, Fixed):
        return c.value
    if isinstance(c, RandomChoice):
        return random.choice(c.values)
    if isinstance(c, RandomRates):
        return random_rates(c.rates)

    raise RuntimeError("given type is not implemented")


def convert_dict_to_class(
    parameter: Union[Dict[str, Any], Any]
) -> Union[ParameterConfiguration, bool]:
    """
    Convert a dict to the correct parameter configuration class
    Returns false and shows a warning, if this is not possible

    If no dict is given, assume it is a fixed value

    :param parameter:
    :return:
    """
    # if parameter is no dict, try to apply the fixed value
    if not isinstance(parameter, dict) or "type" not in parameter:
        parameter = {"type": "fixed", "value": parameter}

    try:
        if parameter["type"] == "random_int":
            return RandomInt(**parameter)
        if parameter["type"] == "random_bool":
            return RandomBool(**parameter)
        if parameter["type"] == "fixed":
            return Fixed(**parameter)
        if parameter["type"] == "random_choice":
            return RandomChoice(**parameter)
        if parameter["type"] == "random_rates":
            return RandomRates(**parameter)
    except TypeError:
        warnings.warn("Could not convert the given parameter configuration")

    return False


def is_valid_parameter_configuration(parameter: Dict[str, Any]):
    """
    Check whether a parameter configuration is syntactically correct

    :param parameter:
    :return:
    """
    c = convert_dict_to_class(parameter)
    if c is False:
        return False

    if isinstance(c, RandomInt):
        return is_int(c.min) and is_int(c.max) and c.min <= c.max

    if isinstance(c, RandomBool):
        return True

    if isinstance(c, Fixed):
        return True

    if isinstance(c, RandomChoice):
        return isinstance(c.values, List)

    if isinstance(c, RandomRates):
        return isinstance(c.rates, Dict) and all(
            [isinstance(v, float) for v in c.rates.values()]
        )

    return False


def is_valid_parameter_value(matching_function: Callable[[Any], bool]):
    """
    Applies the checks on the output values of the configuration
    Note: for random int: only min and max are checked

    :param matching_function:
    :return:
    """

    def valid_on_values(parameter: Dict[str, Any]):
        if not is_valid_parameter_configuration(parameter):
            return False

        c = convert_dict_to_class(parameter)
        if isinstance(c, RandomInt):
            return matching_function(c.min) and matching_function(c.max)

        if isinstance(c, RandomBool):
            return matching_function(True) and matching_function(False)

        if isinstance(c, Fixed):
            return matching_function(c.value)

        if isinstance(c, RandomChoice):
            return all([matching_function(v) for v in c.values])

        if isinstance(c, RandomRates):
            return all([matching_function(v) for v in c.rates.keys()])

        return False

    return valid_on_values


def check_parameter_values_on_specification(
    specification: Dict[str, Callable], parameter_configuration: Dict[str, Any]
):
    """
    Check parameter configuration on specification

    :param specification:
    :param parameter_configuration:
    :return:
    """
    for parameter, check in specification.items():
        if parameter in parameter_configuration:
            if not is_valid_parameter_value(check)(
                parameter_configuration[parameter]
            ):
                return False
        else:
            is_optional = check(None)
            if not is_optional:
                return False

    additional_parameters = set(parameter_configuration.keys()) - set(
        specification.keys()
    )
    if len(additional_parameters) > 0:
        warnings.warn(
            f"The keys {additional_parameters} "
            "are not specified in available parameters"
        )

    return True
