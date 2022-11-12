from typing import Any, Callable, Dict


def is_number(n):
    """
    returns whether a number is an int or float
    :param n:
    :return:
    """
    return is_float(n) or is_int(n)


def is_float(n):
    """
    check if number is a float
    :param n:
    :return:
    """
    return isinstance(n, float)


def is_int(n):
    """
    check if number is an int
    :param n:
    :return:
    """
    # bool evaluates to 0 or 1, but here we want only explicit int types
    return isinstance(n, int) and not is_bool(n)


def is_positive(n):
    """
    a positive number including 0
    :param n:
    :return:
    """
    if not is_number(n):
        return False
    return n >= 0


def is_none(n):
    """
    Only accept None values

    :param n:
    :return:
    """
    return n is None


def greater_than_zero(n):
    """
    a number that is strictly  larger than 0

    :param n:
    :return:
    """
    if not is_number(n):
        return False
    return n > 0


def is_percent(n):
    """
    whether a number is between 0 and 1
    :param n:
    :return:
    """
    return is_number(n) and (0 <= n <= 1)


def is_bool(n):
    """
    whether n is a boolean value
    :param n:
    :return:
    """
    return isinstance(n, bool)


def is_valid_on_all_dict_values(matching_function: Callable[[Any], bool]):
    """
    check whether the matching function is valid on all values
    :param matching_function:
    :return:
    """

    def dict_match(n: Dict):
        for value in n.values():
            if not matching_function(value):
                return False

        return True

    return dict_match


def contains_only_given_keys(keys):
    """
    check whether a dict only contains given keys
    :param keys:
    :return:
    """

    def contain_key(n: Dict):
        for k in n.keys():
            if k not in keys:
                return False
        return True

    return contain_key


def valid_values(values):
    def in_values(a):
        return a in values

    return in_values


def chain_checks(*args: Callable):
    """
    "and" concatenation for multiple checks

    :return:
    """

    def chained(value):
        for c in args:
            if not c(value):
                return False

        return True

    return chained


def any_check(*args: Callable):
    """
    Returns true, if any of the checks is true
    :param args:
    :return:
    """

    def chained(value):
        for c in args:
            if c(value):
                return True

        return False

    return chained
