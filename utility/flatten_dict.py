from collections.abc import MutableMapping


def flatten_dict(dictionary, parent_key=False, separator="."):
    """
    Turn a nested dictionary into a flattened dictionary
    source: https://stackoverflow.com/a/62186053

    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items = []
    for key, value in dictionary.items():
        new_key = (
            str(parent_key) + separator + str(key) if parent_key else str(key)
        )
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
