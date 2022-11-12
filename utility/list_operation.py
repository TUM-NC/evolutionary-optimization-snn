from typing import List, Set


def remove_multiple_indices(l: List, indices: Set):
    """
    Remove multiple indices from a list

    :param l:
    :param indices:
    :return:
    """
    return [v for i, v in enumerate(l) if i not in indices]


def flat_list(li: List):
    """
    Make a list flat in the same order, as displayed
    :param li:
    :return:
    """
    if len(li) == 0:
        return []

    new_list = []
    for element in li:
        if isinstance(element, list):
            flat_element = flat_list(element)
            new_list.extend(flat_element)
        else:
            new_list.append(element)
    return new_list


def count_occurrences(l: list):
    """
    count occurencecs of keys in list
    :param l:
    :return:
    """
    keys = set(l)
    return {k: l.count(k) for k in keys}


def get_depths(l: list):
    """
    Get the depth levels of the list

    :param l:
    :return:
    """
    if not isinstance(l, list) or len(l) == 0:
        return {}

    depths = {0: len(l)}
    for element in l:
        sub_depths = get_depths(element)
        for d, v in sub_depths.items():
            if d + 1 not in depths:
                depths[d + 1] = 0
            depths[d + 1] += v
    return depths
