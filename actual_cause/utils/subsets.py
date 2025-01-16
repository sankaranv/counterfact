from itertools import chain, combinations
import random


def powerset(
    list_or_dict,
    reverse=False,
    include_empty=False,
    include_full=False,
    length=None,
):
    """
    Generate the powerset of a list or dictionary
    :param list_or_dict:
    :param reverse:
    :param include_empty:
    :param include_full:
    :return:
    """
    if list_or_dict.__class__ == dict:
        s = list(list_or_dict.keys())
    elif list_or_dict.__class__ == list:
        s = list_or_dict
    else:
        raise ValueError(
            f"Input must be a list or dictionary, {list_or_dict.__class__} given."
        )

    if length is None:
        lower_bound = 0 if include_empty else 1
        upper_bound = (len(s) + 1) if include_full else len(s)
    elif length < 0 or length > len(s):
        raise ValueError(f"Length must be between 0 and {len(s)}, {length} given.")
    else:
        lower_bound = length
        upper_bound = length

    if reverse:
        return chain.from_iterable(
            combinations(s, r) for r in range(upper_bound - 1, lower_bound - 1, -1)
        )
    return chain.from_iterable(
        combinations(s, r) for r in range(lower_bound, upper_bound + 1)
    )


def get_all_subevents(event, reverse=False, include_empty=False, include_full=False):
    event_combinations = list(powerset(event, reverse, include_empty, include_full))
    subevents = [{k: event[k] for k in subevent} for subevent in event_combinations]
    return subevents


def get_all_subsets(
    var_names: list,
    reverse=False,
    include_empty=False,
    include_full=False,
    shuffle=False,
    shuffle_by_size=False,
):

    if shuffle:
        subsets = list(powerset(var_names, reverse, include_empty, include_full))
        random.shuffle(subsets)
        return subsets
    elif shuffle_by_size:
        # Generate subsets of each size and shuffle them, then put them together
        subsets = []
        lower_bound = 0 if include_empty else 1
        upper_bound = len(var_names) + 1 if include_full else len(var_names)
        for i in range(lower_bound, upper_bound):
            length_wise_subsets = list(powerset(var_names, length=i))
            random.shuffle(length_wise_subsets)
            subsets.extend(length_wise_subsets)
        return subsets
    else:
        subsets = list(powerset(var_names, reverse, include_empty, include_full))
        return subsets
