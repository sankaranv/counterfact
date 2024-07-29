from itertools import chain, combinations


def powerset(iterable, reverse=False):
    s = list(iterable)
    if reverse:
        return chain.from_iterable(combinations(s, r) for r in range(len(s) - 1, 0, -1))
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


def get_all_subevents(event, reverse=False):
    vars = event.keys()
    event_combinations = list(powerset(vars, reverse))
    subevents = [{k: event[k] for k in subevent} for subevent in event_combinations]
    return subevents


def add_info(info, updates):
    """
    Add elements of second dictionary to first dictionary
    If any keys from the updates dict are already in info, the values are overwritten
    :param info: dict
    :param updates: dict
    :return: combined dict
    """
    for k, v in updates.items():
        info[k] = v
    return info
