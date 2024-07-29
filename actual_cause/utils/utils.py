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
