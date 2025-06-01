from counterfact.utils.export import *
from counterfact.utils.subsets import *


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
