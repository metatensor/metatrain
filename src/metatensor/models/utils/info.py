from typing import Dict, Tuple


def update_aggregated_info(
    aggregated_info: Dict[str, Tuple[float, int]],
    new_info: Dict[str, Tuple[float, int]],
):
    """
    Update the aggregated information dictionary with new information.

    For now, the new_info must be a dictionary of tuples, where the first
    element is a sum of squared errors and the second element is the
    number of samples.

    If a key is present in both dictionaries, the values are added.
    If a key is present in ``new_info`` but not ``aggregated_info``,
    it is simply copied.

    :param aggregated_info: The aggregated information dictionary.
    :param new_info: The new information dictionary.

    :returns: The updated aggregated information dictionary.
    """

    for key, value in new_info.items():
        if key in aggregated_info:
            aggregated_info[key] = (
                aggregated_info[key][0] + value[0],
                aggregated_info[key][1] + value[1],
            )
        else:
            aggregated_info[key] = value

    return aggregated_info


def finalize_aggregated_info(aggregated_info):
    """
    Finalize the aggregated information dictionaryby calculating RMSEs.

    For now, the aggregated_info must be a dictionary of tuples, where the first
    element is a sum of squared errors and the second element is the
    number of samples.

    :param aggregated_info: The aggregated information dictionary.

    :returns: The finalized aggregated information dictionary.
    """

    finalized_info = {}
    for key, value in aggregated_info.items():
        finalized_info[key] = (value[0] / value[1]) ** 0.5

    return finalized_info
