import collections.abc
import json

from metatomic.torch import ModelMetadata


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        elif isinstance(v, list):
            if k in d:
                for item in v:
                    if item not in d[k]:
                        d[k].append(item)
            else:
                d[k] = v
        else:
            d[k] = v
    return d


def merge_metadata(self: ModelMetadata, other: ModelMetadata) -> ModelMetadata:
    """Append ``references`` to an existing ModelMetadata object.

    :param self: The metadata object to be updated.
    :param other: The metadata object to merged to self.
    """

    self_dict = json.loads(self._get_method("__getstate__")())
    other_dict = json.loads(other._get_method("__getstate__")())

    self_dict = update(self_dict, other_dict)
    self_dict.pop("class")

    new_metadata = ModelMetadata(**self_dict)

    return new_metadata
