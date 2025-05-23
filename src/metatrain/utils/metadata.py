import json

from metatomic.torch import ModelMetadata


def append_metadata_references(self: ModelMetadata, other: ModelMetadata) -> None:
    """Append ``references`` to an existing ModelMetadata object.

    :param self: The metadata object to be appeneded.
    :param other: The metadata object to update with.
    """

    self_dict = json.loads(self._get_method("__getstate__")())
    other_dict = json.loads(other._get_method("__getstate__")())

    for key, values in other_dict["references"].items():
        if key not in self_dict["references"]:
            self_dict["references"][key] = values
        else:
            for item in values:
                if item not in self_dict["references"][key]:
                    self_dict["references"][key].append(item)

    self._get_method("__setstate__")(json.dumps(self_dict))
