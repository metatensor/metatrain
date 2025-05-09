import json

from metatensor.torch.atomistic import ModelMetadata


def update_metadata_references(self: ModelMetadata, other: ModelMetadata) -> None:
    """Append ``references`` to an existing ModelMetadata object.

    :param self: The metadata object to be appeneded.
    :param other: The metadata object to update with.
    """

    self_dict = json.loads(self._get_method("__getstate__")())
    other_dict = json.loads(other._get_method("__getstate__")())
    self_dict["references"].update(other_dict["references"])

    self._get_method("__setstate__")(json.dumps(self_dict))
