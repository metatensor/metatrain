import pytest

from metatrain.utils.errors import ArchitectureError


def test_architecture_error():
    match = "The error below most likely originates from an architecture"
    with pytest.raises(ArchitectureError, match=match):
        try:
            raise ValueError("An example error from the architecture")
        except Exception as e:
            raise ArchitectureError(e)
