from pathlib import Path

import pytest

from metatrain.utils.extenders.helpers import (
    find_all_extenders,
)


TOML_AVAILABLE = True
try:
    import tomllib
except ModuleNotFoundError:
    TOML_AVAILABLE = False


def test_pyproject_toml_extras():
    """Test that all extenders are included in pyproject.toml extras."""
    if not TOML_AVAILABLE:
        pytest.skip("tomllib is not available in this Python version")

    all_extenders = find_all_extenders()

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        toml = tomllib.load(f)

    for extender in all_extenders:
        extender_name = extender.split(".")[-1].replace("_", "-")
        if f"extender-{extender_name}" not in toml["project"]["optional-dependencies"]:
            raise ValueError(
                f"Extender '{extender}' is not included in pyproject.toml extras."
                f" Please add it to the list of extras in the file."
            )
