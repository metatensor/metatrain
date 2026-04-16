from pathlib import Path

import pytest
from omegaconf import OmegaConf

from metatrain.utils.architectures import find_all_architectures


TOML_AVAILABLE = True
try:
    import tomllib
except ModuleNotFoundError:
    TOML_AVAILABLE = False


def test_codeowners_there():
    """Test that an architeture has owners specified in CODEOWNERS."""
    codeowners_path = Path(__file__).parent.parent / "CODEOWNERS"

    # NOTE: For simplicity, this is a very basic parser. If there is a more complex
    # architecture setup, we can change this test.

    architectures = find_all_architectures()
    with open(codeowners_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            path, *owners = line.split()

            # Fortunately, the architecture names match the directory names.
            path = path[3:]  # Remove leading "**/"
            if not (path in architectures or "experimental." + path in architectures):
                raise ValueError(
                    "Found architecture path in CODEOWNERS that does not match any "
                    "architecture: %s. Was it removed but you forgot to update the "
                    "CODEOWNERS file?" % path
                )
            if path in architectures:
                architectures.remove(path)
            else:
                architectures.remove("experimental." + path)

            assert len(owners) > 0, (
                f"No owners specified for architecture path '{path}' in CODEOWNERS. "
                "Please add at least one owner."
            )

    for architecture in architectures:
        raise ValueError(
            f"Architecture '{architecture}' does not have an entry in CODEOWNERS. "
            "Please add an entry for it with at least one owner."
        )


def test_architecture_in_codecov():
    """Test that all architectures are in codecov.yml for coverage reporting."""
    all_arches = find_all_architectures()

    # Parse codecov.yml
    codecov_path = Path(__file__).parent.parent / ".codecov.yml"
    conf = OmegaConf.load(codecov_path)

    components = conf["component_management"]["individual_components"]
    for arch in all_arches:
        arch_name = arch.split(".")[-1]
        for comp in components:
            if comp["component_id"].lower() == arch_name.lower():
                break
        else:
            raise ValueError(
                f"Architecture '{arch}' is not included in .codecov.yml."
                f" Please add it to the list of components in the file."
            )


def test_architecture_in_tox():
    """Test that all architectures are in tox.ini for testing."""
    all_arches = find_all_architectures()

    tox_ini = Path(__file__).parent.parent / "tox.ini"
    with open(tox_ini, "r") as f:
        ini_content = f.readlines()

    envlist = ini_content[
        ini_content.index("envlist =\n") + 1 : ini_content.index("[testenv]\n")
    ]
    envlist = [line.strip() for line in envlist if line.strip()]

    for arch in all_arches:
        arch_tests = f"{arch.split('.')[-1].replace('_', '-')}-tests"
        if arch_tests not in envlist:
            raise ValueError(
                f"Architecture tests for '{arch}' are not included in tox.ini"
                f" of tox.ini. Please add a {arch_tests} environment and make"
                " sure that the architecture tests are ran."
            )


def test_pyproject_toml_extras():
    """Test that all architectures are included in pyproject.toml extras."""
    if not TOML_AVAILABLE:
        pytest.skip("tomllib is not available in this Python version")

    all_arches = find_all_architectures()

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        toml = tomllib.load(f)

    for arch in all_arches:
        arch_name = arch.split(".")[-1].replace("_", "-")
        if arch_name not in toml["project"]["optional-dependencies"]:
            raise ValueError(
                f"Architecture '{arch}' is not included in pyproject.toml extras."
                f" Please add it to the list of extras in the file."
            )
