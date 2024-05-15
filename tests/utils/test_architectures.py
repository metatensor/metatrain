from pathlib import Path

import pytest

from metatensor.models import PACKAGE_ROOT
from metatensor.models.utils.architectures import (
    check_architecture_name,
    find_all_architectures,
    get_architecture_name,
    get_architecture_path,
    get_default_hypers,
)


def test_find_all_architectures():
    all_arches = find_all_architectures()
    assert "experimental.soap_bpnn" in all_arches
    assert "experimental.pet" in all_arches
    assert "experimental.alchemical_model" in all_arches


def test_get_architecture_path():
    assert (
        get_architecture_path("experimental.soap_bpnn")
        == PACKAGE_ROOT / "experimental" / "soap_bpnn"
    )


@pytest.mark.parametrize("name", find_all_architectures())
def test_get_default_hypers(name):
    """Test that architecture hypers for all arches can be loaded."""
    default_hypers = get_default_hypers(name)
    assert type(default_hypers) is dict


def test_check_architecture_name():
    check_architecture_name("experimental.soap_bpnn")


def test_check_architecture_name_suggest():
    name = "soap-bpnn"
    match = f"Architecture {name!r} is not a valid architecture."
    with pytest.raises(ValueError, match=match):
        check_architecture_name(name)


def test_check_architecture_name_experimental():
    with pytest.raises(
        ValueError, match="experimental architecture with the same name"
    ):
        check_architecture_name("soap_bpnn")


def test_check_architecture_name_deprecated():
    # Create once a deprecated architecture exist
    pass


@pytest.mark.parametrize("path_type", [Path, str])
@pytest.mark.parametrize(
    "path",
    [
        PACKAGE_ROOT / "experimental" / "soap_bpnn",
        PACKAGE_ROOT / "experimental" / "soap_bpnn" / "__init__.py",
        PACKAGE_ROOT / "experimental" / "soap_bpnn" / "default-hypers.yaml",
    ],
)
def test_get_architecture_name(path_type, path):
    assert get_architecture_name(path_type(path)) == "experimental.soap_bpnn"


def test_get_architecture_name_err_no_such_path():
    path = PACKAGE_ROOT / "foo"
    match = f"`path` {str(path)!r} does not exist"
    with pytest.raises(ValueError, match=match):
        get_architecture_name(path)


def test_get_architecture_name_err_no_such_arch():
    path = PACKAGE_ROOT
    match = f"`path` {str(path)!r} does not point to a valid architecture folder"
    with pytest.raises(ValueError, match=match):
        get_architecture_name(path)
