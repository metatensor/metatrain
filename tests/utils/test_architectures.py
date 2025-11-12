import importlib
from pathlib import Path

import pytest

from metatrain import PACKAGE_ROOT
from metatrain.utils.architectures import (
    check_architecture_name,
    check_architecture_options,
    find_all_architectures,
    get_architecture_name,
    get_architecture_path,
    get_default_hypers,
    import_architecture,
)
from metatrain.utils.pydantic import MetatrainValidationError


def is_None(*args, **kwargs) -> None:
    return None


def test_find_all_architectures():
    all_arches = find_all_architectures()

    assert len(all_arches) == 6

    assert "gap" in all_arches
    assert "pet" in all_arches
    assert "soap_bpnn" in all_arches
    assert "deprecated.nanopet" in all_arches
    assert "experimental.flashmd" in all_arches
    assert "llpr" in all_arches


def test_get_architecture_path():
    assert get_architecture_path("soap_bpnn") == PACKAGE_ROOT / "soap_bpnn"


def test_get_default_hypers():
    """Test that architecture default hypers can be loaded.

    We use soap_bpnn as the test architecture to see if the function works.
    Other architectures might have dependencies, and therefore loading their
    default hypers could fail. The loading of default hypers should be
    tested in the tests of each architecture.
    """
    name = "soap_bpnn"
    default_hypers = get_default_hypers(name)
    assert type(default_hypers) is dict
    assert default_hypers["name"] == name


def test_check_architecture_name():
    check_architecture_name("soap_bpnn")


def test_check_architecture_name_suggest():
    name = "soap-bpnn"
    match = (
        rf"Architecture {name!r} is not a valid architecture. "
        r"Did you mean 'soap_bpnn'?"
    )
    with pytest.raises(ValueError, match=match):
        check_architecture_name(name)


def test_check_architecture_no_name_suggest():
    name = "sdlfijwpeofj"
    match = f"Architecture {name!r} is not a valid architecture."
    with pytest.raises(ValueError, match=match):
        check_architecture_name(name)


def test_check_architecture_name_deprecated():
    with pytest.raises(ValueError, match="deprecated architecture with the same name"):
        check_architecture_name("nanopet")


@pytest.mark.parametrize("path_type", [Path, str])
@pytest.mark.parametrize(
    "path",
    [
        PACKAGE_ROOT / "soap_bpnn",
        PACKAGE_ROOT / "soap_bpnn" / "__init__.py",
    ],
)
def test_get_architecture_name(path_type, path):
    assert get_architecture_name(path_type(path)) == "soap_bpnn"


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


def test_check_valid_default_architecture_options():
    """Test that validating architecture options works.

    We use soap_bpnn as the test architecture to see if the function works.
    Other architectures might have dependencies, and therefore loading their
    default hypers could fail. The loading of default hypers should be
    tested in the tests of each architecture.
    """
    name = "soap_bpnn"
    options = get_default_hypers(name)
    check_architecture_options(name=name, options=options)


def test_check_architecture_options_error_raise():
    name = "soap_bpnn"
    options = get_default_hypers(name)

    # Add an unknown parameter
    options["training"]["num_epochxxx"] = 10

    match = r"Unrecognized option 'training\.num_epochxxx'"
    with pytest.raises(MetatrainValidationError, match=match):
        check_architecture_options(name=name, options=options)


def test_import_architecture():
    name = "soap_bpnn"
    architecture_ref = importlib.import_module(f"metatrain.{name}")
    assert import_architecture(name) == architecture_ref
