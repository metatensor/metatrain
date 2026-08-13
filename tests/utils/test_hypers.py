# mypy: ignore-errors
# Satisfying mypy in this file is hard because the fixtures
# define different classes depending on the parametrization.
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from typing_extensions import TypedDict

from metatrain.utils.hypers import (
    get_hypers_diff,
    get_hypers_list,
    init_with_defaults,
    overwrite_defaults,
    raise_if_hypers_mismatch,
)
from metatrain.utils.pydantic import validate_base_options, validate_eval_options


@pytest.fixture(params=["custom_class", "typed_dict"])
def simple_hypers_class(request: pytest.FixtureRequest):
    if request.param == "custom_class":

        class Hypers:
            a: float = 2.0

    elif request.param == "typed_dict":

        class Hypers(TypedDict):
            a: float = 2.0

    return Hypers


@pytest.fixture(params=["custom_class", "typed_dict"])
def nested_hypers_class(request: pytest.FixtureRequest):
    class A(TypedDict):
        x: str = "hello"

    if request.param == "custom_class":

        class Hypers:
            a: A = init_with_defaults(A)

    elif request.param == "typed_dict":

        class Hypers(TypedDict):
            a: A = init_with_defaults(A)

    return Hypers


def test_default_hypers(simple_hypers_class: type):
    hypers = init_with_defaults(simple_hypers_class)
    assert hypers == {"a": 2.0}


def test_get_hypers_list(simple_hypers_class: type):
    hypers_list = get_hypers_list(simple_hypers_class)
    assert hypers_list == ["a"]


def test_default_hypers_nested(nested_hypers_class: type):
    hypers = init_with_defaults(nested_hypers_class)
    assert hypers == {"a": {"x": "hello"}}


def test_default_hypers_inheritance(simple_hypers_class: type):
    class Hypers(simple_hypers_class):
        b: int = 3

    hypers = init_with_defaults(Hypers)
    assert hypers == {"a": 2.0, "b": 3}

    parent_hypers = init_with_defaults(simple_hypers_class)
    assert parent_hypers == {"a": 2.0}


def test_hypers_list_inheritance(simple_hypers_class: type):
    class Hypers(simple_hypers_class):
        b: int = 3

    hypers_list = get_hypers_list(Hypers)
    assert hypers_list == ["a", "b"]

    parent_hypers_list = get_hypers_list(simple_hypers_class)
    assert parent_hypers_list == ["a"]


def test_default_hypers_inheritance_overwrite(simple_hypers_class: type):
    class Hypers(simple_hypers_class):
        b: int = 3

    overwrite_defaults(Hypers, {"a": 5.0})

    hypers = init_with_defaults(Hypers)
    assert hypers == {"a": 5.0, "b": 3}

    parent_hypers = init_with_defaults(simple_hypers_class)
    assert parent_hypers == {"a": 2.0}


@pytest.mark.parametrize(
    "per_atom,mode",
    [(True, "train"), (False, "train"), (True, "eval"), (False, "eval")],
)
def test_per_atom_deprecation(per_atom: bool, mode: str, tmp_path: Path):

    # Write yaml file with a target using per_atom.
    if mode == "eval":
        options_yaml = f"""
        systems:
            read_from: somefile.xyz
        targets:
            mtt::some_target:
                per_atom: {per_atom}
        """
    elif mode == "train":
        options_yaml = f"""
        architecture:
            name: soap_bpnn
            atomic_types: [1]

        training_set:
            systems:
                read_from: somefile.xyz
            targets:
                mtt::some_target:
                    per_atom: {per_atom}
        validation_set: 0.0
        """

    options = OmegaConf.create(options_yaml)

    # Check that a deprecation warning is raised
    with pytest.warns(
        DeprecationWarning,
        match="The `per_atom` key in target specifications is deprecated",
    ):
        if mode == "eval":
            options = validate_eval_options(OmegaConf.to_container(options))
        elif mode == "train":
            options = validate_base_options(OmegaConf.to_container(options))
        options = OmegaConf.create(options)

    sample_kind = "atom" if per_atom else "system"
    if mode == "train":
        assert (
            options.training_set.targets["mtt::some_target"].sample_kind == sample_kind
        )
    elif mode == "eval":
        assert options.targets["mtt::some_target"].sample_kind == sample_kind


def test_hypers_diff_ignores_unspecified_top_level_keys():
    old = {"a": 1, "b": 2}
    assert get_hypers_diff(old, {"a": 1}) == {}
    assert get_hypers_diff(old, {}) == {}


def test_hypers_diff_reports_changed_scalar():
    old = {"a": 1, "b": 2}
    assert get_hypers_diff(old, {"a": 3}) == {"a": (1, 3)}


def test_hypers_diff_partial_nested_dict_matches():
    """Restarting with the same yaml only lists the keys the user wrote.

    Checkpoint soap hypers include cutoff defaults that the input file omitted.
    That is not a mismatch.
    """
    old = {
        "soap": {
            "max_angular": 2,
            "max_radial": 4,
            "cutoff": {"radius": 5.0, "width": 0.5},
        }
    }
    new = {"soap": {"max_radial": 4, "max_angular": 2}}
    assert get_hypers_diff(old, new) == {}


def test_hypers_diff_nested_value_change():
    old = {
        "soap": {
            "max_angular": 2,
            "max_radial": 4,
            "cutoff": {"radius": 5.0, "width": 0.5},
        }
    }
    new = {"soap": {"max_radial": 8}}
    assert get_hypers_diff(old, new) == {"soap.max_radial": (4, 8)}


def test_hypers_diff_unknown_key():
    old = {"a": 1}
    assert get_hypers_diff(old, {"b": 2}) == {"b": ("<not present>", 2)}


def test_raise_if_hypers_mismatch_accepts_partial_nested():
    old = {
        "soap": {
            "max_angular": 2,
            "max_radial": 4,
            "cutoff": {"radius": 5.0, "width": 0.5},
        }
    }
    raise_if_hypers_mismatch(old, {"soap": {"max_radial": 4, "max_angular": 2}})


def test_raise_if_hypers_mismatch_reports_nested_path():
    old = {"soap": {"max_radial": 4, "max_angular": 2}}
    with pytest.raises(ValueError, match=r"soap\.max_radial"):
        raise_if_hypers_mismatch(old, {"soap": {"max_radial": 8}})
