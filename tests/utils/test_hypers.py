# mypy: ignore-errors
# Satisfying mypy in this file is hard because the fixtures
# define different classes depending on the parametrization.
import pytest
from typing_extensions import TypedDict

from metatrain.utils.hypers import (
    get_hypers_list,
    init_with_defaults,
    overwrite_defaults,
)


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
