from collections.abc import Mapping
from typing import Any, Type, TypedDict, TypeVar

from typing_extensions import TypedDict as TE_TypedDict


HypersType = TypeVar("HypersType")


def get_hypers_list(hypers_cls: Type[HypersType]) -> list[str]:
    """Get the list of hyperparameter names defined in a TypedDict hypers class.

    Inheritance of parameters is allowed from parent classes, but make
    sure that the parent classes only contain hyperparameters as
    attributes! (i.e., no methods allowed). Private attributes (starting
    with "_") are not considered as hyperparameters, so one can have
    arbitrary private methods or attributes in the class and its parents,
    although this is not recommended.

    :param hypers_cls: The class defining the hyperparameters.

    :return: A list with the names of the hyperparameters.
    """
    hypers_list = []

    # First find hypers from parent classes
    parent_classes = [*hypers_cls.mro()[1:], *getattr(hypers_cls, "__orig_bases__", [])]
    for base in parent_classes:
        if base not in (TE_TypedDict, TypedDict, dict, object):
            parent_hypers = get_hypers_list(base)
            hypers_list.extend(parent_hypers)

    this_class_vars = vars(hypers_cls)
    # Now get hypers from this class
    for key in this_class_vars.keys():
        # Skip private attributes
        if not key.startswith("_"):
            hypers_list.append(key)

    return hypers_list


def init_with_defaults(hypers_cls: Type[HypersType]) -> dict:
    """Initialize a TypedDict hypers class with its default values.

    Inheritance of parameters is allowed from parent classes, but make
    sure that the parent classes only contain hyperparameters as
    attributes! (i.e., no methods allowed). Private attributes (starting
    with "_") are not considered as hyperparameters, so one can have
    arbitrary private methods or attributes in the class and its parents,
    although this is not recommended.

    :param hypers_cls: The class defining the hyperparameters.

    :return: A dict with the default hyperparameters.
    """
    defaults_dict = {}

    # First find defaults from parent classes
    parent_classes = [*hypers_cls.mro()[1:], *getattr(hypers_cls, "__orig_bases__", [])]
    for base in parent_classes:
        if base not in (TE_TypedDict, TypedDict, dict, object):
            base_defaults = init_with_defaults(base)
            defaults_dict.update(base_defaults)

    this_class_vars = vars(hypers_cls)
    # Now get defaults from this class
    for key, value in this_class_vars.items():
        # Skip private attributes
        if not key.startswith("_"):
            defaults_dict[key] = value

    # Overwrite using the registered overwrites
    to_overwrite = _OVERWRITTEN_DEFAULTS.get(hypers_cls, {})
    for k in to_overwrite:
        if k in defaults_dict:
            defaults_dict[k] = to_overwrite[k]

    return defaults_dict


# Private global dictionary to store overwritten defaults
_OVERWRITTEN_DEFAULTS: dict[Type, dict] = {}


def overwrite_defaults(
    hypers_cls: Type,
    new_defaults: dict,
) -> None:
    """Overwrite the default hyperparameters.

    This function does not check that the new defaults correspond
    to valid hyperparameters of the given hypers class. If the new
    defaults contain keys that are not hyperparameters of the class,
    they will simply be ignored.

    :param hypers_cls: The hypers class whose defaults to overwrite.
    :param new_defaults: A dict with the new default hyperparameters.
    """
    _OVERWRITTEN_DEFAULTS[hypers_cls] = new_defaults


def get_hypers_diff(
    old_hypers: Mapping[str, Any],
    new_hypers: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Get the difference between two hypers dictionaries.

    Only keys present in ``new_hypers`` are compared. Missing keys are treated
    as "keep the checkpoint value". Nested mappings are compared the same way,
    so a partial YAML block such as ``soap: {max_radial: 4}`` does not mismatch
    a checkpoint that also stores ``cutoff``.

    :param old_hypers: The old hyperparameters.
    :param new_hypers: The new hyperparameters.

    :return: A dict of dotted paths to ``(old, new)`` pairs that differ.
        A missing old key is reported as ``"<not present>"``.
    """
    diff: dict[str, tuple[Any, Any]] = {}
    for key, new_value in new_hypers.items():
        if key not in old_hypers:
            diff[key] = ("<not present>", new_value)
            continue
        old_value = old_hypers[key]
        if isinstance(old_value, Mapping) and isinstance(new_value, Mapping):
            for nested_key, nested_pair in get_hypers_diff(
                old_value, new_value
            ).items():
                diff[f"{key}.{nested_key}"] = nested_pair
        elif old_value != new_value:
            diff[key] = (old_value, new_value)
    return diff


def raise_hypers_mismatch(
    hypers_diff: Mapping[str, tuple[Any, Any]],
) -> None:
    """Raise an error if the hypers diff is not empty.

    The error shows a report of the mismatched hyperparameters.

    :param hypers_diff: A dict with the hyperparameters that are different.
        It can be computed using :func:`get_hypers_diff`.
    """
    if hypers_diff:
        n_mismatches = len(hypers_diff)
        raise ValueError(
            f"Found {n_mismatches} mismatch{(n_mismatches != 1) * 'es'} "
            "in model hyperparameters.\n"
            f"Mismatched hypers: {list(hypers_diff.keys())}\n"
            "\n-------- Mismatches --------\n\n"
            + "\n".join(
                f"[Mismatch {i + 1}] {key}\n Previous: {old}\n New: {new}"
                for i, (key, (old, new)) in enumerate(hypers_diff.items())
            )
        )


def raise_if_hypers_mismatch(
    old_hypers: Mapping[str, Any],
    new_hypers: Mapping[str, Any],
) -> None:
    """Raise an error if the new hypers do not match the old hypers.

    :param old_hypers: The old hyperparameters.
    :param new_hypers: The new hyperparameters.
    """
    # Gather mismatchs
    mismatches = get_hypers_diff(old_hypers, new_hypers)

    if mismatches:
        raise_hypers_mismatch(mismatches)
