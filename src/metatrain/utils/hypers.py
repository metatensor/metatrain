from typing import Type, TypedDict, TypeVar

from typing_extensions import TypedDict as TE_TypedDict


HypersType = TypeVar("HypersType")


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
_OVERWRITTEN_DEFAULTS = {}


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
