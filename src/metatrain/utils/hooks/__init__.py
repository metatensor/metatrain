import copy
from typing import Annotated, Any, Callable, TypeVar

from omegaconf import OmegaConf
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Discriminator,
    Tag,
    create_model,
)

from .helpers import get_default_hypers, get_hypers_class, restart_hooks, setup_hooks


__all__ = ["setup_hooks", "restart_hooks"]

HooksHypers = dict[str, Any | dict[str, Any]]


def get_hook_hypers_sanitizer(hook_name: str) -> Callable[[Any], dict[str, Any]]:
    """Get a function that sanitizes the hypers for a hook.

    If a hook's configuration is not a dictionary, we pass the value
    to the "outputs" key of the hook hypers, using defaults for the rest.

    :param hook_name: The name of the hook for which the hypers have
      to be sanitized.

    :return: A function that takes the hypers for a hook and returns
      a dictionary with the sanitized hypers.
    """

    def _sanitize_hook_hypers(hook_hypers: Any) -> dict[str, Any]:

        if not isinstance(hook_hypers, dict):
            hook_hypers = {
                "outputs": hook_hypers,
            }

            hook_hypers = OmegaConf.merge(get_default_hypers(hook_name), hook_hypers)

        return hook_hypers

    return _sanitize_hook_hypers


def get_hook_default_filler(hook_name: str) -> Callable[[Any], Any]:
    """Get a function that fills the hypers for a hook with defaults.

    It only acts if the hypers passed are a dictionary.

    :param hook_name: The name of the hook for which the hypers have
      to be filled with defaults.

    :return: A function that takes the hypers for a hook and returns
      a dictionary with the hypers filled with defaults.
    """

    def _fill_hook_hypers_with_defaults(hook_hypers: Any) -> dict[str, Any]:

        if isinstance(hook_hypers, dict):
            hook_hypers = OmegaConf.merge(get_default_hypers(hook_name), hook_hypers)
            hook_hypers = OmegaConf.to_container(hook_hypers)

        return hook_hypers

    return _fill_hook_hypers_with_defaults


def get_hook_discriminator(only_outputs_possible: bool) -> Callable[[Any], str | None]:
    def hook_hyper_discriminator(v: Any) -> str | None:
        """Discriminator function for the hook hypers

        It helps pydantic determine whether it is dealing with the
        full hook hypers or just the "outputs" value.

        :param v: The value to discriminate.
        :return: The tag of the type that pydantic should validate.
        """
        if isinstance(v, dict):
            return "dict"
        elif only_outputs_possible:
            return "."

        # Could not determine the target type, pydantic will raise a
        # validation error with the appropriate message.
        return None

    return hook_hyper_discriminator


T = TypeVar("T")


def augment_hooks_annotations(
    obj: T,
    values: dict[str, Any],
) -> T:
    """Replace HooksHypers annotations with a pydantic model that
    validates the hypers for every hook.

    :param obj: The object to augment.
    :param values: The values passed as input (which should match the
      annotations of the obj). We will use these to know which hooks
      are in the input, so that we can add validation for them.

    :return: An augmented version of the object, with annotations for
      validating specifically the requested hooks.
    """
    obj = copy.deepcopy(obj)

    is_pydantic_model = isinstance(obj, type) and issubclass(obj, BaseModel)

    for annotation_name, annotation in obj.__annotations__.items():
        if annotation is HooksHypers:
            # This is a hooks hypers annotation, add a pydantic model to validate
            # the hypers for every hook.
            hooks_hypers_annotations = {}
            for name in values.get(annotation_name, {}):
                hook_hypers_class = get_hypers_class(name)

                # If "outputs" is in the hypers, we allow not only passing the
                # full hook hypers but also passing just the outputs
                only_outputs_possible = "outputs" in hook_hypers_class.__annotations__
                if only_outputs_possible:
                    out_annotation = hook_hypers_class.__annotations__["outputs"]
                    hook_annotation = (
                        Annotated[hook_hypers_class, Tag("dict")]  # type: ignore
                        | Annotated[out_annotation, Tag(".")]  # type: ignore
                    )
                else:
                    hook_annotation = Annotated[hook_hypers_class, Tag("dict")]  # type: ignore

                hooks_hypers_annotations[name] = Annotated[
                    hook_annotation,
                    Discriminator(get_hook_discriminator(only_outputs_possible)),
                    BeforeValidator(get_hook_default_filler(name)),
                    AfterValidator(get_hook_hypers_sanitizer(name)),
                ]

            new_annotation = create_model(
                "AugmentedHooksHypers",
                **hooks_hypers_annotations,
                __config__=dict(extra="forbid"),
            )

            obj.__annotations__[annotation_name] = new_annotation

            # attr check in the next line is just for mypy
            if is_pydantic_model and hasattr(obj, "model_fields"):
                obj.model_fields[annotation_name].annotation = new_annotation

    if is_pydantic_model and hasattr(obj, "model_fields"):  # attr check for mypy
        obj = create_model(
            getattr(obj, "__name__", "AugmentedModel"),
            **{k: (v.annotation, v) for k, v in obj.model_fields.items()},
            __config__=dict(extra="forbid"),
        )

    return obj
