from typing import Annotated, Any, Callable

from omegaconf import OmegaConf
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Discriminator,
    Tag,
    create_model,
)

from .helpers import (
    get_default_hypers,
    get_hypers_class,
    restart_extenders,
    setup_extenders,
)


__all__ = ["setup_extenders", "restart_extenders"]

ExtendersHypers = dict[str, Any | dict[str, Any]]


def get_extender_hypers_sanitizer(
    extender_name: str,
) -> Callable[[Any], dict[str, Any]]:
    """Get a function that sanitizes the hypers for an extender.

    If an extender's configuration is not a dictionary, we pass the value
    to the "outputs" key of the extender hypers, using defaults for the rest.

    :param extender_name: The name of the extender for which the hypers have
      to be sanitized.

    :return: A function that takes the hypers for an extender and returns
      a dictionary with the sanitized hypers.
    """

    def _sanitize_extender_hypers(extender_hypers: Any) -> dict[str, Any]:

        if not isinstance(extender_hypers, dict):
            extender_hypers = {
                "outputs": extender_hypers,
            }

            extender_hypers = OmegaConf.merge(
                get_default_hypers(extender_name), extender_hypers
            )

        return extender_hypers

    return _sanitize_extender_hypers


def get_extender_default_filler(extender_name: str) -> Callable[[Any], Any]:
    """Get a function that fills the hypers for an extender with defaults.

    It only acts if the hypers passed are a dictionary.

    :param extender_name: The name of the extender for which the hypers have
      to be filled with defaults.

    :return: A function that takes the hypers for an extender and returns
      a dictionary with the hypers filled with defaults.
    """

    def _fill_extender_hypers_with_defaults(extender_hypers: Any) -> dict[str, Any]:

        if isinstance(extender_hypers, dict):
            extender_hypers = OmegaConf.merge(
                get_default_hypers(extender_name), extender_hypers
            )
            extender_hypers = OmegaConf.to_container(extender_hypers)

        return extender_hypers

    return _fill_extender_hypers_with_defaults


def get_extender_discriminator(
    only_outputs_possible: bool,
) -> Callable[[Any], str | None]:
    def extender_hyper_discriminator(v: Any) -> str | None:
        """Discriminator function for the extender hypers

        It helps pydantic determine whether it is dealing with the
        full extender hypers or just the "outputs" value.

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

    return extender_hyper_discriminator


def augment_extenders_annotations(
    obj: type,
    values: dict[str, Any],
) -> type:
    """Replace extendersHypers annotations with a pydantic model that
    validates the hypers for every extender.

    :param obj: The object to augment.
    :param values: The values passed as input (which should match the
      annotations of the obj). We will use these to know which extenders
      are in the input, so that we can add validation for them.

    :return: An augmented version of the object, with annotations for
      validating specifically the requested extenders.
    """

    is_pydantic_model = isinstance(obj, type) and issubclass(obj, BaseModel)

    modified_annotations = {}
    for annotation_name, annotation in obj.__annotations__.items():
        if annotation is ExtendersHypers:
            # This is an extenders hypers annotation, add a pydantic model to validate
            # the hypers for every extender.
            extenders_hypers_annotations = {}
            for name in values.get(annotation_name, {}):
                extender_hypers_class = get_hypers_class(name)

                # If "outputs" is in the hypers, we allow not only passing the
                # full extender hypers but also passing just the outputs
                only_outputs_possible = (
                    "outputs" in extender_hypers_class.__annotations__
                )
                if only_outputs_possible:
                    out_annotation = extender_hypers_class.__annotations__["outputs"]
                    extender_annotation = (
                        Annotated[extender_hypers_class, Tag("dict")]  # type: ignore
                        | Annotated[out_annotation, Tag(".")]  # type: ignore
                    )
                else:
                    extender_annotation = Annotated[extender_hypers_class, Tag("dict")]  # type: ignore

                extenders_hypers_annotations[name] = Annotated[
                    extender_annotation,
                    Discriminator(get_extender_discriminator(only_outputs_possible)),
                    BeforeValidator(get_extender_default_filler(name)),
                    AfterValidator(get_extender_hypers_sanitizer(name)),
                ]

            new_annotation = create_model(
                "AugmentedextendersHypers",
                **extenders_hypers_annotations,
                __config__=dict(extra="forbid"),
            )

            modified_annotations[annotation_name] = new_annotation

    if is_pydantic_model and hasattr(obj, "model_fields"):  # attr check for mypy
        return create_model(
            getattr(obj, "__name__", "AugmentedModel"),
            **{
                k: (modified_annotations.get(k, v.annotation), v)
                for k, v in obj.model_fields.items()
            },
            __config__=dict(extra="forbid"),
        )
    else:
        return type(
            getattr(obj, "__name__", "AugmentedClass"),
            (obj,),
            {"__annotations__": modified_annotations},
        )
