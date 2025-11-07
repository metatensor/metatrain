from pydantic import TypeAdapter, create_model

from ..share.base_hypers import BaseHypers
from .abc import ModelInterface, TrainerInterface
from .hypers import get_hypers_cls


def validate_architecture_options(
    options: dict, model: ModelInterface, trainer: TrainerInterface
) -> None:
    """Validate architecture-specific options using Pydantic.

    :param options: The architecture options to validate.
    :param model: The model class of the architecture.
    :param trainer: The trainer class of the architecture.

    :raises ValueError: If the architecture name is unknown.
    """
    model_hypers = get_hypers_cls(model)
    trainer_hypers = get_hypers_cls(trainer)

    ArchitectureOptions = create_model(
        "ArchitectureOptions",
        name=str,
        atomic_types=list[int],
        model=model_hypers,
        training=trainer_hypers,
        __config__={"extra": "forbid", "strict": True},
    )

    # Because passing NotRequired[list[int]] to an argument of a pydantic model
    # is not possible, and creating a TypedDict using variables (model_hypers,
    # trainer_hypers) as typehints is also not possible, if atomix_types was
    # not provided we have to add a dummy value for it and remove it after
    # validation.
    added_atomic_types = False
    if "atomic_types" not in options:
        options["atomic_types"] = []
        added_atomic_types = True

    ArchitectureOptions.model_validate(options)

    if added_atomic_types:
        del options["atomic_types"]


def validate_base_options(options: dict) -> None:
    """Validate base options using Pydantic.

    :param options: The base options to validate.

    :raises ValueError: If the options are invalid.
    """
    TypeAdapter(BaseHypers).validate_python(options)
