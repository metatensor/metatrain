from pydantic import create_model

from .hypers import get_hypers_annotation


def validate_architecture_options(options: dict, model, trainer) -> None:
    """Validate architecture-specific options using Pydantic.

    :param name: The name of the architecture.
    :param options: The architecture options to validate.

    :raises ValueError: If the architecture name is unknown.
    """
    model_hypers = get_hypers_annotation(model)
    trainer_hypers = get_hypers_annotation(trainer)

    ArchitectureOptions = create_model(
        "ArchitectureOptions",
        name=str,
        model=model_hypers,
        training=trainer_hypers,
        __config__={"extra": "forbid", "strict": True},
    )

    ArchitectureOptions.model_validate(options)
