import logging
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, create_model
from typing_extensions import NotRequired

from ..share.base_hypers import BaseHypers
from .hypers import init_with_defaults


class MetatrainValidationError(Exception):
    """This class transforms Pydantic validation errors into a
    more user-friendly format.

    :param model: The Pydantic model class or TypedDict that was
        used for validation.
    :param errors: The list of Pydantic error dictionaries.
    """

    def __init__(self, model: Any, errors: list[dict]):
        self.model = model
        self.errors = errors

    def get_error_string(self, error: dict) -> str:
        """Given an individual error from Pydantic, return a user-friendly string.

        :param error: The Pydantic error dictionary.

        :return: The formatted error string to display to the user.
        """

        # This is a field that was not expected
        if error["type"] == "extra_forbidden":
            extra_field = ".".join(error["loc"])
            return f"Unrecognized option '{extra_field}'."

        # If it doesn't match any special case, use the default Pydantic formatting
        return self.default_pydantic(error)

    def default_pydantic(self, err: dict) -> str:
        """Default Pydantic error formatting.

        :param err: The Pydantic error dictionary.
        :return: The formatted error string to display to the user.
        """
        pydantic_error = f"{err['msg']}"
        pydantic_error += f" [type={err['type']}, input_value={err['input']},"
        pydantic_error += f" input_type={err.get('type', 'unknown')}]"

        pydantic_error += f"\n\tFor further information visit {err['url']}"
        return pydantic_error

    def __str__(self) -> str:
        """Return a string representation of all validation errors.

        :return: The formatted error string to display to the user.
        """
        error_str = f"{len(self.errors)} validation errors occurred:\n"
        for i, err in enumerate(self.errors):
            error_str += (
                f"[Error {i}] {'.'.join(err['loc'])}\n\t{self.get_error_string(err)}\n"
            )
        return error_str


def validate(model_cls: Any, data: dict, **kwargs: Any) -> None:
    r"""Validate with pydantic, raising custom metatrain errors.

    :param model_cls: The Pydantic model class to use for validation.
      If it is not a pydantic model, it will be adapted to pydantic
      using ``pydantic.TypeAdapter``.
    :param data: The data to validate.
    :param \*\*kwargs: Additional keyword arguments to pass to the validation method.

    :raises MetatrainValidationError: If validation fails.
    """

    if issubclass(model_cls, BaseModel):
        try:
            model_cls.model_validate(data, **kwargs)
        except ValidationError as e:
            raise MetatrainValidationError(model_cls, e.errors()) from e
    else:
        adapter = TypeAdapter(model_cls)
        try:
            adapter.validate_python(data, **kwargs)
        except ValidationError as e:
            raise MetatrainValidationError(model_cls, e.errors()) from e


def validate_architecture_options(
    options: dict, model_hypers: type, trainer_hypers: type
) -> None:
    """Validate architecture-specific options using Pydantic.

    :param options: The architecture options to validate.
    :param model_hypers: The ModelHypers class of the architecture.
    :param trainer_hypers: The TrainerHypers class of the architecture.
    """

    def _is_validatable(cls: Any) -> bool:
        return issubclass(cls, (BaseModel, dict))

    if not _is_validatable(model_hypers) or not _is_validatable(trainer_hypers):
        logging.warning(
            "Architecture does not provide validation of hyperparameters. "
            "Continuing without validation."
        )
        return

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

    validate(ArchitectureOptions, options)

    if added_atomic_types:
        del options["atomic_types"]


def validate_base_options(options: dict) -> None:
    """Validate base options using Pydantic.

    :param options: The base options to validate.

    :raises ValueError: If the options are invalid.
    """
    validate(BaseHypers, options)


def get_train_json_schema(allow_missing_hypers: bool) -> dict:
    """Generate a JSON schema for the training options.

    This JSON schema is a full specification for the input yaml files of
    ``mtt train``. Therefore, it includes all possible architectures.

    :param allow_missing_hypers: Whether to allow missing hyperparameters.
      If you want to use the JSON schema for validating user input, you
      should set this to ``True``, as it will allow users to omit fields that
      have default values. If you want to use the JSON schema for
      validating the input once filled in with defaults, you should set
      this to ``False``.
    """
    from .architectures import find_all_architectures, preload_documentation_module

    def set_not_required_and_defaults(cls):
        """Helper function to set all fields of a class as NotRequired
        and add default values if they exist.

        This is because ModelHypers and TrainerHypers are written to validate the
        options once all defaults have been filled in, but for a JSON schema to
        validate user input, we want to allow missing fields.
        """
        annotations = {}
        for k, v in cls.__annotations__.items():
            if allow_missing_hypers:
                annotations[k] = NotRequired[v]
            if hasattr(cls, k):
                annotations[k] = Annotated[
                    annotations[k], Field(default=getattr(cls, k))
                ]
        cls.__annotations__ = annotations
        return cls

    # Get the model for the architecture options of each architecture.
    arch_models = []
    for arch_name in find_all_architectures():
        arch_doc = preload_documentation_module(arch_name)

        ModelHypers = set_not_required_and_defaults(arch_doc.ModelHypers)
        TrainerHypers = set_not_required_and_defaults(arch_doc.TrainerHypers)

        ArchModel = create_model(
            f"{arch_name}Architecture",
            name=(
                Literal[arch_name],
                Field(
                    description="Name of the architecture. The architecure options "
                    "will depend on the chosen architecture."
                ),
            ),
            atomic_types=(list[int], Field(default=None)),
            model=(
                ModelHypers,
                Field(
                    default=init_with_defaults(ModelHypers),
                    description=ModelHypers.__doc__,
                ),
            ),
            training=(
                TrainerHypers,
                Field(
                    default=init_with_defaults(TrainerHypers),
                    description=TrainerHypers.__doc__,
                ),
            ),
            __config__={
                "extra": "forbid",
                "strict": True,
                "use_attribute_docstrings": True,
            },
        )

        arch_models.append(ArchModel)

    # Build the global model for the training options, setting the
    # architecture field to be a union of all the possible architectures.
    _baseHypers = set_not_required_and_defaults(BaseHypers)
    _baseHypers.__annotations__["architecture"] = Union[tuple(arch_models)]

    mtttrain_model = TypeAdapter(_baseHypers)

    return mtttrain_model.json_schema()
