import logging
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError, create_model

from ..share.base_hypers import BaseHypers


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
    """Validate with pydantic, raising custom metatrain errors.

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
