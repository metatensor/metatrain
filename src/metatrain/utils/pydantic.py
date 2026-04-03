import logging
from typing import Any, Literal

from pydantic import BaseModel, TypeAdapter, ValidationError, create_model

from ..share import base_hypers
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
            error_loc = error["loc"]
            cls = error_loc[-2]
            field = error_loc[-1]
            return f"Unrecognized option '{field}' for '{cls}'."

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

        pydantic_error += f"\n  For further information visit {err['url']}"
        return pydantic_error

    def __str__(self) -> str:
        """Return a string representation of all validation errors.

        :return: The formatted error string to display to the user.
        """
        errors = self.errors
        error_str = f"{len(errors)} validation errors occurred:\n"
        for i, err in enumerate(errors):
            error_str += (
                f"\n---- [Error {i}] {self.get_loc_path(err['loc'])}"
                f"\n\n  {self.get_error_string(err)}\n"
            )
        return error_str

    def get_loc_path(self, error_loc: tuple) -> str:
        """Convert the error location tuple into a dot-separated string path.

        :param error_loc: The 'loc' field from a Pydantic error dictionary, which
          is a tuple representing the location of the error in the input data.
        :return: A string representing the path to the error location, with
          certain internal ``pydantic`` function calls filtered out for readability.
        """
        return ".".join(
            [
                str(item)
                for item in error_loc
                if not str(item).startswith("function-after")
                or str(item).startswith("function-before")
            ]
        )


def validate(
    model_cls: Any,
    data: dict,
    error_cls: type[MetatrainValidationError] = MetatrainValidationError,
    **kwargs: Any,
) -> None:
    r"""Validate with pydantic, raising custom metatrain errors.

    :param model_cls: The Pydantic model class to use for validation.
      If it is not a pydantic model, it will be adapted to pydantic
      using ``pydantic.TypeAdapter``.
    :param data: The data to validate.
    :param error_cls: The custom error class to raise if validation fails.
    :param \*\*kwargs: Additional keyword arguments to pass to the validation method.

    :raises MetatrainValidationError: If validation fails.
    """

    if issubclass(model_cls, BaseModel):
        try:
            model_cls.model_validate(data, **kwargs)
        except ValidationError as e:
            raise error_cls(model_cls, e.errors()) from e
    else:
        adapter = TypeAdapter(model_cls)
        try:
            adapter.validate_python(data, **kwargs)
        except ValidationError as e:
            raise error_cls(model_cls, e.errors()) from e


class MetatrainArchitectureValidationError(MetatrainValidationError):
    """Custom validation error for architecture options."""

    _architecture: str | None = None

    @classmethod
    def for_architecture(
        cls, name: str | None
    ) -> type["MetatrainArchitectureValidationError"]:
        if name is None:
            return cls
        return type(f"{cls.__name__}{name}", (cls,), {"_architecture": name})

    def architecture_link(self, cls: str) -> str:
        if self._architecture is None:
            return ""

        architecture_name = self._architecture.replace("experimental.", "").replace(
            "deprecated.", ""
        )

        return f"https://docs.metatensor.org/metatrain/latest/architectures/generated/{architecture_name}.html#metatrain.{self._architecture}.documentation.{cls}"

    def get_error_string(self, error: dict) -> str:
        """Given an individual error from Pydantic, return a user-friendly string.

        :param error: The Pydantic error dictionary.

        :return: The formatted error string to display to the user.
        """

        hyper_type: Literal["model", "training"] = error["loc"][0]
        field = error["loc"][1]
        cls = "ModelHypers" if hyper_type == "model" else "TrainerHypers"

        arch_link = self.architecture_link(cls)

        if error["type"] == "extra_forbidden":
            # This is a field that was not expected
            error_loc = error["loc"]
            field = error_loc[-1]

            if len(error_loc) == 2:
                msg = f"Unrecognized option '{field}' for {hyper_type} hyperparameters."
                if arch_link:
                    msg += (
                        f"\n  For the available {hyper_type} hyperparameters see:"
                        f"\n    {arch_link}"
                    )
            else:
                return (
                    f"Unrecognized option '{field}' for '{cls}'."
                    f"\n  See the documentation of {cls}:"
                    f"\n    {arch_link}"
                )
        else:
            # Rest of cases.
            msg = error["msg"]
            msg += f" [type={error['type']}, input_value={error['input']},"
            msg += f" input_type={error.get('type', 'unknown')}]"

            msg += (
                f"\n  For the documentation of '{field}', see:"
                f"\n    {arch_link}.{field}"
                "\n  To understand this pydantic validation error in general, see:"
                f"\n    {error['url']}"
            )
        return msg

    def __str__(self) -> str:
        """Return a string representation of all validation errors.

        :return: The formatted error string to display to the user.
        """
        errors = self.errors

        # Organize errors by their top-level location (model vs training)
        error_dict: dict[Literal["model", "training"], list[dict]] = {}
        for err in errors:
            top_level = err["loc"][0]
            if top_level not in error_dict:
                error_dict[top_level] = []
            error_dict[top_level].append(err)

        error_str = (
            f"{len(errors)} validation errors occurred for the  {self._architecture} "
            if self._architecture
            else " architecture options:\n"
        )

        # Log errors for the model hyperparameters
        if "model" in error_dict:
            n_errors = len(error_dict["model"])
            error_str += f"\n==== Errors in model hyperparameters ({n_errors}):\n"
            for i, err in enumerate(error_dict["model"]):
                error_str += (
                    f"\n---- [Error {i}] {self.get_loc_path(err['loc'])}"
                    f"\n\n  {self.get_error_string(err)}\n"
                )
        # Log errors for the training hyperparameters
        if "training" in error_dict:
            n_errors = len(error_dict["training"])
            error_str += f"\n==== Errors in training hyperparameters ({n_errors}):\n"
            for i, err in enumerate(error_dict["training"]):
                error_str += (
                    f"\n---- [Error {i}] {self.get_loc_path(err['loc'])}"
                    f"\n\n  {self.get_error_string(err)}\n"
                )

        return error_str


def validate_architecture_options(
    options: dict,
    model_hypers: type,
    trainer_hypers: type,
    architecture_name: str | None = None,
) -> None:
    """Validate architecture-specific options using Pydantic.

    :param options: The architecture options to validate.
    :param model_hypers: The ModelHypers class of the architecture.
    :param trainer_hypers: The TrainerHypers class of the architecture.
    :param architecture_name: The name of the architecture. If provided, it is
      used to give more specific error messages with links to the
      architecture documentation.
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

    validate(
        ArchitectureOptions,
        options,
        error_cls=MetatrainArchitectureValidationError.for_architecture(
            architecture_name
        ),
    )

    if added_atomic_types:
        del options["atomic_types"]


class MetatrainBaseValidationError(MetatrainValidationError):
    """Custom validation error for base options."""

    _known_base_hypers_classes = [
        name for name in dir(base_hypers) if not name.startswith("_")
    ]

    def get_error_string(self, error: dict) -> str:
        """Given an individual error from Pydantic, return a user-friendly string.

        :param error: The Pydantic error dictionary.
        :return: The formatted error string to display to the user.
        """
        cls = None
        if len(error["loc"]) > 1 and error["loc"][0] == "architecture":
            cls = "ArchitectureBaseHypers"

        if error["type"] == "extra_forbidden":
            # This is a field that was not expected
            error_loc = error["loc"]

            field = error_loc[-1]

            if len(error_loc) == 1:
                cls = "BaseHypers"
            else:
                cls = cls or error_loc[-2]

            readable_cls = {
                "ArchitectureBaseHypers": "architecture hyperparameters",
                "BaseHypers": "base hyperparameters",
            }.get(cls, cls)

            if cls in self._known_base_hypers_classes:
                return (
                    f"Unrecognized option '{field}' for {readable_cls}."
                    f"\n  For the available options of {readable_cls} see:"
                    f"\n    https://docs.metatensor.org/metatrain/latest/dev-docs/base-hypers.html#metatrain.share.base_hypers.{cls}"
                )
            else:
                return f"Unrecognized option '{field}'"
        elif (
            error["type"] == "union_tag_not_found"
            and error["ctx"]["discriminator"] == "target_type_discriminator()"
        ):
            # Unable to determine the target type.
            target = error["loc"][-3]
            return (
                f"Unable to determine the target type for target '{target}'."
                f"\n  Received target type: {error['input']}."
                f"\n  For the available type specifications see:"
                f"\n    https://docs.metatensor.org/metatrain/latest/dev-docs/base-hypers.html#metatrain.share.base_hypers.TargetHypers.type"
                f"\n  For an accessible tutorial on target types, see:"
                f"\n    https://docs.metatensor.org/metatrain/latest/generated_examples/1-advanced/03-fitting-generic-targets.html"
            )
        elif error["type"] == "union_tag_not_found" and error["ctx"][
            "discriminator"
        ] in (
            "training_set_discriminator()",
            "val_or_test_set_discriminator()",
        ):
            # Unable to identify the kind of dataset specification provided.
            dataset_name = error["loc"][-1]
            base_hypers_link = ""
            if len(error["loc"]) == 1:
                base_hypers_link = (
                    f"\n  For the definition of '{dataset_name}', see:"
                    f"\n    https://docs.metatensor.org/metatrain/latest/dev-docs/base-hypers.html#metatrain.share.base_hypers.BaseHypers.{dataset_name}"
                )
            return (
                f"Unable to process '{dataset_name}'."
                f"\n  Received value: {error['input']}."
                f"{base_hypers_link}"
                f"\n  For a description on how to input datasets, see:"
                f"\n    https://docs.metatensor.org/metatrain/latest/getting-started/train_yaml_config.html#data"
            )
        else:
            # Rest of cases.
            field = error["loc"][-1]

            if len(error["loc"]) == 1:
                cls = "BaseHypers"
            else:
                cls = cls or error["loc"][-2]

            if error["type"] == "missing":
                msg = f"Missing required option '{field}' for '{cls}'."
            else:
                msg = error["msg"]
                msg += f" [type={error['type']}, input_value={error['input']},"
                msg += f" input_type={error.get('type', 'unknown')}]"

            if cls in self._known_base_hypers_classes:
                if error["type"] == "invalid_key":
                    msg += (
                        f"\n  For the available options of {cls}, see:"
                        f"\n    https://docs.metatensor.org/metatrain/latest/dev-docs/base-hypers.html#metatrain.share.base_hypers.{cls}"
                    )
                else:
                    msg += (
                        f"\n  For the documentation of '{field}', see:"
                        f"\n    https://docs.metatensor.org/metatrain/latest/dev-docs/base-hypers.html#metatrain.share.base_hypers.{cls}.{field}"
                    )

            msg += (
                "\n  To understand this pydantic validation error in general, see:"
                f"\n    {error['url']}"
            )
            return msg

    def __str__(self) -> str:
        """Return a string representation of all validation errors.

        :return: The formatted error string to display to the user.
        """
        errors = self.errors
        error_str = f"{len(errors)} validation errors occurred for the base hypers:\n"
        for i, err in enumerate(errors):
            error_str += (
                f"\n---- [Error {i}] {self.get_loc_path(err['loc'])}"
                f"\n\n  {self.get_error_string(err)}\n"
            )
        return error_str


def validate_base_options(options: dict) -> None:
    """Validate base options using Pydantic.

    :param options: The base options to validate.

    :raises ValueError: If the options are invalid.
    """
    validate(BaseHypers, options, error_cls=MetatrainBaseValidationError)
