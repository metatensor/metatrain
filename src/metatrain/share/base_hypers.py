# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
from typing import Annotated, Literal, Optional

from annotated_types import Interval
from pydantic import ConfigDict, NonNegativeInt, with_config
from typing_extensions import NotRequired, TypedDict


@with_config(ConfigDict(extra="forbid", strict=True))
class ArchitectureBaseHypers(TypedDict):
    name: str
    """The name of the architecture."""
    atomic_types: NotRequired[list[int]]
    """List of atomic types present in the dataset.

    If not provided, it will be inferred from the training dataset.
    This hyperparameter is useful if you envision that in the future
    your model will need to be trained/finetuned on datasets with
    additional atomic types.
    """
    model: NotRequired[dict]
    """Model-specific hyperparameters.
    These depend on the architecture used.
    """
    training: NotRequired[dict]
    """Training-specific hyperparameters.

    These depend on the architecture used.
    """


@with_config(ConfigDict(extra="forbid", strict=True))
class SystemDataKeyHypers(TypedDict):
    """Reference to a per-system scalar stored in a memmap ``.bin`` file."""

    key: str
    """Filename stem of the ``.bin`` file (e.g. ``q`` reads ``q.bin``)."""


@with_config(ConfigDict(extra="forbid", strict=True))
class SystemsHypers(TypedDict):
    """Hyperparameters for the systems in the dataset."""

    read_from: NotRequired[str]
    """Path to the file containing the systems."""
    reader: NotRequired[Optional[Literal["ase", "metatensor"]]]
    """The reader library to use for parsing.

    If ``null`` or not provided, the reader will be guessed
    from the file extension. For example, ``.xyz`` and ``.extxyz``
    will be read by ``ase`` and ``.mts`` will be read by
    ``metatensor``."""
    length_unit: NotRequired[Optional[str]]
    """Unit of lengths in the system file, optional but **highly recommended for
    running simulations**. If not given, no unit conversion will be performed when
    running simulations which may lead to severe errors.

    The list of possible length units is available `here
    <https://docs.metatensor.org/metatomic/latest/torch/reference/misc.html#known-quantities-units>`_."""
    charge: NotRequired[SystemDataKeyHypers]
    """Per-system total charge stored in a memmap ``.bin`` file. Only used
    with memmap datasets and PET's ``system_conditioning`` feature."""
    spin: NotRequired[SystemDataKeyHypers]
    """Per-system spin multiplicity (2S+1) stored in a memmap ``.bin`` file.
    Only used with memmap datasets and PET's ``system_conditioning`` feature."""


@with_config(ConfigDict(extra="forbid", strict=True))
class GradientDict(TypedDict):
    read_from: NotRequired[str]
    """The path to the file for gradient data.

    If not provided, the path from its associated target is used.
    """
    reader: NotRequired[Optional[Literal["ase", "metatensor"]] | dict]
    """The reader library to use for parsing.

    If ``null`` or not provided, the reader will be guessed
    from the file extension. For example, ``.xyz`` and ``.extxyz``
    will be read by ``ase`` and ``.mts`` will be read by
    ``metatensor``."""
    key: NotRequired[str]
    """The key under which the target is stored in the file.

    If not provided, it defaults to the key of the gradient in the
    yaml dataset specification."""


ScalarTargetTypeHyper = Literal["scalar"]


@with_config(ConfigDict(extra="forbid", strict=True))
class CartesianTargetConfig(TypedDict):
    rank: int
    """The rank of the cartesian target (e.g. 1 for vectors)."""


@with_config(ConfigDict(extra="forbid", strict=True))
class CartesianTargetTypeHypers(TypedDict):
    """Hyperparameters to specify cartesian target types."""

    cartesian: CartesianTargetConfig


@with_config(ConfigDict(extra="forbid", strict=True))
class SphericalTargetIrrepsConfig(TypedDict):
    o3_lambda: int
    o3_sigma: float


@with_config(ConfigDict(extra="forbid", strict=True))
class SphericalTargetConfig(TypedDict):
    irreps: list[SphericalTargetIrrepsConfig]


@with_config(ConfigDict(extra="forbid", strict=True))
class SphericalTargetTypeHypers(TypedDict):
    """Hyperparameters to specify spherical target types."""

    spherical: SphericalTargetConfig


@with_config(ConfigDict(extra="forbid", strict=True))
class TargetHypers(TypedDict):
    """Hyperparameters for the targets in the dataset."""

    quantity: NotRequired[str] = ""
    """The quantity that the target represents (e.g., ``energy``,
    ``dipole``). Currently only ``energy`` gets a special treatment from
    ``metatrain``, for any other quantity there is no need to specify
    it."""
    read_from: NotRequired[str]
    """The path to the file containing the target data, defaults
    to ``systems.read_from`` path if not provided."""
    reader: NotRequired[Optional[Literal["ase", "metatensor"]] | dict]
    """The reader library to use for parsing.

    If ``null`` or not provided, the reader will be guessed
    from the file extension. For example, ``.xyz`` and ``.extxyz``
    will be read by ``ase`` and ``.mts`` will be read by
    ``metatensor``."""
    key: NotRequired[str]
    """The key under which the target is stored in the file.

    If not provided, it defaults to the key of the target in the
    yaml dataset specification."""
    unit: NotRequired[str] = ""
    """Unit of the target, optional but **highly recommended for running simulations**.
    If not given, no unit conversion will be performed when running simulations which
    may lead to severe errors.

    The list of possible units is available `here
    <https://docs.metatensor.org/metatomic/latest/torch/reference/misc.html#known-quantities-units>`_."""
    per_atom: NotRequired[bool] = False
    """Whether the target is a per-atom quantity, as opposed to a global
    (per-structure) quantity."""
    type: NotRequired[
        ScalarTargetTypeHyper | CartesianTargetTypeHypers | SphericalTargetTypeHypers
    ]
    """Specifies the type of the target.

    See :ref:`Fitting Generic Targets <fitting-generic-targets>` to understand
    in detail how to specify each target type."""
    num_subtargets: NotRequired[int] = 1
    """Specifies the number of sub-targets that need to be learned as part of
    this target.

    Each subtarget is treated as entirely equivalent by models in metatrain
    and they will often be represented as outputs of the same neural
    network layer. A common use case for this field is when you are learning a
    discretization of a continuous target, such as the grid points of a function. In the
    example above, there are 4000 sub-targets for the density of states (DOS). In
    metatensor, these correspond to the number of properties of the target."""
    description: NotRequired[str] = ""
    """A description of this target. A description is highly recommended if there is
    more than one target with the same :attr:`quantity`."""
    forces: NotRequired[bool | str | GradientDict]
    """Specification for the forces associated with the target.

    See :ref:`gradient-subsection`.
    """
    stress: NotRequired[bool | str | GradientDict]
    """Specification for the stress associated with the target.

    See :ref:`gradient-subsection`.
    """
    virial: NotRequired[bool | str | GradientDict]
    """Specification for the virial associated with the target.

    See :ref:`gradient-subsection`.
    """


@with_config(ConfigDict(extra="forbid", strict=True))
class DatasetDictHypers(TypedDict):
    systems: str | SystemsHypers
    """Path to the dataset file or a dictionary specifying the dataset."""
    targets: dict[str, TargetHypers | str]

    extra_data: NotRequired[dict]
    """Additional data to include from the dataset."""


DatasetSpec = DatasetDictHypers | list[DatasetDictHypers] | str

WandbConfig = dict


@with_config(ConfigDict(extra="forbid", strict=True))
class BaseHypers(TypedDict):
    """Base hyperparameters for all models."""

    architecture: ArchitectureBaseHypers
    """Architecture-specific hyperparameters."""

    device: NotRequired[str]
    """The computational device used for model training. If not provided, ``metatrain``
    automatically chooses the best option by default. The available devices and the
    best device option depend on the model architecture. The easiest way to use this
    parameter is to use either either ``"cpu"``, ``"gpu"``, ``"multi-gpu"``. Internally,
    under the choice ``"gpu"``, the script will automatically choose between ``"cuda"``
    or ``"mps"``."""
    base_precision: NotRequired[Literal[16, 32, 64]]
    """The base precision for float values. For example, a value of ``16`` corresponds
    to the data type ``float16``. The datatypes that are supported as well as the
    default datatype depend on the model architecture used."""
    seed: NotRequired[NonNegativeInt]
    """The seed used for non-deterministic operations. It sets the seed of
    ``numpy.random``, ``random``, ``torch`` and ``torch.cuda``. This parameter is
    important for ensuring reproducibility. If not specified, the seed is generated
    randomly and reported in the log.
    """
    wandb: NotRequired[WandbConfig]
    """Configuration for Weights & Biases logging.

    If ``None``, W&B logging is disabled."""

    training_set: DatasetSpec
    """Specification of the training dataset."""
    validation_set: DatasetSpec | Annotated[int | float, Interval(ge=0.0, lt=1.0)]
    """Specification of the validation dataset."""
    test_set: NotRequired[
        DatasetSpec | Annotated[int | float, Interval(ge=0.0, lt=1.0)]
    ]
    """Specification of the test dataset."""
