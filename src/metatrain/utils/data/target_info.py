import itertools
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap, equal_metadata
from metatomic.torch import ModelOutput
from omegaconf import DictConfig

# We explicitly import device because otherwise torch.jit.script doesn't
# recognize the torch.device type
from torch import device as _torch_device


class TargetInfo:
    """A class that contains information about a target.

    :param layout: The layout of the target, as a ``TensorMap`` with 0 samples.
        This ``TensorMap`` will be used to retrieve the names of
        the ``samples``, as well as the ``components`` and ``properties`` of the
        target and their gradients. For example, this allows to infer the type of
        the target (scalar, Cartesian tensor, spherical tensor), whether it is per
        atom, the names of its gradients, etc.
    :param quantity: Quantity of the target (e.g. "energy", "dipole", …). If this is an
        empty string, no unit conversion will be performed.

        The list of possible quantities is available `here`_.
    :param unit: Unit of the target. If this is an empty string, no unit conversion will
        be performed.

        The list of possible units is available `here`_.
    :param description: A description of this target. A description is especially
        recommended for non-standard outputs and variants of a unit.

    .. _here: https://docs.metatensor.org/metatomic/latest/torch/reference/misc.html#known-quantities-units
    """

    def __init__(
        self,
        layout: TensorMap,
        quantity: str = "",
        unit: str = "",
        description: str = "",
    ):
        # one of these will be set to True inside the _check_layout method
        self.is_scalar = False
        self.is_cartesian = False
        self.is_spherical = False
        self.is_atomic_basis = False

        self._check_layout(layout)
        self.layout = layout

        # verify that `quantity`, `unit` and `description` are valid for metatomic
        _ = ModelOutput(quantity=quantity, unit=unit, description=description)

        self.quantity = quantity
        self.unit = unit
        self.description = description

        self.blocks_shape: Dict[str, List[int]] = {}
        self._set_blocks_shape()

    @property
    def gradients(self) -> List[str]:
        """Sorted and unique list of gradient names."""
        if self.is_scalar:
            return sorted(self.layout.block().gradients_list())
        else:
            return []

    @property
    def per_atom(self) -> bool:
        """Whether the target is a per-atom quantity or system wide.

        This is provided for backward compatibility, since ``sample_kind``
        is now a more general version of it. If ``sample_kind`` is not
        one of "system" or "atom", trying to get ``per_atom`` will raise
        an error.
        """
        sample_kind = self.sample_kind

        if sample_kind == "atom":
            return True
        elif sample_kind == "system":
            return False
        else:
            raise ValueError(
                f"Cannot determine whether the target is per-atom or system-wide "
                f"because its sample_kind is '{sample_kind}'."
            )

    @property
    def sample_kind(self) -> Literal["system", "atom"]:
        """The kind of sample the target corresponds to."""
        sample_names = self.layout.block(0).samples.names
        if "atom" in sample_names:
            return "atom"
        else:
            return "system"

    @property
    def component_labels(self) -> List[List[Labels]]:
        """The labels of the components of the target."""
        return [block.components for block in self.layout.blocks()]

    @property
    def property_labels(self) -> List[Labels]:
        """The labels of the properties of the target."""
        return [block.properties for block in self.layout.blocks()]

    def __repr__(self) -> str:
        return (
            f"TargetInfo(layout={self.layout}, quantity='{self.quantity}', "
            f"unit='{self.unit}', description='{self.description}')"
        )

    @torch.jit.unused
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TargetInfo):
            return False
        return (
            self.quantity == other.quantity
            and self.unit == other.unit
            and self.description == other.description
            # we can't use metatensor.torch.equal here because we want to allow
            # for potential gradient mismatches (e.g. energy-0nly vs energy+forces)
            and _is_equal_up_to_gradients(self.layout, other.layout)
        )

    def _check_layout(self, layout: TensorMap) -> None:
        """
        Check that the layout is a valid layout.

        :param layout: The layout TensorMap to check.
        """

        # examine basic properties of all blocks
        for block in layout.blocks():
            for sample_name in block.samples.names:
                if sample_name not in ["system", "atom"]:
                    raise ValueError(
                        "The layout ``TensorMap`` of a target should only have samples "
                        "named 'system' or 'atom', but found "
                        f"'{sample_name}' instead."
                    )
            if len(block.values) != 0:
                raise ValueError(
                    "The layout ``TensorMap`` of a target should have 0 "
                    f"samples, but found {len(block.values)} samples."
                )

        # examine the components of the first block to decide whether this is
        # a scalar, a Cartesian tensor or a spherical tensor

        if len(layout) == 0:
            raise ValueError(
                "The layout ``TensorMap`` of a target should have at least one "
                "block, but found 0 blocks."
            )
        components_first_block = layout.block(0).components
        if len(components_first_block) == 0:
            self.is_scalar = True
        elif components_first_block[0].names[0].startswith("xyz"):
            self.is_cartesian = True
        elif components_first_block[0].names[0].startswith("o3_mu"):
            self.is_spherical = True
        else:
            raise ValueError(
                "The layout ``TensorMap`` of a target should be "
                "either scalars, Cartesian tensors or spherical tensors. The type of "
                "the target could not be determined."
            )

        if self.is_scalar:
            if layout.keys.names != ["_"]:
                raise ValueError(
                    "The layout ``TensorMap`` of a scalar target should have "
                    "a single key sample named '_'."
                )
            if len(layout.blocks()) != 1:
                raise ValueError(
                    "The layout ``TensorMap`` of a scalar target should have "
                    "a single block."
                )
            gradients_names = layout.block(0).gradients_list()
            for gradient_name in gradients_names:
                if gradient_name not in ["positions", "strain"]:
                    raise ValueError(
                        "Only `positions` and `strain` gradients are supported for "
                        "scalar targets. "
                        f"Found '{gradient_name}' instead."
                    )
        if self.is_cartesian:
            if layout.keys.names != ["_"]:
                raise ValueError(
                    "The layout ``TensorMap`` of a Cartesian tensor target should have "
                    "a single key sample named '_'."
                )
            if len(layout.blocks()) != 1:
                raise ValueError(
                    "The layout ``TensorMap`` of a Cartesian tensor target should have "
                    "a single block."
                )
            if len(layout.block(0).gradients_list()) > 0:
                raise ValueError(
                    "Gradients of Cartesian tensor targets are not supported."
                )

        if self.is_spherical:
            # The following loop is ugly but it is for torchscript compatibility.
            keys_names: list[str] = []
            unknown_keys: list[str] = []
            lambdas_indices: list[int] = []
            sigmas_indices: list[int] = []
            for name in layout.keys.names:
                if name.endswith("atom_type"):
                    self.is_atomic_basis = True
                    continue
                elif name.startswith("o3_lambda"):
                    lambdas_indices.append(len(keys_names))
                    keys_names.append(name)
                elif name.startswith("o3_sigma"):
                    sigmas_indices.append(len(keys_names))
                    keys_names.append(name)
                else:
                    unknown_keys.append(name)

            if len(unknown_keys) > 0:
                raise ValueError(
                    "The layout ``TensorMap`` of a spherical tensor target should only "
                    "have keys named 'o3_lambda', 'o3_sigma' or 'atom_type'"
                    f" Found unknown key names: '{unknown_keys}'."
                )

            if len(lambdas_indices) != len(sigmas_indices):
                raise ValueError(
                    "The layout ``TensorMap`` of a spherical tensor target should have "
                    "the same number of 'o3_lambda' and 'o3_sigma' names."
                    f" Found {len(lambdas_indices)} 'o3_lambda' keys and "
                    f"{len(sigmas_indices)} 'o3_sigma' keys."
                    f" Keys found: {layout.keys.names}."
                )

            for key, block in layout.items():
                lambdas = key.values[lambdas_indices]
                sigmas = key.values[sigmas_indices]
                components = block.components

                if len(components) != len(lambdas):
                    raise ValueError(
                        "The layout ``TensorMap`` of a spherical tensor target"
                        " should have as many components as 'o3_lambda' keys"
                        " in the layout ``TensorMap``."
                        f" Found {len(components)} components and {len(lambdas)} "
                        f"'o3_lambda' keys."
                    )

                for i, o3_sigma in enumerate(sigmas):
                    if o3_sigma not in [-1, 1]:
                        raise ValueError(
                            "The layout ``TensorMap`` of a spherical tensor target"
                            "should have 'o3_sigma' key values that are either -1 or 1."
                            f" Found '{o3_sigma}' instead for {layout.keys.names[i]}."
                        )
                for i, o3_lambda in enumerate(lambdas):
                    if o3_lambda < 0:
                        raise ValueError(
                            "The layout ``TensorMap`` of a spherical tensor target"
                            "should have 'o3_lambda' key values that are non-negative"
                            " integers."
                            f" Found '{o3_lambda}' instead for {layout.keys.names[i]}."
                        )

                    if len(components[i]) != 2 * o3_lambda + 1:
                        raise ValueError(
                            "The components of a spherical tensor target should have "
                            f"2*o3_lambda + 1 elements. For {layout.keys.names[i]} with"
                            f" o3_lambda={o3_lambda}, found '{len(components[0])}'"
                            " elements instead."
                        )

                if len(block.gradients_list()) > 0:
                    raise ValueError(
                        "Gradients of spherical tensor targets are not supported."
                    )

    def _set_blocks_shape(self) -> None:
        """Set the attribute storing the shapes of the blocks in layout."""
        for key, block in self.layout.items():
            dict_key = self.quantity
            for n, k in zip(key.names, key.values, strict=True):
                dict_key += f"_{n}_{int(k)}"
            self.blocks_shape[dict_key] = [
                len(comp.values) for comp in block.components
            ] + [len(block.properties.values)]

    def is_compatible_with(self, other: "TargetInfo") -> bool:
        """Check if two targets are compatible.

        Two target infos are compatible if they have the same quantity, unit,
        and layout, except for gradients. This method can be used to check if two
        target infos with the same name can correspond to the same output
        in a model.

        :param other: The target info to compare with.
        :return: :py:obj:`True` if the two target infos are compatible,
            :py:obj:`False` otherwise.
        """

        if self.quantity != other.quantity:
            return False
        if self.unit != other.unit:
            return False
        if self.description != other.description:
            return False
        return equal_metadata(self.layout, other.layout, check_gradients=False)

    @property
    def device(self) -> _torch_device:
        """Return the device of the target info's layout."""
        return self.layout.device

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> "TargetInfo":
        """
        Return a copy with all tensors moved to the device and dtype.

        :param device: The device to move the tensors to.
        :param dtype: The dtype to move the tensors to.
        :return: A copy of the TargetInfo with all tensors moved to the device and
            dtype.
        """
        new_layout = self.layout.to(device=device, dtype=dtype)
        return TargetInfo(
            layout=new_layout,
            quantity=self.quantity,
            unit=self.unit,
            description=self.description,
        )

    @torch.jit.unused
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Set the state of the target info.

        :param state: The state to set.
        """

        self.layout = state["layout"]
        self.is_scalar = state["is_scalar"]
        self.is_cartesian = state["is_cartesian"]
        self.is_spherical = state["is_spherical"]
        self.is_atomic_basis = state.get("is_atomic_basis", False)

        self.quantity = state["quantity"]
        self.unit = state["unit"]
        self.description = state.get("description", "")

        # For backward compatibility, if blocks_shape is not in the state,
        # we build it.
        if "blocks_shape" not in state:
            self.blocks_shape = {}
            self._set_blocks_shape()
        else:
            self.blocks_shape = state["blocks_shape"]


def get_energy_target_info(
    target_name: str,
    target: DictConfig,
    add_position_gradients: bool = False,
    add_strain_gradients: bool = False,
) -> TargetInfo:
    """Get an empty TargetInfo with the layout of an energy target.

    :param target_name: Not used, but kept for consistency with
        :py:func:`get_generic_target_info`.
    :param target: The configuration of the target.
    :param add_position_gradients: Whether to add position gradients to the layout.
    :param add_strain_gradients: Whether to add strain gradients to the layout.

    :return: A `TargetInfo` with the layout of an energy target.
    """
    block = TensorBlock(
        # float64: otherwise metatensor can't serialize
        values=torch.empty(0, 1, dtype=torch.float64),
        samples=Labels(
            names=["system"],
            values=torch.empty((0, 1), dtype=torch.int32),
        ),
        components=[],
        properties=Labels.range("energy", 1),
    )

    if add_position_gradients:
        position_gradient_block = TensorBlock(
            # float64: otherwise metatensor can't serialize
            values=torch.empty(0, 3, 1, dtype=torch.float64),
            samples=Labels(
                names=["sample", "atom"],
                values=torch.empty((0, 2), dtype=torch.int32),
            ),
            components=[
                Labels(
                    names=["xyz"],
                    values=torch.arange(3, dtype=torch.int32).reshape(-1, 1),
                ),
            ],
            properties=Labels.range("energy", 1),
        )
        block.add_gradient("positions", position_gradient_block)

    if add_strain_gradients:
        strain_gradient_block = TensorBlock(
            # float64: otherwise metatensor can't serialize
            values=torch.empty(0, 3, 3, 1, dtype=torch.float64),
            samples=Labels(
                names=["sample"],
                values=torch.empty((0, 1), dtype=torch.int32),
            ),
            components=[
                Labels(
                    names=["xyz_1"],
                    values=torch.arange(3, dtype=torch.int32).reshape(-1, 1),
                ),
                Labels(
                    names=["xyz_2"],
                    values=torch.arange(3, dtype=torch.int32).reshape(-1, 1),
                ),
            ],
            properties=Labels.range("energy", 1),
        )
        block.add_gradient("strain", strain_gradient_block)

    layout = TensorMap(
        keys=Labels.single(),
        blocks=[block],
    )

    return TargetInfo(
        layout=layout,
        quantity="energy",
        unit=target["unit"],
        description=target.get("description", ""),
    )


def get_generic_target_info(target_name: str, target: DictConfig) -> TargetInfo:
    """Get an empty TargetInfo with the appropriate layout.

    :param target_name: The name of the target.
    :param target: The configuration of the target. Based on the ``type`` field,
        this function will create a layout for the appropriate type of target.

    :return: A `TargetInfo` with the layout of the target.
    """
    if target["type"] == "scalar":
        return _get_scalar_target_info(target_name, target)
    elif len(target["type"]) == 1 and next(iter(target["type"])).lower() == "cartesian":
        return _get_cartesian_target_info(target_name, target)
    elif len(target["type"]) == 1 and next(iter(target["type"])) == "spherical":
        return _get_spherical_target_info(target_name, target)
    else:
        raise ValueError(
            f"Target type {target['type']} is not supported. "
            "Supported types are 'scalar', 'cartesian' and 'spherical'."
        )


def _get_scalar_target_info(target_name: str, target: DictConfig) -> TargetInfo:
    sample_names = ["system"]
    if target["sample_kind"] == "atom":
        sample_names.append("atom")

    block = TensorBlock(
        # float64: otherwise metatensor can't serialize
        values=torch.empty(0, target["num_subtargets"], dtype=torch.float64),
        samples=Labels(
            names=sample_names,
            values=torch.empty((0, len(sample_names)), dtype=torch.int32),
        ),
        components=[],
        properties=Labels.range(
            # remove variant and/or mtt:: prefix from target name
            (target_name.split("/")[0] if "/" in target_name else target_name).replace(
                "mtt::", ""
            ),
            target["num_subtargets"],
        ),
    )
    layout = TensorMap(
        keys=Labels.single(),
        blocks=[block],
    )

    return TargetInfo(
        layout=layout,
        quantity=target["quantity"],
        unit=target["unit"],
        description=target.get("description", ""),
    )


def _get_cartesian_target_info(target_name: str, target: DictConfig) -> TargetInfo:
    sample_names = ["system"]
    if target["sample_kind"] == "atom":
        sample_names.append("atom")

    cartesian_key = next(iter(target["type"]))

    if target["type"][cartesian_key]["rank"] == 1:
        components = [Labels(["xyz"], torch.arange(3).reshape(-1, 1))]
    else:
        components = []
        for component in range(1, target["type"][cartesian_key]["rank"] + 1):
            components.append(
                Labels(
                    names=[f"xyz_{component}"],
                    values=torch.arange(3, dtype=torch.int32).reshape(-1, 1),
                )
            )

    block = TensorBlock(
        # float64: otherwise metatensor can't serialize
        values=torch.empty(
            [0] + [3] * len(components) + [target["num_subtargets"]],
            dtype=torch.float64,
        ),
        samples=Labels(
            names=sample_names,
            values=torch.empty((0, len(sample_names)), dtype=torch.int32),
        ),
        components=components,
        properties=Labels.range(
            # remove variant and/or mtt:: prefix from target name
            (target_name.split("/")[0] if "/" in target_name else target_name).replace(
                "mtt::", ""
            ),
            target["num_subtargets"],
        ),
    )
    layout = TensorMap(
        keys=Labels.single(),
        blocks=[block],
    )

    return TargetInfo(
        layout=layout,
        quantity=target["quantity"],
        unit=target["unit"],
        description=target.get("description", ""),
    )


def _get_spherical_target_info(target_name: str, target: DictConfig) -> TargetInfo:
    sample_names = ["system"]
    if target["sample_kind"] == "atom":
        sample_names.append("atom")

    irreps = target["type"]["spherical"]["irreps"]
    product = target["type"]["spherical"].get("product", None)

    keys = []
    blocks = []

    is_atomic_basis = isinstance(irreps, (dict, DictConfig))

    # Define the names of the keys in the tensormap
    if product in [None, "coupled"]:
        keys_names = ["o3_lambda", "o3_sigma"]
    elif product == "cartesian":
        keys_names = ["o3_lambda_1", "o3_lambda_2", "o3_sigma_1", "o3_sigma_2"]
    else:
        raise ValueError(
            f"Unknown product {product!r}. Supported values are 'cartesian', "
            "'coupled' and None."
        )

    if is_atomic_basis:
        keys_names.append("atom_type")

    # Build the tensormap blocks, and store their corresponding keys.
    if not is_atomic_basis:
        irreps_iter, properties = _get_spherical_irreps_iter(irreps, target, product)

        for irrep, props in zip(irreps_iter, properties, strict=True):
            block = _build_spherical_target_block(
                sample_names=sample_names,
                target_name=target_name,
                irreps=irrep,
                properties=props,
            )
            _, lambdas, sigmas = torch.tensor(irrep).T

            keys.append([*lambdas, *sigmas])
            blocks.append(block)
    else:
        # Loop over atomic types
        for atom_type, atom_irreps in irreps.items():
            # For each atomic type, essentially do the same as for the case with
            # no types. We simply have an extra key corresponding to the atomic type.
            irreps_iter, properties = _get_spherical_irreps_iter(
                atom_irreps, target, product=product
            )

            for irrep, props in zip(irreps_iter, properties, strict=True):
                block = _build_spherical_target_block(
                    sample_names=sample_names,
                    target_name=target_name,
                    irreps=irrep,
                    properties=props,
                )
                _, lambdas, sigmas = torch.tensor(irrep).T
                keys.append([*lambdas, *sigmas, atom_type])
                blocks.append(block)

    layout = TensorMap(
        keys=Labels(keys_names, torch.tensor(keys, dtype=torch.int32)),
        blocks=blocks,
    )

    info = TargetInfo(
        layout=layout,
        quantity=target["quantity"],
        unit=target["unit"],
        description=target.get("description", ""),
    )
    return info


def _build_spherical_target_block(
    sample_names: List[str],
    target_name: str,
    irreps: list[tuple[int, int, int]],
    properties: Optional[Labels] = None,
) -> TensorBlock:
    """Builds an empty `TensorBlock` for some given spherical irreps.

    :param sample_names: The names of the sample labels of the block.
    :param target_name: The name of the target (used for the properties labels).
    :param irreps: A list of irreps, where each irrep is a tuple of the form
      (num_subtargets, o3_lambda, o3_sigma).
    :param properties: An optional ``Labels`` object containing the properties
        labels of the block. If `None`, the properties labels will be generated
        automatically with names ``n`` for rank 1 blocks and ``n_1``,
        ``n_2``, ... for higher rank blocks.
    :return: an empty `TensorBlock` with the appropriate samples, components
      and properties labels for the given irreps
    """
    rank = len(irreps)

    if rank == 1:
        lambd = irreps[0][1]
        n_properties = irreps[0][0]
        components = [
            Labels(
                names=["o3_mu"],
                values=torch.arange(-lambd, lambd + 1, dtype=torch.int32).reshape(
                    -1, 1
                ),
            )
        ]
        if properties is None:
            properties = Labels.range("n", n_properties)
    else:
        n_properties = torch.tensor(irreps)[:, 0].prod()
        components = []
        for component in range(1, rank + 1):
            lambd = irreps[component - 1][1]
            components.append(
                Labels(
                    names=[f"o3_mu_{component}"],
                    values=torch.arange(-lambd, lambd + 1, dtype=torch.int32).reshape(
                        -1, 1
                    ),
                )
            )

        if properties is None:
            properties_values = list(
                itertools.product(*[np.arange(irrep[0]) for irrep in irreps])
            )
            properties_values = torch.tensor(properties_values, dtype=torch.int32)

            properties = Labels(
                names=[f"n_{component}" for component in range(1, rank + 1)],
                values=properties_values,
            )

    block = TensorBlock(
        # float64: otherwise metatensor can't serialize
        values=torch.empty(
            0,
            *[len(component) for component in components],
            n_properties,
            dtype=torch.float64,
        ),
        samples=Labels(
            names=sample_names,
            values=torch.empty((0, len(sample_names)), dtype=torch.int32),
        ),
        components=components,
        properties=properties,
    )
    return block


def _get_spherical_irreps_iter(
    irreps: list[dict],
    target: dict,
    product: Optional[Literal["cartesian", "coupled"]] = None,
) -> tuple[list[list[tuple[int, int, int]]], list[Labels]]:
    if product not in [None, "cartesian", "coupled"]:
        raise ValueError(
            f"Product '{product}' is not supported. Supported products are"
            " 'cartesian' and 'coupled'."
        )

    irreps_iter = [
        [
            (
                irrep.get("num", 1) * target["num_subtargets"],
                irrep["o3_lambda"],
                irrep["o3_sigma"],
            )
        ]
        for irrep in irreps
    ]
    if product in ["cartesian", "coupled"]:
        irreps_iter = [
            i_irrep + j_irrep
            for i_irrep, j_irrep in itertools.product(irreps_iter, repeat=2)
        ]
    if product == "coupled":
        coupled_blocks: dict[tuple[int, int], int] = defaultdict(int)
        coupled_properties = defaultdict(list)
        for (num1, l1, sig1), (num2, l2, sig2) in irreps_iter:
            for lam in range(abs(l1 - l2), l1 + l2 + 1):
                sig = sig1 * sig2 * (-1) ** (l1 + l2 + lam)

                # If the two spherical harmonics are in the same center,
                # the coupled values for sigma = -1 are zero. This is true
                # for the Hamiltonian, density matrix and overlap, but might
                # be not true for other targets. In that case, we will need to
                # add an input to specify if the target has this property.
                # For per-pair targets (not introduced yet), this will not be
                # the case.
                if sig == -1:
                    continue

                coupled_blocks[(lam, sig)] += num1 * num2

                # Build properties
                added_properties = []
                for i in range(num1):
                    for j in range(num2):
                        added_properties.append((l1, l2, i, j))
                coupled_properties[(lam, sig)].extend(added_properties)

        irreps_iter = [[(num, lam, sig)] for (lam, sig), num in coupled_blocks.items()]
        properties = [
            Labels(
                names=["l_1", "l_2", "n_1", "n_2"],
                values=torch.tensor(props, dtype=torch.int32),
            )
            for props in coupled_properties.values()
        ]
    else:
        properties = [None] * len(irreps_iter)

    return irreps_iter, properties


def is_auxiliary_output(name: str) -> bool:
    """
    Check if a target name corresponds to an auxiliary output.

    :param name: The name of the target to check.

    :return: `True` if the target is an auxiliary output, `False` otherwise.
    """
    is_auxiliary = (
        name == "features" or name == "energy_ensemble" or name.startswith("mtt::aux::")
    )
    return is_auxiliary


def _is_equal_up_to_gradients(
    layout1: TensorMap,
    layout2: TensorMap,
) -> bool:
    """
    Check if the two layouts are equal up to gradients.

    This includes checking the values, this is why we can't use
    ``metatensor.torch.equal_metadata``. It ignores the values
    of the gradients so we can't use ``metatensor.torch.equal`` either.

    :param layout1: The first layout to compare.
    :param layout2: The second layout to compare.

    :return: `True` if the two layouts are equal up to gradients,
        `False` otherwise.
    """
    if len(layout1) != len(layout2):
        return False
    for key in layout1.keys:
        if key not in layout2.keys:
            return False
        block1 = layout1[key]
        block2 = layout2[key]
        if block1.samples != block2.samples:
            return False
        if block1.components != block2.components:
            return False
        if block1.properties != block2.properties:
            return False
        if not torch.allclose(block1.values, block2.values):
            return False
    return True
