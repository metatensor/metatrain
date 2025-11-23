from typing import Any, Dict, List, Optional

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
        """Whether the target is per atom."""
        return "atom" in self.layout.block(0).samples.names

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
        elif (
            len(components_first_block) == 1
            and components_first_block[0].names[0] == "o3_mu"
        ):
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
            if len(layout.keys.names) < 2 or layout.keys.names[:2] != [
                "o3_lambda",
                "o3_sigma",
            ]:
                raise ValueError(
                    "The layout ``TensorMap`` of a spherical tensor target "
                    "should have  two keys named 'o3_lambda' and 'o3_sigma'."
                    f"Found '{layout.keys.names}' instead."
                )
            for key, block in layout.items():
                o3_lambda, o3_sigma = (
                    int(key.values[0].item()),
                    int(key.values[1].item()),
                )
                if o3_sigma not in [-1, 1]:
                    raise ValueError(
                        "The layout ``TensorMap`` of a spherical tensor target should "
                        "have a key sample 'o3_sigma' that is either -1 or 1."
                        f"Found '{o3_sigma}' instead."
                    )
                if o3_lambda < 0:
                    raise ValueError(
                        "The layout ``TensorMap`` of a spherical tensor target should "
                        "have a key sample 'o3_lambda' that is non-negative."
                        f"Found '{o3_lambda}' instead."
                    )
                components = block.components
                if len(components) != 1:
                    raise ValueError(
                        "The layout ``TensorMap`` of a spherical tensor target should "
                        "have a single component."
                    )
                if len(components[0]) != 2 * o3_lambda + 1:
                    raise ValueError(
                        "Each ``TensorBlock`` of a spherical tensor target should have "
                        "a component with 2*o3_lambda + 1 elements."
                        f"Found '{len(components[0])}' elements instead."
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
    if target["per_atom"]:
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
            target_name.replace("mtt::", ""), target["num_subtargets"]
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
    if target["per_atom"]:
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
            target_name.replace("mtt::", ""), target["num_subtargets"]
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
    if target["per_atom"]:
        sample_names.append("atom")

    irreps = target["type"]["spherical"]["irreps"]
    keys = []
    blocks = []
    for irrep in irreps:
        components = [
            Labels(
                names=["o3_mu"],
                values=torch.arange(
                    -irrep["o3_lambda"], irrep["o3_lambda"] + 1, dtype=torch.int32
                ).reshape(-1, 1),
            )
        ]
        block = TensorBlock(
            # float64: otherwise metatensor can't serialize
            values=torch.empty(
                0,
                2 * irrep["o3_lambda"] + 1,
                target["num_subtargets"],
                dtype=torch.float64,
            ),
            samples=Labels(
                names=sample_names,
                values=torch.empty((0, len(sample_names)), dtype=torch.int32),
            ),
            components=components,
            properties=Labels.range(
                target_name.replace("mtt::", ""), target["num_subtargets"]
            ),
        )
        keys.append([irrep["o3_lambda"], irrep["o3_sigma"]])
        blocks.append(block)

    layout = TensorMap(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor(keys, dtype=torch.int32)),
        blocks=blocks,
    )

    return TargetInfo(
        layout=layout,
        quantity=target["quantity"],
        unit=target["unit"],
        description=target.get("description", ""),
    )


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
