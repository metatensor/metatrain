from typing import List, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from omegaconf import DictConfig

from .layout import get_sample_kind, get_target_type


class TargetInfo:
    """A class that contains information about a target.

    :param quantity: The physical quantity of the target (e.g., "energy").
    :param layout: The layout of the target, as a ``TensorMap`` with 0 samples. This
        ``TensorMap`` will be used to retrieve the names of the ``samples``, as well as
        the ``components`` and ``properties`` of the target and their gradients. For
        example, this allows to infer the type of the target (scalar, Cartesian tensor,
        spherical tensor, spherical tensor on atomic basis), whether it is per atom, the
        names of its gradients, etc.
    :param unit: The unit of the target. If :py:obj:`None` the ``unit`` will be set to
        an empty string ``""``.
    """

    def __init__(
        self,
        quantity: str,
        layout: TensorMap,
        unit: Union[None, str] = "",
    ):
        self.quantity = quantity  # float64: otherwise metatensor can't serialize
        self.layout = layout
        self.unit = unit if unit is not None else ""

        # get the layout type
        self.sample_kind = get_sample_kind(layout)
        self.target_type = get_target_type(layout)

        # check that the layout is valid
        self._check_layout(layout)

    @property
    def gradients(self) -> List[str]:
        """Sorted and unique list of gradient names."""
        if self.target_type == "scalar":
            return sorted(self.layout.block().gradients_list())
        else:
            return []

    @property
    def per_atom(self) -> bool:
        """Whether the target is per atom. Also applies to per-pair quantities."""
        # TODO: separate once per-pair quantities are supported in ModelOutput
        return self.sample_kind == "per_atom" or self.sample_kind == "per_pair"

    def __repr__(self):
        return (
            f"TargetInfo(quantity={self.quantity!r}, unit={self.unit!r}, "
            f"layout={self.layout!r}, target_type={self.target_type!r}, "
            f"sample_kind={self.sample_kind!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, TargetInfo):
            raise NotImplementedError(
                "Comparison between a TargetInfo instance and a "
                f"{type(other).__name__} instance is not implemented."
            )
        return (
            self.quantity == other.quantity
            and self.unit == other.unit
            # we can't use metatensor.torch.equal here because we want to allow
            # for potential gradient mismatches (e.g. energy-0nly vs energy+forces)
            and _is_equal_up_to_gradients(self.layout, other.layout)
        )

    def _check_layout(self, layout: TensorMap) -> None:
        """Check that the layout is a valid layout."""

        if self.target_type == "scalar":
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
        elif self.target_type == "cartesian":
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

        elif self.target_type == "spherical":
            if layout.keys.names != ["o3_lambda", "o3_sigma"]:
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

        elif self.target_type == "spherical_atomic_basis":
            o3_lambda_like_dims = [
                name for name in layout.keys.names if name.startswith("o3_lambda")
            ]
            # If "o3_lambda" is in the keys, there should be one "o3_mu" component
            if o3_lambda_like_dims == ["o3_lambda"]:
                for key, block in layout.items():
                    o3_lambda, o3_sigma = (
                        int(key.values[layout.keys.names.index("o3_lambda")].item()),
                        int(key.values[layout.keys.names.index("o3_sigma")].item()),
                    )
                    if o3_sigma not in [-1, 1]:
                        raise ValueError(
                            "The layout ``TensorMap`` of an atomic-basis spherical "
                            "tensor target should have a key dimension 'o3_sigma' "
                            f"that is either -1 or 1. Found '{o3_sigma}' instead."
                        )
                    if o3_lambda < 0:
                        raise ValueError(
                            "The layout ``TensorMap`` of an atomic-basis spherical "
                            "tensor target should have a key sample 'o3_lambda' that "
                            f"is non-negative. Found '{o3_lambda}' instead."
                        )
                    components = block.components
                    if len(components) != 1:
                        raise ValueError(
                            "The layout ``TensorMap`` of an atomic-basis spherical "
                            "tensor target should have a single component."
                        )
                    if len(components[0]) != 2 * o3_lambda + 1:
                        raise ValueError(
                            "Each ``TensorBlock`` of an atomic-basis spherical "
                            "tensor target should have a component with 2*o3_lambda "
                            f"+ 1 elements. Found '{len(components[0])}' elements "
                            "instead."
                        )
                    if len(block.gradients_list()) > 0:
                        raise ValueError(
                            "Gradients of an atomic-basis spherical tensor "
                            "targets are not supported."
                        )

            else:
                raise ValueError(
                    "atomic basis spherical targets should only have 'o3_lambda' "
                    "key dimension for spherical symmetry"
                )

            # check correct atom types dimensions in the keys
            if self.sample_kind == "per_atom":
                assert "center_type" in layout.keys.names, (
                    "per-atom spherical atomic basis targets must have "
                    "'center_type' in the keys."
                )
            elif self.sample_kind == "per_pair":
                assert (
                    "first_atom_type" in layout.keys.names
                    and "second_atom_type" in layout.keys.names
                ), (
                    "per-pair spherical atomic basis targets must have "
                    "'first_atom_type' and 'second_atom_type' in the keys."
                )
                # check traingularization of atom types for per-pair targets
                assert all(
                    layout.keys.values[:, layout.keys.names.index("first_atom_type")]
                    <= layout.keys.values[
                        :, layout.keys.names.index("second_atom_type")
                    ]
                ), (
                    "atom type key dimensions should be triangularized such that"
                    " 'first_atom_type' <= 'second_atom_type'"
                )
            else:
                raise ValueError(
                    "only per-atom and per-pair sample kinds are supported for "
                    "spherical atomic basis targets"
                )
        else:
            raise ValueError(f"unknown target type for target: {self.target_type}.")

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
        if self.target_type != other.target_type:
            return False
        if self.quantity != other.quantity:
            return False
        if self.unit != other.unit:
            return False
        if self.layout.keys.names != other.layout.keys.names:
            return False
        for key, block in self.layout.items():
            if key not in other.layout.keys:
                return False
            other_block = other.layout[key]
            if not block.samples == other_block.samples:
                return False
            if not block.components == other_block.components:
                return False
            if not block.properties == other_block.properties:
                return False
            # gradients are not checked on purpose
        return True


def get_energy_target_info(
    target: DictConfig,
    add_position_gradients: bool = False,
    add_strain_gradients: bool = False,
) -> TargetInfo:
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
                names=["sample", "atom"],
                values=torch.empty((0, 2), dtype=torch.int32),
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

    target_info = TargetInfo(
        quantity="energy",
        unit=target["unit"],
        layout=layout,
    )
    return target_info


def get_generic_target_info(target: DictConfig) -> TargetInfo:
    if target["type"] == "scalar":
        return _get_scalar_target_info(target)
    elif len(target["type"]) == 1 and next(iter(target["type"])).lower() == "cartesian":
        return _get_cartesian_target_info(target)
    elif len(target["type"]) == 1 and next(iter(target["type"])) == "spherical":
        return _get_spherical_target_info(target)
    elif target["type"].startswith("spherical_atomic_basis"):
        raise ValueError(
            "while 'spherical_atomic_basis' targets are supported, "
            "generic TargetInfo cannot be constructed due to the "
            "flexibility in this target's metadata. Please construct "
            "a TargetInfo object directly using metadata inferred from "
            "target TensorMaps."
        )
    else:
        raise ValueError(
            f"Target type {target['type']} is not supported. "
            "Supported types are 'scalar', 'cartesian', 'spherical', "
            "and 'spherical_atomic_basis'"
        )


def _get_scalar_target_info(target: DictConfig) -> TargetInfo:
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
        properties=Labels.range("properties", target["num_subtargets"]),
    )
    layout = TensorMap(
        keys=Labels.single(),
        blocks=[block],
    )

    target_info = TargetInfo(
        quantity=target["quantity"],
        unit=target["unit"],
        layout=layout,
    )
    return target_info


def _get_cartesian_target_info(target: DictConfig) -> TargetInfo:
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
        properties=Labels.range("properties", target["num_subtargets"]),
    )
    layout = TensorMap(
        keys=Labels.single(),
        blocks=[block],
    )

    target_info = TargetInfo(
        quantity=target["quantity"],
        unit=target["unit"],
        layout=layout,
    )
    return target_info


def _get_spherical_target_info(target: DictConfig) -> TargetInfo:
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
            properties=Labels.range("properties", target["num_subtargets"]),
        )
        keys.append([irrep["o3_lambda"], irrep["o3_sigma"]])
        blocks.append(block)

    layout = TensorMap(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor(keys, dtype=torch.int32)),
        blocks=blocks,
    )

    target_info = TargetInfo(
        quantity=target["quantity"],
        unit=target["unit"],
        layout=layout,
    )
    return target_info


def is_auxiliary_output(name: str) -> bool:
    is_auxiliary = (
        name == "features" or name == "energy_ensemble" or name.startswith("mtt::aux::")
    )
    return is_auxiliary


def _is_equal_up_to_gradients(
    layout1: TensorMap,
    layout2: TensorMap,
) -> bool:
    # checks if the two layouts are equal up to gradients
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
