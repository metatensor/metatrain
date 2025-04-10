from typing import List, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from omegaconf import DictConfig


class TargetInfo:
    """A class that contains information about a target.

    :param quantity: The physical quantity of the target (e.g., "energy").
    :param layout: The layout of the target, as a ``TensorMap`` with 0 samples.
        This ``TensorMap`` will be used to retrieve the names of
        the ``samples``, as well as the ``components`` and ``properties`` of the
        target and their gradients. For example, this allows to infer the type of
        the target (scalar, Cartesian tensor, spherical tensor), whether it is per
        atom, the names of its gradients, etc.
    :param unit: The unit of the target. If :py:obj:`None` the ``unit`` will be set to
        an empty string ``""``.
    """

    def __init__(
        self,
        quantity: str,
        layout: TensorMap,
        unit: Union[None, str] = "",
    ):
        # one of these will be set to True inside the _check_layout method
        self.is_scalar = False
        self.is_cartesian = False
        self.is_spherical = False
        self.is_atomic_basis_spherical_node = False
        self.is_atomic_basis_spherical_edge = False

        self._check_layout(layout)

        self.quantity = quantity  # float64: otherwise metatensor can't serialize
        self.layout = layout
        self.unit = unit if unit is not None else ""

    @property
    def gradients(self) -> List[str]:
        """Sorted and unique list of gradient names."""
        if self.is_scalar:
            return sorted(self.layout.block().gradients_list())
        else:
            return []

    @property
    def per_atom(self) -> bool:
        """Whether the target is per atom. Also applies to per atom pair quantities."""
        return "atom" in self.layout.block(0).samples.names or (
            "first_atom" in self.layout.block(0).samples.names
            and "second_atom" in self.layout.block(0).samples.names
            and "cell_shift_a" in self.layout.block(0).samples.names
            and "cell_shift_b" in self.layout.block(0).samples.names
            and "cell_shift_c" in self.layout.block(0).samples.names
        )

    def __repr__(self):
        return (
            f"TargetInfo(quantity={self.quantity!r}, unit={self.unit!r}, "
            f"layout={self.layout!r})"
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
            and metatensor.torch.equal(self.layout, other.layout)
        )

    def _check_layout(self, layout: TensorMap) -> None:
        """Check that the layout is a valid layout."""

        valid_sample_names = [
            ["system"],
            ["system", "atom"],
            [
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
        ]
        if layout.sample_names not in valid_sample_names:
            raise ValueError(
                "The layout ``TensorMap`` of a target should have samples "
                f"names corresponding to one of: {valid_sample_names}, but found "
                f"'{layout.sample_names}' instead."
            )

        # examine basic properties of all blocks
        for block in layout.blocks():
            if len(block.values) != 0:
                raise ValueError(
                    "The layout ``TensorMap`` of a target should have 0 "
                    f"samples, but found {len(block.values)} samples."
                )

        # examine the components of the first block to decide whether this is
        # a scalar, a Cartesian tensor, a spherical tensor, or an atomic-basis
        # spherical tensor

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
            # keys of just "o3_lambda" and "o3_sigma" is a spherical target
            if layout.keys.names == [
                "o3_lambda",
                "o3_sigma",
            ]:
                self.is_spherical = True

            # keys with "o3_lambda" and "o3_sigma" and keys that indicate center types
            # (i.e. "center_type", "first_atom_type", "second_atom_type") plus
            # (optionally) other arbitrary key dimensions (i.e. "s2_pi", etc) is an
            # atomic-basis spherical target.
            elif "o3_lambda" in layout.keys.names and "o3_sigma" in layout.keys.names:
                if "center_type" in layout.keys.names:
                    self.is_atomic_basis_spherical_node = True
                elif (
                    "first_atom_type" in layout.keys.names
                    and "second_atom_type" in layout.keys.names
                ):
                    self.is_atomic_basis_spherical_edge = True
                else:
                    raise ValueError(
                        "invalid key names. If specifying a spherical target "
                        "on an atomic basis, the keys should include either "
                        "'center_type' or ['first_atom_type', 'second_atom_type']"
                    )
            else:
                raise ValueError(
                    f"invalid key names: {layout.keys.names}. "
                    "Targets with 1 'o3_mu' components axis are "
                    "treated as either spherical targets or atomic "
                    "spherical basis targets, and are expected to have "
                    "at least 'o3_lambda' and 'o3_sigma' key dimensions."
                )
        elif (
            len(components_first_block) == 2
            and components_first_block[0].names[0] == "o3_mu_1"
            and components_first_block[1].names[0] == "o3_mu_2"
        ):
            if (
                "o3_lambda_1" in layout.keys.names
                and "o3_lambda_2" in layout.keys.names
                and "o3_sigma" in layout.keys.names
            ):
                if "center_type" in layout.keys.names:
                    self.is_atomic_basis_spherical_node = True
                elif (
                    "first_atom_type" in layout.keys.names
                    and "second_atom_type" in layout.keys.names
                ):
                    self.is_atomic_basis_spherical_edge = True
                else:
                    raise ValueError(
                        "invalid key names. If specifying a spherical target "
                        "on an atomic basis, the keys should include either "
                        "'center_type' or ['first_atom_type', 'second_atom_type']"
                    )
            else:
                raise ValueError(
                    f"invalid key names: {layout.keys.names}. "
                    "Targets with 2 'o3_mu_{x}' components axes are "
                    "treated as atomic spherical basis targets, and are "
                    "expected to have at least 'o3_lambda_1', 'o3_lambda_2', "
                    "and 'o3_sigma' key dimensions."
                )
        else:
            raise ValueError(
                "The layout ``TensorMap`` of a target should be "
                "either scalars, Cartesian tensors, spherical tensors, "
                "or atomic-basis spherical targets. The type of "
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

        if self.is_atomic_basis_spherical_node or self.is_atomic_basis_spherical_edge:
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

            # If "o3_lambda_1" and "o3_lambda_2" is in the keys, there should be two
            # components, "o3_mu_1" and "o3_mu_2"
            elif o3_lambda_like_dims == ["o3_lambda_1", "o3_lambda_2"]:
                for key, block in layout.items():
                    o3_lambda_1, o3_lambda_2, o3_sigma = (
                        int(key.values[layout.keys.names.index("o3_lambda_1")].item()),
                        int(key.values[layout.keys.names.index("o3_lambda_2")].item()),
                        int(key.values[layout.keys.names.index("o3_sigma")].item()),
                    )
                    if o3_sigma not in [-1, 1]:
                        raise ValueError(
                            "The layout ``TensorMap`` of a spherical tensor target "
                            "should have a key sample 'o3_sigma' that is either -1 "
                            f"or 1. Found '{o3_sigma}' instead."
                        )
                    if len(block.components) != 2:
                        raise ValueError(
                            "The layout ``TensorMap`` of an atomic-basis spherical "
                            "tensor target should have a single component."
                        )
                    for o3_mu_i, o3_lambda in enumerate([o3_lambda_1, o3_lambda_2]):
                        if o3_lambda < 0:
                            raise ValueError(
                                "The layout ``TensorMap`` of an atomic-basis spherical "
                                "tensor target should have a key dimension 'o3_lambda' "
                                f"that is non-negative. Found '{o3_lambda}' instead."
                            )
                        components = block.components[o3_mu_i]

                        if len(components) != 2 * o3_lambda + 1:
                            raise ValueError(
                                "Each ``TensorBlock`` of an atomic-basis spherical "
                                f"tensor target should have component axis {o3_mu_i} "
                                f"with 2*o3_lambda_{o3_mu_i} + 1 elements. Found "
                                f"'{len(components)}' elements instead."
                            )
                    if len(block.gradients_list()) > 0:
                        raise ValueError(
                            "Gradients of an atomic-basis spherical "
                            "targets are not supported."
                        )

            else:
                raise ValueError(
                    "atomic basis spherical targets should only have 'o3_lambda' or "
                    "['o3_lambda_1', 'o3_lambda_2'] key dimensions for spherical "
                    "symmetry"
                )

            # For edges, check that atom types are triangularized in the keys
            if (
                "first_atom" in layout.sample_names
                and "second_atom" in layout.sample_names
            ):
                assert (
                    "first_atom_type" in layout.keys.names
                    and "second_atom_type" in layout.keys.names
                ), (
                    "atomic basis spherical edge targets must have "
                    "'first_atom_type' and 'second_atom_type' in the keys, and "
                    "'first_atom' and 'second_atom' in the samples"
                )
                assert all(
                    layout.keys.values[:, layout.keys.names.index("first_atom_type")]
                    <= layout.keys.values[
                        :, layout.keys.names.index("second_atom_type")
                    ]
                ), (
                    "atom type key dimensions should be triangularized such that"
                    " 'first_atom_type' <= 'second_atom_type'"
                )

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
    elif target["type"] == "atomic_basis_spherical":
        raise ValueError(
            "while 'atomic_basis_spherical' targets are supported, "
            "generic TargetInfo cannot be constructed due to the "
            "flexibility in this target's metadata. Please construct "
            "a TargetInfo object directly using metadata inferred from "
            "target TensorMaps."
        )
    else:
        raise ValueError(
            f"Target type {target['type']} is not supported. "
            "Supported types are 'scalar', 'cartesian', 'spherical', "
            "and 'atomic_basis_spherical'"
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
