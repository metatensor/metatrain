import logging
import warnings
from pathlib import PurePath
from typing import IO, Any, Dict, List, Tuple, Union

import ase.io
import torch
from ase.stress import voigt_6_to_full_3x3_stress
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System, systems_to_torch
from omegaconf import DictConfig

from ..target_info import TargetInfo, get_energy_target_info, get_generic_target_info


def read(
    filename: Union[str, PurePath, IO], *args: Any, **kwargs: Any
) -> List[ase.Atoms]:
    r"""
    Wrapper around the :func:`ase.io.read` function.

    The wrapper provides a more informative error message in case of failure.
    Additionally, it will make the keys ``"energy"``, ``"forces"`` and ``"stress"``
    available from the calculator and the info/arrays dictionary.

    .. warning ::

        Lists of atoms read with this function can NOT be written back to a file with
        :func:`ase.io.write` because of the duplicated keys.

    :param filename: Name of the file to read from or a file descriptor.
    :param \*args: additional positional arguments for :func:`ase.io.read`
    :param \*\*kwargs: additional keyword arguments for :func:`ase.io.read`
    :return: A list of :py:class:`ase.Atoms` objects
    """
    try:
        frames = ase.io.read(filename, *args, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to read '{filename}' with ASE: {e}") from e

    # allow access of "special" keys from calculator and `info`/`arrays` dictionary
    for atoms in frames:
        if hasattr(atoms, "calc") and atoms.calc is not None:
            results = atoms.calc.results
            if "energy" in results:
                atoms.info["energy"] = results["energy"]
            if "forces" in results:
                atoms.arrays["forces"] = results["forces"]
            if "stress" in results:
                atoms.info["stress"] = voigt_6_to_full_3x3_stress(results["stress"])

    return frames


def read_systems(filename: str) -> List[System]:
    """Store system informations using ase.

    :param filename: name of the file to read
    :return: The systems read from the file
    """
    ase_atoms = read(filename, ":")
    systems = systems_to_torch(ase_atoms, dtype=torch.float64)

    # Add momenta (for FlashMD) if available
    if "momenta" in ase_atoms[0].arrays:
        for system, atoms in zip(systems, ase_atoms, strict=False):
            momenta = TensorMap(
                keys=Labels(["_"], torch.tensor([[0]])),
                blocks=[
                    TensorBlock(
                        values=torch.tensor(
                            atoms.arrays["momenta"], dtype=torch.float64
                        ).unsqueeze(-1),
                        samples=Labels(
                            ["system", "atom"],
                            torch.tensor(
                                [[0, a] for a in range(len(atoms.arrays["momenta"]))]
                            ),
                        ),
                        components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
                        properties=Labels("momenta", torch.tensor([[0]])),
                    )
                ],
            )
            system.add_data("momenta", momenta)

    return systems


def _read_energy_ase(filename: str, key: str) -> List[TensorBlock]:
    """Store energy information in a List of :class:`metatensor.TensorBlock`.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file.
    :return: TensorMap containing the energies
    """
    frames = read(filename, ":")

    properties = Labels("energy", torch.tensor([[0]]))

    blocks = []
    for i_system, atoms in enumerate(frames):
        if key not in atoms.info:
            raise ValueError(
                f"energy key {key!r} was not found in system {filename!r} at index "
                f"{i_system}"
            )

        values = torch.tensor([[atoms.info[key]]], dtype=torch.float64)
        samples = Labels(["system"], torch.tensor([[i_system]]))

        block = TensorBlock(
            values=values,
            samples=samples,
            components=[],
            properties=properties,
        )
        blocks.append(block)

    return blocks


def _read_forces_ase(filename: str, key: str = "forces") -> List[TensorBlock]:
    """Store force information in a List of :class:`metatensor.TensorBlock` which can be
    used as ``position`` gradients.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file.
    :return: TensorMap containing the forces
    """
    frames = read(filename, ":")

    components = [Labels(["xyz"], torch.arange(3).reshape(-1, 1))]
    properties = Labels("energy", torch.tensor([[0]]))

    blocks = []
    for i_system, atoms in enumerate(frames):
        if key not in atoms.arrays:
            raise ValueError(
                f"forces key {key!r} was not found in system {filename!r} at index "
                f"{i_system}"
            )

        # We store forces as positions gradients which means we invert the sign
        values = -torch.tensor(atoms.arrays[key], dtype=torch.float64)
        values = values.reshape(-1, 3, 1)

        samples = Labels(
            ["sample", "system", "atom"],
            torch.tensor([[0, i_system, a] for a in range(len(values))]),
            assume_unique=True,
        )

        block = TensorBlock(
            values=values,
            samples=samples,
            components=components,
            properties=properties,
        )

        blocks.append(block)

    return blocks


def _read_virial_ase(filename: str, key: str = "virial") -> List[TensorBlock]:
    """Store virial information in a List of :class:`metatensor.TensorBlock` which can
    be used as ``strain`` gradients.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file
    :return: TensorMap containing the virial
    """
    return _read_virial_stress_ase(filename=filename, key=key, is_virial=True)


def _read_stress_ase(filename: str, key: str = "stress") -> List[TensorBlock]:
    """Store stress information in a List of :class:`metatensor.TensorBlock` which can
    be used as ``strain`` gradients.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file
    :return: TensorMap containing the stress
    """
    return _read_virial_stress_ase(filename=filename, key=key, is_virial=False)


def _read_virial_stress_ase(
    filename: str, key: str, is_virial: bool = True
) -> List[TensorBlock]:
    frames = read(filename, ":")

    samples = Labels(["sample"], torch.tensor([[0]]))
    components = [
        Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
        Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
    ]
    properties = Labels("energy", torch.tensor([[0]]))

    blocks = []
    for i_system, atoms in enumerate(frames):
        if key not in atoms.info:
            target_name = "virial" if is_virial else "stress"
            raise ValueError(
                f"{target_name} key {key!r} was not found in system {filename!r} at "
                f"index {i_system}"
            )

        values = torch.tensor(atoms.info[key].tolist(), dtype=torch.float64)

        if values.shape == (9,):
            warnings.warn(
                "Found 9-long numerical vector for the stress/virial in system "
                f"{i_system}. Assume a row major format for the conversion into a "
                "3 x 3 matrix.",
                stacklevel=2,
            )
        elif values.shape != (3, 3):
            raise ValueError(
                f"Values in system {i_system} has shape {values.shape}. "
                "Stress/virial must be a 3 x 3 matrix or a 9-long numerical vector."
            )

        values = values.reshape(-1, 3, 3, 1)

        if is_virial:
            values *= -1
        else:  # is stress
            if atoms.cell.volume == 0 and not torch.all(torch.isnan(values)).item():
                raise ValueError(
                    f"system {i_system} has zero cell vectors. Stress can only "
                    "be used if cell is non zero."
                )
            values *= atoms.cell.volume

        block = TensorBlock(
            values=values,
            samples=samples,
            components=components,
            properties=properties,
        )
        blocks.append(block)

    return blocks


def read_energy(
    target_name: str, target: DictConfig
) -> Tuple[List[TensorMap], TargetInfo]:
    """Read energy target from an ASE-readable file.

    :param target_name: Name of the target to read.
    :param target: Configuration settings for the target.

    :return: The function returns two outputs:

        1. A list of `TensorMap` objects, each of them being the target for a single
            system.
        2. A `TargetInfo` object containing metadata about the target.

    """
    target_key = target["key"]

    blocks = _read_energy_ase(
        filename=target["read_from"],
        key=target["key"],
    )

    add_position_gradients = False
    if target["forces"]:
        try:
            position_gradients = _read_forces_ase(
                filename=target["forces"]["read_from"],
                key=target["forces"]["key"],
            )
        except Exception:
            logging.warning(f"No forces found in section {target_key!r}.")
            add_position_gradients = False
        else:
            logging.info(
                f"Forces found in section {target_key!r}, "
                "we will use this gradient to train the model"
            )
            for block, position_gradient in zip(
                blocks, position_gradients, strict=True
            ):
                block.add_gradient(parameter="positions", gradient=position_gradient)
            add_position_gradients = True

    if target["stress"] and target["virial"]:
        raise ValueError("Cannot use stress and virial at the same time")

    add_strain_gradients = False

    if target["stress"]:
        try:
            strain_gradients = _read_stress_ase(
                filename=target["stress"]["read_from"],
                key=target["stress"]["key"],
            )
        except Exception:
            logging.warning(f"No stress found in section {target_key!r}.")
            add_strain_gradients = False
        else:
            logging.info(
                f"Stress found in section {target_key!r}, "
                "we will use this gradient to train the model"
            )
            for block, strain_gradient in zip(blocks, strain_gradients, strict=True):
                block.add_gradient(parameter="strain", gradient=strain_gradient)
            add_strain_gradients = True

    if target["virial"]:
        try:
            strain_gradients = _read_virial_ase(
                filename=target["virial"]["read_from"],
                key=target["virial"]["key"],
            )
        except Exception:
            logging.warning(f"No virial found in section {target_key!r}.")
            add_strain_gradients = False
        else:
            logging.info(
                f"Virial found in section {target_key!r}, "
                "we will use this gradient to train the model"
            )
            for block, strain_gradient in zip(blocks, strain_gradients, strict=True):
                block.add_gradient(parameter="strain", gradient=strain_gradient)
            add_strain_gradients = True
    tensor_map_list = [
        TensorMap(
            keys=Labels(["_"], torch.tensor([[0]])),
            blocks=[block],
        )
        for block in blocks
    ]
    target_info = get_energy_target_info(
        target_name, target, add_position_gradients, add_strain_gradients
    )
    return tensor_map_list, target_info


def _read_per_sample_weight_values(
    atoms: ase.Atoms,
    key: str,
    n_samples: int,
    per_atom: bool,
    filename: str,
    i_system: int,
) -> torch.Tensor:
    """Read ``n_samples`` per-sample weight values from a single ASE ``Atoms`` object.

    Per-atom weights are read from ``atoms.arrays`` and per-structure weights from
    ``atoms.info``. A single per-structure value is broadcast to ``n_samples``.

    :param atoms: the ASE ``Atoms`` object to read from.
    :param key: the info/arrays key holding the weights.
    :param n_samples: the expected number of per-sample weights (i.e. the size of the
        first/sample dimension of the block the weights refer to).
    :param per_atom: whether the weights are per-atom (read from ``arrays``) or
        per-structure (read from ``info``).
    :param filename: the file the frame was read from (for error messages).
    :param i_system: the index of the frame (for error messages).
    :return: a 1D ``float64`` tensor of length ``n_samples``.
    """
    container = atoms.arrays if per_atom else atoms.info
    if key not in container:
        location = "arrays" if per_atom else "info"
        raise ValueError(
            f"sample weight key {key!r} was not found in the {location} of system "
            f"{filename!r} at index {i_system}"
        )
    raw = torch.as_tensor(container[key], dtype=torch.float64).reshape(-1)
    if raw.numel() == 1 and n_samples != 1:
        # a single per-structure value is broadcast over all samples
        raw = raw.expand(n_samples)
    if raw.numel() != n_samples:
        raise ValueError(
            f"sample weight key {key!r} in system {filename!r} at index {i_system} has "
            f"{raw.numel()} values, but {n_samples} were expected."
        )
    return raw


def _broadcast_weight_to_block_shape(
    weights_per_sample: torch.Tensor, shape: List[int]
) -> torch.Tensor:
    """Broadcast per-sample weights over the components and properties of a block.

    :param weights_per_sample: 1D tensor of length ``shape[0]``.
    :param shape: the target block (or gradient block) values shape.
    :return: a contiguous tensor of the requested shape.
    """
    view_shape = [shape[0]] + [1] * (len(shape) - 1)
    return weights_per_sample.reshape(view_shape).expand(shape).contiguous()


def read_sample_weights(
    target_name: str,
    target: DictConfig,
    target_tensor_maps: List[TensorMap],
) -> List[TensorMap]:
    """Read per-sample loss weights for a target, mirroring its :py:class:`TensorMap`.

    For each system, the returned :py:class:`TensorMap` has the same keys, blocks,
    components, properties and gradients as the corresponding target
    :py:class:`TensorMap`. The values of each block (and gradient block) are filled by
    broadcasting a single per-sample weight over the components and properties.

    The weights for the main values are read using the target's ``sample_weight_key``.
    The weights for the gradients are read using the ``sample_weight_key`` of the
    corresponding gradient section (``forces`` for the ``positions`` gradient,
    ``stress`` or ``virial`` for the ``strain`` gradient). Any weight that is not
    specified defaults to 1.0 (i.e. an unweighted contribution).

    :param target_name: the name of the target.
    :param target: the (expanded) configuration of the target.
    :param target_tensor_maps: the already-read target TensorMaps, used as templates
        for the structure (metadata) of the weights.
    :return: a list of weight :py:class:`TensorMap` objects, one per system.
    """
    value_key = target.get("sample_weight_key")
    value_file = target["read_from"]

    # Map gradient parameter names (as stored in the target TensorMap) to the
    # (filename, key) from which their weights should be read.
    gradient_specs: Dict[str, Tuple[str, str]] = {}
    forces = target.get("forces")
    if isinstance(forces, DictConfig) and forces.get("sample_weight_key") is not None:
        gradient_specs["positions"] = (
            forces["read_from"],
            forces["sample_weight_key"],
        )
    for strain_section in ("stress", "virial"):
        strain = target.get(strain_section)
        if (
            isinstance(strain, DictConfig)
            and strain.get("sample_weight_key") is not None
        ):
            gradient_specs["strain"] = (
                strain["read_from"],
                strain["sample_weight_key"],
            )

    # Lazily read (and cache) frames per file.
    frame_cache: Dict[str, List[ase.Atoms]] = {}

    def get_frames(filename: str) -> List[ase.Atoms]:
        if filename not in frame_cache:
            frame_cache[filename] = read(filename, ":")
        return frame_cache[filename]

    weight_tensor_maps: List[TensorMap] = []
    for i_system, tensor_map in enumerate(target_tensor_maps):
        new_blocks = []
        for block in tensor_map.blocks():
            n_samples = block.values.shape[0]
            per_atom = "atom" in block.samples.names
            if value_key is not None:
                weights_per_sample = _read_per_sample_weight_values(
                    get_frames(value_file)[i_system],
                    value_key,
                    n_samples,
                    per_atom,
                    value_file,
                    i_system,
                )
            else:
                weights_per_sample = torch.ones(n_samples, dtype=torch.float64)

            new_block = TensorBlock(
                values=_broadcast_weight_to_block_shape(
                    weights_per_sample, list(block.values.shape)
                ),
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )

            for parameter, gradient in block.gradients():
                n_grad_samples = gradient.values.shape[0]
                grad_per_atom = "atom" in gradient.samples.names
                if parameter in gradient_specs:
                    grad_file, grad_key = gradient_specs[parameter]
                    grad_weights = _read_per_sample_weight_values(
                        get_frames(grad_file)[i_system],
                        grad_key,
                        n_grad_samples,
                        grad_per_atom,
                        grad_file,
                        i_system,
                    )
                else:
                    grad_weights = torch.ones(n_grad_samples, dtype=torch.float64)

                new_block.add_gradient(
                    parameter,
                    TensorBlock(
                        values=_broadcast_weight_to_block_shape(
                            grad_weights, list(gradient.values.shape)
                        ),
                        samples=gradient.samples,
                        components=gradient.components,
                        properties=gradient.properties,
                    ),
                )

            new_blocks.append(new_block)

        weight_tensor_maps.append(TensorMap(keys=tensor_map.keys, blocks=new_blocks))

    return weight_tensor_maps


def read_generic(
    target_name: str, target: DictConfig
) -> Tuple[List[TensorMap], TargetInfo]:
    """Read a generic target from an ASE-readable file.

    :param target_name: Name of the target to read.
    :param target: Configuration settings for the target.
    :return: The function returns two outputs:

        1. A list of `TensorMap` objects, each of them being the target for a single
            system.
        2. A `TargetInfo` object containing metadata about the target.

    """
    filename = target["read_from"]
    frames = read(filename, ":")

    # we don't allow ASE to read spherical tensors with more than one irrep,
    # otherwise it's a mess
    if (
        isinstance(target["type"], DictConfig)
        and next(iter(target["type"].keys())) == "spherical"
    ):
        irreps = target["type"]["spherical"]["irreps"]
        if len(irreps) > 1:
            raise ValueError(
                "The metatrain ASE reader does not support reading "
                "spherical tensors with more than one irreducible "
                "representation. Please use the metatensor reader."
            )

    target_info = get_generic_target_info(target_name, target)
    components = target_info.layout.block().components
    properties = target_info.layout.block().properties
    shape_after_samples = target_info.layout.block().shape[1:]
    per_atom = target_info.sample_kind == "atom"
    keys = target_info.layout.keys

    target_key = target["key"]

    tensor_maps = []
    for i_system, atoms in enumerate(frames):
        if (per_atom and target_key not in atoms.arrays) or (
            not per_atom and target_key not in atoms.info
        ):
            raise ValueError(
                f"Target key {target_key!r} was not found in system {filename!r} at "
                f"index {i_system}"
            )

        if per_atom:
            data = atoms.arrays[target_key]
        else:
            data = atoms.info[target_key]

        # here we reshape to allow for more flexibility; this is actually
        # necessary for the `arrays`, which are stored in a 2D array
        values = torch.tensor(data, dtype=torch.float64).reshape(
            [-1] + shape_after_samples
        )

        samples = (
            Labels(
                ["system", "atom"],
                torch.tensor([[i_system, a] for a in range(len(values))]),
                assume_unique=True,
            )
            if per_atom
            else Labels(
                ["system"],
                torch.tensor([[i_system]]),
            )
        )

        block = TensorBlock(
            values=values,
            samples=samples,
            components=components,
            properties=properties,
        )
        tensor_map = TensorMap(
            keys=keys,
            blocks=[block],
        )
        tensor_maps.append(tensor_map)

    return tensor_maps, target_info
