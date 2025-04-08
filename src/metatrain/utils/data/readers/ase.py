import logging
import warnings
from pathlib import PurePath
from typing import IO, List, Tuple, Union

import ase.io
import torch
from ase.stress import voigt_6_to_full_3x3_stress
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System, systems_to_torch
from omegaconf import DictConfig

from ..target_info import TargetInfo, get_energy_target_info, get_generic_target_info


def read(filename: Union[str, PurePath, IO], *args, **kwargs) -> List[ase.Atoms]:
    """Wrapper around the :func:`ase.io.read` function.

    The wrapper provides a more informative error message in case of failure.
    Additionally, it will make the keys ``"energy"``, ``"forces"`` and ``"stress"``
    available from the calculator and the info/arrays dictionary.

    .. warning::

        Lists of atoms read with this function can NOT be written back to a file with
        :func:`ase.io.write` because of the duplicated keys.

    :param filename: Name of the file to read from or a file descriptor.
    :param args: additional positional arguments for :func:`ase.io.read`
    :param kwargs: additional keyword arguments for :func:`ase.io.read`
    :returns: A list of atoms
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
    :returns: A list of systems
    """
    return systems_to_torch(read(filename, ":"), dtype=torch.float64)


def _read_energy_ase(filename: str, key: str) -> List[TensorBlock]:
    """Store energy information in a List of :class:`metatensor.TensorBlock`.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file.
    :returns: TensorMap containing the energies
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
    :returns: TensorMap containing the forces
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
    :returns: TensorMap containing the virial
    """
    return _read_virial_stress_ase(filename=filename, key=key, is_virial=True)


def _read_stress_ase(filename: str, key: str = "stress") -> List[TensorBlock]:
    """Store stress information in a List of :class:`metatensor.TensorBlock` which can
    be used as ``strain`` gradients.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file
    :returns: TensorMap containing the stress
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
            if atoms.cell.volume == 0:
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


def read_energy(target: DictConfig) -> Tuple[List[TensorMap], TargetInfo]:
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
            for block, position_gradient in zip(blocks, position_gradients):
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
            for block, strain_gradient in zip(blocks, strain_gradients):
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
            for block, strain_gradient in zip(blocks, strain_gradients):
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
        target, add_position_gradients, add_strain_gradients
    )
    return tensor_map_list, target_info


def read_generic(target: DictConfig) -> Tuple[List[TensorMap], TargetInfo]:
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

    target_info = get_generic_target_info(target)
    components = target_info.layout.block().components
    properties = target_info.layout.block().properties
    shape_after_samples = target_info.layout.block().shape[1:]
    per_atom = target_info.per_atom
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
