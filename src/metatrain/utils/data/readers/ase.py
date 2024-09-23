import warnings
from typing import List

import ase.io
import torch
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import System, systems_to_torch


def _wrapped_ase_io_read(filename):
    try:
        return ase.io.read(filename, ":")
    except Exception as e:
        raise ValueError(f"Failed to read '{filename}' with ASE: {e}") from e


def read_systems_ase(filename: str) -> List[System]:
    """Store system informations using ase.

    :param filename: name of the file to read
    :returns: A list of systems
    """
    return systems_to_torch(_wrapped_ase_io_read(filename), dtype=torch.float64)


def read_energy_ase(filename: str, key: str) -> List[TensorBlock]:
    """Store energy information in a List of :class:`metatensor.TensorBlock`.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file.
    :returns: TensorMap containing the energies
    """
    frames = _wrapped_ase_io_read(filename)

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


def read_forces_ase(filename: str, key: str = "energy") -> List[TensorBlock]:
    """Store force information in a List of :class:`metatensor.TensorBlock` which can be
    used as ``position`` gradients.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file.
    :returns: TensorMap containing the forces
    """
    frames = _wrapped_ase_io_read(filename)

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


def read_virial_ase(filename: str, key: str = "virial") -> List[TensorBlock]:
    """Store virial information in a List of :class:`metatensor.TensorBlock` which can
    be used as ``strain`` gradients.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file
    :returns: TensorMap containing the virial
    """
    return _read_virial_stress_ase(filename=filename, key=key, is_virial=True)


def read_stress_ase(filename: str, key: str = "stress") -> List[TensorBlock]:
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
    frames = _wrapped_ase_io_read(filename)

    samples = Labels(["sample"], torch.tensor([[0]]))
    components = [
        Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
        Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
    ]
    properties = Labels("energy", torch.tensor([[0]]))

    blocks = []
    for i_system, atoms in enumerate(frames):
        if key not in atoms.info:
            if is_virial:
                target_name = "virial"
            else:
                target_name = "stress"

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
