import warnings
from typing import List

import ase.io
import torch
from metatensor.torch import Labels, TensorBlock


def read_energy_ase(
    filename: str,
    key: str,
) -> List[TensorBlock]:
    """Store energy information in a List of :class:`metatensor.TensorBlock`.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file.

    :returns:
        TensorMap containing the given information
    """
    frames = ase.io.read(filename, ":")

    properties = Labels.single()

    blocks = []
    for i_structure, atoms in enumerate(frames):
        values = torch.tensor([[atoms.info[key]]], dtype=torch.get_default_dtype())
        samples = Labels(["structure"], torch.tensor([[i_structure]]))

        block = TensorBlock(
            values=values,
            samples=samples,
            components=[],
            properties=properties,
        )
        blocks.append(block)

    return blocks


def read_forces_ase(
    filename: str,
    key: str = "energy",
) -> List[TensorBlock]:
    """Store force information in a List of :class:`metatensor.TensorBlock` which can be
    used as ``position`` gradients.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file.

    :returns:
        TensorMap containing the given information
    """
    frames = ase.io.read(filename, ":")

    components = [Labels(["xyz"], torch.arange(3).reshape(-1, 1))]
    properties = Labels.single()

    blocks = []
    for i_structure, atoms in enumerate(frames):
        # We store forces as positions gradients which means we invert the sign
        values = -torch.tensor(atoms.arrays[key], dtype=torch.get_default_dtype())
        values = values.reshape(-1, 3, 1)

        samples = Labels(
            ["sample", "structure", "atom"],
            torch.tensor([[0, i_structure, a] for a in range(len(values))]),
        )

        block = TensorBlock(
            values=values,
            samples=samples,
            components=components,
            properties=properties,
        )

        blocks.append(block)

    return blocks


def read_virial_ase(
    filename: str,
    key: str = "virial",
) -> List[TensorBlock]:
    """Store virial information in a List of :class:`metatensor.TensorBlock` which can
    be used as ``strain`` gradients.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file

    :returns:
        TensorMap containing the given information
    """
    return _read_virial_stress_ase(filename=filename, key=key, is_virial=True)


def read_stress_ase(
    filename: str,
    key: str = "stress",
) -> List[TensorBlock]:
    """Store stress information in a List of :class:`metatensor.TensorBlock` which can
    be used as ``strain`` gradients.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file

    :returns:
        TensorMap containing the given information
    """
    return _read_virial_stress_ase(filename=filename, key=key, is_virial=False)


def _read_virial_stress_ase(
    filename: str,
    key: str,
    is_virial: bool = True,
) -> List[TensorBlock]:
    """Store stress or virial information in a List of :class:`metatensor.TensorBlock`
    which can be used as ``strain`` gradients.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file
    :param is_virial: if target values are stored as stress or virials.

    :returns:
        TensorMap containing the given information
    """
    frames = ase.io.read(filename, ":")

    samples = Labels(["sample"], torch.tensor([[0]]))
    components = [
        Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
        Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
    ]
    properties = Labels.single()

    blocks = []
    for i_structure, atoms in enumerate(frames):

        values = torch.tensor(atoms.info[key].tolist(), dtype=torch.get_default_dtype())

        if values.shape == (9,):
            warnings.warn(
                "Found 9-long numerical vector for the stress/virial in structure "
                f"{i_structure}. Assume a row major format for the conversion into a "
                "3 x 3 matrix.",
                stacklevel=2,
            )
        elif values.shape != (3, 3):
            raise ValueError(
                f"Values in structure {i_structure} has shape {values.shape}. "
                "Stress/virial must be a 3 x 3 matrix or a 9-long numerical vector."
            )

        values = values.reshape(-1, 3, 3, 1)

        if is_virial:
            values *= -1
        else:  # is stress
            if atoms.cell.volume == 0:
                raise ValueError(
                    f"Structure {i_structure} has zero cell vectors. Stress can only "
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
