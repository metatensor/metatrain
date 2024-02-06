import warnings

import ase.io
import torch
from metatensor.torch import Labels, TensorBlock


def read_energy_ase(
    filename: str,
    key: str,
) -> TensorBlock:
    """Store energy information in a :class:`metatensor.TensorBlock`.

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file.

    :returns:
        TensorMap containing the given information
    """
    frames = ase.io.read(filename, ":")
    values = torch.tensor([f.info[key] for f in frames], dtype=torch.get_default_dtype())
    n_structures = len(values)

    block = TensorBlock(
        values=values.reshape(-1, 1),
        samples=Labels(["structure"], torch.arange(n_structures).reshape(-1, 1)),
        components=[],
        properties=Labels.single(),
    )

    return block


def read_forces_ase(
    filename: str,
    key: str = "energy",
) -> TensorBlock:
    """Store force information as a :class:`metatensor.TensorBlock` which can be used as
    ``position`` gradients .

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file.

    :returns:
        TensorMap containing the given information
    """
    frames = ase.io.read(filename, ":")
    n_structures = len(frames)
    # We store forces as positions gradients which means we invert the sign
    values_raw = [-torch.tensor(f.arrays[key], dtype=torch.get_default_dtype()) for f in frames]

    # The `"sample"` label refers to the index of the corresponding value in the
    # block. Here, the number of values is the same as the number of structures so
    # we can keep `"sample"` and `"structure"` the same.
    samples = Labels(
        ["sample", "structure", "atom"],
        torch.tensor(
            [[s, s, a] for s in range(n_structures) for a in range(len(values_raw[s]))]
        ),
    )

    values = torch.concatenate(values_raw, dim=0)
    assert values.shape[1] != (3,)

    block = TensorBlock(
        values=values.reshape(-1, 3, 1),
        samples=samples,
        components=[Labels(["direction"], torch.arange(3).reshape(-1, 1))],
        properties=Labels.single(),
    )

    return block


def read_virial_ase(
    filename: str,
    key: str = "virial",
):
    """Store virial information in :class:`metatensor.TensorBlock` which can be used as
    ``displacement`` gradients

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file

    :returns:
        TensorMap containing the given information
    """
    return _read_virial_stress_ase(filename=filename, key=key, is_virial=True)


def read_stress_ase(
    filename: str,
    key: str = "stress",
):
    """Store stress information in :class:`metatensor.TensorBlock` which can be used as
    ``displacement`` gradients

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
) -> TensorBlock:
    """Store stress or virial information in :class:`metatensor.TensorBlock` which can
    be used as ``displacement`` gradients

    :param filename: name of the file to read
    :param key: target value key name to be parsed from the file
    :param is_virial: if target values are stored as stress or virials.

    :returns:
        TensorMap containing the given information
    """
    frames = ase.io.read(filename, ":")
    n_structures = len(frames)
    values = torch.tensor([f.info[key].tolist() for f in frames], dtype=torch.get_default_dtype())

    if values.shape[1:] == (9,):
        warnings.warn(
            "Found 9-long numerical vector for the stress/virial. Assume a row major "
            "format for the conversion into a 3x3 matrix.",
            stacklevel=2,
        )
        values = values.reshape(n_structures, 3, 3)
    elif values.shape[1:] != (3, 3):
        raise ValueError(
            f"stress/virial must be a 3 x 3 matrix but has shape {values.shape}"
        )

    volumes = torch.tensor([f.cell.volume for f in frames], dtype=torch.get_default_dtype())
    if torch.any(volumes == 0):
        raise ValueError(
            "Found at least one structure with zero cell vectors."
            "Virial/stress can only be used if cell is non zero!"
        )

    if is_virial:
        values *= -1
    else:  # is stress
        values *= volumes.reshape(-1, 1, 1)

    samples = Labels(["sample"], torch.tensor([[s] for s in range(n_structures)]))

    components = [
        Labels(["cell_vector"], torch.arange(3).reshape(-1, 1)),
        Labels(["coordinate"], torch.arange(3).reshape(-1, 1)),
    ]

    block = TensorBlock(
        values=values.reshape(-1, 3, 3, 1),
        samples=samples,
        components=components,
        properties=Labels.single(),
    )

    return block
