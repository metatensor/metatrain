import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from omegaconf import DictConfig
from rascaline.torch.system import System

from .structures import STRUCTURE_READERS
from .targets import ENERGY_READERS, FORCES_READERS, STRESS_READERS, VIRIAL_READERS


logger = logging.getLogger(__name__)


def _base_reader(
    readers: dict, filename: str, fileformat: Optional[str] = None, **reader_kwargs
):
    if fileformat is None:
        fileformat = Path(filename).suffix

    try:
        reader = readers[fileformat]
    except KeyError:
        raise ValueError(f"fileformat {fileformat!r} is not supported")

    return reader(filename, **reader_kwargs)


def read_energy(
    filename: str,
    target_value: str = "energy",
    fileformat: Optional[str] = None,
) -> TensorBlock:
    """Read energy informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param fileformat: format of the structure file. If :py:obj:`None` the format is
        determined from the suffix.
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        readers=ENERGY_READERS,
        filename=filename,
        fileformat=fileformat,
        key=target_value,
    )


def read_forces(
    filename: str,
    target_value: str = "forces",
    fileformat: Optional[str] = None,
) -> TensorBlock:
    """Read force informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param fileformat: format of the structure file. If :py:obj:`None` the format is
        determined from the suffix.
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        readers=FORCES_READERS,
        filename=filename,
        fileformat=fileformat,
        key=target_value,
    )


def read_stress(
    filename: str,
    target_value: str = "stress",
    fileformat: Optional[str] = None,
) -> TensorBlock:
    """Read stress informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param fileformat: format of the structure file. If :py:obj:`None` the format is
        determined from the suffix.
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        readers=STRESS_READERS,
        filename=filename,
        fileformat=fileformat,
        key=target_value,
    )


def read_structures(
    filename: str,
    fileformat: Optional[str] = None,
) -> List[System]:
    """Read structure informations from a file.

    :param filename: name of the file to read
    :param fileformat: format of the structure file. If :py:obj:`None` the format is
        determined from the suffix.
    :returns: list of structures
    """
    return _base_reader(
        readers=STRUCTURE_READERS, filename=filename, fileformat=fileformat
    )


def read_virial(
    filename: str,
    target_value: str = "virial",
    fileformat: Optional[str] = None,
) -> TensorBlock:
    """Read virial informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param fileformat: format of the structure file. If :py:obj:`None` the format is
        determined from the suffix.
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        readers=VIRIAL_READERS,
        filename=filename,
        fileformat=fileformat,
        key=target_value,
    )


def read_targets(conf: DictConfig) -> Dict[str, TensorMap]:  # , slice_samples_by: str
    """Reading all target information from a fully expanded config.

    To get such a config you can use
    :func:`metatensor.models.utils.omegaconf.expand_dataset_config`.

    This function uses subfunctions like :func:`read_energy` to parse the requested
    target quantity. Currently only `energy` is a supported target property. But, within
    the `energy` section gradients such as `forces`, the `stress` or the `virial` can be
    added. Other gradients are silentlty irgnored.

    :param conf: config containing the keys for what should be read.
    :returns: Dictionary containing one TensorMaps for each target section in the
        config."""
    target_dictionary = {}

    for target_key, target in conf.items():
        if target["quantity"] == "energy":
            block = read_energy(
                filename=target["read_from"],
                target_value=target["key"],
                fileformat=target["file_format"],
            )

            if target["forces"]:
                try:
                    position_gradient = read_forces(
                        filename=target["forces"]["read_from"],
                        target_value=target["forces"]["key"],
                        fileformat=target["forces"]["file_format"],
                    )
                except KeyError:
                    logger.warning(
                        f"Forces not found in section {target_key!r}. "
                        "Continue without forces!"
                    )
                else:
                    logger.info(
                        f"Forces found in section {target_key!r}. Forces are taken for "
                        "training!"
                    )
                    block.add_gradient(
                        parameter="positions", gradient=position_gradient
                    )

            if target["stress"] and target["virial"]:
                raise ValueError("Cannot use stress and virial at the same time!")

            if target["stress"]:
                try:
                    displacement_gradient = read_stress(
                        filename=target["stress"]["read_from"],
                        target_value=target["stress"]["key"],
                        fileformat=target["stress"]["file_format"],
                    )
                except KeyError:
                    logger.warning(
                        f"Stress not found in section {target_key!r}. "
                        "Continue without stress!"
                    )
                else:
                    logger.info(
                        f"Stress found in section {target_key!r}. Stress is taken for "
                        f"training!"
                    )
                    block.add_gradient(
                        parameter="displacement", gradient=displacement_gradient
                    )

            if target["virial"]:
                try:
                    displacement_gradient = read_virial(
                        filename=target["virial"]["read_from"],
                        target_value=target["virial"]["key"],
                        fileformat=target["virial"]["file_format"],
                    )
                except KeyError:
                    logger.warning(
                        f"Virial not found in section {target_key!r}. "
                        "Continue without virial!"
                    )
                else:
                    logger.info(
                        f"Virial found in section {target_key!r}. Virial is taken for "
                        f"training!"
                    )
                    block.add_gradient(
                        parameter="displacement", gradient=displacement_gradient
                    )
        else:
            raise ValueError(
                f"Quantity: {target['quantity']!r} is not supported. Choose 'energy'."
            )

        target_dictionary[target_key] = TensorMap(
            keys=Labels(["lambda", "sigma"], torch.tensor([(0, 1)])), blocks=[block]
        )

    return target_dictionary

    # # TODO: slice the targets here - gradients need attention
    # sliced_target_dictionary = {}
    # for target_key, target_tmap in target_dictionary.items():
    #     sample_ids = metatensor.unique_metadata(
    #         target_tmap, "samples", slice_samples_by
    #     )
    #     sample_ids = torch.tensor(sample_ids.values[:, 0])
    #     # sanity check that the sample ids are just a continuous range
    #     assert torch.all(sample_ids == torch.arange(len(sample_ids)))

    #     per_sample_tmaps = []
    #     for sample_id in sample_ids:
    #         print(sample_id)
    #         per_sample_tmaps.append(
    #             metatensor.slice(
    #                 target_tmap,
    #                 axis="samples",
    #                 labels=Labels(
    #                     names=[slice_samples_by],
    #                     values=torch.tensor([sample_id]).reshape(-1, 1),
    #                 ),
    #             )
    #         )
    #     sliced_target_dictionary[target_key] = per_sample_tmaps

    # return sliced_target_dictionary
