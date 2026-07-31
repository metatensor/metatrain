from typing import Optional

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.hooks.abc import HookInterface
from metatrain.utils.hooks.helpers import load_hook


_target_types = ["energy", "local_energy", "global_dipole", "local_multipole"]


class HookTests:
    hook: str
    """Name of the hook to be tested.

    Based on this, the test suite will find the hook class
    as well as the hyperparameters.
    """

    @property
    def hook_cls(self) -> type[HookInterface]:
        """The hook class to be tested."""
        return load_hook(self.hook)

    def build_hypers(
        self, input: Optional[dict], output: Optional[dict]
    ) -> tuple[dict, list, list]:
        raise NotImplementedError("Subclasses must implement build_hypers method.")

    @pytest.fixture
    def dataset_info(self) -> DatasetInfo:

        global_energy = get_energy_target_info(
            "energy",
            dict(
                sample_kind="system",
                unit="eV",
                quantity="energy",
                num_subtargets=1,
            ),
        )
        local_energy = get_generic_target_info(
            "mtt::local_energy",
            dict(
                sample_kind="atom",
                unit="eV",
                quantity="energy",
                num_subtargets=1,
                type="scalar",
            ),
        )
        global_dipole = get_generic_target_info(
            "mtt::global_dipole",
            dict(
                sample_kind="system",
                unit="eV",
                quantity="energy",
                num_subtargets=1,
                type=dict(spherical=dict(irreps=[{"o3_lambda": 1, "o3_sigma": 1}])),
            ),
        )
        local_multipole = get_generic_target_info(
            "mtt::local_multipole",
            dict(
                sample_kind="atom",
                unit="eV",
                quantity="energy",
                num_subtargets=1,
                type=dict(
                    spherical=dict(
                        irreps=[
                            {"o3_lambda": 0, "o3_sigma": 1},
                            {"o3_lambda": 1, "o3_sigma": 1},
                        ]
                    )
                ),
            ),
        )

        return DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={
                "mtt::energy": global_energy,
                "mtt::local_energy": local_energy,
                "mtt::local_energy_1": local_energy,
                "mtt::global_dipole": global_dipole,
                "mtt::local_multipole": local_multipole,
            },
            extra_data={
                "mtt::extra_energy": global_energy,
                "mtt::extra_local_energy": local_energy,
                "mtt::extra_local_energy_1": local_energy,
                "mtt::extra_global_dipole": global_dipole,
                "mtt::extra_local_multipole": local_multipole,
            },
        )

    @pytest.fixture
    def targets(self) -> dict[str, str]:
        return {t: f"mtt::{t}" for t in _target_types}

    @pytest.fixture
    def extra_data(self) -> dict[str, str]:
        return {t: f"mtt::extra_{t}" for t in _target_types}

    @pytest.fixture
    def intermediate(self) -> dict[str, str]:
        return {t: f"mtt::intermediate::{t}" for t in _target_types}

    def get_random(
        self, dataset_info: DatasetInfo, requested: list[str]
    ) -> tuple[list[System], dict[str, TensorMap]]:
        """Get random values for targets in the dataset info.

        :param dataset_info: The dataset info to use.
        :param requested: The requested random tensormaps.
        :return: A tuple of a list of systems and a dictionary
          of random tensormaps.
        """
        return_dict = {}
        all_target_infos = {**dataset_info.targets, **dataset_info.extra_data}

        system = System(
            positions=torch.zeros((1, 3)),
            types=torch.tensor([1]),
            cell=torch.eye(3),
            pbc=torch.tensor([True, True, True]),
        )

        for name in requested:
            target = all_target_infos[name]

            if target.sample_kind == "system":
                tmap = TensorMap(
                    keys=target.layout.keys,
                    blocks=[
                        TensorBlock(
                            values=torch.rand(
                                (1, *block.values.shape[1:]),
                                dtype=block.values.dtype,
                                device=block.values.device,
                            ),
                            samples=Labels(
                                names=["system"],
                                values=torch.tensor([[0]], device=block.values.device),
                            ),
                            components=block.components,
                            properties=block.properties,
                        )
                        for block in target.layout.blocks()
                    ],
                )
            elif target.sample_kind == "atom":
                tmap = TensorMap(
                    keys=target.layout.keys,
                    blocks=[
                        TensorBlock(
                            values=torch.rand(
                                (1, *block.values.shape[1:]),
                                dtype=block.values.dtype,
                                device=block.values.device,
                            ),
                            samples=Labels(
                                names=["system", "atom"],
                                values=torch.tensor(
                                    [[0, 0]], device=block.values.device
                                ),
                            ),
                            components=block.components,
                            properties=block.properties,
                        )
                        for block in target.layout.blocks()
                    ],
                )

            return_dict[name] = tmap

        return [system], return_dict
