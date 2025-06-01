import logging
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System
from metatrain.utils.data import DatasetInfo

import ase.units
import ase.data


logger = logging.getLogger(__name__)


class Baseline(torch.nn.Module):

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()
        self.dataset_info = dataset_info
        self.outputs = {target_name: ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                per_atom=True,
            ) for target_name, target_info in dataset_info.targets.items()
        }
        self.register_buffer("atomic_masses", torch.zeros(100))
        for k in dataset_info.atomic_types:
            self.atomic_masses[k] = ase.data.atomic_masses[int(k)]
        self.register_buffer("fs", torch.tensor(ase.units.fs))

        timesteps = set()
        for target_name in self.dataset_info.targets.keys():
            timesteps.add(
                int(target_name.split("_")[1].split("_")[0])
            )
        self.timesteps = sorted(list(timesteps))

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        device = systems[0].device
        system_indices = torch.concatenate(
            [
                torch.full(
                    (len(system),),
                    i_system,
                    device=device,
                )
                for i_system, system in enumerate(systems)
            ],
        )
        sample_values = torch.stack(
            [
                system_indices,
                torch.concatenate(
                    [
                        torch.arange(
                            len(system),
                            device=device,
                        )
                        for system in systems
                    ],
                ),
            ],
            dim=1,
        )
        sample_labels = Labels(
            names=["system", "atom"],
            values=sample_values,
        )

        types = torch.concatenate([system.types for system in systems])
        masses = self.atomic_masses[types]
        masses = masses.unsqueeze(-1).unsqueeze(-1)
        momenta = torch.concatenate([system.get_data("momenta").block().values for system in systems])
        predicted_positions_base = 0.25 * self.fs * momenta / masses
        predicted_positions = {t: predicted_positions_base * t for t in self.timesteps}

        return_dict: Dict[str, TensorMap] = {}
        for t in self.timesteps:
            original_tmap = systems[0].get_data("momenta")
            if f"mtt::delta_{t}_p" in outputs:
                return_dict[f"mtt::delta_{t}_p"] = TensorMap(
                    keys=original_tmap.keys,
                    blocks=[
                        TensorBlock(
                            values=torch.zeros(
                                sum([len(system) for system in systems]), 3, 1,
                                dtype=original_tmap.block().values.dtype,
                                device=original_tmap.block().values.device,
                            ),
                            samples=sample_labels,
                            components=original_tmap.block().components,
                            properties=original_tmap.block().properties,
                        )
                    ]
                )
            if f"mtt::delta_{t}_q" in outputs:
                return_dict[f"mtt::delta_{t}_q"] = TensorMap(
                    keys=original_tmap.keys,
                    blocks=[
                        TensorBlock(
                            values=predicted_positions[t],
                            samples=sample_labels,
                            components=original_tmap.block().components,
                            properties=original_tmap.block().properties,
                        )
                    ]
                )
        return return_dict

