from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import torch

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.data import Dataset

from .documentation import TrainerHypers


class Trainer(TrainerInterface[TrainerHypers]):
    __checkpoint_version__ = 1

    def __init__(self, hypers: TrainerHypers):
        super().__init__(hypers)

    def train(
        self,
        model: ModelInterface,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        model.train_model(
            train_datasets,
            additive_models=[],
            batch_size=len(train_datasets[0]),
            is_distributed=False,
        )

    def save_checkpoint(
        self, model: ModelInterface, checkpoint_dir: Union[str, Path]
    ) -> None:
        pass

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: TrainerHypers,
        context: Literal["restart", "finetune"],
    ) -> "Trainer":
        raise ValueError("Composition model does not allow restarting training")

    @staticmethod
    def upgrade_checkpoint(checkpoint: Dict) -> Dict:
        raise NotImplementedError(
            "checkpoint upgrade is not implemented for the composition model"
        )
