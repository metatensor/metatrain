import copy
from pathlib import Path

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System
from omegaconf import OmegaConf

from metatrain.experimental.mace.model import MetaMACE
from metatrain.experimental.mace.trainer import Trainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification


def load_mace_model_file(
    mace_model_path: Path,
    mace_head_target: str,
    device: torch.device,
) -> MetaMACE:
    """Set up a metatrain MACE model from a MACE model file.

    Currently, creating a metatrain MACE model that behaves exactly
    like a MACE foundational model requires training with 0 epochs
    to set the composition and scaling weights correctly.

    Since it is not trivial to write the code to do this programatically,
    we keep this function here to ease the process. This might help
    experimenting/testing/debugging. In the future we might want to
    expose this functionality publicly.

    :param mace_model_path: Path to the MACE model file.
    :param mace_head_target: Target name for the predictions of
      the mace internal head.
    :param device: Device to set up the model on.

    :return: The MACE model.
    """

    # Setup a dummy dataset and dataset info.
    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    energies = [1.0]
    energies = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[e]], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(names=["energy"], values=torch.tensor([[0]])),
                )
            ],
        )
        for i, e in enumerate(energies)
    ]
    dataset = Dataset.from_dict({"system": systems, "energy": energies})
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1],
        targets={
            mace_head_target: get_generic_target_info(
                mace_head_target,
                {
                    "quantity": "",
                    "unit": "",
                    "type": "scalar",
                    "num_subtargets": 1,
                    "per_atom": False,
                },
            )
        },
    )

    default_hypers = copy.deepcopy(get_default_hypers("experimental.mace"))

    # Initialize the model
    model_hypers = default_hypers["model"]
    model_hypers["mace_model"] = mace_model_path
    model_hypers["mace_head_target"] = mace_head_target
    model = MetaMACE(model_hypers, dataset_info)

    # Train for 0 epochs to set the composition and scaling weights
    trainer_hypers = default_hypers["training"]
    loss_conf = OmegaConf.create(
        {model.hypers["mace_head_target"]: init_with_defaults(LossSpecification)}
    )
    OmegaConf.resolve(loss_conf)
    trainer_hypers["loss"] = loss_conf
    trainer_hypers["num_epochs"] = 0
    trainer_hypers["batch_size"] = 1
    trainer = Trainer(trainer_hypers)
    trainer.train(
        model,
        dtype=torch.float64,
        devices=[device],
        train_datasets=[dataset],
        val_datasets=[],
        checkpoint_dir=".",
    )

    return model
