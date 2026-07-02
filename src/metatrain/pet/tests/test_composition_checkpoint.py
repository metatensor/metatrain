import copy
import os

import torch
from omegaconf import OmegaConf

from metatrain.composition import CompositionModel
from metatrain.pet import PET, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification

from . import DATASET_PATH, DEFAULT_HYPERS


def test_composition_checkpoint_consistency(tmp_path):
    """atomic_baseline="path" and atomic_baseline={} produce the same
    composition weights when PET is trained on the same dataset."""

    # ── 1. Setup: QM9 data, energy only ──────────────────────────────
    systems = read_systems(DATASET_PATH)

    targets_conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(targets_conf))
    targets = {"energy": targets["energy"]}
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})

    atomic_types = sorted({int(t) for s in systems for t in s.types})
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=atomic_types,
        targets=target_info_dict,
    )

    # ── 2. Train composition standalone, save checkpoint ──────────────
    composition_model = CompositionModel(hypers={}, dataset_info=dataset_info)
    composition_model.train_model(
        datasets=[dataset],
        additive_models=[],
        batch_size=len(dataset),
        is_distributed=False,
    )

    checkpoint_path = tmp_path / "composition.ckpt"
    torch.save(composition_model.get_checkpoint(), checkpoint_path)

    # ── 3. Build PET minimal hypers ──────────────────────────────────
    hypers = copy.deepcopy(DEFAULT_HYPERS)
    model_hypers = hypers["model"]
    model_hypers["d_pet"] = 1
    model_hypers["d_head"] = 1
    model_hypers["d_node"] = 1
    model_hypers["d_feedforward"] = 1
    model_hypers["num_heads"] = 1
    model_hypers["num_attention_layers"] = 1
    model_hypers["num_gnn_layers"] = 1

    training_hypers = hypers["training"]
    training_hypers["num_epochs"] = 1
    training_hypers["num_workers"] = 0
    training_hypers["scheduler_patience"] = 1

    loss_conf = OmegaConf.create({"energy": init_with_defaults(LossSpecification)})
    OmegaConf.resolve(loss_conf)
    training_hypers["loss"] = loss_conf

    # ── 4. PET-A: atomic_baseline = checkpoint path ─────────────────
    training_hypers["atomic_baseline"] = str(checkpoint_path)
    os.makedirs(str(tmp_path / "a"), exist_ok=True)
    model_a = PET(model_hypers, dataset_info)
    trainer_a = Trainer(training_hypers)
    trainer_a.train(
        model=model_a,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=str(tmp_path / "a"),
    )

    # ── 5. PET-B: atomic_baseline = {} (fit from data) ─────────────
    training_hypers["atomic_baseline"] = {}
    os.makedirs(str(tmp_path / "b"), exist_ok=True)
    model_b = PET(model_hypers, dataset_info)
    trainer_b = Trainer(training_hypers)
    trainer_b.train(
        model=model_b,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=str(tmp_path / "b"),
    )

    # ── 6. Assert composition weights match ─────────────────────────
    weights_a = model_a.additive_models[0].model.weights
    weights_b = model_b.additive_models[0].model.weights

    for target_name in weights_a.keys():
        assert target_name in weights_b
        block_a = weights_a[target_name].block()
        block_b = weights_b[target_name].block()
        torch.testing.assert_close(
            block_a.values,
            block_b.values,
            atol=1e-10,
            rtol=1e-10,
        )
