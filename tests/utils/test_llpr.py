import subprocess

import pytest
import torch
from metatomic.torch import (
    AtomisticModel,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    load_atomistic_model,
)
from omegaconf import OmegaConf

from metatrain.utils.architectures import get_default_hypers, import_architecture
from metatrain.utils.data import (
    CollateFn,
    Dataset,
    DatasetInfo,
    read_systems,
    read_targets,
)
from metatrain.utils.io import load_model, model_from_checkpoint
from metatrain.utils.llpr import LLPRUncertaintyModel
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import RESOURCES_PATH


torch.manual_seed(42)


def test_llpr(tmpdir):
    """
    Tests functionality of the LLPRUncertaintyModel.
    """
    model = load_model(str(RESOURCES_PATH / "model-64-bit.ckpt"))
    llpr_model = LLPRUncertaintyModel(model)

    qm9_systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    target_config = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "kcal/mol",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        },
    }
    targets, _ = read_targets(target_config)
    requested_neighbor_lists = get_requested_neighbor_lists(llpr_model)
    qm9_systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in qm9_systems
    ]
    dataset = Dataset.from_dict({"system": qm9_systems, **targets})
    collate_fn = CollateFn(target_keys=list(targets.keys()))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=collate_fn,
    )

    llpr_model.compute_covariance(dataloader)
    llpr_model.compute_inverse_covariance()

    exported_model = AtomisticModel(
        llpr_model.eval(),
        ModelMetadata(),
        llpr_model.capabilities,
    )

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={
            "energy_uncertainty": ModelOutput(per_atom=True),
            "energy": ModelOutput(per_atom=True),
            "mtt::aux::energy_last_layer_features": ModelOutput(per_atom=True),
        },
        selected_atoms=None,
    )

    outputs = exported_model(
        qm9_systems[:5], evaluation_options, check_consistency=True
    )

    assert "energy_uncertainty" in outputs
    assert "energy" in outputs
    assert "mtt::aux::energy_last_layer_features" in outputs

    assert outputs["energy_uncertainty"].block().samples.names == [
        "system",
        "atom",
    ]
    assert outputs["energy"].block().samples.names == ["system", "atom"]
    assert outputs["mtt::aux::energy_last_layer_features"].block().samples.names == [
        "system",
        "atom",
    ]

    # Now test the ensemble approach
    params = []  # One per element, SOAP-BPNN
    for name, param in llpr_model.model.named_parameters():
        if "last_layers" in name and "energy" in name:
            params.append(param.squeeze())
    weights = torch.cat(params)

    n_ensemble_members = 1000000  # converges slowly
    llpr_model.calibrate(dataloader)
    llpr_model.generate_ensemble({"energy": weights}, n_ensemble_members)
    assert "energy_ensemble" in llpr_model.capabilities.outputs

    with tmpdir.as_cwd():
        llpr_model.save_checkpoint("llpr_model.ckpt")
        llpr_model = load_model("llpr_model.ckpt")
        exported_model = llpr_model.export()
        exported_model.save(file="llpr_model.pt", collect_extensions="extensions")
        exported_model = load_model("llpr_model.pt", extensions_directory="extensions")

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={
            "energy": ModelOutput(per_atom=False),
            "energy_uncertainty": ModelOutput(per_atom=False),
            "energy_ensemble": ModelOutput(per_atom=False),
        },
        selected_atoms=None,
    )
    outputs = exported_model(
        qm9_systems[:5], evaluation_options, check_consistency=True
    )

    assert "energy_uncertainty" in outputs
    assert "energy_ensemble" in outputs

    analytical_uncertainty = outputs["energy_uncertainty"].block().values
    ensemble_uncertainty = torch.std(
        outputs["energy_ensemble"].block().values, dim=1, keepdim=True
    )

    torch.testing.assert_close(
        analytical_uncertainty, ensemble_uncertainty, rtol=5e-3, atol=0.0
    )


def test_llpr_metadata_preservation_on_export(tmpdir):
    """
    Tests that the metadata of the wrapped model is preserved
    during save-load-export operations.
    """
    checkpoint = torch.load(
        str(RESOURCES_PATH / "model-64-bit.ckpt"),
        weights_only=False,
        map_location="cpu",
    )
    metadata = ModelMetadata(
        name="test",
        description="test",
        references={"architecture": ["TEST: https://arxiv.org/abs/1234.56789v1"]},
    )
    checkpoint["metadata"] = metadata
    with tmpdir.as_cwd():
        torch.save(checkpoint, "model_with_metadata.ckpt")

    with tmpdir.as_cwd():
        model_with_metadata = load_model("model_with_metadata.ckpt")
    model_without_metadata = load_model(str(RESOURCES_PATH / "model-64-bit.ckpt"))
    llpr_model_with_metadata = LLPRUncertaintyModel(model_with_metadata)
    llpr_model_without_metadata = LLPRUncertaintyModel(model_without_metadata)

    with tmpdir.as_cwd():
        llpr_model_with_metadata.save_checkpoint("llpr_model_with_metadata.ckpt")
        llpr_model_without_metadata.save_checkpoint("llpr_model_without_metadata.ckpt")
        subprocess.run("mtt export llpr_model_with_metadata.ckpt", shell=True)
        subprocess.run("mtt export llpr_model_without_metadata.ckpt", shell=True)
        metadata_1 = load_atomistic_model("llpr_model_with_metadata.pt").metadata()
        metadata_2 = load_atomistic_model("llpr_model_without_metadata.pt").metadata()

    exported_references_1 = metadata_1.references
    exported_references_2 = metadata_2.references

    assert metadata_1.name == "test"
    assert metadata_1.description == "test"
    assert any(["TEST" in ref for ref in exported_references_1["architecture"]])
    assert any(["LLPR" in ref for ref in exported_references_1["architecture"]])
    assert any(["LPR" in ref for ref in exported_references_1["architecture"]])

    assert metadata_2.name == ""
    assert metadata_2.description == ""
    assert all(["TEST" not in ref for ref in exported_references_2["architecture"]])
    assert any(["LLPR" in ref for ref in exported_references_2["architecture"]])
    assert any(["LPR" in ref for ref in exported_references_2["architecture"]])


@pytest.mark.parametrize("context", ["finetune", "restart", "export"])
def test_llpr_loads_wrapped_model(tmpdir, context):
    """
    Test that the wrapped model can be loaded from the LLPR checkpoint
    with the given context.
    """
    model = load_model(str(RESOURCES_PATH / "model-64-bit.ckpt"))
    llpr_model = LLPRUncertaintyModel(model)

    with tmpdir.as_cwd():
        llpr_model.save_checkpoint("llpr_model.ckpt")
        checkpoint = torch.load(
            "llpr_model.ckpt", weights_only=False, map_location="cpu"
        )
        model_from_checkpoint(checkpoint["wrapped_model_checkpoint"], context)


@pytest.mark.parametrize("context", ["finetune", "restart", "export"])
def test_llpr_save_and_load_checkpoint(tmpdir, context):
    """
    Test that the LLPR wrapper can be saved and loaded with the given context.
    """
    model = load_model(str(RESOURCES_PATH / "model-64-bit.ckpt"))
    llpr_model = LLPRUncertaintyModel(model)

    with tmpdir.as_cwd():
        llpr_model.save_checkpoint("llpr_model.ckpt")
        checkpoint = torch.load(
            "llpr_model.ckpt", weights_only=False, map_location="cpu"
        )
        if context == "finetune" or context == "export":
            llpr_model.load_checkpoint(checkpoint, context=context)
        elif context == "restart":
            with pytest.raises(NotImplementedError):
                llpr_model.load_checkpoint(checkpoint, context=context)


def test_llpr_finetuning(tmpdir):
    model = load_model(str(RESOURCES_PATH / "model-pet.ckpt"))
    llpr_model = LLPRUncertaintyModel(model)

    with tmpdir.as_cwd():
        llpr_model.save_checkpoint("llpr_model.ckpt")
        checkpoint = torch.load(
            "llpr_model.ckpt", weights_only=False, map_location="cpu"
        )
        model = llpr_model.load_checkpoint(checkpoint, context="finetune").to(
            torch.float64
        )

    architecture = import_architecture("pet")
    Trainer = architecture.__trainer__

    DATASET_PATH = RESOURCES_PATH / "carbon_reduced_100.xyz"
    systems = read_systems(DATASET_PATH)

    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": {"read_from": DATASET_PATH, "key": "force"},
            "stress": False,
            "virial": False,
        }
    }

    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})
    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=[6], targets=target_info_dict
    )
    model = model.restart(dataset_info)

    hypers = get_default_hypers("pet")
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["scheduler_patience"] = 1
    hypers["training"]["fixed_composition_weights"] = {}

    hypers["training"]["finetune"] = {
        "method": "lora",
        "config": {
            "rank": 4,
            "alpha": 0.1,
        },
    }

    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )
