import subprocess

import torch
from metatomic.torch import (
    AtomisticModel,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    load_atomistic_model,
)

from metatrain.utils.data import CollateFn, Dataset, read_systems, read_targets
from metatrain.utils.io import load_model
from metatrain.utils.llpr import LLPRUncertaintyModel
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import RESOURCES_PATH


torch.manual_seed(42)


def test_llpr(tmpdir):
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


def test_llpr_metadata(tmpdir):
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

    # hack these fields so we can save the models
    llpr_model_with_metadata.covariance_computed = True
    llpr_model_without_metadata.covariance_computed = True
    llpr_model_with_metadata.inv_covariance_computed = True
    llpr_model_without_metadata.inv_covariance_computed = True
    llpr_model_with_metadata.is_calibrated = True
    llpr_model_without_metadata.is_calibrated = True

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
