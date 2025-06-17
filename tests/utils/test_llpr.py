import torch
from metatomic.torch import (
    AtomisticModel,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
)

from metatrain.utils.data import Dataset, collate_fn, read_systems, read_targets
from metatrain.utils.io import load_model
from metatrain.utils.llpr import LLPRUncertaintyModel
from metatrain.utils.loss import TensorMapDictLoss
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import RESOURCES_PATH


torch.manual_seed(42)


def test_llpr(tmpdir):
    model = load_model(
        str(RESOURCES_PATH / "model-64-bit.pt"),
        extensions_directory=str(RESOURCES_PATH / "extensions/"),
    )
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
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    qm9_systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in qm9_systems
    ]
    dataset = Dataset.from_dict({"system": qm9_systems, **targets})
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=collate_fn,
    )

    llpr_model = LLPRUncertaintyModel(model)
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
            "mtt::aux::energy_uncertainty": ModelOutput(per_atom=True),
            "energy": ModelOutput(per_atom=True),
            "mtt::aux::energy_last_layer_features": ModelOutput(per_atom=True),
        },
        selected_atoms=None,
    )

    outputs = exported_model(
        qm9_systems[:5], evaluation_options, check_consistency=True
    )

    assert "mtt::aux::energy_uncertainty" in outputs
    assert "energy" in outputs
    assert "mtt::aux::energy_last_layer_features" in outputs

    assert outputs["mtt::aux::energy_uncertainty"].block().samples.names == [
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

    n_ensemble_members = 10000
    llpr_model.calibrate(dataloader)
    llpr_model.generate_ensemble({"energy": weights}, n_ensemble_members)
    assert "energy_ensemble" in llpr_model.capabilities.outputs

    exported_model = AtomisticModel(
        llpr_model.eval(),
        ModelMetadata(),
        llpr_model.capabilities,
    )

    with tmpdir.as_cwd():
        exported_model.save(file="llpr_model.pt", collect_extensions="extensions")
        llpr_model = load_model("llpr_model.pt", extensions_directory="extensions")

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={
            "energy": ModelOutput(per_atom=False),
            "mtt::aux::energy_uncertainty": ModelOutput(per_atom=False),
            "energy_ensemble": ModelOutput(per_atom=False),
        },
        selected_atoms=None,
    )
    outputs = exported_model(
        qm9_systems[:5], evaluation_options, check_consistency=True
    )

    assert "mtt::aux::energy_uncertainty" in outputs
    assert "energy_ensemble" in outputs

    analytical_uncertainty = outputs["mtt::aux::energy_uncertainty"].block().values
    ensemble_uncertainty = torch.var(
        outputs["energy_ensemble"].block().values, dim=1, keepdim=True
    )

    torch.testing.assert_close(
        analytical_uncertainty, ensemble_uncertainty, rtol=1e-2, atol=1e-2
    )


def test_llpr_covariance_as_pseudo_hessian(tmpdir):
    model = load_model(
        str(RESOURCES_PATH / "model-64-bit.pt"),
        extensions_directory=str(RESOURCES_PATH / "extensions/"),
    )
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
    targets, target_info = read_targets(target_config)
    requested_neighbor_lists = model.requested_neighbor_lists()
    qm9_systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in qm9_systems
    ]
    dataset = Dataset.from_dict({"system": qm9_systems, **targets})
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=collate_fn,
    )

    llpr_model = LLPRUncertaintyModel(model)

    parameters = []
    for name, param in llpr_model.named_parameters():
        if "last_layers" in name:
            parameters.append(param)

    loss_weight_dict = {
        "energy": 1.0,
        "energy_positions_grad": 1.0,
        "energy_grain_grad": 1.0,
    }
    loss_fn = TensorMapDictLoss(loss_weight_dict)

    llpr_model.compute_covariance_as_pseudo_hessian(
        dataloader, target_info, loss_fn, {"energy": parameters}
    )
    llpr_model.compute_inverse_covariance()

    exported_model = AtomisticModel(
        llpr_model.eval(),
        ModelMetadata(),
        llpr_model.capabilities,
    )

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={
            "mtt::aux::energy_uncertainty": ModelOutput(per_atom=True),
            "energy": ModelOutput(per_atom=True),
            "mtt::aux::energy_last_layer_features": ModelOutput(per_atom=True),
        },
        selected_atoms=None,
    )

    outputs = exported_model(
        qm9_systems[:5], evaluation_options, check_consistency=True
    )

    assert "mtt::aux::energy_uncertainty" in outputs
    assert "energy" in outputs
    assert "mtt::aux::energy_last_layer_features" in outputs

    assert outputs["mtt::aux::energy_uncertainty"].block().samples.names == [
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

    n_ensemble_members = 1000000  # converges slowly...
    llpr_model.calibrate(dataloader)
    llpr_model.generate_ensemble({"energy": weights}, n_ensemble_members)
    assert "energy_ensemble" in llpr_model.capabilities.outputs

    exported_model = AtomisticModel(
        llpr_model.eval(),
        ModelMetadata(),
        llpr_model.capabilities,
    )

    with tmpdir.as_cwd():
        exported_model.save(file="llpr_model.pt", collect_extensions="extensions")
        llpr_model = load_model("llpr_model.pt", extensions_directory="extensions")

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={
            "energy": ModelOutput(per_atom=False),
            "mtt::aux::energy_uncertainty": ModelOutput(per_atom=False),
            "energy_ensemble": ModelOutput(per_atom=False),
        },
        selected_atoms=None,
    )
    outputs = exported_model(
        qm9_systems[:5], evaluation_options, check_consistency=True
    )

    assert "mtt::aux::energy_uncertainty" in outputs
    assert "energy_ensemble" in outputs

    predictions = outputs["energy"].block().values
    analytical_uncertainty = outputs["mtt::aux::energy_uncertainty"].block().values
    ensemble_mean = torch.mean(
        outputs["energy_ensemble"].block().values, dim=1, keepdim=True
    )
    ensemble_uncertainty = torch.var(
        outputs["energy_ensemble"].block().values, dim=1, keepdim=True
    )

    print(predictions)
    print(ensemble_mean)
    print(predictions - ensemble_mean)

    torch.testing.assert_close(predictions, ensemble_mean, rtol=5e-3, atol=0.0)
    torch.testing.assert_close(
        analytical_uncertainty, ensemble_uncertainty, rtol=5e-3, atol=0.0
    )
