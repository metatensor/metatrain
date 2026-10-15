import copy
import re
from typing import Dict, List, Optional

import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.llpr.trainer import Trainer, calibrate, compute_covariance
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.testing.llpr import (
    ENSEMBLE,
    LLF,
    TARGET,
    UNCERTAINTY,
    atomic_basis_dataset_info,
    cartesian_dataset_info,
    check_atomic_basis_uncertainty,
    fit_llpr,
    make_systems,
    random_dataset,
    setup_backbone,
    spherical_dataset_info,
    target_dataset_info,
    wrap_backbone,
)


def _soap_bpnn(dataset_info):
    hypers = copy.deepcopy(get_default_hypers("soap_bpnn")["model"])
    hypers["soap"]["max_angular"] = 2
    hypers["soap"]["max_radial"] = 2
    hypers["bpnn"]["num_neurons_per_layer"] = 4
    hypers["bpnn"]["num_hidden_layers"] = 1
    return setup_backbone("soap_bpnn", hypers, dataset_info)


def _pet(dataset_info, width=4):
    hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    for key in ("d_pet", "d_head", "d_node", "d_feedforward"):
        hypers[key] = width
    for key in ("num_heads", "num_attention_layers", "num_gnn_layers"):
        hypers[key] = 1
    return setup_backbone("pet", hypers, dataset_info)


def test_uncertainty_mirrors_the_target_layout():
    """Check the uncertainty has the metadata of the target."""
    dataset_info = spherical_dataset_info()
    model = wrap_backbone(_pet(dataset_info), dataset_info, ensembles=False)
    systems = make_systems(model.model, 8)
    fit_llpr(model, systems)

    outputs = model(
        systems[:1],
        {
            TARGET: ModelOutput(sample_kind="system"),
            UNCERTAINTY: ModelOutput(sample_kind="system"),
        },
    )
    assert mts.equal_metadata(outputs[UNCERTAINTY], outputs[TARGET])


def test_model_without_last_layer_features_is_refused():
    """Check SOAP-BPNN's spherical target, without last-layer features, is refused."""
    dataset_info = spherical_dataset_info()
    message = (
        "The wrapped model exposes no last-layer features, so LLPR can not compute "
        "any uncertainty"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        wrap_backbone(_soap_bpnn(dataset_info), dataset_info, ensembles=False)


def test_last_layer_features_in_another_layout_are_refused():
    """Check features that are neither shared nor in the layout of the target are
    refused."""
    dataset_info = spherical_dataset_info()
    backbone = _pet(dataset_info)
    forward = backbone.core.forward

    def forward_with_features_per_block(
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        results = forward(systems, outputs, selected_atoms)
        if LLF in results:
            block = results[LLF].block()
            results[LLF] = TensorMap(
                keys=dataset_info.targets[TARGET].layout.keys,
                blocks=[block.copy(), block.copy()],
            )
        return results

    # PET's shared features, faked as one block for each block of the target
    backbone.core.forward = forward_with_features_per_block
    message = (
        f"the last-layer features of '{TARGET}' must either be a single block "
        "without components, or have the keys and components of the target"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        wrap_backbone(backbone, dataset_info, ensembles=False)


def test_atomic_basis_uncertainty_with_shared_features():
    """Check the uncertainty and ensemble of an atomic-basis target (PET)."""
    check_atomic_basis_uncertainty(_pet(atomic_basis_dataset_info()))


def test_calibration_is_per_property_of_each_block():
    """Check each property of each block is calibrated against its own residuals."""
    irreps = [{"o3_lambda": 0, "o3_sigma": 1}, {"o3_lambda": 2, "o3_sigma": 1}]
    dataset_info = target_dataset_info(
        {"spherical": {"irreps": irreps}}, "system", num_subtargets=2
    )
    model = wrap_backbone(_pet(dataset_info), dataset_info, ensembles=False)
    systems = make_systems(model.model, 8)

    # references far larger than the untrained model's predictions make the
    # residuals of the two properties ~100x apart, in opposite ways in each block
    layout = dataset_info.targets[TARGET].layout
    reference_values = {0: [100.0, 1.0], 2: [1.0, 100.0]}

    def reference(key: LabelsEntry, block: TensorBlock) -> torch.Tensor:
        values = torch.tensor(reference_values[int(key["o3_lambda"])])
        return values.to(torch.float64).repeat(1, len(block.components[0]), 1)

    references = [
        TensorMap(
            keys=layout.keys,
            blocks=[
                TensorBlock(
                    values=reference(key, block),
                    samples=Labels(["system"], torch.tensor([[system_index]])),
                    components=block.components,
                    properties=block.properties,
                )
                for key, block in layout.items()
            ],
        )
        for system_index in range(len(systems))
    ]

    datasets = [Dataset.from_dict({"system": systems, TARGET: references})]

    compute_covariance(model, datasets, batch_size=2, is_distributed=False)
    model.compute_cholesky_decomposition()
    calibrate(
        model,
        datasets,
        batch_size=2,
        is_distributed=False,
        calibration_method="squared_residuals",
    )

    outputs = model(
        systems,
        {
            TARGET: ModelOutput(sample_kind="system"),
            UNCERTAINTY: ModelOutput(sample_kind="system"),
        },
    )
    for key, block in layout.items():
        residuals = outputs[TARGET].block(key).values.detach() - reference(key, block)
        uncertainties = outputs[UNCERTAINTY].block(key).values.detach()
        torch.testing.assert_close(
            (residuals**2 / uncertainties**2).mean(dim=(0, 1)),
            torch.ones(2, dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )


def test_ensemble_variance_matches_analytic_uncertainty():
    """Check the ensemble variance against the analytic uncertainty."""
    torch.manual_seed(0)
    n_ens = 20000
    dataset_info = cartesian_dataset_info()
    model = LLPRUncertaintyModel(
        {"num_ensemble_members": {TARGET: n_ens}}, dataset_info
    )
    model.set_wrapped_model(_pet(dataset_info, width=1))
    model = model.to(torch.float64)
    systems = make_systems(model.model, 2)

    covariance = model.targets[TARGET].feature_blocks[0].covariance
    features = torch.randn(200, covariance.shape[0], dtype=torch.float64)
    covariance[:] = features.T @ features
    model.compute_cholesky_decomposition(regularizer=1e-8)
    # a multiplier != 1 catches mishandling of the calibration factor
    model.targets[TARGET].multipliers[:] = 2.5
    model.generate_ensemble()

    outputs = model(
        systems[:1],
        {
            UNCERTAINTY: ModelOutput(sample_kind="atom"),
            ENSEMBLE: ModelOutput(sample_kind="atom"),
        },
    )
    uncertainty = outputs[UNCERTAINTY].block().values.detach()
    ensemble = outputs[ENSEMBLE].block().values.detach()
    ensemble_var = ensemble.reshape(ensemble.shape[0], 3, n_ens, -1).var(
        dim=-2, unbiased=True
    )

    # Monte Carlo error on a variance from n_ens samples is ~sqrt(2 / n_ens)
    torch.testing.assert_close(
        ensemble_var, uncertainty**2, rtol=5.0 * (2.0 / n_ens) ** 0.5, atol=0.0
    )


def test_trainer_trains_only_the_ensemble(tmp_path):
    """Check the trainer fits the LLPR of a spherical target and that training the
    ensemble changes nothing else."""
    dataset_info = spherical_dataset_info()
    backbone = _pet(dataset_info)
    checkpoint_path = str(tmp_path / "backbone.ckpt")
    torch.save(backbone.get_checkpoint(), checkpoint_path)
    systems = make_systems(backbone, 8)
    dataset = random_dataset(backbone, systems)

    hypers = copy.deepcopy(get_default_hypers("llpr"))
    hypers["model"]["num_ensemble_members"] = {TARGET: 4}
    hypers["training"]["model_checkpoint"] = checkpoint_path
    hypers["training"]["batch_size"] = 2
    loss = init_with_defaults(LossSpecification)
    loss["type"] = "gaussian_nll_ensemble"
    hypers["training"]["loss"] = {TARGET: loss}
    models = []
    for num_epochs in (None, 2):
        hypers["training"]["num_epochs"] = num_epochs
        trainer = Trainer(hypers["training"])
        model = trainer.setup(hypers["model"], dataset_info)
        torch.manual_seed(0)
        trainer.train(
            model,
            torch.float64,
            [torch.device("cpu")],
            [dataset],
            [dataset],
            str(tmp_path),
        )
        models.append(model.core)

    outputs = {
        TARGET: ModelOutput(sample_kind="system"),
        UNCERTAINTY: ModelOutput(sample_kind="system"),
    }
    calibrated, trained = (model(systems, outputs) for model in models)
    expected = backbone.eval()(systems, {TARGET: outputs[TARGET]})[TARGET]
    mts.allclose_raise(trained[TARGET], expected)
    mts.allclose_raise(trained[UNCERTAINTY], calibrated[UNCERTAINTY])
    for layer, trained_layer in zip(
        models[0].targets[TARGET].ensemble_layers,
        models[1].targets[TARGET].ensemble_layers,
        strict=True,
    ):
        assert not torch.equal(layer.weight, trained_layer.weight)
