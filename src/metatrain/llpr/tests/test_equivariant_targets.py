import copy
import re

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput

from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.pet import PET
from metatrain.soap_bpnn import SoapBpnn
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset
from metatrain.utils.testing.llpr import (
    ENSEMBLE,
    NUM_ENSEMBLE_MEMBERS,
    TARGET,
    UNCERTAINTY,
    atomic_basis_dataset_info,
    cartesian_dataset_info,
    check_atomic_basis_uncertainty,
    make_systems,
    spherical_dataset_info,
    wrap_backbone,
)


def _soap_bpnn(dataset_info):
    hypers = copy.deepcopy(get_default_hypers("soap_bpnn")["model"])
    hypers["soap"]["max_angular"] = 2
    hypers["soap"]["max_radial"] = 2
    hypers["bpnn"]["num_neurons_per_layer"] = 4
    hypers["bpnn"]["num_hidden_layers"] = 1
    return SoapBpnn(hypers, dataset_info).to(torch.float64)


def _pet(dataset_info, width=4):
    hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    for key in ("d_pet", "d_head", "d_node", "d_feedforward"):
        hypers[key] = width
    for key in ("num_heads", "num_attention_layers", "num_gnn_layers"):
        hypers[key] = 1
    return PET(hypers, dataset_info).to(torch.float64)


def _with_identity_covariance(model):
    """A covariance the Cholesky decomposition can be taken of. Its exact value does
    not matter to these tests, only that the uncertainty derives from it."""
    covariance = model._get_covariance(UNCERTAINTY, 0)
    covariance[:] = torch.eye(covariance.shape[0], dtype=covariance.dtype)
    model.compute_cholesky_decomposition(regularizer=1e-8)
    return model


@pytest.mark.parametrize("backbone", [_soap_bpnn, _pet], ids=["soap_bpnn", "pet"])
def test_uncertainty_mirrors_the_target_layout(backbone):
    """Check the uncertainty has the target's own keys, components and
    properties."""
    dataset_info = spherical_dataset_info()
    model = wrap_backbone(backbone(dataset_info), dataset_info, ensembles=False)
    model = _with_identity_covariance(model)
    system = make_systems(model.model, 1)[0]

    outputs = model(
        [system],
        {
            TARGET: ModelOutput(sample_kind="system"),
            UNCERTAINTY: ModelOutput(sample_kind="system"),
        },
    )
    prediction = outputs[TARGET]
    uncertainty = outputs[UNCERTAINTY]

    assert uncertainty.keys == prediction.keys
    for index in range(len(prediction.keys)):
        prediction_block = prediction.block(index)
        uncertainty_block = uncertainty.block(index)

        assert uncertainty_block.values.shape == prediction_block.values.shape
        assert uncertainty_block.components == prediction_block.components
        assert uncertainty_block.properties == prediction_block.properties
        assert torch.all(uncertainty_block.values > 0.0)


def test_ensemble_mirrors_the_target_layout():
    """Check the ensemble has the target's keys and its mean reproduces the
    prediction block by block (PET)."""
    dataset_info = spherical_dataset_info()
    model = wrap_backbone(_pet(dataset_info), dataset_info, ensembles=True)
    model = _with_identity_covariance(model)
    model.generate_ensemble()
    system = make_systems(model.model, 1)[0]

    outputs = model(
        [system],
        {
            TARGET: ModelOutput(sample_kind="system"),
            ENSEMBLE: ModelOutput(sample_kind="system"),
        },
    )
    prediction = outputs[TARGET]
    ensemble = outputs[ENSEMBLE]

    assert ensemble.keys == prediction.keys

    for index in range(len(prediction.keys)):
        prediction_block = prediction.block(index)
        ensemble_block = ensemble.block(index)

        num_properties = prediction_block.values.shape[-1]
        assert ensemble_block.components == prediction_block.components

        # the ensemble is re-centered on the prediction, so its mean is exact
        members = ensemble_block.values.reshape(
            list(ensemble_block.values.shape[:-1])
            + [NUM_ENSEMBLE_MEMBERS, num_properties]
        )
        torch.testing.assert_close(
            members.mean(dim=-2), prediction_block.values, rtol=1e-10, atol=1e-10
        )


def test_ensemble_refused_when_not_a_linear_readout():
    """Check requesting ensembles for SOAP-BPNN's spherical target fails loudly."""
    dataset_info = spherical_dataset_info()
    message = (
        f"Cannot generate LLPR ensembles for '{TARGET}': it is not a linear "
        "function of the last-layer features of the wrapped model. Uncertainties "
        "are still available for this target; remove it from the "
        "`num_ensemble_members` section."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        wrap_backbone(_soap_bpnn(dataset_info), dataset_info, ensembles=True)


def test_atomic_basis_uncertainty_with_shared_features():
    """Check the uncertainty and ensemble of an atomic-basis target (PET)."""
    check_atomic_basis_uncertainty(_pet(atomic_basis_dataset_info()))


def test_calibration_is_per_block():
    """Check each block is calibrated against its own residuals: with residuals
    differing 100x between blocks, both must come out calibrated."""
    dataset_info = spherical_dataset_info()
    model = wrap_backbone(_pet(dataset_info), dataset_info, ensembles=False)
    systems = make_systems(model.model, 8)

    # references far larger than the untrained model's predictions make the
    # lambda=0 residuals ~100x the lambda=2 ones
    layout = dataset_info.targets[TARGET].layout
    scales = {0: 100.0, 2: 1.0}
    references = []
    for system_index in range(len(systems)):
        blocks = []
        for key, layout_block in layout.items():
            shape = (1, len(layout_block.components[0]), 1)
            blocks.append(
                TensorBlock(
                    values=torch.full(
                        shape, scales[int(key["o3_lambda"])], dtype=torch.float64
                    ),
                    samples=Labels(
                        names=["system"], values=torch.tensor([[system_index]])
                    ),
                    components=layout_block.components,
                    properties=layout_block.properties,
                )
            )
        references.append(TensorMap(keys=layout.keys, blocks=blocks))

    datasets = [Dataset.from_dict({"system": systems, TARGET: references})]

    model.compute_covariance(datasets, batch_size=2, is_distributed=False)
    model.compute_cholesky_decomposition()
    model.calibrate(
        datasets,
        batch_size=2,
        is_distributed=False,
        calibration_method="squared_residuals",
    )

    # every block ends up calibrated: its residuals are of the size its own
    # uncertainty claims
    outputs = model(
        systems,
        {
            TARGET: ModelOutput(sample_kind="system"),
            UNCERTAINTY: ModelOutput(sample_kind="system"),
        },
    )
    for index in range(len(layout.keys)):
        # every reference is the same constant, so `references[0]` serves them all
        residuals = (
            outputs[TARGET].block(index).values.detach()
            - references[0].block(index).values
        )
        uncertainties = outputs[UNCERTAINTY].block(index).values.detach()
        assert torch.allclose(
            (residuals**2 / uncertainties**2).mean(),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-6,
        )


def test_ensemble_variance_matches_analytic_uncertainty():
    """Check the ensemble variance against the analytic ``alpha^2 f^T C^-1 f``
    uncertainty for a vector target, with a non-unit calibration factor."""
    torch.manual_seed(0)
    n_ens = 20000
    dataset_info = cartesian_dataset_info()
    model = LLPRUncertaintyModel(
        {"num_ensemble_members": {TARGET: n_ens}}, dataset_info
    )
    model.set_wrapped_model(_pet(dataset_info, width=1))
    model = model.to(torch.float64)

    # inject a known covariance and multiplier through the private buffers
    covariance = model._get_covariance(UNCERTAINTY, 0)
    features = torch.randn(200, covariance.shape[0], dtype=torch.float64)
    covariance[:] = features.T @ features
    model.compute_cholesky_decomposition(regularizer=1e-8)
    # a multiplier != 1 catches mishandling of the calibration factor
    model._get_multiplier(UNCERTAINTY, 0)[:] = 2.5
    model.generate_ensemble()

    system = make_systems(model.model, 1)[0]
    outputs = model(
        [system],
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
