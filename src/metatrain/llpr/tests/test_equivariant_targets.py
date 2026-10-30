import re
from pathlib import Path

import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap
from metatomic.torch import ModelEvaluationOptions, ModelOutput, load_atomistic_model
from metatomic.torch.o3 import random_transformations

from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.llpr.trainer import calibrate, compute_covariance
from metatrain.utils import io
from metatrain.utils.data import Dataset, DatasetInfo

from .helpers import (
    ENSEMBLE,
    LLF,
    TARGET,
    UNCERTAINTY,
    atomic_basis_dataset_info,
    dummy_backbone,
    fit_llpr,
    make_systems,
    spherical_dataset_info,
    target_dataset_info,
    wrap_backbone,
)


DATASET_INFOS = [spherical_dataset_info("system"), spherical_dataset_info("atom")]


@pytest.fixture
def dummy_architecture(monkeypatch):
    """Make the checkpoints of the dummy model loadable like those of an
    architecture."""
    monkeypatch.setitem(
        io._MODEL_REGISTRY, "dummy", "metatrain.llpr.tests.dummy_model.DummyModel"
    )
    architectures = io.find_all_architectures() + ["dummy"]
    monkeypatch.setattr(io, "find_all_architectures", lambda: architectures)


def test_model_without_last_layer_features_is_refused():
    """Check a model without last-layer features is refused."""
    dataset_info = spherical_dataset_info()
    message = (
        "The wrapped model exposes no last-layer features, so LLPR can not compute "
        "any uncertainty"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        wrap_backbone(
            dummy_backbone(dataset_info, last_layer_features=False),
            dataset_info,
            ensembles=False,
        )


@pytest.mark.parametrize("shared", [False, True], ids=["per-block", "shared"])
def test_atomic_basis_uncertainty(shared):
    """Check the uncertainty of an atomic-basis target against the last-layer
    features, with different scales for each atomic type."""
    dataset_info = atomic_basis_dataset_info()
    backbone = dummy_backbone(dataset_info, shared=shared)
    model = wrap_backbone(backbone, dataset_info, ensembles=False)
    systems = make_systems(model.model, 8)
    fit_llpr(model, systems)

    # different scales for each atomic type and property
    scaler = model.model.scaler
    assert scaler is not None
    scales = scaler.model.scales[TARGET]
    generator = torch.Generator().manual_seed(0)
    for block in scales.blocks():
        block.values[:] = 1.0 + torch.rand(
            block.values.shape, dtype=block.values.dtype, generator=generator
        )

    outputs = {
        name: ModelOutput(sample_kind="atom") for name in (TARGET, UNCERTAINTY, LLF)
    }
    out = model([systems[0]], outputs)
    types = systems[0].types
    for key, block in out[TARGET].items():
        n_samples = block.values.shape[0]
        prediction = block.values.reshape(n_samples, -1, block.values.shape[-1])

        # the features and scales of the irrep, for the atoms and properties of this
        # block
        dense_key = [int(key["o3_lambda"]), int(key["o3_sigma"])]
        dense_index = scales.keys.position(dense_key)
        assert dense_index is not None
        feature_index = 0 if shared else dense_index
        feature_block = out[LLF].block(feature_index)
        features = feature_block.values[feature_block.samples.select(block.samples)]
        features = features.reshape(n_samples, -1, features.shape[-1])
        scales_block = scales.block(dense_index)
        dense_properties = scales_block.properties.select(block.properties)
        type_index = scaler.model.type_to_index[types[block.samples.column("atom")]]
        block_scales = scales_block.values[type_index][:, dense_properties].unsqueeze(1)

        cholesky = model.targets[TARGET].covariances[feature_index].cholesky
        one_over_pr = torch.einsum(
            "scf,fg,scg->sc",
            features,
            torch.linalg.inv(cholesky @ cholesky.T),
            features,
        )
        uncertainty = out[UNCERTAINTY].block(key).values
        torch.testing.assert_close(
            uncertainty.reshape(prediction.shape),
            (torch.sqrt(one_over_pr).unsqueeze(-1) * block_scales).expand_as(
                prediction
            ),
        )


@pytest.mark.usefixtures("dummy_architecture")
def test_covariance_of_a_float32_model_is_summed_in_float64():
    """Check a float32 model sums its covariance in float64, also after a reload."""
    dataset_info = spherical_dataset_info("atom")
    backbone = dummy_backbone(dataset_info, shared=True)
    model = wrap_backbone(backbone, dataset_info, ensembles=False)
    model.to(torch.float32)
    systems = make_systems(model.model, 8)
    fit_llpr(model, systems)

    systems = [system.to(torch.float32) for system in systems]
    outputs = {LLF: ModelOutput(sample_kind="atom")}
    features = model.model(systems, outputs)[LLF].block().values.to(torch.float64)
    reloaded = LLPRUncertaintyModel.load_checkpoint(model.get_checkpoint(), "restart")
    for llpr_model in (model, reloaded):
        torch.testing.assert_close(
            llpr_model.targets[TARGET].covariances[0].covariance,
            features.T @ features,
        )


def test_calibration_is_per_property_of_each_block():
    """Check each property of each block is calibrated against its own residuals."""
    irreps = [{"o3_lambda": 0, "o3_sigma": 1}, {"o3_lambda": 1, "o3_sigma": 1}]
    dataset_info = target_dataset_info(
        {"spherical": {"irreps": irreps}}, "system", num_subtargets=2
    )
    model = wrap_backbone(dummy_backbone(dataset_info), dataset_info, ensembles=False)
    systems = make_systems(model.model, 8)

    # references far larger than the untrained model's predictions make the
    # residuals of the two properties ~100x apart, in opposite ways in each block
    layout = dataset_info.targets[TARGET].layout
    reference_values = {0: [100.0, 1.0], 1: [1.0, 100.0]}

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
    irreps = [{"o3_lambda": 1, "o3_sigma": 1}]
    dataset_info = target_dataset_info({"spherical": {"irreps": irreps}}, "atom")
    model = LLPRUncertaintyModel(
        {"num_ensemble_members": {TARGET: n_ens}}, dataset_info
    )
    model.set_wrapped_model(dummy_backbone(dataset_info))
    model = model.to(torch.float64)
    systems = make_systems(model.model, 2)

    covariance = model.targets[TARGET].covariances[0].covariance
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


@pytest.mark.parametrize("dataset_info", DATASET_INFOS, ids=["per-system", "per-atom"])
def test_uncertainty_component_resolved_and_rotation_invariant(
    dataset_info: DatasetInfo,
) -> None:
    """Check the uncertainty has the metadata of the target, resolves the components
    and is rotation-invariant."""
    model = wrap_backbone(dummy_backbone(dataset_info), dataset_info, ensembles=False)
    systems = make_systems(model.model, 8)
    fit_llpr(model, systems)

    sample_kind = dataset_info.targets[TARGET].sample_kind
    outputs = {
        TARGET: ModelOutput(sample_kind=sample_kind),
        UNCERTAINTY: ModelOutput(sample_kind=sample_kind),
    }
    out = model([systems[0]], outputs)
    unc = out[UNCERTAINTY]
    assert mts.equal_metadata(unc, out[TARGET])
    rotation = random_transformations(
        1,
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(0),
    )
    rotated_system = rotation.transform_systems([systems[0]])[0]
    unc_rot = model([rotated_system], outputs)[UNCERTAINTY]

    for block, block_rot in zip(unc.blocks(), unc_rot.blocks(), strict=True):
        values = block.values.reshape(block.values.shape[0], -1)
        values_rot = block_rot.values.reshape(block_rot.values.shape[0], -1)
        if values.shape[1] > 1:
            assert not torch.allclose(values, values.mean(dim=1, keepdim=True))
        # the sum of the variances over the components is invariant
        torch.testing.assert_close(
            (values**2).sum(dim=1), (values_rot**2).sum(dim=1), rtol=1e-6, atol=1e-8
        )


@pytest.mark.usefixtures("dummy_architecture")
@pytest.mark.parametrize("dataset_info", DATASET_INFOS, ids=["per-system", "per-atom"])
def test_export_and_checkpoint(dataset_info: DatasetInfo, tmp_path: Path) -> None:
    """Check the exported model, and the model reloaded from its checkpoint, give
    the same predictions, uncertainties and ensembles."""
    model = wrap_backbone(dummy_backbone(dataset_info), dataset_info, ensembles=True)
    systems = make_systems(model.model, 8)
    fit_llpr(model, systems)
    model.generate_ensemble()

    path = str(tmp_path / "model.pt")
    model.export().save(path)

    sample_kind = dataset_info.targets[TARGET].sample_kind
    outputs = {
        name: ModelOutput(sample_kind=sample_kind)
        for name in (TARGET, UNCERTAINTY, ENSEMBLE)
    }
    expected = model(systems[:1], outputs)
    exported = load_atomistic_model(path)(
        systems[:1],
        ModelEvaluationOptions(length_unit="angstrom", outputs=outputs),
        check_consistency=True,
    )
    reloaded = [
        LLPRUncertaintyModel.load_checkpoint(model.get_checkpoint(), context).to(
            torch.float64
        )(systems[:1], outputs)
        for context in ("restart", "export")
    ]
    for actual in [exported] + reloaded:
        for name in outputs:
            for block, expected_block in zip(
                actual[name].blocks(), expected[name].blocks(), strict=True
            ):
                torch.testing.assert_close(block.values, expected_block.values)
