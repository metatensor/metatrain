"""Tests for the heteroscedastic variance heads of FlashMD, and for their use as
aleatoric noise in the LLPR."""

import copy

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.experimental.flashmd import FlashMD, Trainer
from metatrain.llpr import Trainer as LLPRTrainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import (
    DatasetInfo,
    get_atomic_types,
    get_dataset,
    unpack_batch,
)
from metatrain.utils.data.target_info import TargetInfo
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import DATASET_PATH, MODEL_HYPERS


DTYPE = torch.float64
TARGETS = ["position", "momentum"]
POSITION = "mtt::aux::position_variance"

SMALL_FLASHMD = {
    "d_pet": 16,
    "d_head": 16,
    "d_node": 16,
    "d_feedforward": 16,
    "num_heads": 1,
    "num_attention_layers": 1,
    "num_gnn_layers": 1,
}


def _dataset_info():
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={
            name: TargetInfo(
                layout=TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=torch.empty((0, 3, 1), dtype=torch.float64),
                            samples=Labels(
                                names=["system", "atom"],
                                values=torch.empty((0, 2), dtype=int),
                            ),
                            components=[Labels.range("xyz", 3)],
                            properties=Labels.range(name, 1),
                        )
                    ],
                ),
                quantity="length",
                unit="angstrom",
            )
            for name in TARGETS
        },
    )


def _systems(model):
    systems = [
        System(
            types=torch.tensor([1, 6, 1]),
            positions=torch.randn(3, 3),
            cell=5.0 * torch.eye(3),
            pbc=torch.tensor([True] * 3),
        ),
        System(
            types=torch.tensor([6, 1]),
            positions=torch.randn(2, 3),
            cell=5.0 * torch.eye(3),
            pbc=torch.tensor([True] * 3),
        ),
    ]

    neighbor_lists = get_requested_neighbor_lists(model)
    systems = [get_system_with_neighbor_lists(s, neighbor_lists) for s in systems]

    for system in systems:
        n_atoms = len(system)
        system.add_data(
            "momentum",
            TensorMap(
                keys=Labels.single(),
                blocks=[
                    TensorBlock(
                        values=torch.randn(n_atoms, 3, 1),
                        samples=Labels(
                            names=["system", "atom"],
                            values=torch.stack(
                                [
                                    torch.zeros(n_atoms, dtype=int),
                                    torch.arange(n_atoms),
                                ],
                                dim=1,
                            ),
                        ),
                        components=[Labels.range("xyz", 3)],
                        properties=Labels.range("momentum", 1),
                    )
                ],
            ),
        )

    return systems


def test_variance_output():
    """The variance output is positive, isotropic, summed over atoms for
    per-structure outputs, and survives TorchScript and checkpointing."""
    model = FlashMD(MODEL_HYPERS, _dataset_info())
    model.set_masses({1: 1.008, 6: 12.011})

    assert POSITION not in model.supported_outputs()

    model.add_variance_output("position")
    model.add_variance_output("position")  # adding twice is a no-op

    assert model.variance_target_names == ["position"]
    assert model.supported_outputs()[POSITION].unit == "(angstrom)^2"

    systems = _systems(model)
    per_atom = model(
        systems,
        {
            "position": ModelOutput(sample_kind="atom"),
            POSITION: ModelOutput(sample_kind="atom"),
        },
    )

    variance = per_atom[POSITION].block()
    assert variance.values.shape == (5, 3, 1)
    assert variance.samples == per_atom["position"].block().samples
    assert torch.all(variance.values > 0.0)
    torch.testing.assert_close(variance.values[:, 0], variance.values[:, 2])

    # a new head starts from a unit variance (in normalized units) for all atoms, so
    # that its softplus is not saturated
    torch.testing.assert_close(
        variance.values, torch.full_like(variance.values, 1.0 + 1e-6)
    )

    # requesting only the variance gives the same result
    alone = model(systems, {POSITION: ModelOutput(sample_kind="atom")})

    assert list(alone.keys()) == [POSITION]
    torch.testing.assert_close(alone[POSITION].block().values, variance.values)

    # independent per-atom noise: per-structure variances are sums
    per_system = model(systems, {POSITION: ModelOutput(sample_kind="system")})

    torch.testing.assert_close(
        per_system[POSITION].block().values,
        torch.stack([variance.values[:3].sum(0), variance.values[3:].sum(0)]),
    )

    scripted = torch.jit.script(model.eval())
    scripted_output = scripted(systems, {POSITION: ModelOutput(sample_kind="atom")})

    torch.testing.assert_close(
        scripted_output[POSITION].block().values, variance.values
    )

    checkpoint = model.get_checkpoint()
    checkpoint["epoch"] = checkpoint["best_epoch"] = 0
    reloaded = FlashMD.load_checkpoint(checkpoint, "restart")
    assert reloaded.variance_target_names == ["position"]

    reloaded_output = reloaded(systems, {POSITION: ModelOutput(sample_kind="atom")})
    torch.testing.assert_close(
        reloaded_output[POSITION].block().values, variance.values
    )

    # removing the target also removes its variance head
    model.remove_output("position")

    assert POSITION not in model.supported_outputs()
    assert model.variance_target_names == []
    assert "position__variance" not in model.node_heads


def test_variance_output_unknown_target():
    model = FlashMD(MODEL_HYPERS, _dataset_info())
    with pytest.raises(ValueError, match="not a target of this model"):
        model.add_variance_output("energy")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A FlashMD model, the same model with trained variance heads, the LLPR wrapping
    the latter, and the dataset they were trained on."""
    torch.manual_seed(0)
    tmp_path = tmp_path_factory.mktemp("flashmd_variance")

    targets = {
        name: {
            "quantity": quantity,
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": key,
            "unit": unit,
            "type": {"cartesian": {"rank": 1}},
            "sample_kind": "atom",
            "num_subtargets": 1,
        }
        for name, quantity, key, unit in [
            ("position", "length", "future_positions", "A"),
            ("momentum", "momentum", "future_momenta", "(eV*u)^(1/2)"),
        ]
    }
    dataset, targets_info, _ = get_dataset(
        {
            "systems": {"read_from": DATASET_PATH, "reader": "ase"},
            "targets": targets,
        }
    )
    dataset_info = DatasetInfo(
        length_unit="A", atomic_types=get_atomic_types(dataset), targets=targets_info
    )

    train_kwargs = {
        "dtype": DTYPE,
        "devices": [torch.device("cpu")],
        "train_datasets": [dataset],
        "val_datasets": [dataset],
        "checkpoint_dir": str(tmp_path),
    }

    hypers = copy.deepcopy(get_default_hypers("experimental.flashmd"))
    hypers["model"].update(SMALL_FLASHMD)
    hypers["training"].update(timestep=32.0, batch_size=2, num_epochs=2)
    hypers["training"]["loss"] = OmegaConf.to_container(
        OmegaConf.create(
            {name: init_with_defaults(LossSpecification) for name in TARGETS}
        ),
        resolve=True,
    )

    # 1. the model with a mean head only
    base_path = str(tmp_path / "base.ckpt")
    trainer = Trainer(hypers["training"])
    base_model = trainer.setup(hypers["model"], dataset_info)

    trainer.train(base_model, **train_kwargs)
    trainer.save_checkpoint(base_model, base_path)

    # 2. variance heads on top of the frozen model
    variance_path = str(tmp_path / "variance.ckpt")
    variance_hypers = copy.deepcopy(hypers["training"])
    variance_hypers["finetune"]["read_from"] = base_path
    variance_hypers.update(variance_heads=TARGETS, num_epochs=20, learning_rate=1e-2)

    trainer = Trainer(variance_hypers)
    variance_model = model_from_checkpoint(
        torch.load(base_path, weights_only=False), "finetune"
    )
    variance_model = trainer.restart(variance_model, dataset_info, {})

    trainer.train(variance_model, **train_kwargs)
    trainer.save_checkpoint(variance_model, variance_path)

    # 3. LLPR on top of the model with variance heads
    llpr_hypers = copy.deepcopy(get_default_hypers("llpr"))
    llpr_hypers["training"].update(model_checkpoint=variance_path, batch_size=2)
    llpr_trainer = LLPRTrainer(llpr_hypers["training"])
    llpr = llpr_trainer.setup(llpr_hypers["model"], dataset_info)

    llpr_trainer.train(llpr, **train_kwargs)

    return {
        "base": model_from_checkpoint(
            torch.load(base_path, weights_only=False), "export"
        ),
        "variance": model_from_checkpoint(
            torch.load(variance_path, weights_only=False), "export"
        ),
        "llpr": llpr,
        "dataset": dataset,
    }


def _batches(trained):
    """The batches of the dataset, with neighbor lists, in double precision."""
    loader = trained["llpr"].core._get_dataloader(
        [trained["dataset"]], batch_size=2, is_distributed=False
    )
    for batch in loader:
        systems, targets, _ = unpack_batch(batch)
        yield [s.to(dtype=DTYPE) for s in systems], targets


def test_only_variance_heads_are_trained(trained):
    """The backbone, mean heads, atomic baseline and scaler are left untouched."""
    base = trained["base"].state_dict()
    variance = trained["variance"].state_dict()

    new_keys = set(variance) - set(base)
    assert len(new_keys) > 0
    assert all("__variance" in key for key in new_keys)
    for key, value in base.items():
        if key == "core.masses":
            # re-set by every training, in the (here different) dtype of the model,
            # and NaN for the atomic numbers which are not in the model
            torch.testing.assert_close(variance[key], value, equal_nan=True)
        else:
            _assert_identical(variance[key], value, key)


def _assert_identical(actual, expected, key):
    """Exact comparison of state dict entries, which can be nested containers (e.g.
    the serialized TensorMaps of the scaler)."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=key)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), key
        for k in expected:
            _assert_identical(actual[k], expected[k], key)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), key
        for a, e in zip(actual, expected, strict=True):
            _assert_identical(a, e, key)
    else:
        assert actual == expected, key


def test_variance_is_rescaled_at_evaluation(trained):
    """At evaluation, the variance in normalized units is multiplied by the square of
    the target's scale, while the mean is multiplied by the scale."""
    model = trained["variance"].to(dtype=DTYPE)
    systems, _ = next(_batches(trained))
    outputs = {name: ModelOutput(sample_kind="atom") for name in [*TARGETS, POSITION]}

    model.train()
    normalized = model(systems, outputs)
    model.eval()
    physical = model(systems, outputs)

    scale = model.scaler.get_scales(systems, {"position": outputs["position"]})
    scale = scale["position"].block().values
    torch.testing.assert_close(
        physical[POSITION].block().values,
        normalized[POSITION].block().values * scale**2,
    )


def _features_and_variances(trained, name):
    """Last-layer features, predicted variances, residuals and squared scales of the
    wrapped model, concatenated over the dataset."""
    model = trained["variance"].to(dtype=DTYPE).eval()

    features, variances, residuals, squared_scales = [], [], [], []
    with torch.no_grad():
        for systems, targets in _batches(trained):
            outputs = model(
                systems,
                {
                    name: ModelOutput(sample_kind="atom"),
                    f"mtt::aux::{name}_variance": ModelOutput(sample_kind="atom"),
                    f"mtt::aux::{name}_last_layer_features": ModelOutput(
                        sample_kind="atom"
                    ),
                },
            )

            features.append(
                outputs[f"mtt::aux::{name}_last_layer_features"].block().values
            )
            variances.append(outputs[f"mtt::aux::{name}_variance"].block().values)
            residuals.append(
                outputs[name].block().values
                - targets[name].block().values.to(dtype=DTYPE)
            )

            scales = model.scaler.get_scales(
                systems, {name: ModelOutput(sample_kind="atom")}
            )
            squared_scales.append(scales[name].block().values ** 2)

    return (
        torch.cat(features),
        torch.cat(variances),
        torch.cat(residuals),
        torch.cat(squared_scales),
    )


@pytest.mark.parametrize("name", TARGETS)
def test_llpr_covariance_is_weighted_by_variance(trained, name):
    """The LLPR Hessian is the Gauss-Newton matrix of the Gaussian NLL with predicted
    variance, for the weights of the unscaled last layer: sum_i c_i^2 f_i f_i^T / var_i,
    with c_i the scale of the target."""
    llpr = trained["llpr"].core
    assert llpr.heteroscedastic_outputs == TARGETS

    features, variances, _, squared_scales = _features_and_variances(trained, name)
    weights = squared_scales[:, 0, 0] / variances[:, 0, 0]
    expected = (features * weights[:, None]).T @ features
    torch.testing.assert_close(
        llpr._get_covariance(f"mtt::aux::{name}_uncertainty").to(DTYPE), expected
    )


@pytest.mark.parametrize("name", TARGETS)
def test_llpr_uncertainty_is_total(trained, name):
    """The LLPR uncertainty is sqrt(aleatoric var + alpha^2 c^2 f^T Lambda^-1 f)."""
    llpr = trained["llpr"].to(dtype=DTYPE).eval()
    uncertainty_name = f"mtt::aux::{name}_uncertainty"
    cholesky = llpr.core._get_cholesky(uncertainty_name).to(DTYPE)
    alpha = llpr.core._get_multiplier(uncertainty_name).to(DTYPE)
    precision_inverse = torch.cholesky_inverse(cholesky)

    scaler = trained["variance"].scaler
    for systems, _ in _batches(trained):
        outputs = llpr(
            systems,
            {
                uncertainty_name: ModelOutput(sample_kind="atom"),
                f"mtt::aux::{name}_variance": ModelOutput(sample_kind="atom"),
                f"mtt::aux::{name}_last_layer_features": ModelOutput(
                    sample_kind="atom"
                ),
            },
        )

        features = outputs[f"mtt::aux::{name}_last_layer_features"].block().values
        aleatoric = outputs[f"mtt::aux::{name}_variance"].block().values

        epistemic = torch.einsum("if,fg,ig->i", features, precision_inverse, features)
        scales = scaler.get_scales(systems, {name: ModelOutput(sample_kind="atom")})
        epistemic = scales[name].block().values ** 2 * epistemic[:, None, None]

        expected = torch.sqrt(aleatoric + alpha**2 * epistemic)
        torch.testing.assert_close(outputs[uncertainty_name].block().values, expected)


@pytest.mark.parametrize("name", TARGETS)
def test_llpr_calibration_minimizes_nll(trained, name):
    """The calibrated alpha minimizes the Gaussian NLL of the total variance."""
    llpr = trained["llpr"].core
    uncertainty_name = f"mtt::aux::{name}_uncertainty"
    alpha = llpr._get_multiplier(uncertainty_name).to(DTYPE)
    precision_inverse = torch.cholesky_inverse(
        llpr._get_cholesky(uncertainty_name).to(DTYPE)
    )

    features, variances, residuals, squared_scales = _features_and_variances(
        trained, name
    )
    epistemic = torch.einsum("if,fg,ig->i", features, precision_inverse, features)
    epistemic = squared_scales * epistemic[:, None, None]

    def nll(a):
        total = variances + a**2 * epistemic
        return torch.sum(torch.log(total) + residuals**2 / total)

    for factor in [0.9, 1.1]:
        assert nll(alpha) <= nll(alpha * factor)


def test_per_structure_variance_of_per_atom_target_is_not_supported(trained):
    """The scales of a per-atom target depend on the atomic type, so they can not be
    applied to a variance that was already summed over the atoms."""
    model = trained["variance"].to(dtype=DTYPE).eval()
    systems, _ = next(_batches(trained))

    with pytest.raises(NotImplementedError, match="per-atom target"):
        model(systems, {POSITION: ModelOutput(sample_kind="system")})


def test_variance_with_different_scaler_layout_is_not_supported(trained, monkeypatch):
    model = trained["variance"].to(dtype=DTYPE).eval()
    systems, _ = next(_batches(trained))

    # pretend that the layout of the model differs from the one of the scaler
    monkeypatch.setitem(model._eval_transforms, "pre_scaler", ["position"])

    with pytest.raises(NotImplementedError, match="layout"):
        model(systems, {POSITION: ModelOutput(sample_kind="atom")})


def test_only_the_heads_of_requested_outputs_are_evaluated():
    model = FlashMD(MODEL_HYPERS, _dataset_info())
    model.set_masses({1: 1.008, 6: 12.011})
    model.add_variance_output("position")
    systems = _systems(model)

    evaluated = []
    for head_name, heads in model.node_heads.items():
        heads[0].register_forward_hook(
            lambda module, args, output, name=head_name: evaluated.append(name)
        )

    def evaluated_heads(outputs):
        evaluated.clear()
        model(systems, outputs)
        return sorted(set(evaluated))

    atom = ModelOutput(sample_kind="atom")
    features = "mtt::aux::position_last_layer_features"

    assert evaluated_heads({"position": atom}) == ["position"]
    assert evaluated_heads({POSITION: atom}) == ["position__variance"]
    assert evaluated_heads({"momentum": atom}) == ["momentum"]
    assert evaluated_heads({features: atom}) == ["position"]
    assert evaluated_heads({"position": atom, POSITION: atom}) == [
        "position",
        "position__variance",
    ]
