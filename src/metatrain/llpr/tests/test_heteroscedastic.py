"""Heteroscedastic LLPR against the closed-form solution of Bayesian linear
regression, using a wrapped model whose last layer is a known linear model, and a
variance head that predicts the known noise variance."""

import types
from typing import Dict, List, Optional

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelCapabilities, ModelOutput, System

from metatrain.llpr import LLPRUncertaintyModel
from metatrain.llpr.trainer import calibrate, compute_covariance, train_variance_heads
from metatrain.scaler import Scaler
from metatrain.scaler.utils.samples import get_samples_labels
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo, TargetInfo
from metatrain.utils.data.target_info import get_features_info, get_generic_target_info
from metatrain.utils.wrapper import MetatrainModel


DTYPE = torch.float64
TARGET = "mtt::y"
FEATURES = "mtt::aux::y_last_layer_features"
VARIANCE = "mtt::aux::y_variance"
UNCERTAINTY = "mtt::aux::y_uncertainty"
ENSEMBLE = "mtt::aux::y_ensemble"
WEIGHTS = torch.tensor([0.3, -1.0, 0.5, 2.0, -0.7], dtype=DTYPE)
REGULARIZER = 0.1
NUM_ENSEMBLE_MEMBERS = 20_000


def features(positions: torch.Tensor) -> torch.Tensor:
    """phi(r) = [1, x, y, z, x y]"""
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    return torch.stack([torch.ones_like(x), x, y, z, x * y], dim=1)


def normalized_variance(positions: torch.Tensor) -> torch.Tensor:
    """Noise variance, in the normalized units of the (unscaled) model."""
    return torch.nn.functional.softplus(positions[:, 0] - 2.0)


def _scalar_info(name: str) -> TargetInfo:
    return get_generic_target_info(
        name,
        {
            "quantity": "",
            "unit": "",
            "type": "scalar",
            "num_subtargets": 1,
            "sample_kind": "atom",
        },
    )


class KnownVariance(torch.nn.Module):
    """A variance head that predicts the noise variance, in normalized units, from
    the features ``phi(r)``, whose first two entries are ``1`` and ``x``."""

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(phi[:, 1] - 2.0 * phi[:, 0]).unsqueeze(1)


class LinearModel(torch.nn.Module):
    """Per-atom model ``w^T phi(r)``, in normalized units (the scaler of the wrapper
    converts them to physical units)."""

    def __init__(self, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.dataset_info = dataset_info
        self.atomic_types = dataset_info.atomic_types
        self.last_layer_feature_size = len(WEIGHTS)
        # shape (num_properties, num_features), as expected for the LLPR ensemble
        self.weights = torch.nn.Parameter(
            WEIGHTS.clone().unsqueeze(0), requires_grad=False
        )
        self.last_layer_parameter_names = {TARGET: ["weights"]}
        self.outputs = {
            TARGET: dataset_info.targets[TARGET],
            FEATURES: get_features_info(len(WEIGHTS)),
        }

    def supported_outputs(self) -> Dict[str, TargetInfo]:
        return self.outputs

    def requested_neighbor_lists(self) -> List:
        return []

    def export(self):
        capabilities = ModelCapabilities(
            outputs={
                name: info.to_model_output() for name, info in self.outputs.items()
            },
            atomic_types=self.atomic_types,
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        )
        return types.SimpleNamespace(capabilities=lambda: capabilities)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        assert selected_atoms is None
        assert all(output.sample_kind == "atom" for output in outputs.values())

        positions = torch.cat([system.positions for system in systems])
        samples = get_samples_labels(systems)["atom"]

        phi = features(positions)
        values = {
            TARGET: phi @ self.weights.T,
            FEATURES: phi,
        }

        properties = {
            TARGET: Labels.range("y", 1),
            FEATURES: Labels.range("feature", phi.shape[1]),
        }

        return {
            name: TensorMap(
                Labels.single(),
                [TensorBlock(values[name], samples, [], properties[name])],
            )
            for name in outputs
        }


def _systems(n_systems, seed):
    generator = torch.Generator().manual_seed(seed)

    systems = []
    for _ in range(n_systems):
        n_atoms = int(torch.randint(2, 6, (1,), generator=generator))
        systems.append(
            System(
                types=torch.tensor([1, 6])[
                    torch.randint(2, (n_atoms,), generator=generator)
                ],
                positions=torch.randn(n_atoms, 3, dtype=DTYPE, generator=generator),
                cell=torch.zeros(3, 3, dtype=DTYPE),
                pbc=torch.tensor([False] * 3),
            )
        )

    return systems


def _per_atom_tensor_map(index, system, values):
    return TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values.reshape(-1, 1),
                Labels(
                    ["system", "atom"],
                    torch.stack(
                        [
                            torch.full((len(system),), index),
                            torch.arange(len(system)),
                        ],
                        dim=1,
                    ),
                ),
                [],
                Labels.range("y", 1),
            )
        ],
    )


def _make_dataset(systems, scales):
    """Data in physical units: y = c_t w^T phi + noise of variance c_t^2 v(r)."""
    generator = torch.Generator().manual_seed(0)

    targets = []
    for index, system in enumerate(systems):
        c = torch.tensor([scales[int(t)] for t in system.types], dtype=DTYPE)
        mean = c * (features(system.positions) @ WEIGHTS)
        std = c * normalized_variance(system.positions).sqrt()
        noise = torch.randn(len(system), dtype=DTYPE, generator=generator)

        targets.append(_per_atom_tensor_map(index, system, mean + std * noise))

    return Dataset.from_dict({"system": systems, TARGET: targets})


def _wrap(dataset, scales):
    """The LLPR around the linear model, with a variance head for the target."""
    target_info = _scalar_info(TARGET)
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6], targets={TARGET: target_info}
    )

    scaler = Scaler(get_default_hypers("scaler")["model"], dataset_info)
    with pytest.warns(FutureWarning):
        scaler.train_model(
            [dataset],
            [],
            batch_size=4,
            is_distributed=False,
            fixed_weights={TARGET: scales},
        )

    wrapped = MetatrainModel(
        core=LinearModel(dataset_info),
        additive_models=[],
        scaler=scaler,
        dataset_info=dataset_info,
    ).to(dtype=DTYPE)

    llpr_hypers = get_default_hypers("llpr")["model"]
    llpr_hypers["num_ensemble_members"] = {TARGET: NUM_ENSEMBLE_MEMBERS}
    llpr_hypers["variance_heads"] = [TARGET]
    llpr = LLPRUncertaintyModel(llpr_hypers, dataset_info)
    llpr.set_wrapped_model(wrapped)

    return llpr.to(dtype=DTYPE)  # as done by the LLPR trainer


@pytest.fixture(params=[{1: 1.0, 6: 1.0}, {1: 2.0, 6: 0.5}], ids=["unscaled", "scaled"])
def setup(request):
    """The LLPR around the linear model, trained on noisy data from that model, with
    a variance head that predicts the known noise variance."""
    scales = request.param
    systems = _systems(40, seed=1)
    dataset = _make_dataset(systems, scales)

    llpr = _wrap(dataset, scales)
    llpr.variance_heads[TARGET] = KnownVariance()

    compute_covariance(llpr, [dataset], batch_size=4, is_distributed=False)
    llpr.compute_cholesky_decomposition(regularizer=REGULARIZER)
    # the calibration constant is 1, as in the closed form

    return llpr, systems, scales


def _closed_form_precision(systems, scales):
    """Lambda = sum_i c_i^2 phi_i phi_i^T / s_i + lambda I, with s_i = c_i^2 v(r_i)
    the noise variance in physical units."""
    precision = REGULARIZER * torch.eye(len(WEIGHTS), dtype=DTYPE)

    for system in systems:
        c = torch.tensor([scales[int(t)] for t in system.types], dtype=DTYPE)
        phi = features(system.positions)
        noise_variance = c**2 * normalized_variance(system.positions)

        precision += (phi * (c**2 / noise_variance)[:, None]).T @ phi

    return precision


def test_precision_matches_closed_form(setup):
    llpr, systems, scales = setup
    assert llpr.heteroscedastic_outputs == [TARGET]

    cholesky = llpr.targets[TARGET].covariances[0].cholesky.to(DTYPE)
    torch.testing.assert_close(
        cholesky @ cholesky.T, _closed_form_precision(systems, scales)
    )


def test_predictive_variance_matches_closed_form(setup):
    """At new points, the uncertainty is the epistemic one, c* sqrt(phi*^T Lambda^-1
    phi*), and the total variance is s(r*) + c*^2 phi*^T Lambda^-1 phi*."""
    llpr, systems, scales = setup
    precision_inverse = torch.linalg.inv(_closed_form_precision(systems, scales))

    test_systems = _systems(5, seed=2)
    outputs = llpr.eval()(
        test_systems,
        {
            TARGET: ModelOutput(sample_kind="atom"),
            VARIANCE: ModelOutput(sample_kind="atom"),
            UNCERTAINTY: ModelOutput(sample_kind="atom"),
        },
    )

    positions = torch.cat([s.positions for s in test_systems])
    c = torch.tensor(
        [scales[int(t)] for s in test_systems for t in s.types], dtype=DTYPE
    )
    phi = features(positions)

    expected_mean = c * (phi @ WEIGHTS)
    expected_aleatoric = c**2 * normalized_variance(positions)
    expected_epistemic = c**2 * torch.einsum("if,fg,ig->i", phi, precision_inverse, phi)

    aleatoric = outputs[VARIANCE].block().values[:, 0]
    epistemic = outputs[UNCERTAINTY].block().values[:, 0] ** 2

    torch.testing.assert_close(outputs[TARGET].block().values[:, 0], expected_mean)
    torch.testing.assert_close(aleatoric, expected_aleatoric)
    torch.testing.assert_close(epistemic, expected_epistemic)
    torch.testing.assert_close(
        aleatoric + epistemic, expected_aleatoric + expected_epistemic
    )


def test_ensemble_spread_matches_epistemic_uncertainty(setup):
    """The spread of the ensemble is the epistemic part of the uncertainty,
    c* sqrt(phi*^T Lambda^-1 phi*), also when the targets are scaled."""
    llpr, systems, scales = setup
    precision_inverse = torch.linalg.inv(_closed_form_precision(systems, scales))

    # the ensemble weights are drawn from the global random number generator, which
    # generate_ensemble does not let us replace by our own
    torch.manual_seed(0)
    llpr.generate_ensemble()

    test_systems = _systems(5, seed=2)
    outputs = llpr.eval()(test_systems, {ENSEMBLE: ModelOutput(sample_kind="atom")})

    positions = torch.cat([s.positions for s in test_systems])
    c = torch.tensor(
        [scales[int(t)] for s in test_systems for t in s.types], dtype=DTYPE
    )
    phi = features(positions)
    expected_std = c * torch.einsum("if,fg,ig->i", phi, precision_inverse, phi).sqrt()

    ensemble = outputs[ENSEMBLE].block().values
    assert ensemble.shape == (len(positions), NUM_ENSEMBLE_MEMBERS)
    torch.testing.assert_close(ensemble.std(dim=1), expected_std, rtol=0.05, atol=0)


def test_calibration_minimizes_the_nll_of_the_total_variance(setup):
    """For a target with a predicted variance, alpha minimizes the Gaussian NLL of
    s + alpha^2 e on the calibration data, whatever the calibration method."""
    llpr, _, scales = setup

    systems = _systems(40, seed=4)
    dataset = _make_dataset(systems, scales)
    calibrate(
        llpr, [dataset], batch_size=4, is_distributed=False, calibration_method="crps"
    )
    alpha = llpr.targets[TARGET].multipliers[0]

    outputs = llpr.eval()(
        systems,
        {
            TARGET: ModelOutput(sample_kind="atom"),
            VARIANCE: ModelOutput(sample_kind="atom"),
            UNCERTAINTY: ModelOutput(sample_kind="atom"),
        },
    )
    reference = torch.cat([dataset[i][TARGET].block().values for i in range(40)])
    r2 = (outputs[TARGET].block().values - reference)[:, 0] ** 2
    s = outputs[VARIANCE].block().values[:, 0]
    # the uncertainty is already calibrated
    e = outputs[UNCERTAINTY].block().values[:, 0] ** 2 / alpha**2

    def nll(a):
        v = s + a**2 * e
        return 0.5 * torch.sum(torch.log(v) + r2 / v)

    assert alpha > 0
    for factor in [0.9, 1.1]:
        assert nll(alpha) <= nll(factor * alpha)


def test_variance_head_starts_at_unit_variance():
    """Before training, the variance is 1 in normalized units, i.e. c^2."""
    scales = {1: 2.0, 6: 0.5}
    systems = _systems(5, seed=2)
    llpr = _wrap(_make_dataset(_systems(10, seed=1), scales), scales)

    outputs = llpr.eval()(systems, {VARIANCE: ModelOutput(sample_kind="atom")})

    c = torch.tensor([scales[int(t)] for s in systems for t in s.types], dtype=DTYPE)
    torch.testing.assert_close(
        outputs[VARIANCE].block().values[:, 0], c**2 + 1e-6 * c**2
    )


@pytest.mark.parametrize("scales", [{1: 1.0, 6: 1.0}, {1: 2.0, 6: 0.5}])
def test_trained_variance_head_learns_the_noise(scales):
    """The variance head, trained with the Gaussian NLL, recovers the noise variance,
    and the mean stays the same."""
    train = _make_dataset(_systems(400, seed=1), scales)
    val = _make_dataset(_systems(100, seed=3), scales)
    llpr = _wrap(train, scales)
    weights = llpr.model.core.weights.clone()

    hypers = get_default_hypers("llpr")["training"]
    hypers["variance_num_epochs"] = 30
    hypers["variance_learning_rate"] = 1e-2
    train_variance_heads(llpr, [train], [val], hypers, is_distributed=False)

    torch.testing.assert_close(llpr.model.core.weights, weights)

    test_systems = _systems(50, seed=4)
    outputs = llpr.eval()(test_systems, {VARIANCE: ModelOutput(sample_kind="atom")})
    positions = torch.cat([s.positions for s in test_systems])
    c = torch.tensor(
        [scales[int(t)] for s in test_systems for t in s.types], dtype=DTYPE
    )
    expected = c**2 * normalized_variance(positions)
    relative_error = (outputs[VARIANCE].block().values[:, 0] - expected) / expected
    assert relative_error.abs().median() < 0.15


def test_variance_has_to_be_requested_per_atom(setup):
    llpr, systems, _ = setup
    with pytest.raises(ValueError, match="can only be requested per atom"):
        llpr(systems[:1], {VARIANCE: ModelOutput(sample_kind="system")})


def test_variance_head_for_an_unknown_target():
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6],
        targets={TARGET: _scalar_info(TARGET)},
    )
    hypers = get_default_hypers("llpr")["model"]
    hypers["variance_heads"] = ["mtt::other"]
    llpr = LLPRUncertaintyModel(hypers, dataset_info)
    wrapped = MetatrainModel(
        core=LinearModel(dataset_info),
        additive_models=[],
        scaler=None,
        dataset_info=dataset_info,
    )
    with pytest.raises(ValueError, match="Cannot add a variance head"):
        llpr.set_wrapped_model(wrapped)
