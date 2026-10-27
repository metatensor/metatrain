"""Heteroscedastic LLPR against the closed-form solution of Bayesian linear
regression, using a wrapped model whose last layer is a known linear model."""

import types
from typing import Dict, List, Optional

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelCapabilities, ModelOutput, System

from metatrain.llpr import LLPRUncertaintyModel
from metatrain.scaler import Scaler
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
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
    return 0.05 + 0.1 * positions[:, 0] ** 2


class LinearModel(torch.nn.Module):
    """Per-atom model ``w^T phi(r)`` with a predicted noise variance, in normalized
    units (the scaler of the wrapper converts them to physical units)."""

    def __init__(self, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.dataset_info = dataset_info
        self.atomic_types = dataset_info.atomic_types
        self.last_layer_feature_size = len(WEIGHTS)
        # shape (num_properties, num_features), as expected for the LLPR ensemble
        self.register_buffer("weights", WEIGHTS.clone().unsqueeze(0))
        self.last_layer_parameter_names = {TARGET: ["weights"]}
        self.outputs = {
            TARGET: ModelOutput(sample_kind="atom"),
            FEATURES: ModelOutput(sample_kind="atom"),
            VARIANCE: ModelOutput(sample_kind="atom"),
        }

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def requested_neighbor_lists(self) -> List:
        return []

    def export(self):
        capabilities = ModelCapabilities(
            outputs=self.outputs,
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
        samples = Labels(
            ["system", "atom"],
            torch.cat(
                [
                    torch.stack([torch.full((len(s),), i), torch.arange(len(s))], dim=1)
                    for i, s in enumerate(systems)
                ]
            ),
        )

        phi = features(positions)
        values = {
            TARGET: phi @ self.weights.T,
            FEATURES: phi,
            VARIANCE: normalized_variance(positions).unsqueeze(1),
        }

        properties = {
            TARGET: Labels.range("y", 1),
            FEATURES: Labels.range("feature", phi.shape[1]),
            VARIANCE: Labels.range("y", 1),
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


@pytest.fixture(params=[{1: 1.0, 6: 1.0}, {1: 2.0, 6: 0.5}], ids=["unscaled", "scaled"])
def setup(request):
    """The LLPR around the linear model, trained on noisy data from that model."""
    scales = request.param

    target_info = get_generic_target_info(
        TARGET,
        {
            "quantity": "",
            "unit": "",
            "type": "scalar",
            "num_subtargets": 1,
            "sample_kind": "atom",
        },
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6], targets={TARGET: target_info}
    )

    systems = _systems(40, seed=1)
    dataset = _make_dataset(systems, scales)

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
    llpr = LLPRUncertaintyModel(llpr_hypers, dataset_info)
    llpr.set_wrapped_model(wrapped)
    llpr.to(dtype=DTYPE)  # as done by the LLPR trainer

    llpr.compute_covariance([dataset], batch_size=4, is_distributed=False)
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

    cholesky = llpr._get_cholesky(UNCERTAINTY).to(DTYPE)
    torch.testing.assert_close(
        cholesky @ cholesky.T, _closed_form_precision(systems, scales)
    )


def test_predictive_variance_matches_closed_form(setup):
    """At new points, total variance = s(r*) + c*^2 phi*^T Lambda^-1 phi*."""
    llpr, systems, scales = setup
    precision_inverse = torch.linalg.inv(_closed_form_precision(systems, scales))

    test_systems = _systems(5, seed=2)
    outputs = llpr.eval()(
        test_systems,
        {
            TARGET: ModelOutput(sample_kind="atom"),
            UNCERTAINTY: ModelOutput(sample_kind="atom"),
        },
    )

    positions = torch.cat([s.positions for s in test_systems])
    c = torch.tensor(
        [scales[int(t)] for s in test_systems for t in s.types], dtype=DTYPE
    )
    phi = features(positions)

    expected_mean = c * (phi @ WEIGHTS)
    expected_variance = c**2 * normalized_variance(positions) + c**2 * torch.einsum(
        "if,fg,ig->i", phi, precision_inverse, phi
    )

    torch.testing.assert_close(outputs[TARGET].block().values[:, 0], expected_mean)
    torch.testing.assert_close(
        outputs[UNCERTAINTY].block().values[:, 0] ** 2, expected_variance
    )


def test_ensemble_spread_matches_epistemic_uncertainty(setup):
    """The spread of the ensemble is the epistemic part of the uncertainty,
    c* sqrt(phi*^T Lambda^-1 phi*), also when the targets are scaled."""
    llpr, systems, scales = setup
    precision_inverse = torch.linalg.inv(_closed_form_precision(systems, scales))

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


@pytest.mark.parametrize("training", [True, False])
def test_covariance_and_calibration_keep_the_mode_of_the_wrapped_model(setup, training):
    llpr, systems, scales = setup
    dataset = _make_dataset(systems, scales)

    llpr.model.train(training)

    llpr.compute_covariance([dataset], batch_size=4, is_distributed=False)
    assert llpr.model.training == training

    llpr.compute_cholesky_decomposition(regularizer=REGULARIZER)
    llpr.calibrate(
        [dataset],
        batch_size=4,
        is_distributed=False,
        calibration_method="squared_residuals",
    )
    assert llpr.model.training == training


def test_variance_and_uncertainty_with_different_sample_kinds(setup):
    llpr, _, _ = setup

    with pytest.raises(ValueError, match="sample_kind"):
        llpr.eval()(
            _systems(2, seed=3),
            {
                VARIANCE: ModelOutput(sample_kind="atom"),
                UNCERTAINTY: ModelOutput(sample_kind="system"),
            },
        )
