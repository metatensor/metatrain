"""LLPR on a tiny se_e2_a model.

The fitting network is two units wide and the system contains two elements,
so each atom's last-layer feature has length 2 * (2 + 1) = 6.
"""

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.experimental.se_e2_a import SeE2A, Trainer
from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.llpr.trainer import compute_covariance
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from .test_basic import _minimal_hypers


def _dataset_info() -> DatasetInfo:
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )


def _systems() -> list[System]:
    """Dimers and monomers with both elements, so the type blocks all vary."""
    specs = [
        ([1, 6], [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0]]),
        ([1, 6], [[0.0, 0.0, 0.0], [1.4, -0.2, 0.1]]),
        ([6, 6], [[0.0, 0.0, 0.0], [1.2, 0.3, 0.0]]),
        ([1, 1], [[0.0, 0.0, 0.0], [0.8, 0.0, 0.2]]),
        ([1], [[0.0, 0.0, 0.0]]),
        ([6], [[0.2, 0.0, 0.0]]),
        ([1, 6, 1], [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.2, 1.0, 0.1]]),
        ([6, 1, 6], [[0.0, 0.0, 0.0], [0.7, 0.7, 0.0], [1.5, 0.1, 0.2]]),
    ]
    return [
        System(
            types=torch.tensor(types),
            positions=torch.tensor(positions, dtype=torch.float64),
            cell=torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        for types, positions in specs
    ]


def _energy_targets(systems: list[System], layout: TensorMap) -> list[TensorMap]:
    targets = []
    for index, _system in enumerate(systems):
        samples = Labels(["system"], torch.tensor([[index]]))
        block = layout.block()
        targets.append(
            TensorMap(
                layout.keys,
                [
                    TensorBlock(
                        values=torch.zeros(
                            [len(samples)] + list(block.values.shape[1:]),
                            dtype=torch.float64,
                        ),
                        samples=samples,
                        components=block.components,
                        properties=block.properties,
                    )
                ],
            )
        )
    return targets


def test_last_layer_features_are_affine_and_llpr_runs():
    """A trained-from-scratch tiny model exposes features LLPR can consume."""
    torch.manual_seed(0)
    hypers = _minimal_hypers("experimental.se_e2_a")
    hypers["descriptor"]["precision"] = 64
    hypers["fitting_net"]["precision"] = 64
    dataset_info = _dataset_info()
    training = get_default_hypers("experimental.se_e2_a")["training"]
    wrapped = Trainer(training).setup(hypers, dataset_info).to(torch.float64)
    model = wrapped.core
    assert isinstance(model, SeE2A)

    systems = _systems()
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    systems = [system.to(torch.float64) for system in systems]

    features_name = "mtt::aux::energy_last_layer_features"
    both = model(
        systems,
        {
            "energy": ModelOutput(sample_kind="system"),
            features_name: ModelOutput(sample_kind="system"),
        },
    )
    energy = both["energy"].block().values.detach()
    features = both[features_name].block().values.detach()
    # Six type-gated features: hidden width 2, two elements, plus the one-hot.
    assert features.shape == (len(systems), 2 * (2 + 1))
    design = torch.cat([features, torch.ones_like(features[:, :1])], dim=1)
    weights = torch.linalg.lstsq(design, energy).solution
    torch.testing.assert_close(design @ weights, energy, atol=1e-5, rtol=1e-5)

    n_atoms = sum(len(system) for system in systems)
    per_atom = model(
        systems,
        {features_name: ModelOutput(sample_kind="atom")},
    )
    assert per_atom[features_name].block().values.shape == (n_atoms, 6)

    llpr = LLPRUncertaintyModel({"num_ensemble_members": {}}, dataset_info)
    llpr.set_wrapped_model(wrapped)
    dataset = Dataset.from_dict(
        {
            "system": systems,
            "energy": _energy_targets(systems, dataset_info.targets["energy"].layout),
        }
    )
    compute_covariance(llpr, [dataset], batch_size=2, is_distributed=False)
    llpr.compute_cholesky_decomposition(regularizer=1e-4)

    predicted = llpr(
        systems,
        {
            "energy": ModelOutput(sample_kind="system"),
            "energy_uncertainty": ModelOutput(sample_kind="system"),
        },
    )
    uncertainty = predicted["energy_uncertainty"].block().values
    assert uncertainty.shape == (len(systems), 1)
    assert torch.isfinite(uncertainty).all()
    assert torch.all(uncertainty >= 0)
    # Distinct geometries must not collapse to one uncertainty.
    assert torch.std(uncertainty) > 0
