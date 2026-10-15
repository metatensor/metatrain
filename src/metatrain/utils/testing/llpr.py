"""Shared behavioral tests of an architecture's last-layer (LLPR) interface.

Architectures with last-layer features that have the keys and components of their
targets opt in by subclassing :class:`LLPRInterfaceTests` next to their own tests,
setting ``architecture`` and, for faster tests, ``model_hypers``; the suite then
runs in the architecture's own test environment, with its own dependencies.

The dataset infos and systems are smaller than those of the other shared suites (one
atomic type, a target with a lambda=0 and a lambda=2 block, four-atom clusters), to
keep the fits of the LLPR fast.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
)
from metatomic.torch.o3 import random_transformations

from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.llpr.trainer import compute_covariance
from metatrain.utils.abc import ModelInterface
from metatrain.utils.architectures import get_default_hypers, import_architecture
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.wrapper import MetatrainModel

from .architectures import ArchitectureTests


TARGET = "mtt::target"
UNCERTAINTY = "mtt::aux::target_uncertainty"
ENSEMBLE = "mtt::aux::target_ensemble"
LLF = "mtt::aux::target_last_layer_features"
NUM_ENSEMBLE_MEMBERS = 32


def target_dataset_info(
    target_type: dict, sample_kind: str, num_subtargets: int = 1
) -> DatasetInfo:
    """A single-element dataset info with one target.

    :param target_type: the target's type.
    :param sample_kind: the target's sample kind.
    :param num_subtargets: the target's number of properties.
    :return: the dataset info.
    """
    target = get_generic_target_info(
        TARGET,
        {
            "quantity": "",
            "unit": "",
            "num_subtargets": num_subtargets,
            "sample_kind": sample_kind,
            "type": target_type,
        },
    )
    return DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets={TARGET: target}
    )


def spherical_dataset_info(
    irreps: Optional[List[Dict[str, int]]] = None,
) -> DatasetInfo:
    """A single-element dataset info with one per-system spherical target.

    :param irreps: the target's irreps; a lambda=0 and a lambda=2 block by
        default.
    :return: the dataset info.
    """
    if irreps is None:
        irreps = [
            {"o3_lambda": 0, "o3_sigma": 1},
            {"o3_lambda": 2, "o3_sigma": 1},
        ]
    return target_dataset_info({"spherical": {"irreps": irreps}}, "system")


def cartesian_dataset_info() -> DatasetInfo:
    """A single-element dataset info with one per-atom rank-1 Cartesian target.

    :return: the dataset info.
    """
    return target_dataset_info({"cartesian": {"rank": 1}}, "atom")


def atomic_basis_dataset_info() -> DatasetInfo:
    """A dataset info with one per-atom spherical target in an atomic basis, whose
    lambda=1 block only exists for carbon.

    :return: the dataset info.
    """
    target = get_generic_target_info(
        TARGET,
        {
            "quantity": "",
            "unit": "",
            "num_subtargets": 1,
            "sample_kind": "atom",
            "type": {
                "spherical": {
                    "irreps": {
                        1: [{"num": 1, "o3_lambda": 0, "o3_sigma": 1}],
                        6: [
                            {"num": 2, "o3_lambda": 0, "o3_sigma": 1},
                            {"num": 1, "o3_lambda": 1, "o3_sigma": 1},
                        ],
                    }
                }
            },
        },
    )
    return DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6], targets={TARGET: target}
    )


def make_systems(model: ModelInterface, n_systems: int) -> List[System]:
    """Random four-atom clusters of the model's atomic types, with its neighbor lists.

    :param model: the model the systems are built for.
    :param n_systems: how many systems to build.
    :return: the systems.
    """
    generator = torch.Generator().manual_seed(0)
    atomic_types = model.dataset_info.atomic_types
    systems = []
    for _ in range(n_systems):
        system = System(
            types=torch.tensor([atomic_types[i % len(atomic_types)] for i in range(4)]),
            positions=torch.eye(4, 3, dtype=torch.float64)
            + 0.4 * torch.randn(4, 3, dtype=torch.float64, generator=generator),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        systems.append(
            get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
        )
    return systems


def _rotate_system(model: ModelInterface, system: System) -> System:
    """The system under a random rotation, with fresh neighbor lists.

    :param model: the model the system is built for.
    :param system: the system to rotate.
    :return: the rotated system.
    """
    rotation = random_transformations(
        1,
        device=system.positions.device,
        dtype=system.positions.dtype,
        generator=torch.Generator().manual_seed(0),
    )
    rotated = rotation.transform_systems([system])[0]
    return get_system_with_neighbor_lists(rotated, model.requested_neighbor_lists())


def setup_backbone(
    architecture_name: str, model_hypers: dict, dataset_info: DatasetInfo
) -> MetatrainModel:
    """The model of an architecture, as set up by its trainer, in float64.

    :param architecture_name: name of the architecture.
    :param model_hypers: the hypers of the model.
    :param dataset_info: the dataset info to build the model with.
    :return: the model.
    """
    architecture = import_architecture(architecture_name)
    trainer = architecture.__trainer__(
        get_default_hypers(architecture_name)["training"]
    )
    return trainer.setup(model_hypers, dataset_info).to(torch.float64)


def wrap_backbone(
    backbone: MetatrainModel,
    dataset_info: DatasetInfo,
    ensembles: bool,
) -> LLPRUncertaintyModel:
    """The backbone wrapped in an LLPR model.

    :param backbone: the backbone to wrap.
    :param dataset_info: the dataset info the backbone was built with.
    :param ensembles: whether to request ensemble members for the target.
    :return: the wrapped model.
    """
    num_ensemble_members = {TARGET: NUM_ENSEMBLE_MEMBERS} if ensembles else {}
    model = LLPRUncertaintyModel(
        {"num_ensemble_members": num_ensemble_members}, dataset_info
    )
    model.set_wrapped_model(backbone)
    return model.to(torch.float64)


def _random_target_like(
    layout: TensorMap, system_index: int, num_atoms: int
) -> TensorMap:
    """A random target TensorMap with the layout's blocks for one system.

    :param layout: the target's layout.
    :param system_index: the system the samples point at.
    :param num_atoms: number of atoms, for per-atom layouts.
    :return: the random target.
    """
    blocks = []
    for block in layout.blocks():
        if "atom" in block.samples.names:
            samples = Labels(
                ["system", "atom"],
                torch.tensor([[system_index, a] for a in range(num_atoms)]),
            )
        else:
            samples = Labels(["system"], torch.tensor([[system_index]]))
        shape = [len(samples)] + list(block.values.shape[1:])
        blocks.append(
            TensorBlock(
                values=torch.randn(shape, dtype=torch.float64),
                samples=samples,
                components=block.components,
                properties=block.properties,
            )
        )
    return TensorMap(layout.keys, blocks)


def random_dataset(model: ModelInterface, systems: List[System]) -> Dataset:
    """A dataset of ``systems`` with random values of the target.

    :param model: the model whose dataset info gives the layout of the target.
    :param systems: the systems of the dataset.
    :return: the dataset.
    """
    layout = model.dataset_info.targets[TARGET].layout
    targets = [
        _random_target_like(layout, i, num_atoms=len(systems[i].positions))
        for i in range(len(systems))
    ]
    return Dataset.from_dict({"system": systems, TARGET: targets})


def fit_llpr(model: LLPRUncertaintyModel, systems: List[System]) -> None:
    """Fit the model's covariance and Cholesky factor on random targets.

    :param model: the wrapped model.
    :param systems: the systems to fit on.
    """
    dataset = random_dataset(model, systems)
    compute_covariance(model, [dataset], batch_size=2, is_distributed=False)
    model.compute_cholesky_decomposition(regularizer=1e-3)


def check_atomic_basis_uncertainty(backbone: MetatrainModel) -> None:
    """Check the uncertainty and ensemble of an atomic-basis target against the
    last-layer features, with different scales for each atomic type.

    :param backbone: the model, built with :func:`atomic_basis_dataset_info`.
    """
    model = wrap_backbone(backbone, backbone.dataset_info, ensembles=True)
    systems = make_systems(model.model, 8)
    fit_llpr(model, systems)
    model.generate_ensemble()

    # the uncertainty and ensemble are recomputed below from the features, the
    # Cholesky factor and the ensemble layers, as a reference implementation

    # different scales for each atomic type and property
    scaler = model.model.scaler
    assert scaler is not None
    scales = scaler.model.scales[TARGET]
    for block in scales.blocks():
        block.values[:] = 1.0 + torch.rand_like(block.values)

    outputs = {
        name: ModelOutput(sample_kind="atom")
        for name in (TARGET, UNCERTAINTY, ENSEMBLE, LLF)
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
        shared = model.targets[TARGET].shared
        feature_index = 0 if shared else dense_index
        feature_block = out[LLF].block(feature_index)
        features = feature_block.values[feature_block.samples.select(block.samples)]
        features = features.reshape(n_samples, -1, features.shape[-1])
        scales_block = scales.block(dense_index)
        dense_properties = scales_block.properties.select(block.properties)
        type_index = scaler.model.type_to_index[types[block.samples.column("atom")]]
        block_scales = scales_block.values[type_index][:, dense_properties].unsqueeze(1)

        cholesky = model.targets[TARGET].feature_blocks[feature_index].cholesky
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

        members = (
            out[ENSEMBLE]
            .block(key)
            .values.reshape(n_samples, -1, NUM_ENSEMBLE_MEMBERS, prediction.shape[-1])
        )
        layer = model.targets[TARGET].ensemble_layers[dense_index]
        num_components = prediction.shape[1]
        if not shared:
            deviations = layer(features).reshape(
                n_samples, num_components, NUM_ENSEMBLE_MEMBERS, -1
            )
        else:
            deviations = (
                layer(features[:, 0])
                .reshape(n_samples, NUM_ENSEMBLE_MEMBERS, num_components, -1)
                .movedim(1, 2)
            )
        deviations = deviations[..., dense_properties]
        deviations = deviations - deviations.mean(dim=2, keepdim=True)
        torch.testing.assert_close(
            members, prediction.unsqueeze(2) + deviations * block_scales.unsqueeze(2)
        )


class LLPRInterfaceTests(ArchitectureTests):
    """Tests of the LLPR interface of architectures with equivariant last-layer
    features."""

    @pytest.fixture(
        params=[spherical_dataset_info, cartesian_dataset_info],
        ids=["spherical", "cartesian"],
    )
    def dataset_info(self, request: pytest.FixtureRequest) -> DatasetInfo:
        """Fixture that provides the dataset info of the target under test.

        :param request: the pytest request, whose parameter builds the dataset info.
        :return: the dataset info.
        """
        return request.param()

    @pytest.fixture
    def backbone(self, model_hypers: dict, dataset_info: DatasetInfo) -> MetatrainModel:
        """Fixture that provides the architecture's model, in float64.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        :return: the model.
        """
        return setup_backbone(self.architecture, model_hypers, dataset_info)

    def test_features_are_linear_readout(self, backbone: MetatrainModel) -> None:
        """Check each block of the prediction is a linear readout of its features.

        :param backbone: The model to test.
        """
        outputs = {
            TARGET: ModelOutput(sample_kind="atom"),
            LLF: ModelOutput(sample_kind="atom"),
        }
        out = backbone(make_systems(backbone, 16), outputs)
        llf = out[LLF]
        prediction = out[TARGET]
        assert llf.keys == prediction.keys
        for index in range(len(prediction.keys)):
            features = llf.block(index).values
            features = features.reshape(-1, features.shape[-1])
            # constant feature for additive contributions
            features = torch.cat([features, torch.ones_like(features[:, :1])], dim=1)
            values = prediction.block(index).values
            values = values.reshape(-1, values.shape[-1])
            assert features.shape[0] > features.shape[1]
            weights = torch.linalg.lstsq(features, values).solution
            torch.testing.assert_close(features @ weights, values)

    def test_uncertainty_component_resolved_and_rotation_invariant(
        self, backbone: MetatrainModel, dataset_info: DatasetInfo
    ) -> None:
        """Check the uncertainty resolves the components and is rotation-invariant.

        :param backbone: The model to test.
        :param dataset_info: Dataset information to initialize the model.
        """
        model = wrap_backbone(backbone, dataset_info, ensembles=False)
        systems = make_systems(model.model, 8)
        fit_llpr(model, systems)

        sample_kind = dataset_info.targets[TARGET].sample_kind
        outputs = {UNCERTAINTY: ModelOutput(sample_kind=sample_kind)}
        unc = model([systems[0]], outputs)[UNCERTAINTY]
        rotated_system = _rotate_system(model.model, systems[0])
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

    def test_ensemble_recentered_and_rotation_invariant(
        self, backbone: MetatrainModel, dataset_info: DatasetInfo
    ) -> None:
        """Check the ensemble is centered on the prediction and rotation-invariant.

        :param backbone: The model to test.
        :param dataset_info: Dataset information to initialize the model.
        """
        model = wrap_backbone(backbone, dataset_info, ensembles=True)
        systems = make_systems(model.model, 8)
        fit_llpr(model, systems)
        model.generate_ensemble()

        sample_kind = dataset_info.targets[TARGET].sample_kind
        outputs = {
            TARGET: ModelOutput(sample_kind=sample_kind),
            ENSEMBLE: ModelOutput(sample_kind=sample_kind),
        }
        out = model([systems[0]], outputs)
        rotated_system = _rotate_system(model.model, systems[0])
        out_rot = model([rotated_system], outputs)

        for index in range(len(out[TARGET].keys)):
            prediction = out[TARGET].block(index).values
            n_samples = prediction.shape[0]
            prediction = prediction.reshape(n_samples, -1, 1)
            members = (
                out[ENSEMBLE]
                .block(index)
                .values.reshape(n_samples, -1, NUM_ENSEMBLE_MEMBERS)
            )
            members_rot = (
                out_rot[ENSEMBLE]
                .block(index)
                .values.reshape(n_samples, -1, NUM_ENSEMBLE_MEMBERS)
            )
            torch.testing.assert_close(members.mean(dim=-1, keepdim=True), prediction)
            centered = members - members.mean(dim=-1, keepdim=True)
            centered_rot = members_rot - members_rot.mean(dim=-1, keepdim=True)
            # loose tolerance: the members sample poorly constrained directions of
            # the weights, where rounding errors are amplified
            torch.testing.assert_close(
                (centered**2).sum(dim=1),
                (centered_rot**2).sum(dim=1),
                rtol=1e-4,
                atol=1e-8,
            )

    def test_export_and_checkpoint(
        self, backbone: MetatrainModel, dataset_info: DatasetInfo, tmp_path: Path
    ) -> None:
        """Check the exported model, and the model reloaded from its checkpoint, give
        the same predictions, uncertainties and ensembles.

        :param backbone: The model to test.
        :param dataset_info: Dataset information to initialize the model.
        :param tmp_path: The pytest tmp_path fixture.
        """
        model = wrap_backbone(backbone, dataset_info, ensembles=True)
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

    def test_atomic_basis_uncertainty(self, model_hypers: dict) -> None:
        """Check the uncertainty and ensemble of an atomic-basis target.

        :param model_hypers: Hyperparameters to initialize the model.
        """
        check_atomic_basis_uncertainty(
            setup_backbone(self.architecture, model_hypers, atomic_basis_dataset_info())
        )
