"""Shared behavioral tests of an architecture's last-layer (LLPR) interface.

Architectures with block-aligned last-layer features (see
:mod:`metatrain.utils.last_layer`) opt in by subclassing
:class:`LLPRInterfaceTests` next to their own tests and implementing
``make_backbone``; the suite then runs in the architecture's own test
environment, with its own dependencies.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


TARGET = "mtt::target"
UNCERTAINTY = "mtt::aux::target_uncertainty"
ENSEMBLE = "mtt::aux::target_ensemble"
LLF = "mtt::aux::target_last_layer_features"
NUM_ENSEMBLE_MEMBERS = 32


def target_dataset_info(target_type: dict, sample_kind: str) -> DatasetInfo:
    """A single-element dataset info with one target.

    :param target_type: the target's type.
    :param sample_kind: the target's sample kind.
    :return: the dataset info.
    """
    target = get_generic_target_info(
        TARGET,
        {
            "quantity": "",
            "unit": "",
            "num_subtargets": 1,
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


def _sample_kind(dataset_info: DatasetInfo) -> str:
    return dataset_info.targets[TARGET].sample_kind


def make_systems(model: ModelInterface, n_systems: int) -> List[System]:
    """Random four-atom clusters of the model's atomic types, with its neighbor lists.

    :param model: the model the systems are built for.
    :param n_systems: how many systems to build.
    :return: the systems.
    """
    torch.manual_seed(0)
    atomic_types = model.dataset_info.atomic_types
    systems = []
    for _ in range(n_systems):
        system = System(
            types=torch.tensor([atomic_types[i % len(atomic_types)] for i in range(4)]),
            positions=torch.eye(4, 3, dtype=torch.float64)
            + 0.4 * torch.randn(4, 3, dtype=torch.float64),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        systems.append(
            get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
        )
    return systems


# a fixed non-trivial rotation (orthogonal, determinant +1)
ROTATION = torch.tensor(
    [
        [-0.37308922665846000, 0.78897943879082055, 0.48817606876690917],
        [-0.35050433284023819, -0.60703397465603248, 0.71320156076211638],
        [0.85904082651036540, 0.09497999146462899, 0.50301854797787238],
    ],
    dtype=torch.float64,
)


def rotate_system(model: ModelInterface, system: System) -> System:
    """The system under the fixed ``ROTATION``, with fresh neighbor lists.

    :param model: the model the system is built for.
    :param system: the system to rotate.
    :return: the rotated system.
    """
    rotated = System(
        types=system.types,
        positions=system.positions @ ROTATION.T,
        cell=system.cell,
        pbc=system.pbc,
    )
    return get_system_with_neighbor_lists(rotated, model.requested_neighbor_lists())


def wrap_backbone(
    backbone: ModelInterface,
    dataset_info: DatasetInfo,
    ensembles: bool,
    target: str = TARGET,
) -> LLPRUncertaintyModel:
    """The backbone wrapped in an LLPR model.

    :param backbone: the backbone to wrap.
    :param dataset_info: the dataset info the backbone was built with.
    :param ensembles: whether to request ensemble members for the target.
    :param target: the target ensembles are requested for.
    :return: the wrapped model.
    """
    num_ensemble_members = {target: NUM_ENSEMBLE_MEMBERS} if ensembles else {}
    model = LLPRUncertaintyModel(
        {"num_ensemble_members": num_ensemble_members}, dataset_info
    )
    model.set_wrapped_model(backbone)
    return model.to(torch.float64)


def random_target_like(
    layout: TensorMap, system_index: int, num_atoms: int = 1
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


def fit_llpr(
    model: LLPRUncertaintyModel,
    systems: List[System],
    target: str = TARGET,
) -> None:
    """Fit the model's covariance and Cholesky factor on random targets.

    :param model: the wrapped model.
    :param systems: the systems to fit on.
    :param target: the target to fit.
    """
    layout = model.dataset_info.targets[target].layout
    targets = [
        random_target_like(layout, i, num_atoms=len(systems[i].positions))
        for i in range(len(systems))
    ]
    dataset = Dataset.from_dict({"system": systems, target: targets})
    model.compute_covariance([dataset], batch_size=2, is_distributed=False)
    model.compute_cholesky_decomposition(regularizer=1e-3)


def check_atomic_basis_uncertainty(backbone: ModelInterface) -> None:
    """Check the uncertainty of an atomic-basis target, whose blocks only hold the
    atoms of one atomic type, against its analytic expression from the last-layer
    features, the mean of its ensemble against the prediction, and the sharing of the
    ensemble weights between atomic types.

    :param backbone: the model, built with :func:`atomic_basis_dataset_info`.
    """
    model = wrap_backbone(backbone, backbone.dataset_info, ensembles=True)
    systems = make_systems(model.model, 8)
    fit_llpr(model, systems)
    model.generate_ensemble()

    outputs = {
        name: ModelOutput(sample_kind="atom")
        for name in (TARGET, UNCERTAINTY, ENSEMBLE, LLF)
    }
    out = model([systems[0]], outputs)
    assert out[UNCERTAINTY].keys == out[TARGET].keys
    for index, block in enumerate(out[TARGET].blocks()):
        n_samples = block.values.shape[0]
        prediction = block.values.reshape(n_samples, -1, block.values.shape[-1])

        feature_index = model.block_feature_index[TARGET][index]
        feature_block = out[LLF].block(feature_index)
        features = feature_block.values[feature_block.samples.select(block.samples)]
        features = features.reshape(n_samples, -1, features.shape[-1])
        cholesky = model._get_cholesky(UNCERTAINTY, feature_index)
        one_over_pr = torch.einsum(
            "scf,fg,scg->sc",
            features,
            torch.linalg.inv(cholesky @ cholesky.T),
            features,
        )
        uncertainty = out[UNCERTAINTY].block(index).values
        torch.testing.assert_close(
            uncertainty.reshape(prediction.shape),
            torch.sqrt(one_over_pr).unsqueeze(-1).expand_as(prediction),
        )

        members = (
            out[ENSEMBLE]
            .block(index)
            .values.reshape(n_samples, -1, NUM_ENSEMBLE_MEMBERS, prediction.shape[-1])
        )
        torch.testing.assert_close(members.mean(dim=2), prediction)

    # the atomic types of an irrep share the ensemble weights of their common
    # properties, as they share the corresponding last layer
    shared_weights: Dict[tuple, torch.Tensor] = {}
    layout = backbone.dataset_info.targets[TARGET].layout
    for index, (key, block) in enumerate(layout.items()):
        weight = model.llpr_ensemble_layers[f"{TARGET}_{index}"].weight
        weight = weight.reshape(
            NUM_ENSEMBLE_MEMBERS, -1, len(block.properties), weight.shape[-1]
        )
        for p, entry in enumerate(block.properties.values.tolist()):
            property_key = (int(key["o3_lambda"]), int(key["o3_sigma"]), *entry)
            reference = shared_weights.setdefault(property_key, weight[:, :, p])
            torch.testing.assert_close(weight[:, :, p], reference)
    assert len(shared_weights) < sum(len(block.properties) for block in layout.blocks())


class LLPRInterfaceTests:
    """Contract of an architecture's LLPR interface on spherical and Cartesian
    targets: the predictions are a linear readout of the block-aligned last-layer
    features, and uncertainties and ensembles are O(3)-consistent."""

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

    def make_backbone(self, dataset_info: DatasetInfo) -> ModelInterface:
        """The architecture's model, small and in float64.

        :param dataset_info: the dataset info to build the model with.
        :return: the model.
        """
        raise NotImplementedError

    def test_features_are_linear_readout(self, dataset_info: DatasetInfo) -> None:
        """Check every block of the prediction is a linear function of the
        corresponding block of the last-layer features, with the same weights for
        all components.

        :param dataset_info: Dataset information to initialize the model.
        """
        backbone = self.make_backbone(dataset_info)
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
        self, dataset_info: DatasetInfo
    ) -> None:
        """Check the uncertainty resolves the components of each block and its
        norm over them is rotation-invariant.

        :param dataset_info: Dataset information to initialize the model.
        """
        model = wrap_backbone(
            self.make_backbone(dataset_info), dataset_info, ensembles=False
        )
        systems = make_systems(model.model, 8)
        fit_llpr(model, systems)

        outputs = {UNCERTAINTY: ModelOutput(sample_kind=_sample_kind(dataset_info))}
        unc = model([systems[0]], outputs)[UNCERTAINTY]
        rotated_system = rotate_system(model.model, systems[0])
        unc_rot = model([rotated_system], outputs)[UNCERTAINTY]

        for block, block_rot in zip(unc.blocks(), unc_rot.blocks(), strict=True):
            values = block.values.reshape(block.values.shape[0], -1)
            values_rot = block_rot.values.reshape(block_rot.values.shape[0], -1)
            if values.shape[1] > 1:
                assert not torch.allclose(values, values.mean(dim=1, keepdim=True))
            # the sum of the variances over the components is invariant
            assert torch.allclose(
                (values**2).sum(dim=1), (values_rot**2).sum(dim=1), rtol=1e-6
            )

    def test_ensemble_recentered_and_rotation_invariant(
        self, dataset_info: DatasetInfo, tmp_path: Path
    ) -> None:
        """Check ensemble members re-center on the prediction and their centered
        norm over the components is rotation-invariant, and that the model can be
        exported.

        :param dataset_info: Dataset information to initialize the model.
        :param tmp_path: The pytest tmp_path fixture.
        """
        model = wrap_backbone(
            self.make_backbone(dataset_info), dataset_info, ensembles=True
        )
        systems = make_systems(model.model, 8)
        fit_llpr(model, systems)
        model.generate_ensemble()

        sample_kind = _sample_kind(dataset_info)
        outputs = {
            TARGET: ModelOutput(sample_kind=sample_kind),
            ENSEMBLE: ModelOutput(sample_kind=sample_kind),
        }
        out = model([systems[0]], outputs)
        rotated_system = rotate_system(model.model, systems[0])
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
            assert torch.allclose(
                (centered**2).sum(dim=1), (centered_rot**2).sum(dim=1), rtol=1e-4
            )

        model.export().save(str(tmp_path / "model.pt"))
        # exporting leaves the model usable, e.g. for checkpointing
        LLPRUncertaintyModel.load_checkpoint(model.get_checkpoint(), context="export")

    def test_atomic_basis_uncertainty(self) -> None:
        """Check the uncertainty and ensemble of an atomic-basis target."""
        check_atomic_basis_uncertainty(self.make_backbone(atomic_basis_dataset_info()))
