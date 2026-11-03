"""Small dataset infos, systems and models for the LLPR tests."""

from typing import List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.composition import CompositionModel
from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.llpr.trainer import compute_covariance
from metatrain.scaler import Scaler
from metatrain.utils.abc import ModelInterface
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.atomic_basis_helpers import densify_atomic_basis_dataset_info
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.wrapper import MetatrainModel

from .dummy_model import DummyModel


TARGET = "mtt::target"
UNCERTAINTY = "mtt::aux::target_uncertainty"
ENSEMBLE = "mtt::aux::target_ensemble"
LLF = "mtt::aux::target_last_layer_features"
NUM_ENSEMBLE_MEMBERS = 32


def target_dataset_info(
    target_type: dict,
    sample_kind: str,
    num_subtargets: int = 1,
    atomic_types: Optional[List[int]] = None,
) -> DatasetInfo:
    """A dataset info with one target.

    :param target_type: the target's type.
    :param sample_kind: the target's sample kind.
    :param num_subtargets: the target's number of properties.
    :param atomic_types: the atomic types, carbon only by default.
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
        length_unit="Angstrom",
        atomic_types=[6] if atomic_types is None else atomic_types,
        targets={TARGET: target},
    )


def spherical_dataset_info(sample_kind: str = "system") -> DatasetInfo:
    """A single-element dataset info with one spherical target, with a lambda=0 and
    a lambda=1 block.

    :param sample_kind: the target's sample kind.
    :return: the dataset info.
    """
    irreps = [{"o3_lambda": 0, "o3_sigma": 1}, {"o3_lambda": 1, "o3_sigma": 1}]
    return target_dataset_info({"spherical": {"irreps": irreps}}, sample_kind)


def atomic_basis_dataset_info() -> DatasetInfo:
    """A dataset info with one per-atom spherical target in an atomic basis, whose
    lambda=1 block only exists for carbon.

    :return: the dataset info.
    """
    irreps = {
        1: [{"num": 1, "o3_lambda": 0, "o3_sigma": 1}],
        6: [
            {"num": 2, "o3_lambda": 0, "o3_sigma": 1},
            {"num": 1, "o3_lambda": 1, "o3_sigma": 1},
        ],
    }
    return target_dataset_info(
        {"spherical": {"irreps": irreps}}, "atom", atomic_types=[1, 6]
    )


def make_systems(model: ModelInterface, n_systems: int) -> List[System]:
    """Random four-atom clusters of the model's atomic types.

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
        systems.append(system)
    return systems


def dummy_backbone(
    dataset_info: DatasetInfo,
    shared: bool = False,
    last_layer_features: bool = True,
) -> MetatrainModel:
    """The dummy model with a composition model and a scaler, as trainers set up
    models, in float64.

    :param dataset_info: the dataset info to build the model with.
    :param shared: whether the last-layer features are shared by all the blocks.
    :param last_layer_features: whether the model declares last-layer features.
    :return: the model.
    """
    model_dataset_info = densify_atomic_basis_dataset_info(dataset_info)
    hypers = {
        "num_features": 8,
        "shared": shared,
        "last_layer_features": last_layer_features,
    }
    return MetatrainModel(
        core=DummyModel(hypers, model_dataset_info),
        additive_models=[
            CompositionModel.from_valid_targets(
                model_dataset_info, model_dataset_info.atomic_types
            )
        ],
        scaler=Scaler.from_valid_targets(
            hypers=get_default_hypers("scaler")["model"],
            dataset_info=model_dataset_info,
        ),
        dataset_info=dataset_info,
    ).to(torch.float64)


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


def fit_llpr(model: LLPRUncertaintyModel, systems: List[System]) -> None:
    """Fit the model's covariance and Cholesky factor on ``systems``.

    :param model: the wrapped model.
    :param systems: the systems to fit on.
    """
    # the covariance only depends on the features, so the target can be zero
    layout = model.dataset_info.targets[TARGET].layout
    targets = []
    for index, system in enumerate(systems):
        if "atom" in layout.sample_names:
            samples = Labels(
                ["system", "atom"],
                torch.tensor([[index, atom] for atom in range(len(system))]),
            )
        else:
            samples = Labels(["system"], torch.tensor([[index]]))
        blocks = [
            TensorBlock(
                values=torch.zeros(
                    [len(samples)] + list(block.values.shape[1:]), dtype=torch.float64
                ),
                samples=samples,
                components=block.components,
                properties=block.properties,
            )
            for block in layout.blocks()
        ]
        targets.append(TensorMap(layout.keys, blocks))
    dataset = Dataset.from_dict({"system": systems, TARGET: targets})
    compute_covariance(model, [dataset], batch_size=2, is_distributed=False)
    model.compute_cholesky_decomposition(regularizer=1e-3)
