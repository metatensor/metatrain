from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional

import pytest
import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)

from metatrain.scaler import Scaler
from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.atomic_basis_helpers import densify_atomic_basis_dataset_info
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.wrapper import MetatrainModel


class Model1(ModelInterface):
    __supported_devices__ = ["cpu", "cuda"]
    __checkpoint_version__ = 1
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(authors=["Some person"])

    def __init__(self, hypers, dataset_info):
        super().__init__(hypers, dataset_info, self.__default_metadata__)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        raise NotImplementedError()

    def supported_outputs(self) -> Dict[str, TargetInfo]:
        return dict(self.dataset_info.targets)

    def restart(
        self,
        dataset_info: DatasetInfo,
        model_hypers: Optional[Mapping[str, Any]] = None,
    ) -> ModelInterface:
        raise NotImplementedError()

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> ModelInterface:
        raise NotImplementedError()

    def export(
        self,
        metadata: Optional[ModelMetadata] = None,
    ) -> AtomisticModel:

        capabilities = ModelCapabilities(
            outputs={
                k: ModelOutput(sample_kind=target.sample_kind, unit=target.unit)
                for k, target in self.dataset_info.targets.items()
            },
            atomic_types=self.dataset_info.atomic_types,
            interaction_range=int(self.__class__.__name__[-1]),
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(self.__supported_dtypes__[0]),
        )

        metadata = merge_metadata(self.metadata, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)

    @staticmethod
    def upgrade_checkpoint(
        checkpoint: Dict["str", Any], version: int | None = None
    ) -> Dict["str", Any]:
        raise NotImplementedError()

    def get_checkpoint(
        self, best_model_state_Dict: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        raise NotImplementedError()


class Model2(Model1):
    __supported_devices__ = ["cuda", "cpu"]
    __checkpoint_version__ = 2
    __supported_dtypes__ = [torch.float32]
    __default_metadata__ = ModelMetadata(authors=["Another person"])


class Model3(Model1):
    __supported_devices__ = ["cpu"]
    __checkpoint_version__ = 3
    __supported_dtypes__ = [torch.float64]
    __default_metadata__ = ModelMetadata(authors=["Yet another person"])


def test_only_core():
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )

    core = Model1({}, dataset_info)

    model = MetatrainModel(
        core=core,
        additive_models=[],
        scaler=None,
        dataset_info=core.dataset_info,
    )

    assert model.core is core
    assert len(model.additive_models) == 0
    assert model.scaler is None
    assert model.dataset_info == core.dataset_info

    # Check that there are no eval transforms
    assert all(len(v) == 0 for v in model._eval_transforms.values())

    # Check that scripting doesn't fail
    torch.jit.script(model)

    # Check that you can export the model.
    extra_metadata = ModelMetadata(name="Cool model")
    atomistic_model = model.export(extra_metadata)

    # Check that the metadata is correct
    metadata = atomistic_model.metadata()
    assert metadata.authors == core.__default_metadata__.authors
    assert metadata.name == "Cool model"

    # Check that the capabilities are correct
    capabilities = atomistic_model.capabilities()
    assert capabilities.atomic_types == core.dataset_info.atomic_types
    assert capabilities.interaction_range == 1
    assert capabilities.supported_devices == core.__supported_devices__
    for output_name, output in capabilities.outputs.items():
        assert output.sample_kind == core.dataset_info.targets[output_name].sample_kind
        assert output.unit == core.dataset_info.targets[output_name].unit


def test_with_additive_and_scaler():
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )
    core = Model1({}, dataset_info=dataset_info)

    additive_models = [
        Model2({}, dataset_info),
        Model3({}, dataset_info),
    ]

    scaler = Scaler({}, dataset_info)

    model = MetatrainModel(
        core=core,
        additive_models=additive_models,
        scaler=scaler,
        dataset_info=core.dataset_info,
    )

    assert model.core is core
    assert len(model.additive_models) == 2
    assert model.additive_models[0] is additive_models[0]
    assert model.additive_models[1] is additive_models[1]
    assert model.scaler is scaler
    assert model.dataset_info == core.dataset_info

    # Check that there are no eval transforms
    assert all(len(v) == 0 for v in model._eval_transforms.values())

    # Check that scripting doesn't fail
    torch.jit.script(model)

    # Check that you can export the model.
    extra_metadata = ModelMetadata(name="Cool model")
    atomistic_model = model.export(extra_metadata)

    # Check that the metadata is correct
    metadata = atomistic_model.metadata()
    assert metadata.authors == core.__default_metadata__.authors
    assert metadata.name == "Cool model"

    # Check that the capabilities are correct
    capabilities = atomistic_model.capabilities()
    assert capabilities.atomic_types == core.dataset_info.atomic_types
    assert capabilities.interaction_range == 3
    assert capabilities.supported_devices == ["cpu"]
    for output_name, output in capabilities.outputs.items():
        assert output.sample_kind == core.dataset_info.targets[output_name].sample_kind
        assert output.unit == core.dataset_info.targets[output_name].unit


@pytest.mark.parametrize(
    "target_change", ["scaler", "additive_0", "additive_1", "return"]
)
def test_with_different_dataset_infos(target_change):

    atomic_basis_target = get_generic_target_info(
        "spherical_atomic_basis",
        {
            "quantity": "",
            "unit": "",
            "type": {
                "spherical": {
                    "irreps": {
                        1: [{"o3_lambda": 0, "o3_sigma": 1}],
                        8: [{"o3_lambda": 0, "o3_sigma": 1}],
                    }
                }
            },
            "num_subtargets": 1,
            "sample_kind": "atom",
        },
    )

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={
            "energy": get_energy_target_info("energy", {"unit": "eV"}),
            "mtt::spherical_atomic_basis": atomic_basis_target,
        },
    )

    dense_dataset_info = densify_atomic_basis_dataset_info(dataset_info)

    core = Model1({}, dataset_info=dense_dataset_info)

    scaler_info = dense_dataset_info if target_change != "scaler" else dataset_info
    scaler = Scaler({}, dataset_info=scaler_info)

    add1_info = scaler_info if target_change != "additive_0" else dataset_info
    add2_info = add1_info if target_change != "additive_1" else dataset_info
    additive_models = [
        Model2({}, dataset_info=add1_info),
        Model3({}, dataset_info=add2_info),
    ]

    wrapper_info = add2_info if target_change != "return" else dataset_info
    model = MetatrainModel(
        core=core,
        additive_models=additive_models,
        scaler=scaler,
        dataset_info=wrapper_info,
    )

    assert model.core is core
    assert len(model.additive_models) == 2
    assert model.additive_models[0] is additive_models[0]
    assert model.additive_models[1] is additive_models[1]
    assert model.scaler is scaler
    assert model.dataset_info != core.dataset_info
    assert model.dataset_info == wrapper_info

    # Check that there is an eval transform at the right point
    for k, v in model._eval_transforms.items():
        if k != f"pre_{target_change}":
            assert len(v) == 0
        else:
            assert v == ["mtt::spherical_atomic_basis"]

    # Check that scripting doesn't fail
    torch.jit.script(model)

    # Check that you can export the model.
    extra_metadata = ModelMetadata(name="Cool model")
    atomistic_model = model.export(extra_metadata)

    # Check that the metadata is correct
    metadata = atomistic_model.metadata()
    assert metadata.authors == core.__default_metadata__.authors
    assert metadata.name == "Cool model"

    # Check that the capabilities are correct
    capabilities = atomistic_model.capabilities()
    assert capabilities.atomic_types == core.dataset_info.atomic_types
    assert capabilities.interaction_range == 3
    assert capabilities.supported_devices == ["cpu"]
    for output_name, output in capabilities.outputs.items():
        assert output.sample_kind == core.dataset_info.targets[output_name].sample_kind
        assert output.unit == core.dataset_info.targets[output_name].unit
