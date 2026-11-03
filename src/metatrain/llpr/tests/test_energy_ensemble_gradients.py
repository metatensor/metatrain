import copy
import re

import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels
from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
)
from omegaconf import OmegaConf

from metatrain.llpr import LLPRUncertaintyModel
from metatrain.llpr import Trainer as LLPRTrainer
from metatrain.pet import PET
from metatrain.pet import Trainer as PETTrainer
from metatrain.utils.data import DatasetInfo, get_atomic_types, get_dataset
from metatrain.utils.data.readers import read_systems
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatrain.utils.wrapper import MetatrainModel

from . import DATASET_WITH_FORCES_PATH, DEFAULT_HYPERS_LLPR, DEFAULT_HYPERS_PET


DTYPE = torch.float64
GRADIENTS = ["positions", "strain"]
ENSEMBLE = "energy_ensemble"

# gradients are computed once per member, so runtime scales with this
NUM_ENSEMBLE_MEMBERS = 8

# two runs of the same model should differ by numerical noise only
SAME_MODEL = {"atol": 1e-10, "rtol": 1e-8}

SMALL_PET = {
    "d_pet": 16,
    "d_head": 16,
    "d_node": 16,
    "d_feedforward": 16,
    "num_heads": 2,
    "num_attention_layers": 1,
    "num_gnn_layers": 1,
}


@pytest.fixture(scope="module")
def wrapped_pet(tmp_path_factory):
    """A one-epoch PET, as the checkpoint and the arguments needed to wrap it."""
    torch.manual_seed(0)

    targets = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_WITH_FORCES_PATH,
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    dataset, targets_info, _ = get_dataset(
        {
            "systems": {"read_from": DATASET_WITH_FORCES_PATH, "reader": "ase"},
            "targets": targets,
        }
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=get_atomic_types(dataset),
        targets=targets_info,
    )

    pet_hypers = copy.deepcopy(DEFAULT_HYPERS_PET)
    pet_hypers["model"].update(SMALL_PET)
    pet_hypers["training"]["num_epochs"] = 1
    pet_hypers["training"]["loss"] = OmegaConf.to_container(
        OmegaConf.create({"energy": init_with_defaults(LossSpecification)}),
        resolve=True,
    )

    train_kwargs = {
        "dtype": DTYPE,
        "devices": [torch.device("cpu")],
        "train_datasets": [dataset],
        "val_datasets": [dataset],
        "checkpoint_dir": "",
    }
    pet_trainer = PETTrainer(pet_hypers["training"])
    pet_model = pet_trainer.setup(pet_hypers["model"], dataset_info)
    pet_trainer.train(pet_model, **train_kwargs)

    checkpoint_path = str(tmp_path_factory.mktemp("llpr") / "pet.ckpt")
    pet_trainer.save_checkpoint(pet_model, checkpoint_path)

    return checkpoint_path, dataset_info, train_kwargs


def wrap_in_llpr(wrapped_pet, num_ensemble_members) -> LLPRUncertaintyModel:
    """The trained LLPR wrapper around ``wrapped_pet``, with those ensemble sizes."""
    checkpoint_path, dataset_info, train_kwargs = wrapped_pet

    llpr_hypers = copy.deepcopy(DEFAULT_HYPERS_LLPR)
    llpr_hypers["model"]["num_ensemble_members"] = num_ensemble_members
    llpr_hypers["training"].update(model_checkpoint=checkpoint_path, batch_size=4)
    trainer = LLPRTrainer(llpr_hypers["training"])
    model = trainer.setup(llpr_hypers["model"], dataset_info)
    trainer.train(model, **train_kwargs)

    return model


@pytest.fixture(scope="module")
def llpr_model(wrapped_pet) -> LLPRUncertaintyModel:
    """A one-epoch PET wrapped in LLPR, with an ensemble over the energy."""
    return wrap_in_llpr(wrapped_pet, {"energy": NUM_ENSEMBLE_MEMBERS})


def ensemble_output(gradients=GRADIENTS, sample_kind="system", name=ENSEMBLE):
    """The ``outputs`` argument requesting the ensemble ``name`` with ``gradients``."""
    return {
        name: ModelOutput(sample_kind=sample_kind, explicit_gradients=list(gradients))
    }


def make_systems(model, n=2, *, requires_grad=False, device="cpu"):
    """``n`` systems from the carbon dataset, with neighbor lists attached.

    ``requires_grad`` is a bool, or the tuple of ``"positions"`` and ``"cell"`` that
    should require gradients.
    """
    if isinstance(requires_grad, bool):
        requires_grad = ("positions", "cell") if requires_grad else ()

    requested_neighbor_lists = get_requested_neighbor_lists(model)
    systems = []
    for system in read_systems(DATASET_WITH_FORCES_PATH)[:n]:
        system = system.to(dtype=DTYPE)
        system = System(
            system.types,
            system.positions.detach().requires_grad_("positions" in requires_grad),
            system.cell.detach().requires_grad_("cell" in requires_grad),
            system.pbc,
        )
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        systems.append(system if device == "cpu" else system.to(device=device))
    return systems


def ensemble_block(model, systems, gradients=GRADIENTS, name=ENSEMBLE):
    return model(systems, ensemble_output(gradients, name=name))[name].block()


@pytest.mark.parametrize(
    "requires_grad",
    [("positions", "cell"), ("positions",)],
    ids=["positions+cell", "positions-only"],
)
def test_gradients_average_to_the_energy_gradients(llpr_model, requires_grad):
    """Check the mean of the ensemble gradients against the caller's own gradients
    of the energy, on the same graph."""
    systems = make_systems(llpr_model, requires_grad=requires_grad)
    outputs = llpr_model(
        systems, {"energy": ModelOutput(sample_kind="system"), **ensemble_output()}
    )
    ensemble = outputs[ENSEMBLE].block()

    inputs = [system.positions for system in systems]
    if "cell" in requires_grad:
        inputs += [system.cell for system in systems]
    gradients = torch.autograd.grad(outputs["energy"].block().values.sum(), inputs)
    positions_gradients = gradients[: len(systems)]

    torch.testing.assert_close(
        ensemble.gradient("positions").values.mean(dim=-1),
        torch.cat(positions_gradients, dim=0),
        atol=1e-8,
        rtol=1e-6,
    )
    if "cell" in requires_grad:
        # at strain = identity, dE/dstrain = positions^T dE/dpositions + cell^T dE/dcell
        strain_gradients = torch.stack(
            [
                system.positions.detach().t() @ positions_gradient
                + system.cell.detach().t() @ cell_gradient
                for system, positions_gradient, cell_gradient in zip(
                    systems, positions_gradients, gradients[len(systems) :], strict=True
                )
            ]
        )
        torch.testing.assert_close(
            ensemble.gradient("strain").values.mean(dim=-1),
            strain_gradients,
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize("requested", [["positions"], ["strain"]])
def test_only_requested_gradients_are_attached(llpr_model, requested):
    """Check only the requested gradients are computed."""
    block = ensemble_block(llpr_model, make_systems(llpr_model, n=1), requested)
    assert block.gradients_list() == requested


def test_batching_does_not_mix_systems(llpr_model):
    """Check the gradients of a system do not change when batched with another."""
    single = ensemble_block(llpr_model, make_systems(llpr_model, n=1))
    batched = ensemble_block(llpr_model, make_systems(llpr_model, n=2))

    for gradient in GRADIENTS:
        expected = single.gradient(gradient).values
        torch.testing.assert_close(
            batched.gradient(gradient).values[: expected.shape[0]],
            expected,
            **SAME_MODEL,
        )


def test_exported_model_gives_the_same_gradients(llpr_model, tmp_path):
    """Check the exported model gives the gradients of the eager one, from systems
    without gradients enabled."""
    path = str(tmp_path / "llpr_ensemble.pt")
    llpr_model = copy.deepcopy(llpr_model)
    llpr_model.export().save(path)

    options = ModelEvaluationOptions(
        length_unit="angstrom", outputs=ensemble_output(), selected_atoms=None
    )
    exported = load_atomistic_model(path)(
        make_systems(llpr_model), options, check_consistency=True
    )

    mts.allclose_block_raise(
        exported[ENSEMBLE].block(),
        ensemble_block(llpr_model, make_systems(llpr_model, requires_grad=True)),
        **SAME_MODEL,
    )


def test_model_without_any_ensemble_can_be_exported(wrapped_pet, tmp_path):
    """Check a model without ensembles, and so without gradient outputs, exports."""
    model = wrap_in_llpr(wrapped_pet, {})
    assert model.core.ensemble_gradient_outputs == []

    model.export().save(str(tmp_path / "llpr_without_ensemble.pt"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_gradients_on_cuda_match_cpu(llpr_model):
    """Check the gradients on CUDA, whose metadata must live on the device."""
    cuda_model = copy.deepcopy(llpr_model).to(device="cuda", dtype=DTYPE)
    block = ensemble_block(cuda_model, make_systems(cuda_model, device="cuda"))

    mts.allclose_block_raise(
        block.to(device="cpu"),
        ensemble_block(llpr_model, make_systems(llpr_model, requires_grad=True)),
        atol=1e-8,
        rtol=1e-6,
    )


def test_per_atom_energy_ensemble_is_rejected(llpr_model):
    """Check gradients of a per-atom energy ensemble are refused."""
    message = (
        f"explicit gradients of '{ENSEMBLE}' are only supported for a per-system "
        "energy, but this output was requested with sample_kind 'atom'"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        llpr_model(
            make_systems(llpr_model, n=1),
            ensemble_output(["positions"], sample_kind="atom"),
        )


def test_selected_atoms_are_rejected(llpr_model):
    """Check gradients together with ``selected_atoms`` are refused."""
    selected_atoms = Labels(
        names=["system", "atom"], values=torch.tensor([[0, 0], [0, 1]])
    )
    message = (
        f"explicit gradients of '{ENSEMBLE}' are not supported together with "
        "'selected_atoms'"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        llpr_model(
            make_systems(llpr_model, n=2),
            ensemble_output(["positions"]),
            selected_atoms,
        )


def test_multi_property_energy_ensemble_is_rejected():
    """Check gradients of an energy ensemble with several properties are refused."""
    model, ensemble_name = untrained_llpr_model(
        "mtt::my_energy", "energy", num_subtargets=3
    )
    message = (
        "explicit gradients of an energy ensemble are only supported for a "
        "single-property energy without components"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        ensemble_block(
            model, make_systems(model, n=1), ["positions"], name=ensemble_name
        )


def untrained_llpr_model(target_name, quantity, num_subtargets=1):
    """A minimal untrained LLPR model with a single target, and its ensemble name."""
    target_info = get_generic_target_info(
        target_name,
        {
            "quantity": quantity,
            "unit": "",
            "type": "scalar",
            "num_subtargets": num_subtargets,
            "sample_kind": "system",
        },
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets={target_name: target_info}
    )
    pet_hypers = copy.deepcopy(DEFAULT_HYPERS_PET["model"])
    pet_hypers.update(SMALL_PET)

    model = LLPRUncertaintyModel(
        {"num_ensemble_members": {target_name: 4}}, dataset_info
    )
    model.set_wrapped_model(
        MetatrainModel(
            core=PET(pet_hypers, dataset_info).to(DTYPE),
            additive_models=[],
            scaler=None,
            dataset_info=dataset_info,
        )
    )
    model = model.to(DTYPE)

    target = model.targets[target_name]
    cholesky = target.covariances[0].cholesky
    cholesky[:] = torch.eye(cholesky.shape[0], dtype=DTYPE)
    model.generate_ensemble()

    return model, target.ensemble_name


@pytest.mark.parametrize(
    "target_name, quantity, expected",
    [
        ("mtt::my_energy", "energy", GRADIENTS),
        ("energy/pbesol", "energy", GRADIENTS),
        ("mtt::my_charge", "charge", []),
    ],
    ids=["custom-name", "variant", "non-energy"],
)
def test_gradients_are_selected_by_quantity_not_by_name(
    target_name, quantity, expected
):
    """Check energy ensembles of any name or variant get gradients, others not."""
    model, ensemble_name = untrained_llpr_model(target_name, quantity)

    assert model.capabilities.outputs[ensemble_name].explicit_gradients == expected

    block = ensemble_block(
        model, make_systems(model, n=1), gradients=expected, name=ensemble_name
    )
    assert block.gradients_list() == expected
