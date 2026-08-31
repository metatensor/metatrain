import copy

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
from metatrain.llpr.model import _get_uncertainty_name
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
    pet_model = PET(pet_hypers["model"], dataset_info)
    pet_trainer = PETTrainer(pet_hypers["training"])
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
    model = LLPRUncertaintyModel(llpr_hypers["model"], dataset_info)
    LLPRTrainer(llpr_hypers["training"]).train(model, **train_kwargs)

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

    ``requires_grad`` selects how a caller hands systems over: ``True`` for an engine
    running its own autograd, ``False`` for one that does not and so relies on
    ``_systems_with_grad``. It also accepts ``("positions",)``, as sent by an engine
    computing forces but not stress.
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


def deterministic_energy_gradients(model, systems):
    """``dE/dr`` and ``dE/dstrain`` of the plain ``energy`` output.

    Differentiates that output directly, independently of
    ``_add_energy_ensemble_gradients``, so it can serve as its reference. The strain
    gradient uses the identity the implementation relies on: at strain = identity,
    ``dE/dstrain == positions^T @ dE/dpositions + cell^T @ dE/dcell``.
    """
    energy = model(systems, {"energy": ModelOutput(sample_kind="system")})
    gradients = torch.autograd.grad(
        energy["energy"].block().values.sum(),
        [s.positions for s in systems] + [s.cell for s in systems],
    )
    positions_grad, cell_grad = gradients[: len(systems)], gradients[len(systems) :]
    return {
        "positions": torch.cat(positions_grad, dim=0),
        "strain": torch.stack(
            [
                system.positions.detach().t() @ positions_grad[i]
                + system.cell.detach().t() @ cell_grad[i]
                for i, system in enumerate(systems)
            ]
        ),
    }


@pytest.mark.parametrize("gradient", GRADIENTS)
def test_gradients_average_to_the_deterministic_energy(llpr_model, gradient):
    """The ensemble is recentered on the model's own prediction, so the mean over
    members must reproduce the gradient of the deterministic ``energy`` output.
    """
    systems = make_systems(llpr_model, requires_grad=True)
    reference = deterministic_energy_gradients(llpr_model, systems)[gradient]

    block = ensemble_block(llpr_model, make_systems(llpr_model, requires_grad=True))
    torch.testing.assert_close(
        block.gradient(gradient).values.mean(dim=-1), reference, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("requested", [[], ["positions"], ["strain"]])
def test_only_requested_gradients_are_attached(llpr_model, requested):
    """Asking for both is what every other test does, so only the partial requests
    are worth spelling out here: each skips work the other one does."""
    block = ensemble_block(llpr_model, make_systems(llpr_model, n=1), requested)
    assert block.gradients_list() == requested


@pytest.mark.parametrize("gradient", GRADIENTS)
def test_batching_does_not_mix_systems(llpr_model, gradient):
    """Guards the atom-offset bookkeeping: the first system's gradients must not
    change when a second, independent system is batched alongside it."""
    single = ensemble_block(llpr_model, make_systems(llpr_model, n=1))
    batched = ensemble_block(llpr_model, make_systems(llpr_model, n=2))

    expected = single.gradient(gradient).values
    torch.testing.assert_close(
        batched.gradient(gradient).values[: expected.shape[0]], expected, **SAME_MODEL
    )


def test_gradients_do_not_depend_on_the_callers_grad_setup(llpr_model):
    """Systems handed over without grad enabled are rebuilt by ``_systems_with_grad``
    and must give what caller-prepared ones give, so that an engine can request
    ensemble gradients without any autograd setup of its own."""
    mts.allclose_block_raise(
        ensemble_block(llpr_model, make_systems(llpr_model)),
        ensemble_block(llpr_model, make_systems(llpr_model, requires_grad=True)),
        **SAME_MODEL,
    )


def test_exported_model_gives_the_same_gradients(llpr_model, tmp_path):
    """The gradient path must survive ``torch.jit.script`` at export, which is how
    engines actually load the model."""
    path = str(tmp_path / "llpr_ensemble.pt")
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
    """Requesting no ensembles at all leaves ``ensemble_gradient_outputs`` empty, and
    an empty list gives ``torch.jit.script`` no element type to infer at export. Every
    other test here configures an ensemble, so this is the only one that covers the
    plain uncertainty-only model, which is what the LLPR example trains."""
    model = wrap_in_llpr(wrapped_pet, {})
    assert model.ensemble_gradient_outputs == []

    model.export().save(str(tmp_path / "llpr_without_ensemble.pt"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_gradients_on_cuda_match_cpu(llpr_model):
    """The gradient blocks must be assembled on the device the values live on: their
    Labels were built on the CPU, which metatensor rejects."""
    cuda_model = copy.deepcopy(llpr_model).to(device="cuda", dtype=DTYPE)
    block = ensemble_block(cuda_model, make_systems(cuda_model, device="cuda"))

    for name in GRADIENTS:
        gradient = block.gradient(name)
        assert gradient.values.device.type == "cuda"
        assert gradient.samples.device.type == "cuda"
        assert gradient.properties.device.type == "cuda"
        assert all(c.device.type == "cuda" for c in gradient.components)

    mts.allclose_block_raise(
        block.to(device="cpu"),
        ensemble_block(llpr_model, make_systems(llpr_model, requires_grad=True)),
        **SAME_MODEL,
    )


@pytest.mark.parametrize(
    "requires_grad",
    [("positions", "cell"), ("positions",)],
    ids=["positions+cell", "positions-only"],
)
def test_caller_keeps_its_own_autograd_graph(llpr_model, requires_grad):
    """Requesting ensemble gradients must leave the caller's graph usable, so that an
    engine differentiating the energy itself can still run its own backward pass.
    """
    system = make_systems(llpr_model, n=1, requires_grad=requires_grad)[0]

    outputs = llpr_model(
        [system],
        {"energy": ModelOutput(sample_kind="system"), **ensemble_output()},
    )
    (caller_gradient,) = torch.autograd.grad(
        outputs["energy"].block().values.sum(), [system.positions]
    )

    # the ensemble is recentered on the energy, so the two must agree
    ensemble_gradient = (
        outputs[ENSEMBLE].block().gradient("positions").values.mean(dim=-1)
    )
    torch.testing.assert_close(caller_gradient, ensemble_gradient, atol=1e-8, rtol=1e-6)


def test_per_atom_energy_ensemble_is_rejected(llpr_model):
    """``_add_energy_ensemble_gradients`` assumes one sample per system; for a
    per-atom energy it would differentiate the summed energy instead."""
    with pytest.raises(ValueError, match="only supported for a per-system energy"):
        llpr_model(
            make_systems(llpr_model, n=1),
            ensemble_output(["positions"], sample_kind="atom"),
        )


def test_selected_atoms_are_rejected(llpr_model):
    """The gradient samples cover every atom of every system and index the value
    samples by system, so a selection that drops a system from the value block leaves
    the two pointing at different rows."""
    selected_atoms = Labels(
        names=["system", "atom"], values=torch.tensor([[0, 0], [0, 1]])
    )
    with pytest.raises(
        ValueError, match="not supported together with 'selected_atoms'"
    ):
        llpr_model(
            make_systems(llpr_model, n=2),
            ensemble_output(["positions"]),
            selected_atoms,
        )


def test_multi_property_energy_ensemble_is_rejected():
    """The ensemble values hold one column per member *per property*, so with more
    than one property the member-by-member differentiation would read the wrong
    columns."""
    model, ensemble_name = untrained_llpr_model(
        "mtt::my_energy", "energy", num_subtargets=3
    )
    with pytest.raises(ValueError, match="single-property energy without components"):
        ensemble_block(
            model, make_systems(model, n=1), ["positions"], name=ensemble_name
        )


def untrained_llpr_model(target_name, quantity, num_subtargets=1):
    """A minimal untrained LLPR model with a single target, and its ensemble name.

    Training is irrelevant to which outputs may carry gradients, so this skips it and
    seeds the covariance directly.
    """
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
    model.set_wrapped_model(PET(pet_hypers, dataset_info).to(DTYPE))
    model = model.to(DTYPE)

    uncertainty_name = _get_uncertainty_name(target_name)
    model._get_cholesky(uncertainty_name)[:] = torch.eye(
        model.ll_feat_size, dtype=DTYPE
    )
    model.generate_ensemble()

    return model, uncertainty_name.replace("_uncertainty", "_ensemble")


@pytest.mark.parametrize(
    "target_name, quantity, expected",
    [
        ("energy", "energy", GRADIENTS),
        ("mtt::my_energy", "energy", GRADIENTS),
        ("energy/pbesol", "energy", GRADIENTS),
        ("mtt::my_charge", "charge", []),
    ],
    ids=["default-name", "custom-name", "variant", "non-energy"],
)
def test_gradients_are_selected_by_quantity_not_by_name(
    target_name, quantity, expected
):
    """Energy targets may carry any name (``mtt::my_energy``) or variant
    (``energy/pbesol``), so the gradients follow the quantity. Keying on the name
    dropped the feature for all but the default one, and silently: the capabilities
    then agreed with ``forward`` that there was nothing to compute.
    """
    model, ensemble_name = untrained_llpr_model(target_name, quantity)

    assert model.capabilities.outputs[ensemble_name].explicit_gradients == expected

    block = ensemble_block(
        model, make_systems(model, n=1), gradients=expected, name=ensemble_name
    )
    assert block.gradients_list() == expected
