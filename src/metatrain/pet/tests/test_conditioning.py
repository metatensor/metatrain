"""Tests for charge and spin conditioning embeddings in PET."""

import copy

import ase
import ase.io
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch import load as mts_load
from metatomic.torch import ModelOutput, System
from metatomic_ase import MetatomicCalculator
from omegaconf import OmegaConf

from metatrain.cli.eval import eval_model
from metatrain.pet import PET, Trainer
from metatrain.pet.checkpoints import model_update_from_max_atom_sampler
from metatrain.pet.modules.conditioning import SystemConditioningEmbedding
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import CollateFn, Dataset, DatasetInfo, unpack_batch
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import (
    get_system_with_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.system_data import get_system_data_transform


def _small_hypers(system_conditioning=True, **kwargs):
    """Return minimal PET hypers with system conditioning."""
    hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    hypers["d_pet"] = 8
    hypers["d_head"] = 8
    hypers["d_node"] = 8
    hypers["d_feedforward"] = 8
    hypers["num_heads"] = 1
    hypers["num_attention_layers"] = 1
    hypers["num_gnn_layers"] = 1
    hypers["system_conditioning"] = system_conditioning
    hypers.update(kwargs)
    return hypers


def _dataset_info():
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )


def _make_scalar_tmap(value: int, property_name: str = "value") -> TensorMap:
    """Create a scalar TensorMap with a single float value (System requires float)."""
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[float(value)]]),
                samples=Labels("system", torch.tensor([[0]])),
                components=[],
                properties=Labels(property_name, torch.tensor([[0]])),
            )
        ],
    )


def _make_system(model, charge=None, spin_multiplicity=None):
    """Create a simple 2-atom system with optional charge/spin_multiplicity."""
    system = System(
        types=torch.tensor([6, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    if charge is not None:
        system.add_data("charge", _make_scalar_tmap(charge, "charge"))
    if spin_multiplicity is not None:
        system.add_data(
            "spin_multiplicity",
            _make_scalar_tmap(spin_multiplicity, "spin_multiplicity"),
        )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def test_conditioning_shapes():
    """SystemConditioningEmbedding produces [n_atoms, d_out]."""
    d_out = 16
    module = SystemConditioningEmbedding(
        d_out=d_out, max_charge=5, max_spin_multiplicity=5
    )

    charge = torch.tensor([0, 2])  # 2 systems
    spin_multiplicity = torch.tensor([1, 3])  # 2 systems
    system_indices = torch.tensor([0, 0, 0, 1, 1])  # 5 atoms total

    out = module(charge, spin_multiplicity, system_indices)
    assert out.shape == (5, d_out)
    assert out.dtype == torch.float32


def test_conditioning_different_d_node_d_pet():
    """Conditioning works when d_node != d_pet (the common case)."""
    hypers = _small_hypers(d_pet=8, d_node=16)
    model = PET(hypers, _dataset_info())
    model.eval()

    system = _make_system(model, charge=1, spin_multiplicity=2)
    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        result = model([system], outputs)
    assert "energy" in result


def _train_steps(model, n_steps=20):
    """Do a few optimizer steps to break the zero-init of the conditioning gate."""
    torch.manual_seed(42)
    model.train()
    for _ in range(n_steps):
        system = _make_system(model, charge=2, spin_multiplicity=3)
        outputs = {"energy": ModelOutput(per_atom=False)}
        result = model([system], outputs)
        loss = result["energy"].block().values.sum()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= 0.01 * p.grad
                    p.grad.zero_()
    model.eval()


def test_conditioning_changes_output():
    """Same structure with different charges should produce different predictions."""
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    _train_steps(model)

    system_neutral = _make_system(model, charge=0, spin_multiplicity=1)
    system_charged = _make_system(model, charge=2, spin_multiplicity=1)

    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        result_neutral = model([system_neutral], outputs)
        result_charged = model([system_charged], outputs)

    e_neutral = result_neutral["energy"].block().values
    e_charged = result_charged["energy"].block().values
    assert not torch.allclose(e_neutral, e_charged), (
        "Different charges should produce different energies"
    )


def test_conditioning_disabled_unchanged():
    """With system_conditioning=False, no conditioning module should exist."""
    hypers_off = _small_hypers(system_conditioning=False)
    model = PET(hypers_off, _dataset_info())

    assert model.system_conditioning is None

    # Model should still run fine
    model.eval()
    system = _make_system(model)
    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        result = model([system], outputs)
    assert "energy" in result


def test_conditioning_gradients_flow():
    """Gradients should flow through the conditioning embeddings."""
    module = SystemConditioningEmbedding(d_out=8, max_charge=5, max_spin_multiplicity=5)

    charge = torch.tensor([1])
    spin_multiplicity = torch.tensor([2])
    system_indices = torch.tensor([0, 0])

    out = module(charge, spin_multiplicity, system_indices)
    loss = out.sum()
    loss.backward()

    assert module.charge_embedding.weight.grad is not None
    assert module.spin_multiplicity_embedding.weight.grad is not None
    assert module.project[0].weight.grad is not None


def test_conditioning_batch_independence():
    """Changing charge of one system in a batch should not affect others."""
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    _train_steps(model)

    system_a = _make_system(model, charge=0, spin_multiplicity=1)
    system_b_v1 = _make_system(model, charge=1, spin_multiplicity=1)
    system_b_v2 = _make_system(model, charge=3, spin_multiplicity=2)

    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        result_v1 = model([system_a, system_b_v1], outputs)
        result_v2 = model([system_a, system_b_v2], outputs)

    # Energy of system_a should be the same in both batches
    e_a_v1 = result_v1["energy"].block().values[0]
    e_a_v2 = result_v2["energy"].block().values[0]
    torch.testing.assert_close(e_a_v1, e_a_v2)

    # Energy of system_b should differ between batches
    e_b_v1 = result_v1["energy"].block().values[1]
    e_b_v2 = result_v2["energy"].block().values[1]
    assert not torch.allclose(e_b_v1, e_b_v2)


def test_conditioning_default_values():
    """Systems without explicit charge/spin_multiplicity should use defaults
    (charge=0, spin_multiplicity=1)."""
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    model.eval()

    # System with no charge/spin data (should default to charge=0, spin_multiplicity=1)
    system_default = _make_system(model)
    # System with explicit charge=0, spin_multiplicity=1
    system_explicit = _make_system(model, charge=0, spin_multiplicity=1)

    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        result_default = model([system_default], outputs)
        result_explicit = model([system_explicit], outputs)

    e_default = result_default["energy"].block().values
    e_explicit = result_explicit["energy"].block().values
    torch.testing.assert_close(e_default, e_explicit)


def test_conditioning_out_of_range():
    """Charges or spins outside the supported range raise ValueError."""
    module = SystemConditioningEmbedding(d_out=8, max_charge=3, max_spin_multiplicity=4)

    # charge too positive
    with pytest.raises(ValueError, match=r"charge values must be in \[-3, 3\]"):
        module.validate(torch.tensor([5]), torch.tensor([1]))

    # charge too negative
    with pytest.raises(ValueError, match=r"charge values must be in \[-3, 3\]"):
        module.validate(torch.tensor([-4]), torch.tensor([1]))

    # spin too high
    with pytest.raises(
        ValueError, match=r"spin_multiplicity values must be in \[1, 4\]"
    ):
        module.validate(torch.tensor([0]), torch.tensor([5]))

    # spin too low (0 is invalid, minimum is 1)
    with pytest.raises(
        ValueError, match=r"spin_multiplicity values must be in \[1, 4\]"
    ):
        module.validate(torch.tensor([0]), torch.tensor([0]))


def _make_raw_system() -> System:
    """2-atom system without neighbor lists (as stored in a Dataset).

    Uses float64 to match what read_systems() produces; batch_to() later
    converts to the model's float32.
    """
    return System(
        types=torch.tensor([6, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]], dtype=torch.float64),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


def _make_extra_tmap(value: float, prop_name: str, system_index: int = 0) -> TensorMap:
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[value]], dtype=torch.float64),
                samples=Labels("system", torch.tensor([[system_index]])),
                components=[],
                properties=Labels(prop_name, torch.tensor([[0]])),
            )
        ],
    )


def test_eval_routes_extra_data_to_conditioning():
    """Regression: extra_data charge/spin must reach system conditioning during eval.

    Simulates the eval data pipeline (Dataset + CollateFn with
    get_system_data_transform) and verifies that a non-default charge produces
    different model predictions than the default (charge=0).

    This would silently fail if the transform were not applied (or applied with
    an empty key list): all systems would fall back to charge=0 /
    spin_multiplicity=1 without any error. The actual ``mtt eval`` wiring of
    the transform is covered by ``test_eval_model_end_to_end_with_extra_data``.
    """
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    _train_steps(model)
    model.eval()

    extra_data_keys = ["charge", "spin_multiplicity"]

    dataset_neutral = Dataset.from_dict(
        {
            "system": [_make_raw_system()],
            "charge": [_make_extra_tmap(0.0, "charge")],
            "spin_multiplicity": [_make_extra_tmap(1.0, "spin_multiplicity")],
        }
    )
    dataset_charged = Dataset.from_dict(
        {
            "system": [_make_raw_system()],
            "charge": [_make_extra_tmap(2.0, "charge")],  # non-default
            "spin_multiplicity": [_make_extra_tmap(1.0, "spin_multiplicity")],
        }
    )

    callables = [
        get_system_with_neighbor_lists_transform(model.requested_neighbor_lists()),
        get_system_data_transform(extra_data_keys),
    ]
    collate_fn = CollateFn(["energy"], callables=callables)

    systems_neutral, _, _ = unpack_batch(collate_fn([dataset_neutral[0]]))
    systems_charged, _, _ = unpack_batch(collate_fn([dataset_charged[0]]))

    # Verify charge was actually attached to systems — if the transform is
    # skipped, known_data() will be empty and the assertion fails here.
    assert "charge" in systems_neutral[0].known_data(), (
        "charge not attached to neutral system — "
        "get_system_data_transform was not applied"
    )
    assert "charge" in systems_charged[0].known_data(), (
        "charge not attached to charged system — "
        "get_system_data_transform was not applied"
    )

    charge_neutral = systems_neutral[0].get_data("charge").block().values.item()
    charge_charged = systems_charged[0].get_data("charge").block().values.item()
    assert charge_neutral == 0.0
    assert charge_charged == 2.0

    # Convert to float32 (model dtype), mirroring batch_to() in the eval loop
    systems_neutral = [s.to(dtype=torch.float32) for s in systems_neutral]
    systems_charged = [s.to(dtype=torch.float32) for s in systems_charged]

    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        e_neutral = model(systems_neutral, outputs)["energy"].block().values
        e_charged = model(systems_charged, outputs)["energy"].block().values

    assert not torch.allclose(e_neutral, e_charged), (
        "charge=0 and charge=2 produce identical energies — "
        "conditioning is not being applied (extra_data not routed to systems)"
    )


@pytest.mark.filterwarnings(
    "ignore:the 'features' output name is deprecated:UserWarning",
    "ignore:`per_atom` is deprecated:DeprecationWarning",
    "ignore:ModelOutput.quantity is deprecated:UserWarning",
    "ignore:Found metatomic.torch v.*vesin.metatomic was only tested:UserWarning",
)
def test_export_with_conditioning_preserves_validate(tmp_path):
    """Exporting (TorchScript) a PET model with system_conditioning must keep the
    in-forward ``validate(...)`` call working: ``mtt export`` succeeds, valid
    inputs run through, and an out-of-range value raises a clear error from the
    scripted module (driven via ``MetatomicCalculator`` for end-to-end coverage).
    """
    hypers = _small_hypers(max_charge=3, max_spin_multiplicity=4)
    model = PET(hypers, _dataset_info())
    model.eval()

    path = str(tmp_path / "pet_conditioning.pt")
    # If validate() did not survive scripting, this line raises during export.
    model.export().save(path)

    calculator = MetatomicCalculator(path)

    # Valid inputs (spin_multiplicity within [1, max_spin_multiplicity]) work.
    atoms = ase.Atoms("CH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    atoms.info["charge"] = 2
    atoms.info["spin"] = 3
    atoms.calc = calculator
    _ = atoms.get_potential_energy()

    # Out-of-range spin_multiplicity must be rejected by the scripted validate().
    atoms_bad = ase.Atoms("CH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    atoms_bad.info["charge"] = 0
    atoms_bad.info["spin"] = 99
    atoms_bad.calc = MetatomicCalculator(path)
    with pytest.raises(Exception, match="spin_multiplicity"):
        atoms_bad.get_potential_energy()


def _to_max_atom_sampler_shape(checkpoint):
    """Rewrite a checkpoint produced on this branch so it looks like one that
    came out of the ``origin/max-atom-sampler`` fork.

    Reverses every rename that
    :func:`metatrain.pet.checkpoints.model_update_from_max_atom_sampler`
    is expected to undo, so we can exercise the function on a realistic
    payload without depending on an external .ckpt artifact.
    """
    hypers = checkpoint["model_data"]["model_hypers"]
    hypers["max_spin"] = hypers.pop("max_spin_multiplicity")
    hypers.pop("adaptive_cutoff_method", None)

    for key in ("model_state_dict", "best_model_state_dict"):
        sd = checkpoint.get(key)
        if sd is None:
            continue
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("system_conditioning.spin_multiplicity_embedding."):
                k = k.replace(
                    "system_conditioning.spin_multiplicity_embedding.",
                    "system_conditioning.spin_embedding.",
                )
            if "gnn_layers" in k and ".edge_embedder." in k:
                k = k.replace(".edge_embedder.", ".edge_linear.")
            new_sd[k] = v
        checkpoint[key] = new_sd

    # Per-property scaler buffers did not exist before PR #1107 either.
    for key in ("model_state_dict", "best_model_state_dict"):
        sd = checkpoint.get(key)
        if sd is None:
            continue
        for k in list(sd):
            if "_per_target_scaler_buffer" in k or "_per_property_scaler_buffer" in k:
                del sd[k]

    checkpoint["model_ckpt_version"] = 11
    return checkpoint


def test_model_update_from_max_atom_sampler():
    """Round-trip: build a v14 checkpoint with conditioning, downgrade it to
    a max-atom-sampler-shaped v11 payload, then verify our migration restores
    something the standard loader accepts."""
    hypers = _small_hypers(
        system_conditioning=True, max_charge=3, max_spin_multiplicity=4
    )
    model = PET(hypers, _dataset_info())
    checkpoint = model.get_checkpoint()

    legacy = _to_max_atom_sampler_shape(copy.deepcopy(checkpoint))
    assert legacy["model_ckpt_version"] == 11
    assert "max_spin" in legacy["model_data"]["model_hypers"]
    assert "max_spin_multiplicity" not in legacy["model_data"]["model_hypers"]
    assert any(
        k.startswith("system_conditioning.spin_embedding.")
        for k in legacy["model_state_dict"]
    )

    upgraded = model_update_from_max_atom_sampler(legacy)

    assert upgraded["model_ckpt_version"] == PET.__checkpoint_version__
    new_hypers = upgraded["model_data"]["model_hypers"]
    assert new_hypers["max_spin_multiplicity"] == 4
    assert "max_spin" not in new_hypers
    assert new_hypers["adaptive_cutoff_method"] == "grid"
    assert new_hypers["max_charge"] == 3
    assert new_hypers["system_conditioning"] is True
    assert not any(
        k.startswith("system_conditioning.spin_embedding.")
        for k in upgraded["model_state_dict"]
    )
    assert any(
        k.startswith("system_conditioning.spin_multiplicity_embedding.")
        for k in upgraded["model_state_dict"]
    )

    # The upgraded checkpoint must load cleanly via the standard loader.
    PET.load_checkpoint(upgraded, "export")


def test_non_integer_charge_raises():
    """Non-integer charge or spin_multiplicity values raise a clear ValueError
    instead of being silently truncated by the long() cast."""
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    model.eval()

    outputs = {"energy": ModelOutput(per_atom=False)}

    system_bad_charge = _make_system(model, charge=0.5, spin_multiplicity=1)
    with pytest.raises(ValueError, match="charge must be an integer value"):
        model([system_bad_charge], outputs)

    system_bad_spin = _make_system(model, charge=0, spin_multiplicity=1.5)
    with pytest.raises(ValueError, match="spin_multiplicity must be an integer value"):
        model([system_bad_spin], outputs)


@pytest.mark.filterwarnings(
    "ignore:the 'features' output name is deprecated:UserWarning",
    "ignore:`per_atom` is deprecated:DeprecationWarning",
    "ignore:ModelOutput.quantity is deprecated:UserWarning",
    "ignore:Found metatomic.torch v.*vesin.metatomic was only tested:UserWarning",
)
def test_eval_model_end_to_end_with_extra_data(tmp_path, monkeypatch):
    """``mtt eval`` routes extra_data charge/spin to system conditioning.

    Runs the real eval path (``eval_model`` on an exported model, with an
    ``extra_data`` section in the options) on two xyz files differing only in
    the charge stored in ``atoms.info``, and checks the predicted energies
    differ. This covers the ``requested_inputs(use_new_names=True)`` gating in
    ``cli/eval.py``: if the exported model stopped declaring its requested
    inputs (or the gate filtered them out), both runs would silently fall back
    to charge=0 and predict identical energies.
    """
    monkeypatch.chdir(tmp_path)

    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    _train_steps(model)
    model.eval()

    exported = model.export()
    # The contract cli/eval.py relies on: the exported model still declares
    # its requested inputs.
    assert set(exported.requested_inputs(use_new_names=True).keys()) == {
        "charge",
        "spin_multiplicity",
    }

    for name, charge in [("neutral", 0), ("charged", 2)]:
        atoms = ase.Atoms("CH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
        atoms.info["charge"] = charge
        atoms.info["spin_multiplicity"] = 1
        ase.io.write(f"{name}.xyz", atoms)

        options = OmegaConf.create(
            {
                "systems": {"read_from": f"{name}.xyz", "reader": "ase"},
                "extra_data": {
                    "charge": {"key": "charge"},
                    "spin_multiplicity": {"key": "spin_multiplicity"},
                },
            }
        )
        # .mts output: the MetatensorWriter writes one file per predicted target
        eval_model(model=exported, options=options, output=f"out_{name}.mts")

    e_neutral = mts_load("out_neutral_energy.mts").block().values.item()
    e_charged = mts_load("out_charged_energy.mts").block().values.item()
    assert e_neutral != pytest.approx(e_charged), (
        "charge=0 and charge=2 gave identical energies through mtt eval — "
        "extra_data is not reaching the conditioning module"
    )


def test_trainer_wires_conditioning_transform(tmp_path, monkeypatch):
    """The PET trainer attaches charge/spin extra_data to systems during training.

    Trains briefly on systems that all carry charge=2 / spin_multiplicity=3 and
    checks that exactly the embedding rows for those values received updates,
    while the rows for the default values (charge=0, spin_multiplicity=1) are
    untouched. If the trainer did not wire ``get_system_data_transform`` into
    its collate functions, the defaults' rows would be the ones to change.
    """
    monkeypatch.chdir(tmp_path)

    max_charge = 4
    hypers = _small_hypers(max_charge=max_charge, max_spin_multiplicity=4)
    model = PET(hypers, _dataset_info())

    n_systems = 4
    systems = [_make_raw_system() for _ in range(n_systems)]
    # Sample labels carry the dataset index, like the readers produce them;
    # batching joins them and duplicate labels would fail.
    energies = [
        TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[float(i)]], dtype=torch.float64),
                    samples=Labels("system", torch.tensor([[i]])),
                    components=[],
                    properties=Labels("energy", torch.tensor([[0]])),
                )
            ],
        )
        for i in range(n_systems)
    ]
    dataset = Dataset.from_dict(
        {
            "system": systems,
            "energy": energies,
            "charge": [
                _make_extra_tmap(2.0, "charge", system_index=i)
                for i in range(n_systems)
            ],
            "spin_multiplicity": [
                _make_extra_tmap(3.0, "spin_multiplicity", system_index=i)
                for i in range(n_systems)
            ],
        }
    )

    train_hypers = copy.deepcopy(get_default_hypers("pet")["training"])
    train_hypers["num_epochs"] = 4
    train_hypers["num_workers"] = 0
    train_hypers["batch_size"] = 2
    train_hypers["atomic_baseline"] = {}
    loss_conf = OmegaConf.create({"energy": init_with_defaults(LossSpecification)})
    OmegaConf.resolve(loss_conf)
    train_hypers["loss"] = loss_conf

    charge_weights_before = (
        model.system_conditioning.charge_embedding.weight.detach().clone()
    )
    spin_weights_before = (
        model.system_conditioning.spin_multiplicity_embedding.weight.detach().clone()
    )

    trainer = Trainer(train_hypers)
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    charge_weights_after = model.system_conditioning.charge_embedding.weight.detach()
    spin_weights_after = (
        model.system_conditioning.spin_multiplicity_embedding.weight.detach()
    )

    # Rows for the values present in the dataset must have been updated ...
    assert not torch.equal(
        charge_weights_before[2 + max_charge], charge_weights_after[2 + max_charge]
    ), "charge=2 embedding row unchanged — charge did not reach the module"
    assert not torch.equal(spin_weights_before[3 - 1], spin_weights_after[3 - 1]), (
        "spin_multiplicity=3 embedding row unchanged — spin did not reach the module"
    )
    # ... while the rows for the default values must be untouched (plain Adam
    # leaves rows with zero gradient exactly as they were).
    assert torch.equal(
        charge_weights_before[0 + max_charge], charge_weights_after[0 + max_charge]
    ), "charge=0 (default) embedding row changed — systems fell back to defaults"
    assert torch.equal(spin_weights_before[1 - 1], spin_weights_after[1 - 1]), (
        "spin_multiplicity=1 (default) embedding row changed — "
        "systems fell back to defaults"
    )
