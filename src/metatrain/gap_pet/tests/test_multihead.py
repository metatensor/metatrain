"""Multi-head GapPET tests.

Cover the pieces that make multi-head finetuning work -- several gap targets
(e.g. two excitation levels) sharing one backbone:

1. ``test_multiple_targets_get_independent_heads``: each target owns its own
   HOMO/LUMO head pair and the predictions differ.
2. ``test_shared_backbone``: the backbone is shared, not duplicated per target.
3. ``test_only_requested_heads_are_evaluated``: asking for one target does not
   return the other.
4. ``test_restart_adds_new_target``: ``restart`` grows new heads (the finetune
   path) while keeping the old ones.
5. ``test_inherit_heads_expansion``: a target-level ``inherit_heads`` mapping
   expands to both internal heads and copies the weights.
6. ``test_upgrade_v1_checkpoint``: a single-head v1 checkpoint loads into the
   multi-head model.
"""

import ase.build
import pytest
import torch
from metatomic.torch import ModelOutput, systems_to_torch

from metatrain.gap_pet.documentation import ModelHypers
from metatrain.gap_pet.model import (
    HOMO_HEAD_PREFIX,
    LUMO_HEAD_PREFIX,
    GapPET,
    homo_per_atom_output_name,
    lumo_per_atom_output_name,
)
from metatrain.gap_pet.trainer import Trainer
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


DTYPE = torch.float64
S1_TARGET = "mtt::gap_s1"
S2_TARGET = "mtt::gap_s2"


def _minimal_hypers() -> dict:
    h = init_with_defaults(ModelHypers)
    h["d_pet"] = 4
    h["d_node"] = 4
    h["d_head"] = 4
    h["d_feedforward"] = 4
    h["num_heads"] = 1
    h["num_attention_layers"] = 1
    h["num_gnn_layers"] = 2
    h["cutoff"] = 4.0
    h["cutoff_width"] = 0.5
    h["zbl"] = False
    return h


def _target_info():
    return get_energy_target_info(
        "__gap__",
        {"quantity": "energy", "unit": "eV"},
        add_position_gradients=True,
    )


def _dataset_info(target_names, atomic_types):
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=sorted(set(atomic_types)),
        targets={name: _target_info() for name in target_names},
    )


def _build_water():
    atoms = ase.build.molecule("H2O", vacuum=0.0)
    atoms.set_cell([4.0, 4.0, 4.0])
    atoms.center()
    atoms.set_pbc(True)
    systems = systems_to_torch(atoms, dtype=DTYPE)
    return systems if isinstance(systems, list) else [systems]


def _make_model(target_names, atomic_types, seed=0):
    torch.manual_seed(seed)
    model = GapPET(_minimal_hypers(), _dataset_info(target_names, atomic_types))
    return model.to(DTYPE).eval()


def _attach_nl(model, systems):
    return [
        get_system_with_neighbor_lists(s, model.requested_neighbor_lists())
        for s in systems
    ]


def test_multiple_targets_get_independent_heads():
    """Two excitation levels must each own a HOMO/LUMO head pair, and -- being
    independently initialized -- must not predict the same gap."""
    systems = _build_water()
    model = _make_model([S1_TARGET, S2_TARGET], systems[0].types.tolist())
    systems = _attach_nl(model, systems)

    for target in (S1_TARGET, S2_TARGET):
        assert HOMO_HEAD_PREFIX + target in model.node_heads
        assert LUMO_HEAD_PREFIX + target in model.node_heads
        # The user-visible target is intensive, not per-atom.
        assert model.outputs[target].per_atom is False
        assert homo_per_atom_output_name(target) in model.outputs
        assert lumo_per_atom_output_name(target) in model.outputs

    outputs = {
        S1_TARGET: ModelOutput(per_atom=False),
        S2_TARGET: ModelOutput(per_atom=False),
    }
    with torch.no_grad():
        out = model(systems, outputs)

    assert set(out.keys()) == {S1_TARGET, S2_TARGET}
    e_s1 = out[S1_TARGET].block().values
    e_s2 = out[S2_TARGET].block().values
    assert e_s1.shape == (1, 1)
    assert not torch.allclose(e_s1, e_s2)


def test_shared_backbone():
    """Adding a second target must add head parameters only -- the GNN backbone
    is shared across excitation levels, which is the point of multi-head."""
    types = _build_water()[0].types.tolist()
    one = _make_model([S1_TARGET], types)
    two = _make_model([S1_TARGET, S2_TARGET], types)

    def backbone_params(model):
        return {
            name: p.numel()
            for name, p in model.named_parameters()
            if not name.startswith(
                ("node_heads.", "edge_heads.", "node_last_layers.", "edge_last_layers.")
            )
        }

    assert backbone_params(one) == backbone_params(two)

    def head_params(model):
        return sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith(
                ("node_heads.", "edge_heads.", "node_last_layers.", "edge_last_layers.")
            )
        )

    # Exactly one extra head pair's worth of parameters.
    assert head_params(two) == 2 * head_params(one)


def test_only_requested_heads_are_evaluated():
    """Requesting a single target must not compute or return the other one."""
    systems = _build_water()
    model = _make_model([S1_TARGET, S2_TARGET], systems[0].types.tolist())
    systems = _attach_nl(model, systems)

    with torch.no_grad():
        out = model(systems, {S1_TARGET: ModelOutput(per_atom=False)})
    assert set(out.keys()) == {S1_TARGET}

    # A per-atom auxiliary alone is enough to trigger its target's heads.
    with torch.no_grad():
        out = model(
            systems, {lumo_per_atom_output_name(S2_TARGET): ModelOutput(per_atom=True)}
        )
    assert set(out.keys()) == {lumo_per_atom_output_name(S2_TARGET)}


def test_restart_adds_new_target():
    """``restart`` is the finetune path: it must grow heads for unseen targets
    while leaving the existing target's heads in place."""
    types = _build_water()[0].types.tolist()
    model = _make_model([S1_TARGET], types)

    old_homo = model.node_heads[HOMO_HEAD_PREFIX + S1_TARGET][0][0].weight.clone()

    model.restart(_dataset_info([S1_TARGET, S2_TARGET], types))
    # Heads added by ``restart`` are built in the default dtype; the trainer
    # casts the whole model afterwards (pet/trainer.py), so mirror that here.
    model.to(DTYPE)

    assert model.has_new_targets
    assert set(model._gap_target_names) == {S1_TARGET, S2_TARGET}
    assert HOMO_HEAD_PREFIX + S2_TARGET in model.node_heads
    # the pre-existing head must be untouched
    torch.testing.assert_close(
        model.node_heads[HOMO_HEAD_PREFIX + S1_TARGET][0][0].weight, old_homo
    )

    systems = _attach_nl(model, _build_water())
    with torch.no_grad():
        out = model(systems.copy(), {S2_TARGET: ModelOutput(per_atom=False)})
    assert S2_TARGET in out


def test_restart_rejects_new_atomic_types():
    types = _build_water()[0].types.tolist()
    model = _make_model([S1_TARGET], types)
    with pytest.raises(ValueError, match="New atomic types"):
        model.restart(_dataset_info([S1_TARGET], list(types) + [79]))


def test_inherit_heads_expansion():
    """A target-level ``inherit_heads`` entry must expand to both internal heads
    and actually copy the source weights into the destination."""
    types = _build_water()[0].types.tolist()
    model = _make_model([S1_TARGET], types)
    model.restart(_dataset_info([S1_TARGET, S2_TARGET], types))
    model.to(DTYPE)

    expanded = Trainer._expand_inherit_heads(
        model, {"inherit_heads": {S2_TARGET: S1_TARGET}}
    )
    assert expanded["inherit_heads"] == {
        HOMO_HEAD_PREFIX + S2_TARGET: HOMO_HEAD_PREFIX + S1_TARGET,
        LUMO_HEAD_PREFIX + S2_TARGET: LUMO_HEAD_PREFIX + S1_TARGET,
    }

    # The expanded mapping must be consumable by PET's finetuning machinery.
    from metatrain.pet.modules.finetuning import apply_finetuning_strategy

    strategy = {"method": "full", "read_from": "unused", **expanded}
    apply_finetuning_strategy(model, strategy)

    torch.testing.assert_close(
        model.node_heads[HOMO_HEAD_PREFIX + S2_TARGET][0][0].weight,
        model.node_heads[HOMO_HEAD_PREFIX + S1_TARGET][0][0].weight,
    )

    # ...including the per-layer linear last layers, whose ModuleDict key is
    # derived from the head name (single scalar block, so a single key).
    def only_last_layer(target):
        module_dict = model.edge_last_layers[LUMO_HEAD_PREFIX + target][0]
        (key,) = module_dict.keys()
        return module_dict[key]

    torch.testing.assert_close(
        only_last_layer(S2_TARGET).weight, only_last_layer(S1_TARGET).weight
    )


def test_inherit_heads_unknown_target_raises():
    types = _build_water()[0].types.tolist()
    model = _make_model([S1_TARGET], types)
    with pytest.raises(ValueError, match="not a gap target"):
        Trainer._expand_inherit_heads(
            model, {"inherit_heads": {S1_TARGET: "mtt::does_not_exist"}}
        )


def test_upgrade_v1_checkpoint():
    """An existing single-head (v1) checkpoint must load into the multi-head
    model, with its heads rebound to the checkpoint's own target."""
    types = _build_water()[0].types.tolist()
    model = _make_model([S1_TARGET], types)
    checkpoint = model.get_checkpoint()

    # Fake a v1 checkpoint by renaming the heads back to the old, target-agnostic
    # internal keys.
    def to_v1(state_dict):
        out = {}
        for name, value in state_dict.items():
            name = name.replace(
                HOMO_HEAD_PREFIX + S1_TARGET, "__gap_pet_homo_internal__"
            )
            name = name.replace(
                LUMO_HEAD_PREFIX + S1_TARGET, "__gap_pet_lumo_internal__"
            )
            out[name] = value
        return out

    checkpoint["model_ckpt_version"] = 1
    checkpoint["model_state_dict"] = to_v1(checkpoint["model_state_dict"])
    checkpoint["best_model_state_dict"] = to_v1(checkpoint["best_model_state_dict"])
    assert any("__gap_pet_homo_internal__" in k for k in checkpoint["model_state_dict"])

    upgraded = GapPET.upgrade_checkpoint(checkpoint)
    assert upgraded["model_ckpt_version"] == GapPET.__checkpoint_version__

    restored = GapPET.load_checkpoint(upgraded, context="finetune")
    torch.testing.assert_close(
        restored.node_heads[HOMO_HEAD_PREFIX + S1_TARGET][0][0].weight.to(DTYPE),
        model.node_heads[HOMO_HEAD_PREFIX + S1_TARGET][0][0].weight,
    )

    # ...and the restored model must then be growable to a second head.
    restored.restart(_dataset_info([S1_TARGET, S2_TARGET], types))
    assert HOMO_HEAD_PREFIX + S2_TARGET in restored.node_heads
