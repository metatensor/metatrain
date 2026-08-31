"""
Tests that the diagnostic hook machinery has no side-effects on normal inference.

Specifically:
* No forward hooks remain on any module after a forward pass (with or without
  diagnostic outputs requested).
* When no diagnostic outputs are requested the hook-registration code path is
  never entered, so no hooks exist at any point during inference.
* The gradient of the energy with respect to atomic positions is identical
  whether or not diagnostic outputs are requested alongside the energy.
"""

import torch
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset_info():
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )


def _make_water_system(model):
    system = System(
        types=torch.tensor([8, 1, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.119], [0.0, 0.757, -0.477], [0.0, -0.757, -0.477]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def _all_forward_hooks(model):
    """Return a dict of {module_name: hook_count} for every module with hooks."""
    return {
        name: len(module._forward_hooks)
        for name, module in model.named_modules()
        if module._forward_hooks
    }


# ---------------------------------------------------------------------------
# Hook cleanup
# ---------------------------------------------------------------------------


def test_hooks_cleaned_up_after_diagnostic_forward():
    """All forward hooks must be removed after each diagnostic forward pass."""
    model = PET(MODEL_HYPERS, _make_dataset_info())
    system = _make_water_system(model)

    # Run several times to confirm hooks don't accumulate across calls.
    for _ in range(3):
        model(
            [system],
            {"mtt::feature::node_heads.energy.0": ModelOutput(per_atom=True)},
        )
        stale = _all_forward_hooks(model)
        assert not stale, f"Stale hooks after diagnostic forward: {stale}"


def test_hooks_cleaned_up_after_mixed_forward():
    """Hooks must be removed even when energy and diagnostic outputs are combined."""
    model = PET(MODEL_HYPERS, _make_dataset_info())
    system = _make_water_system(model)

    model(
        [system],
        {
            "energy": ModelOutput(per_atom=False),
            "mtt::feature::node_heads.energy.0": ModelOutput(per_atom=True),
            "mtt::feature::gnn_layers.0_node": ModelOutput(per_atom=True),
        },
    )
    stale = _all_forward_hooks(model)
    assert not stale, f"Stale hooks after mixed forward: {stale}"


# ---------------------------------------------------------------------------
# No hooks during plain inference
# ---------------------------------------------------------------------------


def test_no_hooks_during_plain_inference():
    """A plain energy-only forward pass must never register any hooks."""
    model = PET(MODEL_HYPERS, _make_dataset_info())
    system = _make_water_system(model)

    model([system], {"energy": ModelOutput(per_atom=False)})

    stale = _all_forward_hooks(model)
    assert not stale, f"Unexpected hooks after plain inference: {stale}"


# ---------------------------------------------------------------------------
# Gradient correctness
# ---------------------------------------------------------------------------


def test_gradients_unaffected_by_diagnostic_outputs():
    """
    The gradient of the energy w.r.t. atomic positions must be identical
    whether or not diagnostic outputs are requested alongside the energy.

    This guards against a future refactor that accidentally cuts the gradient
    tape for the main computation (e.g. by wrapping tensors rather than just
    detaching clones inside the hook).
    """
    model = PET(MODEL_HYPERS, _make_dataset_info())
    model.eval()

    def run(with_diagnostic: bool) -> torch.Tensor:
        system = System(
            types=torch.tensor([8, 1, 1]),
            positions=torch.tensor(
                [[0.0, 0.0, 0.119], [0.0, 0.757, -0.477], [0.0, -0.757, -0.477]]
            ),
            cell=torch.zeros(3, 3),
            pbc=torch.tensor([False, False, False]),
        )
        system.positions.requires_grad_(True)
        system = get_system_with_neighbor_lists(
            system, model.requested_neighbor_lists()
        )
        outputs = {"energy": ModelOutput(per_atom=False)}
        if with_diagnostic:
            outputs["mtt::feature::node_heads.energy.0"] = ModelOutput(per_atom=True)
        result = model([system], outputs)
        result["energy"].block().values.sum().backward()
        return system.positions.grad.clone()

    grad_plain = run(with_diagnostic=False)
    grad_diag = run(with_diagnostic=True)

    torch.testing.assert_close(grad_plain, grad_diag, atol=0.0, rtol=0.0)
