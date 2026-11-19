"""Numerical tests for the learnable per-atom sharpness of the intensive gap
extender."""

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.extenders.helpers import get_default_hypers, load_extender
from metatrain.utils.extenders.intensive_gap.extender import _scatter_softmax_pool


N_ATOMS = [3, 5, 1, 7]

NAMES = {
    "bottom": "mtt::extender_aux::intensive_gap::bottom::gap",
    "top": "mtt::extender_aux::intensive_gap::top::gap",
    "log_sharpness_bottom": (
        "mtt::extender_aux::intensive_gap::log_sharpness_bottom::gap"
    ),
    "log_sharpness_top": "mtt::extender_aux::intensive_gap::log_sharpness_top::gap",
}


def _make_extender(
    learnable_sharpness, sharpness_bottom: float = 10.0, sharpness_top: float = -10.0
):
    gap = get_generic_target_info(
        "mtt::gap",
        dict(
            sample_kind="system",
            unit="eV",
            quantity="energy",
            num_subtargets=1,
            type="scalar",
        ),
    )
    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=[1, 6], targets={"mtt::gap": gap}
    )
    hypers = {
        **get_default_hypers("intensive_gap"),
        "outputs": "mtt::gap",
        "sharpness_bottom": sharpness_bottom,
        "sharpness_top": sharpness_top,
    }
    if learnable_sharpness is None:
        hypers.pop("learnable_sharpness", None)
    else:
        hypers["learnable_sharpness"] = learnable_sharpness
    return load_extender("intensive_gap")(hypers, dataset_info)


def _systems() -> list[System]:
    return [
        System(
            types=torch.ones(n, dtype=torch.int32),
            positions=torch.zeros((n, 3), dtype=torch.float64),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        for n in N_ATOMS
    ]


def _system_indices() -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(len(N_ATOMS)), torch.tensor(N_ATOMS))


def _per_atom(values: torch.Tensor) -> TensorMap:
    samples = torch.stack(
        [_system_indices(), torch.cat([torch.arange(n) for n in N_ATOMS])], dim=1
    )
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=values.reshape(-1, 1),
                samples=Labels(["system", "atom"], samples.to(torch.int32)),
                components=[],
                properties=Labels.range("energy", 1),
            )
        ],
    )


def _random_values(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    n = sum(N_ATOMS)

    def randn() -> torch.Tensor:
        return torch.randn(n, generator=generator, dtype=torch.float64)

    return {
        "bottom": randn() - 6.0,
        "top": randn() - 1.0,
        "log_sharpness_bottom": randn(),
        "log_sharpness_top": randn(),
    }


def _gap(extender, values: dict[str, torch.Tensor]) -> torch.Tensor:
    inputs = {NAMES[key]: _per_atom(value) for key, value in values.items()}
    outputs = {"mtt::gap": ModelOutput(sample_kind="system")}
    result = extender(_systems(), outputs, inputs)
    return result["mtt::gap"].block().values.ravel()


def test_requested_inputs():
    """Fixed sharpness requests 2 inputs, learnable sharpness 4 (u is unitless)."""
    outputs = {"mtt::gap": ModelOutput(sample_kind="system")}
    fixed_names = {NAMES["bottom"], NAMES["top"]}

    fixed = _make_extender(learnable_sharpness=False)
    assert set(fixed.requested_extender_inputs(outputs)) == fixed_names
    assert set(fixed.requested_extender_inputs()) == fixed_names

    learnable = _make_extender(learnable_sharpness=True)
    requested = learnable.requested_extender_inputs(outputs)
    assert set(requested) == set(NAMES.values())
    assert set(learnable.requested_extender_inputs()) == set(NAMES.values())
    for key in ["log_sharpness_bottom", "log_sharpness_top"]:
        assert requested[NAMES[key]].sample_kind == "atom"
        assert requested[NAMES[key]].unit == ""
        assert requested[NAMES[key]].quantity == ""

    assert learnable.requested_extender_inputs({}) == {}


def test_hypers_without_learnable_sharpness():
    """Extenders saved before the option existed keep the fixed sharpness."""
    values = _random_values(seed=7)
    fixed_values = {"bottom": values["bottom"], "top": values["top"]}
    old = _make_extender(learnable_sharpness=None)
    assert set(old.requested_extender_inputs()) == {NAMES["bottom"], NAMES["top"]}
    torch.testing.assert_close(
        _gap(old, fixed_values),
        _gap(_make_extender(learnable_sharpness=False), fixed_values),
        rtol=0,
        atol=0,
    )


def test_u_zero_matches_fixed():
    """With u = 0 the learnable pooling reduces to the fixed-sharpness pooling."""
    values = _random_values(seed=0)
    values["log_sharpness_bottom"] = torch.zeros_like(values["log_sharpness_bottom"])
    values["log_sharpness_top"] = torch.zeros_like(values["log_sharpness_top"])

    fixed_values = {"bottom": values["bottom"], "top": values["top"]}
    gap_fixed = _gap(_make_extender(learnable_sharpness=False), fixed_values)
    gap_learnable = _gap(_make_extender(learnable_sharpness=True), values)

    torch.testing.assert_close(gap_learnable, gap_fixed, rtol=1e-12, atol=1e-12)


def test_not_shift_invariant():
    """A constant shift of all values changes the gap."""
    values = _random_values(seed=1)
    extender = _make_extender(learnable_sharpness=True)

    shifted = dict(values)
    shifted["bottom"] = values["bottom"] + 3.7
    shifted["top"] = values["top"] + 3.7
    difference = (_gap(extender, shifted) - _gap(extender, values)).abs()
    assert difference.max() > 1e-3


def test_intensive():
    """Duplicating all atoms of a system leaves the pooled value unchanged."""
    values = _random_values(seed=2)["bottom"][:5]
    alphas = 10.0 * torch.exp(_random_values(seed=3)["log_sharpness_bottom"][:5])

    pooled = _scatter_softmax_pool(values, alphas, torch.zeros(5, dtype=torch.long), 1)
    pooled_twice = _scatter_softmax_pool(
        torch.cat([values, values]),
        torch.cat([alphas, alphas]),
        torch.zeros(10, dtype=torch.long),
        1,
    )
    torch.testing.assert_close(pooled_twice, pooled)


@pytest.mark.parametrize("maximum", [True, False])
def test_hard_limit(maximum):
    """Very sharp pooling with a uniform sharpness recovers the hard per-system
    max/min."""
    values = _random_values(seed=4)
    sign = 1.0 if maximum else -1.0
    alphas = sign * 1e5 * torch.ones_like(values["bottom"])
    system_indices = _system_indices()

    pooled = _scatter_softmax_pool(
        values["bottom"], alphas, system_indices, len(N_ATOMS)
    )

    reduce = "amax" if maximum else "amin"
    expected = torch.zeros(len(N_ATOMS), dtype=torch.float64).scatter_reduce(
        0, system_indices, values["bottom"], reduce=reduce, include_self=False
    )
    torch.testing.assert_close(pooled, expected, atol=1e-4, rtol=0)


def test_gradients():
    """Gradients flow to both the values and the log-sharpness u."""
    values = {
        key: value.requires_grad_(True) for key, value in _random_values(seed=5).items()
    }
    gap = _gap(_make_extender(learnable_sharpness=True), values)
    gap.sum().backward()

    for key, value in values.items():
        assert value.grad is not None, key
        assert torch.all(torch.isfinite(value.grad)), key
        assert torch.any(value.grad != 0), key


def test_torchscript_matches_eager():
    extender = _make_extender(learnable_sharpness=True)
    scripted = torch.jit.script(extender)
    values = _random_values(seed=6)
    torch.testing.assert_close(_gap(scripted, values), _gap(extender, values))


@pytest.mark.parametrize("name", ["sharpness_bottom", "sharpness_top"])
def test_learnable_zero_sharpness_raises(name):
    with pytest.raises(ValueError, match=f"non-zero '{name}'"):
        _make_extender(learnable_sharpness=True, **{name: 0.0})
