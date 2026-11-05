"""Numerical tests for the learnable per-atom sharpness of the intensive gap hook."""

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.hooks.helpers import load_hook
from metatrain.utils.hooks.intensive_gap.hook import IntensiveGap, _scatter_softmax_pool


N_ATOMS = [3, 5, 1, 7]

NAMES = {
    "bottom": "mtt::aux::gap_bottom::gap",
    "top": "mtt::aux::gap_top::gap",
    "alpha_bottom": "mtt::aux::gap_bottom_logalpha::gap",
    "alpha_top": "mtt::aux::gap_top_logalpha::gap",
}


def _make_hook(
    learnable_alpha: bool,
    alpha_bottom: float = 10.0,
    alpha_top: float = -10.0,
):
    # A fresh dataset info for every hook, since the hook stores its parameters
    # in the layout of the output target.
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
        "outputs": "mtt::gap",
        "alpha_bottom": alpha_bottom,
        "alpha_top": alpha_top,
        "learnable_alpha": learnable_alpha,
    }
    return load_hook("intensive_gap")(hypers, dataset_info)


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
        "alpha_bottom": randn(),
        "alpha_top": randn(),
    }


def _gap(hook, values: dict[str, torch.Tensor]) -> torch.Tensor:
    inputs = {NAMES[key]: _per_atom(value) for key, value in values.items()}
    outputs = {"mtt::gap": ModelOutput(sample_kind="system")}
    result = hook(_systems(), outputs, inputs)
    return result["mtt::gap"].block().values.ravel()


def test_requested_inputs():
    """Fixed sharpness requests 2 inputs, learnable sharpness 4 (u is unitless)."""
    outputs = {"mtt::gap": ModelOutput(sample_kind="system")}

    fixed = _make_hook(learnable_alpha=False)
    assert set(fixed.requested_hook_inputs(outputs)) == {NAMES["bottom"], NAMES["top"]}
    assert set(fixed.requested_target_infos()) == {NAMES["bottom"], NAMES["top"]}

    learnable = _make_hook(learnable_alpha=True)
    requested = learnable.requested_hook_inputs(outputs)
    assert set(requested) == set(NAMES.values())
    assert set(learnable.requested_target_infos()) == set(NAMES.values())
    for key in ["alpha_bottom", "alpha_top"]:
        assert requested[NAMES[key]].sample_kind == "atom"
        assert requested[NAMES[key]].unit == ""
        target_info = learnable.requested_target_infos()[NAMES[key]]
        assert target_info.sample_kind == "atom"
        assert target_info.unit == ""

    # Nothing is requested if the hook's output is not requested.
    assert learnable.requested_hook_inputs({}) == {}


def test_u_zero_matches_fixed():
    """With u = 0 the learnable pooling reduces to the fixed-sharpness pooling."""
    values = _random_values(seed=0)
    values["alpha_bottom"] = torch.zeros_like(values["alpha_bottom"])
    values["alpha_top"] = torch.zeros_like(values["alpha_top"])

    fixed_values = {"bottom": values["bottom"], "top": values["top"]}
    gap_fixed = _gap(_make_hook(learnable_alpha=False), fixed_values)
    gap_learnable = _gap(_make_hook(learnable_alpha=True), values)

    torch.testing.assert_close(gap_learnable, gap_fixed, rtol=1e-12, atol=1e-12)


def test_not_shift_invariant():
    """A constant shift of all values changes the gap."""
    values = _random_values(seed=1)
    hook = _make_hook(learnable_alpha=True)

    shifted = dict(values)
    shifted["bottom"] = values["bottom"] + 3.7
    shifted["top"] = values["top"] + 3.7
    difference = (_gap(hook, shifted) - _gap(hook, values)).abs()
    assert difference.max() > 1e-3


def test_intensive():
    """Duplicating all atoms of a system leaves the pooled value unchanged."""
    values = _random_values(seed=2)["bottom"][:5]
    alphas = 10.0 * torch.exp(_random_values(seed=3)["alpha_bottom"][:5])

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
    gap = _gap(_make_hook(learnable_alpha=True), values)
    gap.sum().backward()

    for key, value in values.items():
        assert value.grad is not None, key
        assert torch.all(torch.isfinite(value.grad)), key
        assert torch.any(value.grad != 0), key


def test_torchscript_matches_eager():
    hook = _make_hook(learnable_alpha=True)
    scripted = torch.jit.script(hook)
    values = _random_values(seed=6)
    torch.testing.assert_close(_gap(scripted, values), _gap(hook, values))


def test_pooling_section_raises():
    """The old nested ``pooling`` hypers are rejected instead of silently ignored."""
    hook = _make_hook(learnable_alpha=False)
    hypers = {"outputs": "mtt::gap", "pooling": {"alpha_bottom": 5.0}}
    with pytest.raises(ValueError, match="'pooling' section"):
        IntensiveGap(hypers, hook.dataset_info)


def _v1_checkpoint(learnable_alpha: bool, pooling_type: str = "softmax") -> dict:
    """A version 1 checkpoint, with the pooling hypers in a ``pooling`` section."""
    hook = _make_hook(learnable_alpha=learnable_alpha, alpha_bottom=3.0, alpha_top=-7.0)
    checkpoint = hook.get_checkpoint()
    hypers = checkpoint["hook_data"]["hypers"]
    checkpoint["hook_data"]["hypers"] = {
        "outputs": hypers["outputs"],
        "pooling": {
            "type": pooling_type,
            "alpha_bottom": hypers["alpha_bottom"],
            "alpha_top": hypers["alpha_top"],
            "learnable_alpha": hypers["learnable_alpha"],
        },
    }
    checkpoint["hook_ckpt_version"] = 1
    return checkpoint


@pytest.mark.parametrize("learnable_alpha", [False, True])
def test_upgrade_v1_checkpoint(learnable_alpha):
    """A version 1 checkpoint loads with the same sharpness and gives the same gap."""
    reference = _make_hook(
        learnable_alpha=learnable_alpha, alpha_bottom=3.0, alpha_top=-7.0
    )
    hook = IntensiveGap.load_checkpoint(_v1_checkpoint(learnable_alpha), "restart")

    assert hook.hypers["alpha_bottom"] == 3.0
    assert hook.hypers["alpha_top"] == -7.0
    assert hook.hypers["learnable_alpha"] == learnable_alpha
    assert "pooling" not in hook.hypers
    assert hook.get_checkpoint()["hook_ckpt_version"] == 2

    values = _random_values(seed=7)
    if not learnable_alpha:
        values = {"bottom": values["bottom"], "top": values["top"]}
    torch.testing.assert_close(_gap(hook, values), _gap(reference, values))


def test_default_alphas():
    """The default sharpness is +-10, and it is stored in the hook's hypers."""
    hook = _make_hook(learnable_alpha=False)
    hook = IntensiveGap({"outputs": "mtt::gap"}, hook.dataset_info)
    assert hook.hypers["alpha_bottom"] == 10.0
    assert hook.hypers["alpha_top"] == -10.0
    assert hook.get_checkpoint()["hook_data"]["hypers"]["alpha_bottom"] == 10.0


def test_upgrade_v1_checkpoint_default_alphas():
    """A version 1 checkpoint without alphas keeps the version 1 defaults (+-20)."""
    checkpoint = _v1_checkpoint(learnable_alpha=False)
    del checkpoint["hook_data"]["hypers"]["pooling"]["alpha_bottom"]
    del checkpoint["hook_data"]["hypers"]["pooling"]["alpha_top"]
    hook = IntensiveGap.load_checkpoint(checkpoint, "restart")
    assert hook.hypers["alpha_bottom"] == 20.0
    assert hook.hypers["alpha_top"] == -20.0


def test_upgrade_v1_smoothmax_checkpoint_raises():
    checkpoint = _v1_checkpoint(learnable_alpha=False, pooling_type="smoothmax")
    with pytest.raises(ValueError, match="only the softmax pooling"):
        IntensiveGap.load_checkpoint(checkpoint, "restart")


@pytest.mark.parametrize("alpha_name", ["alpha_bottom", "alpha_top"])
def test_learnable_zero_alpha_raises(alpha_name):
    with pytest.raises(ValueError, match=f"non-zero '{alpha_name}'"):
        _make_hook(learnable_alpha=True, **{alpha_name: 0.0})
