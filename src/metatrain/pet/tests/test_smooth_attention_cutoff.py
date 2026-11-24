"""Tests for the ``smooth_attention_cutoff`` hyperparameter of PET.

With a smooth attention cutoff, the cutoff factor of a neighbor does not damp the
attention of its own token to itself (the diagonal of the query-key cutoff matrix is
one), and the attention received by a neighbor in the tail of the cutoff function is
capped, so that it grows in proportion to its cutoff factor.
"""

import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.pet.checkpoints import model_update_v18_v19
from metatrain.pet.modules.transformer import (
    AttentionBlock,
    CartesianTransformer,
    smooth_cutoff_attention,
)
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


CUTOFF = 4.0
CUTOFF_WIDTH = 1.0


def _hypers(smooth_attention_cutoff, num_neighbors_adaptive=None):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["cutoff"] = CUTOFF
    hypers["cutoff_width"] = CUTOFF_WIDTH
    hypers["num_neighbors_adaptive"] = num_neighbors_adaptive
    hypers["smooth_attention_cutoff"] = smooth_attention_cutoff
    return hypers


def _make_model(smooth_attention_cutoff, num_neighbors_adaptive=None):
    torch.manual_seed(0)
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    hypers = _hypers(smooth_attention_cutoff, num_neighbors_adaptive)
    return PET(hypers, dataset_info).to(torch.float64).eval()


def _make_pair_of_models(num_neighbors_adaptive=None):
    """Two models with the same weights, differing only by the flag."""
    legacy = _make_model(False, num_neighbors_adaptive)
    smooth = _make_model(True, num_neighbors_adaptive)
    smooth.load_state_dict(legacy.state_dict())
    return legacy, smooth


def _energy(model, positions):
    system = System(
        types=torch.full((len(positions),), 6),
        positions=torch.tensor(positions, dtype=torch.float64),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    with torch.no_grad():
        outputs = model([system], {"energy": ModelOutput(per_atom=False)})
    return float(outputs["energy"].block().values.squeeze())


# a compact cluster: all the distances are below CUTOFF - CUTOFF_WIDTH
COMPACT = [
    [0.0, 0.0, 0.0],
    [1.4, 0.0, 0.0],
    [0.0, 1.5, 0.0],
    [0.0, 0.0, 1.3],
    [1.2, 1.3, 0.2],
]


def _with_far_atom(distance):
    return COMPACT + [[-distance, 0.0, 0.0]]


def _captured_factors(transformer, cutoff_factors, padding_mask):
    """Run a ``CartesianTransformer`` and return the query-key cutoff matrix that
    reaches its attention layers."""
    captured = {}

    def hook(module, args, kwargs):
        captured["factors"] = kwargs["cutoff_factors"].detach().clone()

    handle = transformer.trans.register_forward_pre_hook(hook, with_kwargs=True)
    n_nodes, n_neighbors = cutoff_factors.shape
    d_model = transformer.edge_embedder.out_features
    torch.manual_seed(1)
    edge_vectors = torch.randn(n_nodes, n_neighbors, 3)
    transformer(
        torch.randn(n_nodes, d_model),
        torch.randn(n_nodes, n_neighbors, d_model),
        torch.zeros(n_nodes, n_neighbors, dtype=torch.long),
        edge_vectors,
        padding_mask,
        edge_vectors.norm(dim=-1),
        cutoff_factors,
        use_manual_attention=False,
    )
    handle.remove()
    return captured["factors"]


def _transformer(smooth_attention_cutoff):
    torch.manual_seed(0)
    return CartesianTransformer(
        cutoff=CUTOFF,
        cutoff_width=CUTOFF_WIDTH,
        d_model=8,
        n_head=2,
        dim_node_features=8,
        dim_feedforward=16,
        n_layers=1,
        norm="RMSNorm",
        activation="SwiGLU",
        attention_temperature=1.0,
        transformer_type="PreLN",
        n_atomic_species=1,
        is_first=True,
        smooth_attention_cutoff=smooth_attention_cutoff,
    ).eval()


def test_self_key_has_no_cutoff_bias():
    # two nodes, three neighbor slots; the last slot of the second node is padding
    cutoff_factors = torch.tensor([[1.0, 0.3, 1e-4], [0.7, 0.02, 0.0]])
    padding_mask = torch.tensor([[True, True, True], [True, True, False]])

    legacy = _captured_factors(_transformer(False), cutoff_factors, padding_mask)
    fixed = _captured_factors(_transformer(True), cutoff_factors, padding_mask)

    # legacy: every query sees the same key factors (centre token first)
    expected = torch.tensor([[1.0, 1.0, 0.3, 1e-4], [1.0, 0.7, 0.02, 0.0]])
    assert torch.equal(legacy, expected[:, None, :].expand(-1, 4, -1))

    diagonal = torch.diagonal(fixed, dim1=1, dim2=2)
    # real tokens attend to themselves at full weight, padded tokens stay excluded
    assert torch.equal(diagonal, torch.tensor([[1.0] * 4, [1.0, 1.0, 1.0, 0.0]]))

    # nothing else changes
    off_diagonal = ~torch.eye(4, dtype=torch.bool)[None].expand(2, -1, -1)
    assert torch.equal(fixed[off_diagonal], legacy[off_diagonal])


def test_default_is_smooth():
    assert MODEL_HYPERS["smooth_attention_cutoff"] is True


def test_no_effect_without_neighbors_in_the_tail():
    legacy, smooth = _make_pair_of_models()
    # not bit-identical, since the two settings use different attention kernels
    assert _energy(legacy, COMPACT) == pytest.approx(_energy(smooth, COMPACT), 1e-12)


def test_changes_the_model_with_a_neighbor_in_the_tail():
    legacy, smooth = _make_pair_of_models()
    positions = _with_far_atom(CUTOFF - 0.4 * CUTOFF_WIDTH)
    assert abs(_energy(legacy, positions) - _energy(smooth, positions)) > 1e-8


@pytest.mark.parametrize("num_neighbors_adaptive", [None, 4.0])
def test_energy_is_continuous_when_an_atom_enters_the_cutoff(num_neighbors_adaptive):
    # A token that just entered the cutoff attends to itself at full weight although
    # its cutoff factor is zero. This must not create a jump in the energy, since
    # nothing else can attend to it yet.
    model = _make_model(True, num_neighbors_adaptive)

    if num_neighbors_adaptive is None:
        boundary = CUTOFF
    else:
        # the far atom drops out when it crosses the pair cutoff, which is the mean
        # of the two adaptive atomic cutoffs: locate it by bisection on the energy
        outside = _energy(model, COMPACT + [[-50.0, 0.0, 0.0]])
        lower, upper = 1.5, CUTOFF + 0.5
        for _ in range(40):
            middle = 0.5 * (lower + upper)
            if abs(_energy(model, _with_far_atom(middle)) - outside) < 1e-12:
                upper = middle
            else:
                lower = middle
        boundary = upper

    inside = _energy(model, _with_far_atom(boundary - 1e-3))
    outside = _energy(model, _with_far_atom(boundary + 1e-3))
    well_inside = _energy(model, _with_far_atom(boundary - 0.5 * CUTOFF_WIDTH))

    # the atom does matter when it is well inside the cutoff...
    assert abs(well_inside - outside) > 1e-6
    # ...and its contribution vanishes at the boundary
    assert abs(inside - outside) < 1e-10


def test_torchscript():
    model = _make_model(True)
    scripted = torch.jit.script(model)

    system = System(
        types=torch.full((len(COMPACT) + 1,), 6),
        positions=torch.tensor(_with_far_atom(3.5), dtype=torch.float64),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(per_atom=False)}

    reference = model([system], outputs)["energy"].block().values
    result = scripted([system], outputs)["energy"].block().values
    assert torch.allclose(result, reference, atol=1e-12, rtol=1e-12)


def test_checkpoint_update_keeps_legacy_behavior():
    checkpoint = {"model_data": {"model_hypers": {"cutoff": 4.5}}}
    model_update_v18_v19(checkpoint)
    assert checkpoint["model_data"]["model_hypers"]["smooth_attention_cutoff"] is False

    # an explicit choice is not overwritten
    checkpoint["model_data"]["model_hypers"]["smooth_attention_cutoff"] = True
    model_update_v18_v19(checkpoint)
    assert checkpoint["model_data"]["model_hypers"]["smooth_attention_cutoff"] is True


def test_keys_inside_the_cutoff_are_not_affected():
    torch.manual_seed(0)
    logits = 5.0 * torch.randn(4, 2, 6, 6, dtype=torch.float64)
    factors = torch.ones(4, 1, 6, 6, dtype=torch.float64)

    weights = smooth_cutoff_attention(logits, factors)
    assert torch.allclose(weights, logits.softmax(dim=-1), atol=1e-14, rtol=1e-12)

    # a key in the tail gets at most the weight it has with the plain log(f) bias,
    # and the weights of the other keys keep their proportions
    factors[..., 5] = 0.3
    weights = smooth_cutoff_attention(logits, factors)
    plain = (logits + torch.log(factors)).softmax(dim=-1)
    assert torch.all(weights[..., 5] <= plain[..., 5] * (1 + 1e-12))
    ratios = weights[..., :5] / plain[..., :5]
    assert torch.allclose(ratios, ratios[..., :1].expand_as(ratios), rtol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("preference", [-3.0, 0.0, 8.0, 25.0])
@pytest.mark.parametrize("factor", [1e-6, 1e-3, 0.1, 0.7, 1.0])
def test_single_tail_key_gets_a_share_proportional_to_its_factor(
    dtype, preference, factor
):
    # one query, three established keys and one key in the tail, which the query
    # prefers over the others by exp(preference)
    logits = torch.tensor([[0.3, -1.0, 0.5, preference]], dtype=dtype)
    factors = torch.tensor([[1.0, 1.0, 1.0, factor]], dtype=dtype)

    weights = smooth_cutoff_attention(logits, factors)
    share_inside = logits.softmax(dim=-1)[0, 3]

    assert torch.allclose(weights[0, 3], factor * share_inside, rtol=1e-4, atol=0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0, dtype=dtype))

    # with the plain log(f) bias, a strongly preferred key takes most of the
    # attention even when it has barely entered the cutoff
    if preference == 25.0 and factor == 1e-3:
        plain = (logits + torch.log(factors)).softmax(dim=-1)
        assert plain[0, 3] > 0.99
        assert weights[0, 3] < 2e-3


def test_excluded_keys_do_not_matter():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5, 5, dtype=torch.float64)
    factors = torch.rand(2, 1, 5, 5, dtype=torch.float64)
    factors[..., 0] = 1.0
    factors[..., 4] = 0.0
    logits[..., 4] = 50.0  # would dominate if it were not excluded

    weights = smooth_cutoff_attention(logits, factors)
    reference = smooth_cutoff_attention(logits[..., :4], factors[..., :4])

    assert torch.all(weights[..., 4] == 0.0)
    assert torch.allclose(weights[..., :4], reference, atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_are_finite(dtype):
    # includes keys with factors of exactly zero and one, an isolated token and a
    # key that dominates all the others
    logits = torch.tensor(
        [[[0.0, 1.0, 60.0, 2.0], [0.0, 0.0, 0.0, 0.0]]], dtype=dtype, requires_grad=True
    )
    factors = torch.tensor(
        [[[1.0, 1e-4, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=dtype, requires_grad=True
    )

    weights = smooth_cutoff_attention(logits, factors)
    assert torch.isfinite(weights).all()

    values = torch.arange(4, dtype=dtype)
    (grad_logits, grad_factors) = torch.autograd.grad(
        (weights * values).sum(), [logits, factors], create_graph=True
    )
    assert torch.isfinite(grad_logits).all()
    assert torch.isfinite(grad_factors).all()

    # second derivatives, needed to train on forces
    (grad_logits.sum() + grad_factors.sum()).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(factors.grad).all()


def _attention_block(smooth_cutoff):
    torch.manual_seed(0)
    block = AttentionBlock(8, 2, temperature=1.0, smooth_cutoff=smooth_cutoff)
    block = block.to(torch.float64).eval()
    # larger logits, so that the heads have a marked preference for one of the tokens
    with torch.no_grad():
        block.input_linear.weight *= 4.0
    return block


def test_attention_is_linear_in_the_factor_of_a_single_neighbor():
    # a center token and a single neighbor: the output for the center must
    # interpolate linearly between the neighbor being absent and fully present
    torch.manual_seed(1)
    tokens = torch.randn(1, 2, 8, dtype=torch.float64)

    def center_output(block, factor):
        factors = torch.tensor([[[1.0, factor], [1.0, 1.0]]], dtype=torch.float64)
        return block(tokens, factors, use_manual_attention=False)[0, 0]

    smooth = _attention_block(True)
    absent, present = center_output(smooth, 0.0), center_output(smooth, 1.0)
    assert (present - absent).norm() > 1e-3
    for factor in [1e-6, 1e-3, 0.2, 0.9]:
        expected = absent + factor * (present - absent)
        assert torch.allclose(center_output(smooth, factor), expected, atol=1e-12)

    # this is not the case with the plain log(f) bias
    legacy = _attention_block(False)
    absent, present = center_output(legacy, 0.0), center_output(legacy, 1.0)
    expected = absent + 1e-3 * (present - absent)
    assert not torch.allclose(center_output(legacy, 1e-3), expected, atol=1e-3)


def test_only_the_central_token_uses_the_smooth_weights():
    # the neighbor tokens attend with the plain log(f) bias, as without the option
    torch.manual_seed(1)
    tokens = torch.randn(1, 4, 8, dtype=torch.float64)
    factors = torch.tensor([1.0, 1.0, 0.3, 1e-3], dtype=torch.float64).repeat(1, 4, 1)

    smooth = _attention_block(True)(tokens, factors, use_manual_attention=False)
    legacy = _attention_block(False)(tokens, factors, use_manual_attention=False)

    assert torch.allclose(smooth[0, 1:], legacy[0, 1:], atol=1e-12)
    assert not torch.allclose(smooth[0, 0], legacy[0, 0], atol=1e-3)
