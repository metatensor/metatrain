"""Bernstein / e3x SH, flax transfer, PET trunk, jax-pme tiling helpers."""

import torch

from metatrain.experimental.lorem import LOREM
from metatrain.experimental.lorem.flax_io import apply_flax_params, flatten_flax_tree
from metatrain.experimental.lorem.modules.pme_batch import (
    pad_to_tile,
    split_pbc_indices,
)
from metatrain.experimental.lorem.modules.radial import bernstein_basis, binomial_row
from metatrain.experimental.lorem.modules.spherical import to_racah
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from .test_long_range import _chain_system, _energy_dataset_info, _small_lr_hypers


def test_bernstein_is_zero_outside_cutoff_and_partition_of_unity():
    cutoff = 4.0
    r = torch.tensor([-0.2, 0.0, 1.0, 4.0, 4.2])
    values = bernstein_basis(r, cutoff, binomial_row(5).to(torch.float32))
    assert values.shape == (5, 5)
    torch.testing.assert_close(values[0], torch.zeros(5), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(values[-1], torch.zeros(5), atol=1e-6, rtol=1e-6)
    inside = values[1:4].sum(dim=-1)
    torch.testing.assert_close(inside, torch.ones(3), atol=1e-5, rtol=1e-5)


def test_e3x_racah_scales_l0_to_one():
    y00 = torch.full((2, 1), 0.28209479177387814)
    racah = to_racah(y00, max_degree=0)
    torch.testing.assert_close(racah, torch.ones(2, 1), atol=1e-5, rtol=1e-5)


def test_e3x_racah_l1_has_unit_cartesian_coeffs():
    c1 = 0.4886025119029199
    sh = torch.tensor([[c1, 0.0, 0.0]])
    racah = to_racah(torch.cat([torch.zeros(1, 1), sh], dim=-1), max_degree=1)
    torch.testing.assert_close(racah[0, 1], torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_pad_to_tile_rounds_up_to_multiple():
    assert pad_to_tile(0, 32) == 0
    assert pad_to_tile(1, 32) == 32
    assert pad_to_tile(32, 32) == 32
    assert pad_to_tile(33, 32) == 64


def test_split_pbc_indices():
    pbc = _chain_system(pbc=True)
    open_ = _chain_system(pbc=False)
    periodic, nonperiodic = split_pbc_indices([open_, pbc, open_])
    assert periodic == [1]
    assert nonperiodic == [0, 2]


def test_flax_linear_kernel_is_transposed():
    layer = torch.nn.Linear(3, 2, bias=False)
    flax = {"Dense_0": {"kernel": torch.arange(6.0).reshape(3, 2)}}
    transferred = apply_flax_params(layer, flax)
    assert transferred == ["weight"]
    expected = torch.arange(6.0).reshape(3, 2).T
    torch.testing.assert_close(layer.weight, expected)


def test_flax_flatten_unwraps_params():
    flat = flatten_flax_tree({"params": {"Embed_0": {"embedding": torch.ones(2, 3)}}})
    assert list(flat) == ["Embed_0/embedding"]
    assert flat["Embed_0/embedding"].shape == (2, 3)


def test_lorem_load_flax_weights_embedding():
    hypers = _small_lr_hypers()
    model = LOREM(hypers, _energy_dataset_info())
    table = torch.arange(
        model.sr.species_embedding.weight.numel(), dtype=torch.float32
    ).reshape_as(model.sr.species_embedding.weight)
    flax_name = "Initial_0/ChemicalEmbedding_0/Embed_0/embedding"
    transferred = model.load_flax_weights(
        {
            "params": {
                "Initial_0": {"ChemicalEmbedding_0": {"Embed_0": {"embedding": table}}}
            }
        },
        name_map={"sr.species_embedding.weight": flax_name},
    )
    assert "sr.species_embedding.weight" in transferred
    torch.testing.assert_close(model.sr.species_embedding.weight, table)


def test_pet_trunk_forward_shapes():
    hypers = _small_lr_hypers(max_degree=1, max_degree_lr=0)
    hypers["trunk"] = "pet"
    hypers["num_features"] = 8
    hypers["pet"] = {
        "d_pet": 8,
        "d_feedforward": 8,
        "d_head": 8,
        "num_gnn_layers": 1,
        "num_attention_layers": 1,
        "num_heads": 1,
        "featurizer_type": "feedforward",
    }
    model = LOREM(hypers, _energy_dataset_info())
    assert type(model.sr).__name__ == "PetTrunk"
    system = get_system_with_neighbor_lists(
        _chain_system(pbc=False), model.requested_neighbor_lists()
    )
    features, distances, spherical = model.sr([system])
    assert features.shape[0] == 4
    assert features.shape[1] == 8
    assert spherical.shape == (4, 4, int(hypers["num_spherical_features"]))
    assert distances.ndim == 1


def test_bernstein_default_is_paper_basis():
    from metatrain.utils.architectures import get_default_hypers

    hypers = get_default_hypers("experimental.lorem")["model"]
    assert hypers["radial_basis"] == "basic_bernstein"
    assert hypers["sh_convention"] == "e3x"
    assert hypers["trunk"] == "spherical"
