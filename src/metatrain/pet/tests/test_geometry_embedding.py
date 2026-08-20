"""Tests for the optional ``geometry_embedding_l_max`` edge embedding."""

import copy
import math

import torch
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.pet.modules.transformer import SphericalHarmonicsNoSphericart
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def test_spherical_harmonics_no_sphericart_norms():
    """Each ell block of the normalized harmonics, scaled by |r|, should have a
    norm equal to |r| (sanity check from the feature request)."""

    l_max = 4
    calc = SphericalHarmonicsNoSphericart(l_max)

    vectors = torch.rand(10, 3)
    distances = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    harmonics = calc(vectors)

    for ell in range(l_max + 1):
        harmonics[..., ell**2 : (ell + 1) ** 2] *= math.sqrt(
            4 * math.pi / (2 * ell + 1)
        )
    features = distances * harmonics

    for ell in range(l_max + 1):
        features_ell = features[..., ell**2 : (ell + 1) ** 2]
        assert torch.allclose(
            torch.linalg.norm(features_ell, dim=-1, keepdim=True), distances
        )


def test_geometry_embedding_forward_and_padding():
    """The PET model should run with ``geometry_embedding_l_max`` set, produce no
    NaNs (including for padded edges), and predict the same energy independently
    of the padding size, just like the standard embedding."""

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["geometry_embedding_l_max"] = 3

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )

    model = PET(hypers, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(sample_kind="system")}
    lone_output = model([system], outputs)
    lone_energy = lone_output["energy"].block().values.squeeze(-1)[0]
    assert torch.isfinite(lone_energy).all()

    system_2 = System(
        types=torch.tensor([6, 6, 6, 6, 6, 6, 6]),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 6.0],
            ]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system_2 = get_system_with_neighbor_lists(
        system_2, model.requested_neighbor_lists()
    )
    padded_output = model([system, system_2], outputs)
    padded_energy = padded_output["energy"].block().values.squeeze(-1)[0]

    assert torch.isfinite(padded_energy).all()
    assert torch.allclose(lone_energy, padded_energy, atol=1e-6, rtol=1e-6)
