import warnings

import pytest


# pytest.skip("Activate this test only if needed", allow_module_level=True)

warnings.filterwarnings(
    "ignore",
    message="This system's positions or cell requires grad, but the neighbors does not.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="PET assumes that Cartesian tensors of rank 2 are stress-like",
    category=UserWarning,
)

import random
from urllib.parse import urlparse
from urllib.request import urlretrieve

import ase.io
import numpy as np
import torch
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator

from metatrain.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo, read_systems
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.output_gradient import compute_gradient

from . import DATASET_PATH, DATASET_WITH_FORCES_PATH


DEFAULT_PET_HYPERS = get_default_hypers("pet")


def get_test_environment():
    # reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )
    model = PET(DEFAULT_PET_HYPERS["model"], dataset_info)

    systems_1 = read_systems(DATASET_PATH)[:5]
    systems_2 = read_systems(DATASET_WITH_FORCES_PATH)[:5]
    systems = systems_1 + systems_2
    for system in systems:
        system.positions.requires_grad_(True)
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    systems = [system.to(torch.float32) for system in systems]
    return model, systems


def test_predictions_compatibility():
    """Tests that the predictions of the PET and NativePET models
    are the same"""

    model, systems = get_test_environment()

    outputs = {"energy": ModelOutput(per_atom=False)}

    predictions = model(systems, outputs)
    expeted_precitions = torch.tensor(
        [
            [3.015289306641],
            [1.845376491547],
            [0.776397109032],
            [1.858697652817],
            [1.014654874802],
            [27.288089752197],
            [27.207054138184],
            [27.232028961182],
            [27.417749404907],
            [27.313308715820],
        ]
    )

    # torch.set_printoptions(precision=12)
    # print(predictions["energy"].block().values)

    torch.testing.assert_close(predictions["energy"].block().values, expeted_precitions)


def test_positions_gradients_compatibility():
    """Tests that the gradients w.r.t positions of the
    PET and NativePET models are the same"""

    model, systems = get_test_environment()
    system = systems[0]

    outputs = {"energy": ModelOutput(per_atom=False)}

    predictions = model([system], outputs)

    gradients = -torch.autograd.grad(
        predictions["energy"].block().values[0][0],
        system.positions,
        torch.ones_like(predictions["energy"].block().values[0][0]),
        create_graph=True,
        retain_graph=True,
    )[0]

    expeted_gradients = torch.tensor(
        [
            [-0.049563053995, 0.024843461812, -0.023552317172],
            [-0.000238502864, 0.007088285871, 0.024732787162],
            [-0.047641716897, 0.002346446738, 0.004111178219],
            [0.043148320168, -0.001909041777, 0.055015549064],
            [0.054294951260, -0.032369159162, -0.060307204723],
        ]
    )

    # torch.set_printoptions(precision=12)
    # print(gradients)

    torch.testing.assert_close(
        gradients,
        expeted_gradients,
    )


def test_features_compatibility():
    """
    Test the compatibility of the features predictions.
    We are testing the features sum instead of the raw features
    in order to keep the testing array small.
    """

    pet_model, systems = get_test_environment()

    outputs = {
        "energy": ModelOutput(per_atom=False),
        "features": ModelOutput(per_atom=False),
    }

    predictions = pet_model(systems, outputs)

    # torch.set_printoptions(precision=12)
    # print(predictions["features"].block().values.sum(axis=1))

    expeted_features_sum = torch.tensor(
        [
            -1.907348632812e-05,
            2.670288085938e-05,
            7.629394531250e-06,
            -1.525878906250e-05,
            -2.288818359375e-05,
            -6.103515625000e-04,
            6.103515625000e-04,
            0.000000000000e00,
            -1.220703125000e-04,
            1.220703125000e-04,
        ]
    )

    torch.testing.assert_close(
        predictions["features"].block().values.sum(axis=1), expeted_features_sum
    )


def test_last_layer_features_compatibility():
    """
    Test the compatibility of the features predictions.
    We are testing the features sum instead of the raw features
    in order to keep the testing array small.
    """

    pet_model, systems = get_test_environment()

    outputs = {
        "energy": ModelOutput(per_atom=False),
        "mtt::aux::energy_last_layer_features": ModelOutput(per_atom=False),
    }

    predictions = pet_model(systems, outputs)

    # torch.set_printoptions(precision=12)
    # print(
    #     predictions["mtt::aux::energy_last_layer_features"].block().values.sum(axis=1)
    # )

    expeted_llf_sum = torch.tensor(
        [
            86.886886596680,
            50.872245788574,
            27.573354721069,
            64.920928955078,
            34.371223449707,
            718.131347656250,
            715.955078125000,
            717.168823242188,
            715.163452148438,
            714.825561523438,
        ]
    )

    torch.testing.assert_close(
        predictions["mtt::aux::energy_last_layer_features"].block().values.sum(axis=1),
        expeted_llf_sum,
    )


def test_pet_mad_model_compatibility(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    path = "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt"

    if urlparse(path).scheme:
        path, _ = urlretrieve(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # Remap checkpoint keys to match new architecture
    new_state_dict = {}
    for key, value in checkpoint["best_model_state_dict"].items():
        new_key = key
        new_key = new_key.replace("gnn_layers.", "gnn_layers.")
        new_key = new_key.replace(".edge_embedder", ".token_encoder.edge_embedder")
        new_key = new_key.replace(".node_embedder", ".token_encoder.node_embedder")
        new_key = new_key.replace(".neighbor_embedder", ".token_encoder.neighbor_embedder")
        new_key = new_key.replace(".compress", ".token_encoder.compress")
        new_state_dict[new_key] = value
    checkpoint["best_model_state_dict"] = new_state_dict
    model = PET.load_checkpoint(checkpoint, context="export").eval()

    _, systems = get_test_environment()
    system = systems[0]

    outputs = {"energy": ModelOutput(per_atom=True)}

    predictions = model([system], outputs)

    gradients = compute_gradient(
        predictions["energy"].block().values,
        [system.positions],
        is_training=True,
    )[0]

    # torch.set_printoptions(precision=12)
    # print(predictions["energy"].block().values)
    # print(gradients)

    expeted_predictions = torch.tensor(
        [
            [-9.708948135376],
            [-3.548597574234],
            [-3.546929836273],
            [-3.550716400146],
            [-3.550122737885],
        ]
    )

    expeted_gradients = torch.tensor(
        [
            [-0.000917881727, -0.003729507327, 0.007607728243],
            [-0.000858668238, 0.153381973505, -0.001641030191],
            [-0.137988328934, -0.051602333784, -0.001579828560],
            [0.072291240096, -0.050593279302, 0.120229497552],
            [0.067473590374, -0.047456823289, -0.124616414309],
        ]
    )

    torch.testing.assert_close(
        predictions["energy"].block().values, expeted_predictions
    )

    torch.testing.assert_close(gradients, expeted_gradients)


def test_pet_mad_non_conservative_heads_compatibility(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    path = "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt"

    if urlparse(path).scheme:
        path, _ = urlretrieve(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # Remap checkpoint keys to match new architecture
    new_state_dict = {}
    for key, value in checkpoint["best_model_state_dict"].items():
        new_key = key
        new_key = new_key.replace(".edge_embedder", ".token_encoder.edge_embedder")
        new_key = new_key.replace(".node_embedder", ".token_encoder.node_embedder")
        new_key = new_key.replace(".neighbor_embedder", ".token_encoder.neighbor_embedder")
        new_key = new_key.replace(".compress", ".token_encoder.compress")
        new_state_dict[new_key] = value
    checkpoint["best_model_state_dict"] = new_state_dict
    model = PET.load_checkpoint(checkpoint, context="export").eval()

    _, systems = get_test_environment()
    system = systems[-1]

    outputs = {
        "non_conservative_forces": ModelOutput(per_atom=True),
        "non_conservative_stress": ModelOutput(per_atom=False),
    }

    predictions = model([system], outputs)

    # torch.set_printoptions(precision=12)
    # print(predictions["non_conservative_forces"].block().values.squeeze(-1))
    # print(predictions["non_conservative_stress"].block().values.squeeze(-1).squeeze(0))

    expeted_nc_forces_predictions = torch.tensor(
        [
            [-0.313198447227, -0.596738040447, 0.012323617935],
            [0.331029176712, 0.680125236511, 0.031832695007],
            [0.331488728523, 0.671775341034, 0.025099277496],
            [-0.311262369156, -0.601662576199, 0.018292188644],
        ]
    )

    expeted_nc_stress_predictions = torch.tensor(
        [
            [-2.711088955402e-01, 4.658789932728e-02, 8.484579622746e-03],
            [4.658789932728e-02, -3.988451361656e-01, 9.212930686772e-03],
            [8.484579622746e-03, 9.212930686772e-03, -1.589092426002e-04],
        ]
    )

    torch.testing.assert_close(
        predictions["non_conservative_forces"].block().values.squeeze(-1),
        expeted_nc_forces_predictions,
    )

    torch.testing.assert_close(
        predictions["non_conservative_stress"].block().values.squeeze(-1).squeeze(0),
        expeted_nc_stress_predictions,
    )


def test_pet_mad_ase_calculator_inference_timings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    path = "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt"

    if urlparse(path).scheme:
        path, _ = urlretrieve(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    # Change the model architecture
    # Remap checkpoint keys to match new architecture
    # TODO: Remove this when the model is updated or add to load old checkpoints
    new_state_dict = {}
    for key, value in checkpoint["best_model_state_dict"].items():
        new_key = key
        new_key = new_key.replace("gnn_layers.", "gnn_layers.")
        new_key = new_key.replace(".edge_embedder", ".token_encoder.edge_embedder")
        new_key = new_key.replace(".node_embedder", ".token_encoder.node_embedder")
        new_key = new_key.replace(".neighbor_embedder", ".token_encoder.neighbor_embedder")
        new_key = new_key.replace(".compress", ".token_encoder.compress")
        new_state_dict[new_key] = value
    checkpoint["best_model_state_dict"] = new_state_dict



    model = PET.load_checkpoint(checkpoint, context="export").export()
    model.save("model.pt")
    calc = MetatomicCalculator("model.pt", device="cpu")

    atoms = ase.io.read(DATASET_WITH_FORCES_PATH)
    atoms.calc = calc

    # Warm-up
    for _ in range(10):
        calc.compute_energy(atoms)

    with torch.profiler.profile() as energy_profiler:
        for _ in range(10):
            atoms.positions += np.random.rand(*atoms.positions.shape)
            atoms.get_potential_energy()

    print(
        energy_profiler.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=10
        )
    )
