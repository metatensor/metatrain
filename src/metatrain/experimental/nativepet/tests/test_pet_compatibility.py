import pytest


pytest.importorskip("pet_neighbors_convert")

from urllib.parse import urlparse
from urllib.request import urlretrieve

import metatensor.torch
import torch
from metatensor.torch.atomistic import ModelOutput

from metatrain.experimental.nativepet import NativePET
from metatrain.experimental.nativepet.modules.compatibility import (
    convert_checkpoint_from_legacy_pet,
)
from metatrain.experimental.nativepet.modules.structures import systems_to_batch
from metatrain.experimental.nativepet.modules.utilities import cutoff_func
from metatrain.pet import PET
from metatrain.pet.modules.hypers import Hypers
from metatrain.pet.modules.pet import PET as RawPET
from metatrain.pet.utils import (
    systems_to_batch_dict,
)
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo, read_systems
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.output_gradient import compute_gradient

from . import DATASET_PATH, DATASET_WITH_FORCES_PATH


DEFAULT_PET_HYPERS = get_default_hypers("pet")
DEFAULT_NATIVEPET_HYPERS = get_default_hypers("experimental.nativepet")

DEFAULT_PET_HYPERS["model"]["R_CUT"] = DEFAULT_NATIVEPET_HYPERS["model"]["cutoff"]
DEFAULT_PET_HYPERS["model"]["BLEND_NEIGHBOR_SPECIES"] = True
DEFAULT_PET_HYPERS["model"]["TRANSFORMER_N_HEAD"] = DEFAULT_NATIVEPET_HYPERS["model"][
    "num_heads"
]
DEFAULT_PET_HYPERS["model"]["N_GNN_LAYERS"] = DEFAULT_NATIVEPET_HYPERS["model"][
    "num_gnn_layers"
]
DEFAULT_PET_HYPERS["model"]["N_TRANS_LAYERS"] = DEFAULT_NATIVEPET_HYPERS["model"][
    "num_attention_layers"
]


def set_embedding_weights(nativepet_state_dict, pet_state_dict):
    weight = nativepet_state_dict["embedding.weight"].detach().clone()
    weight.requires_grad = True
    pet_state_dict["pet.embedding.weight"] = weight
    return pet_state_dict


def set_gnn_weights(nativepet_state_dict, pet_state_dict):
    for key in nativepet_state_dict.keys():
        if "gnn_layers" in key:
            weight = nativepet_state_dict[key].detach().clone()
            weight.requires_grad = True
            pet_key = "pet." + key
            if "edge_embedder" in pet_key:
                pet_key = pet_key.replace("edge_embedder", "r_embedding")
            if "node_embedder" in pet_key:
                pet_key = pet_key.replace("node_embedder", "central_embedder")
            pet_state_dict[pet_key] = weight
    return pet_state_dict


def set_heads_weights(nativepet_state_dict, pet_state_dict):
    for key in nativepet_state_dict.keys():
        if "head" in key and "linear" not in key:
            pet_key = "pet." + key.replace("energy.", "")
            if "node" in key:
                pet_key = pet_key.replace("node_", "")
            if "edge" in key:
                pet_key = pet_key.replace("edge_", "bond_")
            pet_key = pet_key.replace("node_", "")
            split_key = pet_key.split(".")
            split_key.insert(3, "nn")
            pet_key = ".".join(split_key)
            weight = nativepet_state_dict[key].detach().clone()
            weight.requires_grad = True
            pet_state_dict[pet_key] = weight
    return pet_state_dict


def set_last_layers_weights(nativepet_state_dict, pet_state_dict):
    for key in nativepet_state_dict.keys():
        if "last_layer" in key:
            layer_num = key.split(".")[2]
            if "edge" in key:
                pet_key = f"pet.bond_heads.{layer_num}.nn.4.linear."
            else:
                pet_key = f"pet.heads.{layer_num}.nn.4.linear."
            if "weight" in key:
                pet_key += "weight"
            else:
                pet_key += "bias"
            weight = nativepet_state_dict[key].detach().clone()
            weight.requires_grad = True
            pet_state_dict[pet_key] = weight
    return pet_state_dict


def ensure_embeddings_weights_equality(nativepet_model, pet_model):
    torch.testing.assert_close(
        nativepet_model.embedding.weight, pet_model.pet.embedding.weight
    )


def ensure_gnn_layers_weights_equality(nativepet_model, pet_model):
    for i in range(len(nativepet_model.gnn_layers)):
        nativepet_gnn_layer = nativepet_model.gnn_layers[i]
        pet_gnn_layer = pet_model.pet.gnn_layers[i]
        # Testing if the r_embedding layers are the same
        torch.testing.assert_close(
            nativepet_gnn_layer.edge_embedder.weight, pet_gnn_layer.r_embedding.weight
        )
        torch.testing.assert_close(
            nativepet_gnn_layer.edge_embedder.bias, pet_gnn_layer.r_embedding.bias
        )
        # Testing if the compress layers are the same
        for j in range(len(nativepet_gnn_layer.compress)):
            if not hasattr(nativepet_gnn_layer.compress[j], "weight"):
                continue
            nativepet_compress_layer = nativepet_gnn_layer.compress[j]
            pet_compress_layer = pet_gnn_layer.compress[j]
            torch.testing.assert_close(
                nativepet_compress_layer.weight, pet_compress_layer.weight
            )
            torch.testing.assert_close(
                nativepet_compress_layer.bias, pet_compress_layer.bias
            )
        # Testing if the neighbor_embedder layers are the same
        if hasattr(nativepet_gnn_layer.neighbor_embedder, "weight"):
            torch.testing.assert_close(
                nativepet_gnn_layer.neighbor_embedder.weight,
                pet_gnn_layer.neighbor_embedder.weight,
            )
        # Testing if the transformer layers are the same
        for j in range(len(nativepet_gnn_layer.trans.layers)):
            nativepet_trans_layer = nativepet_gnn_layer.trans.layers[j]
            pet_trans_layer = pet_gnn_layer.trans.layers[j]
            for k in range(len(nativepet_trans_layer.mlp)):
                nativepet_mlp_layer = nativepet_trans_layer.mlp[k]
                pet_mlp_layer = pet_trans_layer.mlp[k]
                if not hasattr(nativepet_mlp_layer, "weight"):
                    continue
                # Testing if the mlp layers are the same
                torch.testing.assert_close(
                    nativepet_mlp_layer.weight, pet_mlp_layer.weight
                )
                torch.testing.assert_close(nativepet_mlp_layer.bias, pet_mlp_layer.bias)
            # Testing if the attention layers are the same
            torch.testing.assert_close(
                nativepet_trans_layer.attention.input_linear.weight,
                pet_trans_layer.attention.input_linear.weight,
            )
            torch.testing.assert_close(
                nativepet_trans_layer.attention.output_linear.weight,
                pet_trans_layer.attention.output_linear.weight,
            )
            torch.testing.assert_close(
                nativepet_trans_layer.attention.input_linear.bias,
                pet_trans_layer.attention.input_linear.bias,
            )
            torch.testing.assert_close(
                nativepet_trans_layer.attention.output_linear.bias,
                pet_trans_layer.attention.output_linear.bias,
            )
            # Testing if attention layernorms are the same
            torch.testing.assert_close(
                nativepet_trans_layer.norm_attention.weight,
                pet_trans_layer.norm_attention.weight,
            )
            torch.testing.assert_close(
                nativepet_trans_layer.norm_attention.bias,
                pet_trans_layer.norm_attention.bias,
            )
            # Testing if mlp layernorms are the same
            torch.testing.assert_close(
                nativepet_trans_layer.norm_mlp.weight,
                pet_trans_layer.norm_mlp.weight,
            )
            torch.testing.assert_close(
                nativepet_trans_layer.norm_mlp.bias,
                pet_trans_layer.norm_mlp.bias,
            )


def ensure_node_heads_weights_equality(nativepet_model, pet_model):
    for i in range(len(nativepet_model.node_heads["energy"])):
        nativepet_head = nativepet_model.node_heads["energy"][i]
        pet_head = pet_model.pet.heads[i]
        # Testing if the linear layers are the same
        for j in range(len(nativepet_head)):
            if not hasattr(nativepet_head[j], "weight"):
                continue
            nativepet_linear_layer = nativepet_head[j]
            pet_linear_layer = pet_head.nn[j]
            torch.testing.assert_close(
                nativepet_linear_layer.weight, pet_linear_layer.weight
            )
            torch.testing.assert_close(
                nativepet_linear_layer.bias, pet_linear_layer.bias
            )


def ensure_edge_heads_weights_equality(nativepet_model, pet_model):
    for i in range(len(nativepet_model.edge_heads["energy"])):
        nativepet_bond_head = nativepet_model.edge_heads["energy"][i]
        pet_bond_head = pet_model.pet.bond_heads[i]
        # Testing if the linear layers are the same
        for j in range(len(nativepet_bond_head)):
            if not hasattr(nativepet_bond_head[j], "weight"):
                continue
            nativepet_linear_layer = nativepet_bond_head[j]
            pet_linear_layer = pet_bond_head.nn[j]
            torch.testing.assert_close(
                nativepet_linear_layer.weight, pet_linear_layer.weight
            )
            torch.testing.assert_close(
                nativepet_linear_layer.bias, pet_linear_layer.bias
            )


def ensure_node_last_layers_weights_equality(nativepet_model, pet_model):
    for i in range(len(nativepet_model.node_last_layers["energy"])):
        nativepet_last_layer = nativepet_model.node_last_layers["energy"][i][
            "energy___0"
        ]
        pet_last_layer = pet_model.pet.heads[i].nn[-1].linear
        torch.testing.assert_close(nativepet_last_layer.weight, pet_last_layer.weight)
        torch.testing.assert_close(nativepet_last_layer.bias, pet_last_layer.bias)


def ensure_edge_last_layers_weights_equality(nativepet_model, pet_model):
    for i in range(len(nativepet_model.edge_last_layers["energy"])):
        nativepet_last_layer = nativepet_model.edge_last_layers["energy"][i][
            "energy___0"
        ]
        pet_last_layer = pet_model.pet.bond_heads[i].nn[-1].linear
        torch.testing.assert_close(nativepet_last_layer.weight, pet_last_layer.weight)
        torch.testing.assert_close(nativepet_last_layer.bias, pet_last_layer.bias)


def get_identical_pet_models():
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )
    nativepet_model = NativePET(DEFAULT_NATIVEPET_HYPERS["model"], dataset_info)
    pet_model = PET(DEFAULT_PET_HYPERS["model"], dataset_info)
    raw_pet_hypers = Hypers(pet_model.hypers)
    raw_pet = RawPET(raw_pet_hypers, 0.0, len(pet_model.atomic_types))
    pet_model.set_trained_model(raw_pet)

    nativepet_state_dict = nativepet_model.state_dict()
    pet_state_dict = pet_model.state_dict()

    # Setting the embedding layers to be the same weights
    pet_state_dict = set_embedding_weights(nativepet_state_dict, pet_state_dict)
    # Setting the GNN layers to be the same weights
    pet_state_dict = set_gnn_weights(nativepet_state_dict, pet_state_dict)
    # Setting the heads to be the same weights
    pet_state_dict = set_heads_weights(nativepet_state_dict, pet_state_dict)
    # Setting the last layers to be the same weights
    pet_state_dict = set_last_layers_weights(nativepet_state_dict, pet_state_dict)

    # Loading the updated state dict
    pet_model.load_state_dict(pet_state_dict)

    # Testing if embedding layers are the same
    ensure_embeddings_weights_equality(nativepet_model, pet_model)
    # Testing if GNN layers are the same
    ensure_gnn_layers_weights_equality(nativepet_model, pet_model)
    # Testing if the heads are the same
    ensure_node_heads_weights_equality(nativepet_model, pet_model)
    # Testing if the bond heads are the same
    ensure_edge_heads_weights_equality(nativepet_model, pet_model)
    # Testing if the last layers are the same
    ensure_node_last_layers_weights_equality(nativepet_model, pet_model)
    # Testing if the bond last layers are the same
    ensure_edge_last_layers_weights_equality(nativepet_model, pet_model)

    return nativepet_model, pet_model


def get_test_environment():
    nativepet_model, pet_model = get_identical_pet_models()

    systems_1 = read_systems(DATASET_PATH)[:5]
    systems_2 = read_systems(DATASET_WITH_FORCES_PATH)[:5]
    systems = systems_1 + systems_2
    for system in systems:
        system.positions.requires_grad_(True)
        get_system_with_neighbor_lists(
            system, nativepet_model.requested_neighbor_lists()
        )
    systems = [system.to(torch.float32) for system in systems]
    return nativepet_model, pet_model, systems


def test_batch_dict_compatability():
    """Tests that the batch dictionaries created by the
    PET and NativePET models are the same"""
    nativepet_model, pet_model, systems = get_test_environment()

    nl_options = nativepet_model.requested_neighbor_lists()[0]

    system_indices, sample_labels = nativepet_model._get_system_indices_and_labels(
        systems, device="cpu"
    )

    (
        element_indices_nodes,
        element_indices_neighbors,
        edge_vectors,
        padding_mask,
        neighbors_index,
        num_neghbors,
        reversed_neighbor_list,
    ) = systems_to_batch(
        systems,
        nl_options,
        nativepet_model.atomic_types,
        system_indices,
        nativepet_model.species_to_species_index,
        None,
    )

    nativepet_batch_dict = {
        "central_species": element_indices_nodes,
        "neighbor_species": element_indices_neighbors,
        "x": edge_vectors,
        "mask": ~padding_mask,
        "neighbors_index": neighbors_index,
        "nums": num_neghbors,
        "batch": system_indices,
        "neighbors_pos": reversed_neighbor_list,
    }

    pet_batch_dict = systems_to_batch_dict(
        systems, nl_options, pet_model.atomic_types, None
    )

    for key in nativepet_batch_dict:
        if key in ["nums", "batch", "central_species", "mask"]:
            assert torch.allclose(nativepet_batch_dict[key], pet_batch_dict[key])
        else:
            nativepet_mask = nativepet_batch_dict["mask"] is False
            pet_mask = pet_batch_dict["mask"] is False
            assert torch.allclose(
                nativepet_batch_dict[key][nativepet_mask],
                pet_batch_dict[key][pet_mask],
            )


def test_embeddings_compatibility():
    """Tests that neighbor species embeddings are the same for
    the PET and NativePET models"""
    nativepet_model, pet_model, systems = get_test_environment()

    nl_options = nativepet_model.requested_neighbor_lists()[0]
    batch_dict = systems_to_batch_dict(
        systems, nl_options, nativepet_model.atomic_types, None
    )

    nativepet_embeddings = nativepet_model.embedding(batch_dict["neighbor_species"])
    pet_embeddings = pet_model.pet.embedding(batch_dict["neighbor_species"])

    torch.testing.assert_close(nativepet_embeddings, pet_embeddings)


def test_cartesian_transformer_compatibility():
    """Tests that the Cartesian transformer layers in the PET
    and NativePET models give the same predictions"""
    torch.manual_seed(0)
    nativepet_model, pet_model, systems = get_test_environment()

    nl_options = nativepet_model.requested_neighbor_lists()[0]
    batch_dict = systems_to_batch_dict(
        systems, nl_options, nativepet_model.atomic_types, None
    )
    batch_dict["input_messages"] = nativepet_model.embedding(
        batch_dict["neighbor_species"]
    )

    nativepet_cartesian_transformer = nativepet_model.gnn_layers[0]
    pet_cartesian_transformer = pet_model.pet.gnn_layers[0]

    edge_distances = torch.sqrt(torch.sum(batch_dict["x"] ** 2, dim=2) + 1e-16)
    cutoff_factors = cutoff_func(
        edge_distances, nativepet_model.cutoff, nativepet_model.cutoff_width
    )
    cutoff_factors[batch_dict["mask"]] = 0.0

    nativepet_output_node_embeddings, nativepet_output_messages = (
        nativepet_cartesian_transformer(
            input_messages=batch_dict["input_messages"],
            element_indices_nodes=batch_dict["central_species"],
            element_indices_neighbors=batch_dict["neighbor_species"],
            edge_vectors=batch_dict["x"],
            padding_mask=~batch_dict["mask"],
            edge_distances=edge_distances,
            cutoff_factors=cutoff_factors,
            use_manual_attention=False,
        )
    )
    pet_result = pet_cartesian_transformer(batch_dict)
    torch.testing.assert_close(
        nativepet_output_node_embeddings, pet_result["central_token"]
    )
    torch.testing.assert_close(nativepet_output_messages, pet_result["output_messages"])


def test_gnn_layers_compatibility():
    """Tests that the GNN layers in the PET and NativePET models
    give the same predictions"""
    nativepet_model, pet_model, systems = get_test_environment()

    nl_options = nativepet_model.requested_neighbor_lists()[0]
    nativepet_batch_dict = systems_to_batch_dict(
        systems, nl_options, nativepet_model.atomic_types, None
    )
    pet_batch_dict = systems_to_batch_dict(
        systems, nl_options, pet_model.atomic_types, None
    )

    neighbors_index = nativepet_batch_dict["neighbors_index"]
    neighbors_pos = nativepet_batch_dict["neighbors_pos"]

    nativepet_batch_dict["input_messages"] = nativepet_model.embedding(
        nativepet_batch_dict["neighbor_species"]
    )

    pet_batch_dict["input_messages"] = pet_model.pet.embedding(
        pet_batch_dict["neighbor_species"]
    )

    edge_distances = torch.sqrt(
        torch.sum(nativepet_batch_dict["x"] ** 2, dim=2) + 1e-15
    )
    cutoff_factors = cutoff_func(
        edge_distances, nativepet_model.cutoff, nativepet_model.cutoff_width
    )
    cutoff_factors[nativepet_batch_dict["mask"]] = 0.0

    for nativepet_gnn_layer, pet_gnn_layer in zip(
        nativepet_model.gnn_layers, pet_model.pet.gnn_layers
    ):
        output_node_embeddings, output_edge_embeddings = nativepet_gnn_layer(
            input_messages=nativepet_batch_dict["input_messages"],
            element_indices_nodes=nativepet_batch_dict["central_species"],
            element_indices_neighbors=nativepet_batch_dict["neighbor_species"],
            edge_vectors=nativepet_batch_dict["x"],
            padding_mask=~nativepet_batch_dict["mask"],
            edge_distances=edge_distances,
            cutoff_factors=cutoff_factors,
            use_manual_attention=False,
        )
        pet_result = pet_gnn_layer(pet_batch_dict)
        new_nativepet_input_messages = output_edge_embeddings[
            neighbors_index, neighbors_pos
        ]
        new_pet_input_messages = pet_result["output_messages"][
            neighbors_index, neighbors_pos
        ]

        nativepet_batch_dict["input_messages"] = 0.5 * (
            nativepet_batch_dict["input_messages"] + new_nativepet_input_messages
        )

        pet_batch_dict["input_messages"] = pet_model.pet.RESIDUAL_FACTOR * (
            pet_batch_dict["input_messages"] + new_pet_input_messages
        )

        torch.testing.assert_close(output_node_embeddings, pet_result["central_token"])
        torch.testing.assert_close(
            output_edge_embeddings, pet_result["output_messages"]
        )
        torch.testing.assert_close(
            nativepet_batch_dict["input_messages"], pet_batch_dict["input_messages"]
        )


def test_node_heads():
    """Tests that the heads in the PET and NativePET models
    give the same predictions"""
    nativepet_model, pet_model, systems = get_test_environment()

    nl_options = nativepet_model.requested_neighbor_lists()[0]
    batch_dict = systems_to_batch_dict(
        systems, nl_options, nativepet_model.atomic_types, None
    )
    batch_dict["input_messages"] = nativepet_model.embedding(
        batch_dict["neighbor_species"]
    )
    gnn_result = pet_model.pet.gnn_layers[0](batch_dict)
    pet_atomic_predictions = torch.zeros(1)
    nativepet_atomic_predictions = torch.zeros(1)

    pet_precitor_output = pet_model.pet.central_tokens_predictors[0](
        gnn_result["central_token"], batch_dict["central_species"]
    )
    pet_atomic_predictions = (
        pet_atomic_predictions + pet_precitor_output["atomic_predictions"]
    )
    nativepet_head_output = nativepet_model.node_heads["energy"][0](
        gnn_result["central_token"]
    )
    nativepet_atomic_predictions = (
        nativepet_atomic_predictions
        + nativepet_model.node_last_layers["energy"][0]["energy___0"](
            nativepet_head_output
        )
    )

    torch.testing.assert_close(nativepet_atomic_predictions, pet_atomic_predictions)


def test_edge_heads():
    """Tests that the bond heads in the PET and NativePET models
    give the same predictions"""
    nativepet_model, pet_model, systems = get_test_environment()

    nl_options = nativepet_model.requested_neighbor_lists()[0]
    batch_dict = systems_to_batch_dict(
        systems, nl_options, nativepet_model.atomic_types, None
    )
    x = batch_dict["x"]
    central_species = batch_dict["central_species"]
    mask = batch_dict["mask"]
    nums = batch_dict["nums"]

    lengths = torch.sqrt(torch.sum(x * x, dim=2) + 1e-16)
    multipliers = cutoff_func(lengths, pet_model.pet.R_CUT, pet_model.pet.CUTOFF_DELTA)
    multipliers[mask] = 0.0

    batch_dict["input_messages"] = pet_model.pet.embedding(
        batch_dict["neighbor_species"]
    )

    gnn_result = pet_model.pet.gnn_layers[0](batch_dict)

    pet_atomic_predictions = torch.zeros(1)
    nativepet_atomic_predictions = torch.zeros(1)

    pet_precitor_output = pet_model.pet.messages_bonds_predictors[0](
        gnn_result["output_messages"],
        mask,
        nums,
        central_species,
        multipliers,
    )

    pet_atomic_predictions = (
        pet_atomic_predictions + pet_precitor_output["atomic_predictions"]
    )

    nativepet_head_output = nativepet_model.edge_heads["energy"][0](
        gnn_result["output_messages"]
    )
    nativepet_last_layer_output = nativepet_model.edge_last_layers["energy"][0][
        "energy___0"
    ](nativepet_head_output)

    mask_expanded = mask[..., None].repeat(1, 1, nativepet_last_layer_output.shape[2])
    nativepet_last_layer_output = torch.where(
        mask_expanded, 0.0, nativepet_last_layer_output
    )
    nativepet_last_layer_output = nativepet_last_layer_output * multipliers[:, :, None]
    nativepet_last_layer_output = nativepet_last_layer_output.sum(dim=1)

    nativepet_atomic_predictions = (
        nativepet_atomic_predictions + nativepet_last_layer_output
    )

    torch.testing.assert_close(nativepet_atomic_predictions, pet_atomic_predictions)


def test_predictions_compatibility():
    """Tests that the predictions of the PET and NativePET models
    are the same"""
    nativepet_model, pet_model, systems = get_test_environment()

    outputs = {"energy": ModelOutput(per_atom=False)}

    nativepet_predictions = nativepet_model(systems, outputs)
    pet_predictions = pet_model(systems, outputs)

    torch.testing.assert_close(
        nativepet_predictions["energy"].block().values,
        pet_predictions["energy"].block().values,
    )


def test_positions_gradients_compatibility():
    """Tests that the gradients w.r.t positions of the
    PET and NativePET models are the same"""
    nativepet_model, pet_model, systems = get_test_environment()
    system = systems[0]

    outputs = {"energy": ModelOutput(per_atom=False)}

    nativepet_predictions = nativepet_model([system], outputs)

    nativepet_gradients = -torch.autograd.grad(
        nativepet_predictions["energy"].block().values[0][0],
        system.positions,
        torch.ones_like(nativepet_predictions["energy"].block().values[0][0]),
        create_graph=True,
        retain_graph=True,
    )[0]

    nl_options = nativepet_model.requested_neighbor_lists()[0]
    batch_dict = systems_to_batch_dict(
        [system], nl_options, nativepet_model.atomic_types, None
    )
    x = batch_dict["x"]
    x.requires_grad_(True)

    pet_predictions = pet_model.pet(batch_dict)["prediction"]

    pet_grads_wrt_x = torch.autograd.grad(
        pet_predictions,
        x,
        grad_outputs=torch.ones_like(pet_predictions),
        create_graph=True,
        retain_graph=True,
    )[0]

    neighbors_index = batch_dict["neighbors_index"]  # .transpose(0, 1)
    neighbors_pos = batch_dict["neighbors_pos"]
    grads_messaged = pet_grads_wrt_x[neighbors_index, neighbors_pos]
    pet_grads_wrt_x[batch_dict["mask"]] = 0.0

    grads_messaged[batch_dict["mask"]] = 0.0
    first = pet_grads_wrt_x.sum(dim=1)
    second = grads_messaged.sum(dim=1)
    pet_gradients = first - second

    torch.testing.assert_close(
        nativepet_gradients,
        pet_gradients,
    )


def test_energy_loss_grads_compatibility():
    """Tests that the gradients of the energy loss produced by
    PET and NativePET models are the same"""
    nativepet_model, pet_model, systems = get_test_environment()

    outputs = {"energy": ModelOutput(per_atom=False)}

    nativepet_predictions = nativepet_model(systems, outputs)
    nativepet_loss = (
        nativepet_predictions["energy"].block().values.sum()
    )  # Artificial loss
    nativepet_loss.backward()

    pet_predictions = pet_model(systems, outputs)
    pet_loss = pet_predictions["energy"].block().values.sum()  # Artificial loss
    pet_loss.backward()

    nativepet_params = dict(nativepet_model.named_parameters())
    pet_params = dict(pet_model.named_parameters())

    for key, nativepet_param in nativepet_params.items():
        if "last_layers" in key:
            continue
        pet_key = "pet." + key
        if "heads" in key:
            pet_key = pet_key.replace(".energy", "")
        if pet_key in pet_params:
            pet_param = pet_params.pop(pet_key)
            torch.testing.assert_close(nativepet_param.grad, pet_param.grad)


def test_forces_loss_grads_compatibility():
    """Tests that the gradients of the loss produced by PET and NativePET
    models are the same"""
    nativepet_model, pet_model, systems = get_test_environment()
    system = systems[0]

    outputs = {"energy": ModelOutput(per_atom=False)}

    nativepet_predictions = nativepet_model([system], outputs)

    nativepet_gradients = -torch.autograd.grad(
        nativepet_predictions["energy"].block().values[0][0],
        system.positions,
        torch.ones_like(nativepet_predictions["energy"].block().values[0][0]),
        create_graph=True,
        retain_graph=True,
    )[0]

    nl_options = nativepet_model.requested_neighbor_lists()[0]
    batch_dict = systems_to_batch_dict(
        [system], nl_options, nativepet_model.atomic_types, None
    )
    x = batch_dict["x"]
    x.requires_grad_(True)

    pet_predictions = pet_model.pet(batch_dict)["prediction"]

    pet_grads_wrt_x = torch.autograd.grad(
        pet_predictions,
        x,
        grad_outputs=torch.ones_like(pet_predictions),
        create_graph=True,
        retain_graph=True,
    )[0]

    neighbors_index = batch_dict["neighbors_index"]  # .transpose(0, 1)
    neighbors_pos = batch_dict["neighbors_pos"]
    grads_messaged = pet_grads_wrt_x[neighbors_index, neighbors_pos]
    pet_grads_wrt_x[batch_dict["mask"]] = 0.0

    grads_messaged[batch_dict["mask"]] = 0.0
    first = pet_grads_wrt_x.sum(dim=1)
    second = grads_messaged.sum(dim=1)
    pet_gradients = first - second

    nativepet_loss = nativepet_gradients.sum()
    nativepet_loss.backward()

    pet_loss = pet_gradients.sum()
    pet_loss.backward()

    nativepet_params = dict(nativepet_model.named_parameters())
    pet_params = dict(pet_model.named_parameters())

    for key, nativepet_param in nativepet_params.items():
        if "last_layers" in key:
            continue
        pet_key = "pet." + key
        if "heads" in key:
            pet_key = pet_key.replace(".energy", "")
        if pet_key in pet_params:
            pet_param = pet_params.pop(pet_key)
            torch.testing.assert_close(nativepet_param.grad, pet_param.grad)


def test_last_layer_features_compatibility():
    nativepet_model, pet_model, systems = get_test_environment()
    system = systems[0]
    outputs = {
        "energy": ModelOutput(per_atom=False),
        "mtt::aux::energy_last_layer_features": ModelOutput(per_atom=False),
    }

    nativepet_predictions = nativepet_model([system], outputs)
    pet_predictions = pet_model([system], outputs)

    nativepet_last_layer_features = nativepet_predictions[
        "mtt::aux::energy_last_layer_features"
    ]

    pet_last_layer_features = pet_predictions["mtt::aux::energy_last_layer_features"]

    assert metatensor.torch.allclose(
        nativepet_last_layer_features, pet_last_layer_features, atol=1e-6
    )


def test_pet_mad_model_compatibility(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    path = "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt"

    if urlparse(path).scheme:
        path, _ = urlretrieve(path)

    pet_model = PET.load_checkpoint(path).eval()

    nativepet_checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    nativepet_checkpoint = convert_checkpoint_from_legacy_pet(nativepet_checkpoint)
    torch.save(
        nativepet_checkpoint,
        "nativepet_checkpoint.ckpt",
    )
    nativepet_model = NativePET.load_checkpoint("nativepet_checkpoint.ckpt").eval()

    systems_1 = read_systems(DATASET_PATH)[:5]
    systems_2 = read_systems(DATASET_WITH_FORCES_PATH)[:5]
    systems = systems_1 + systems_2
    for system in systems:
        system.positions.requires_grad_(True)
        get_system_with_neighbor_lists(
            system, nativepet_model.requested_neighbor_lists()
        )
    systems = [system.to(torch.float32) for system in systems]

    outputs = {"energy": ModelOutput(per_atom=False)}

    nativepet_predictions = nativepet_model(systems, outputs)
    pet_predictions = pet_model(systems, outputs)

    nativepet_gradients = compute_gradient(
        nativepet_predictions["energy"].block().values,
        [system.positions for system in systems],
        is_training=True,
    )

    pet_gradients = compute_gradient(
        pet_predictions["energy"].block().values,
        [system.positions for system in systems],
        is_training=True,
    )

    torch.testing.assert_close(
        nativepet_predictions["energy"].block().values,
        pet_predictions["energy"].block().values,
    )

    torch.testing.assert_close(nativepet_gradients, pet_gradients)
