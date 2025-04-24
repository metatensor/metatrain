from typing import Dict

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def convert_model_state_dict_from_legacy_pet(
    checkpoint: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Converts the model state dict from the metatain.pet format
    to the metatrain.experimental.nativepet format.
    """

    new_model_state_dict: Dict[str, torch.Tensor] = {}
    model_state_dict = checkpoint["model_state_dict"]
    for key, value in model_state_dict.items():
        new_key = key.replace("module.model.model.pet_model.", "")
        if "embedding" in new_key and "gnn_layers" not in new_key:
            new_model_state_dict[new_key] = value
        if "gnn_layers" in new_key and "trans_layer" not in new_key:
            if "r_embedding" in new_key:
                new_key = new_key.replace("r_embedding", "edge_embedder")
            if "central_embedder" in new_key:
                new_key = new_key.replace("central_embedder", "node_embedder")
            new_model_state_dict[new_key] = value

        if "heads" in new_key:
            if "linear" in new_key:
                layer_num = int(new_key.split(".")[1])
                if "bond" in new_key:
                    last_layer_key = f"edge_last_layers.energy.{layer_num}.energy___0"
                else:
                    last_layer_key = f"node_last_layers.energy.{layer_num}.energy___0"
                if "weight" in new_key:
                    last_layer_key += ".weight"
                if "bias" in new_key:
                    last_layer_key += ".bias"
                new_model_state_dict[last_layer_key] = value
            else:
                if "bond" in new_key:
                    new_key = new_key.replace("bond_heads.", "edge_heads.energy.")
                else:
                    new_key = new_key.replace("heads.", "node_heads.energy.")
                new_key = new_key.replace("nn.", "")
                new_model_state_dict[new_key] = value

    new_model_state_dict["scaler.scales"] = torch.tensor([1.0], dtype=torch.float32)
    new_model_state_dict["additive_models.0.dummy_buffer"] = torch.tensor(
        [1.0], dtype=torch.float64
    )

    atomic_types = torch.tensor(checkpoint["dataset_info"].atomic_types)

    samples = Labels(names=["center_type"], values=atomic_types.unsqueeze(1))
    weights = TensorMap(
        keys=Labels.range("_", 1),
        blocks=[
            TensorBlock(
                samples=samples,
                components=[],
                properties=Labels.range("energy", 1),
                values=torch.tensor(
                    checkpoint["self_contributions"], dtype=torch.float64
                ).unsqueeze(1),
            )
        ],
    )

    new_model_state_dict["additive_models.0.energy_composition_buffer"] = (
        metatensor.torch.save_buffer(weights)
    )

    species_to_species_index = torch.full(
        (max(atomic_types) + 1,),
        -1,
        dtype=torch.int64,
    )
    for i, species in enumerate(atomic_types):
        species_to_species_index[species] = i

    new_model_state_dict["species_to_species_index"] = species_to_species_index
    new_model_state_dict["additive_models.0.type_to_index"] = species_to_species_index
    return new_model_state_dict


def convert_model_data_from_legacy_pet(
    checkpoint: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Converts the model data from the metatain.pet format
    to the metatrain.experimental.nativepet format.
    """

    new_model_data: Dict[str, torch.Tensor] = {}
    new_model_data["dataset_info"] = checkpoint["dataset_info"]
    hypers = checkpoint["hypers"]["ARCHITECTURAL_HYPERS"]
    new_model_data["model_hypers"] = {
        "cutoff": hypers["R_CUT"],
        "cutoff_width": hypers["CUTOFF_DELTA"],
        "d_pet": hypers["TRANSFORMER_D_MODEL"],
        "d_head": hypers["HEAD_N_NEURONS"],
        "d_feedforward": hypers["TRANSFORMER_DIM_FEEDFORWARD"],
        "num_heads": hypers["TRANSFORMER_N_HEAD"],
        "num_attention_layers": hypers["N_TRANS_LAYERS"],
        "num_gnn_layers": hypers["N_GNN_LAYERS"],
        "residual_factor": hypers["RESIDUAL_FACTOR"],
        "zbl": hypers["USE_ZBL"],
        "long_range": {
            "enable": False,
            "use_ewald": False,
            "smearing": 1.4,
            "kspace_resolution": 1.33,
            "interpolation_nodes": 5,
        },
    }

    return new_model_data


def convert_train_hypers_from_legacy_pet(
    checkpoint: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Converts the training hypers from the metatain.pet format
    to the metatrain.experimental.nativepet format.
    """
    hypers = checkpoint["hypers"]["FITTING_SCHEME"]
    new_train_hypers: Dict[str, torch.Tensor] = {
        "distributed": False,
        "distributed_port": 39591,
        "batch_size": 16,
        "num_epochs": hypers["EPOCH_NUM"],
        "num_epochs_warmup": hypers["EPOCHS_WARMUP"],
        "learning_rate": hypers["INITIAL_LR"],
        "weight_decay": hypers["WEIGHT_DECAY"] if hypers["USE_WEIGHT_DECAY"] else None,
        "scheduler_patience": hypers["SCHEDULER_STEP_SIZE"],
        "log_interval": 1,
        "checkpoint_interval": hypers["CHECKPOINT_INTERVAL"],
        "scale_targets": False,
        "fixed_composition_weights": {},
        "per_structure_targets": [],
        "log_mae": True,
        "log_separate_blocks": False,
        "best_model_metric": "rmse_prod",
        "loss": {
            "type": "mse",
            "weights": {"energy": hypers["ENERGY_WEIGHT"]},
            "reduction": "mean",
            "sliding_factor": hypers["SLIDING_FACTOR"],
        },
    }

    return new_train_hypers


def convert_checkpoint_from_legacy_pet(
    checkpoint: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Converts the state dict from the metatain.pet format
    to the metatrain.experimental.nativepet format.
    """

    new_checkpoint: Dict[str, torch.Tensor] = {}
    new_checkpoint["architecture_name"] = "experimental.nativepet"
    new_checkpoint["epoch"] = 0
    new_checkpoint["optimizer_state_dict"] = None
    new_checkpoint["scheduler_state_dict"] = None
    new_checkpoint["best_metric"] = float("inf")
    new_checkpoint["best_model_state_dict"] = None
    new_checkpoint["best_optimizer_state_dict"] = None
    new_checkpoint["model_state_dict"] = convert_model_state_dict_from_legacy_pet(
        checkpoint
    )
    new_checkpoint["model_data"] = convert_model_data_from_legacy_pet(checkpoint)
    new_checkpoint["train_hypers"] = convert_train_hypers_from_legacy_pet(checkpoint)

    return new_checkpoint
