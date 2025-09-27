# ===== Model checkpoint updates =====

# ...

# ===== Trainer checkpoint updates =====


def trainer_update_v1_v2(checkpoint):
    old_loss_hypers = checkpoint["train_hypers"]["loss"].copy()
    dataset_info = checkpoint["model_data"]["dataset_info"]
    new_loss_hypers = {}

    for target_name in dataset_info.targets.keys():
        new_loss_hypers[target_name] = {
            "type": old_loss_hypers["type"],
            "weight": old_loss_hypers["weights"].get(target_name, 1.0),
            "reduction": old_loss_hypers["reduction"],
            "sliding_factor": old_loss_hypers.get("sliding_factor", None),
        }
    checkpoint["train_hypers"]["loss"] = new_loss_hypers


def trainer_update_v2_v3(checkpoint):
    # num_workers=0 means that the main process will do the data loading, which is
    # equivalent to not setting it (this was the behavior before v3)
    checkpoint["train_hypers"]["num_workers"] = 0
