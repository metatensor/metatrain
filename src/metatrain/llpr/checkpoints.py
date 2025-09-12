def model_update_v1_v2(checkpoint):
    print(checkpoint.keys())
    # Here we need to add model_data to the old checkpoint:
    checkpoint["model_data"] = {}
    checkpoint["model_data"]["hypers"] = {
        "ensembles": {
            "num_members": {"energy": 128},
            "means": {"energy": []},
        },
    }
    checkpoint["model_data"]["dataset_info"] = checkpoint["wrapped_model_checkpoint"][
        "model_data"
    ]["dataset_info"]
    return checkpoint
