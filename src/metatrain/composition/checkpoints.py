def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 Trainer checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["distributed"] = checkpoint["train_hypers"].get(
        "distributed", False
    )
    checkpoint["train_hypers"]["distributed_port"] = checkpoint["train_hypers"].get(
        "distributed_port", 39591
    )
    checkpoint["train_hypers"]["num_workers"] = checkpoint["train_hypers"].get(
        "num_workers", None
    )
