def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 Trainer checkpoint to v2.

    v2 added the ``distributed``, ``distributed_port`` and ``num_workers``
    training hypers. A missing ``distributed`` or ``num_workers`` means
    automatic behavior, so only the port needs a value.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["distributed_port"] = checkpoint["train_hypers"].get(
        "distributed_port", 39591
    )
