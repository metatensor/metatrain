import logging
from typing import Dict, List, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from metatensor.learn.data.dataset import _BaseDataset
from metatensor.torch.atomistic import ModelCapabilities

from ...utils.composition import calculate_composition_weights
from ...utils.data import check_datasets, get_all_targets
from .model import DEFAULT_HYPERS
from .pet.models import PET, PET_energy_force
from .pet.utils.augmentation import apply_random_augmentation
from .pet.utils.dataloader import dataloader
from .pet.utils.jax_batch import calculate_padding_sizes, jax_structures_to_batch
from .pet.utils.jax_structure import structure_to_jax
from .pet.utils.mts_to_structure import mts_to_structure


logger = logging.getLogger(__name__)


def train(
    train_datasets: List[Union[_BaseDataset, torch.utils.data.Subset]],
    validation_datasets: List[Union[_BaseDataset, torch.utils.data.Subset]],
    requested_capabilities: ModelCapabilities,
    hypers: Dict = DEFAULT_HYPERS,
    continue_from: Optional[str] = None,
    output_dir: str = ".",
    device_str: str = "cpu",
):
    logger.info(
        "This is a JAX version of the PET architecture. "
        "It does not support message passing yet."
    )

    # Random seed
    logger.warn(
        "The random seed is not being set from outside, but it is hardcoded for now."
    )
    key = jax.random.PRNGKey(1337)

    # Device
    if device_str == "gpu":
        device_str = "cuda"
    jax.config.update("jax_platform_name", device_str)
    logger.info(
        "Running on device "
        f"{list(jnp.array([1, 2, 3]).addressable_data(0).devices())[0]}"
    )

    # Dtype
    if torch.get_default_dtype() == torch.float64:
        jax.config.update("jax_enable_x64", True)
    elif torch.get_default_dtype() == torch.float32:
        pass
    else:
        raise ValueError(f"Unsupported dtype {torch.get_default_dtype()} in PET-JAX.")

    if len(train_datasets) != 1:
        raise NotImplementedError(
            "Only one training dataset is supported in PET-JAX for the moment."
        )
    if len(validation_datasets) != 1:
        raise NotImplementedError(
            "Only one validation dataset is supported in PET-JAX for the moment."
        )

    if continue_from is not None:
        raise NotImplementedError(
            "Continuing from a previous run is not supported yet in PET-JAX."
        )
    model_capabilities = requested_capabilities
    # TODO: implement restarting

    # Perform checks on the datasets:
    logger.info("Checking datasets for consistency")
    check_datasets(
        train_datasets,
        validation_datasets,
        model_capabilities,
    )

    # Check capabilities:
    if len(model_capabilities.outputs) != 1:
        raise NotImplementedError(
            "Only one output is supported in PET-JAX for the moment."
        )
    if next(iter(model_capabilities.outputs.values())).quantity != "energy":
        raise NotImplementedError(
            "Only energy outputs are supported in PET-JAX for the moment."
        )

    # Extract whether we're also training on forces
    do_forces = next(iter(train_datasets[0]))[1].block(0).has_gradient("positions")

    # Calculate and set the composition weights for all targets:
    logger.info("Calculating composition weights")
    target_name = next(iter(model_capabilities.outputs.keys()))
    train_datasets_with_target = []
    for dataset in train_datasets:
        if target_name in get_all_targets(dataset):
            train_datasets_with_target.append(dataset)
    if len(train_datasets_with_target) == 0:
        raise ValueError(
            f"Target {target_name} in the model's new capabilities is not "
            "present in any of the training datasets."
        )
    composition_weights = calculate_composition_weights(
        train_datasets_with_target, target_name
    )
    composition_weights_jax = jnp.array(composition_weights.numpy())

    # Extract the training and validation sets from metatensor format
    cutoff = hypers["model"]["cutoff"]
    training_set = [
        mts_to_structure(
            structure,
            float(targets.block().values),
            (
                targets.block().gradient("positions").values.reshape(-1, 3).numpy()
                if do_forces
                else np.zeros((0, 3))
            ),
            cutoff,
        )
        for structure, targets in train_datasets[0]
    ]
    valid_set = [
        mts_to_structure(
            structure,
            float(targets.block().values),
            (
                targets.block().gradient("positions").values.reshape(-1, 3).numpy()
                if do_forces
                else np.zeros((0, 3))
            ),
            cutoff,
        )
        for structure, targets in validation_datasets[0]
    ]

    def loss_fn(model, structures, n_edges_per_node, do_forces, force_weight, key):
        predictions = model(structures, n_edges_per_node, is_training=True, key=key)
        loss = jnp.sum((predictions["energies"] - structures.energies) ** 2)
        if do_forces:
            loss += force_weight * jnp.sum(
                (predictions["forces"] - structures.forces) ** 2
            )
        return loss

    grad_loss_fn = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def train_step(
        model,
        structures,
        n_edges_per_node_array,
        optimizer,
        opt_state,
        do_forces,
        force_weight,
        key,
    ):
        n_edges_per_node = len(n_edges_per_node_array)
        loss, grads = grad_loss_fn(
            model, structures, n_edges_per_node, do_forces, force_weight, key
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    # Initialize the model
    all_species = jnp.array(model_capabilities.species)
    if do_forces:
        print("hello")
        model = PET_energy_force(
            all_species, hypers["model"], composition_weights_jax, key=key
        )
    else:
        model = PET(all_species, hypers["model"], composition_weights_jax, key=key)

    training_hypers = hypers["training"]
    learning_rate = training_hypers["learning_rate"]
    force_weight = 1.0
    num_epochs = training_hypers["num_epochs"]
    batch_size = training_hypers["batch_size"]
    num_warmup_steps = training_hypers["num_warmup_steps"]

    schedule = optax.linear_schedule(0.0, learning_rate, num_warmup_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(learning_rate=schedule),
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def _evaluate_model(model, jax_batch, n_edges_per_node_array):
        n_edges_per_node = len(n_edges_per_node_array)
        return model(jax_batch, n_edges_per_node, is_training=False)

    def evaluate_model(model, dataset, do_forces):
        energy_sse = 0.0
        energy_sae = 0.0
        if do_forces:
            force_sse = 0.0
            force_sae = 0.0
            number_of_forces = 0
        for batch in dataloader(dataset, batch_size, shuffle=False):
            jax_batch = jax_structures_to_batch(
                [structure_to_jax(structure) for structure in batch]
            )
            n_nodes, n_edges, n_edges_per_node = calculate_padding_sizes(jax_batch)
            # TODO: pad the batch
            # jax_batch = pad_batch(jax_batch, n_nodes, n_edges)
            predictions = _evaluate_model(
                model, jax_batch, jnp.zeros((n_edges_per_node,))
            )
            energy_sse += jnp.sum((predictions["energies"] - jax_batch.energies) ** 2)
            energy_sae += jnp.sum(jnp.abs(predictions["energies"] - jax_batch.energies))
            if do_forces:
                force_sse += jnp.sum((predictions["forces"] - jax_batch.forces) ** 2)
                force_sae += jnp.sum(jnp.abs(predictions["forces"] - jax_batch.forces))
                number_of_forces += 3 * len(jax_batch.forces)
        energy_mse = energy_sse / len(dataset)
        energy_mae = energy_sae / len(dataset)
        energy_rmse = jnp.sqrt(energy_mse)
        result_dict = {}
        result_dict["energy_rmse"] = energy_rmse
        result_dict["energy_mae"] = energy_mae
        if do_forces:
            force_mse = force_sse / number_of_forces
            force_mae = force_sae / number_of_forces
            force_rmse = jnp.sqrt(force_mse)
            result_dict["force_rmse"] = force_rmse
            result_dict["force_mae"] = force_mae
        return result_dict

    train_metrics = evaluate_model(model, training_set, do_forces)
    valid_metrics = evaluate_model(model, valid_set, do_forces)

    if do_forces:
        print(
            f"Epoch 0 | Train loss:    N/A    | Train energy RMSE: {train_metrics['energy_rmse']:8.3e} | Train force RMSE: {train_metrics['force_rmse']:8.3e} | Valid energy RMSE: {valid_metrics['energy_rmse']:8.3e} | Valid force RMSE: {valid_metrics['force_rmse']:8.3e}"  # noqa: E501
        )
        print(
            f"Epoch 0 | Train loss:    N/A    | Train energy MAE:  {train_metrics['energy_mae']:8.3e} | Train force MAE:  {train_metrics['force_mae']:8.3e} | Valid energy MAE:  {valid_metrics['energy_mae']:8.3e} | Valid force MAE:  {valid_metrics['force_mae']:8.3e}"  # noqa: E501
        )
        print()
    else:
        print(
            f"Epoch 0 | Train loss:    N/A    | Train energy RMSE: {train_metrics['energy_rmse']:8.3e} | Valid energy RMSE: {valid_metrics['energy_rmse']:8.3e}"  # noqa: E501
        )
        print(
            f"Epoch 0 | Train loss:    N/A    | Train energy MAE:  {train_metrics['energy_mae']:8.3e} | Valid energy MAE:  {valid_metrics['energy_mae']:8.3e}"  # noqa: E501
        )
        print()

    key = jax.random.PRNGKey(0)

    for epoch in range(1, num_epochs):
        train_loss = 0.0
        for batch in dataloader(training_set, batch_size, shuffle=True):
            jax_batch = jax_structures_to_batch(
                [
                    structure_to_jax(apply_random_augmentation(structure))
                    for structure in batch
                ]
            )
            n_nodes, n_edges, n_edges_per_node = calculate_padding_sizes(jax_batch)
            # TODO: pad the batch
            # jax_batch = pad_batch(jax_batch, n_nodes, n_edges)
            subkey, key = jax.random.split(key)
            loss, model, opt_state = train_step(
                model,
                jax_batch,
                jnp.zeros((n_edges_per_node,)),
                optimizer,
                opt_state,
                do_forces,
                force_weight,
                subkey,
            )
            train_loss += loss

        if epoch % training_hypers["log_interval"] == 0:
            train_metrics = evaluate_model(model, training_set, do_forces)
            valid_metrics = evaluate_model(model, valid_set, do_forces)
            if do_forces:
                print(
                    f"Epoch {epoch} | Train loss: {train_loss:8.3e} | Train energy RMSE: {train_metrics['energy_rmse']:8.3e} | Train force RMSE: {train_metrics['force_rmse']:8.3e} | Valid energy RMSE: {valid_metrics['energy_rmse']:8.3e} | Valid force RMSE: {valid_metrics['force_rmse']:8.3e}"  # noqa: E501
                )
                print(
                    f"Epoch {epoch} | Train loss: {train_loss:8.3e} | Train energy MAE:  {train_metrics['energy_mae']:8.3e} | Train force MAE:  {train_metrics['force_mae']:8.3e} | Valid energy MAE:  {valid_metrics['energy_mae']:8.3e} | Valid force MAE:  {valid_metrics['force_mae']:8.3e}"  # noqa: E501
                )
                print()
            else:
                print(
                    f"Epoch {epoch} | Train loss: {train_loss:8.3e} | Train energy RMSE: {train_metrics['energy_rmse']:8.3e} | Valid energy RMSE: {valid_metrics['energy_rmse']:8.3e}"  # noqa: E501
                )
                print(
                    f"Epoch {epoch} | Train loss: {train_loss:8.3e} | Train energy MAE:  {train_metrics['energy_mae']:8.3e} | Valid energy MAE:  {valid_metrics['energy_mae']:8.3e}"  # noqa: E501
                )
                print()
