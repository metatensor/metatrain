import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from metatrain.utils.additive import remove_additive
from metatrain.utils.augmentation import RotationalAugmenter
from metatrain.utils.data import (
    CombinedDataLoader,
    Dataset,
    _is_disk_dataset,
    collate_fn,
)
from metatrain.utils.distributed.distributed_data_parallel import (
    DistributedDataParallel,
)
from metatrain.utils.distributed.slurm import DistributedEnvironment
from metatrain.utils.evaluate_model import evaluate_model, _prepare_system
from metatrain.utils.external_naming import to_external_name
from metatrain.utils.io import check_file_extension
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
from metatrain.utils.loss import TensorMapDictLoss
from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.scaler import remove_scale
from metatrain.utils.transfer import (
    systems_and_targets_to_device,
    systems_and_targets_to_dtype,
)
from metatrain.utils.output_gradient import compute_gradient
from .model import PET
from .modules.finetuning import apply_finetuning_strategy

# CHANGE: IMPORT THE loss function FROM DOSutils
from .DOSutils import get_dynamic_shift_agnostic_mse

def get_scheduler(optimizer, train_hypers):
    def func_lr_scheduler(epoch):
        if epoch < train_hypers["num_epochs_warmup"]:
            return epoch / train_hypers["num_epochs_warmup"]
        delta = epoch - train_hypers["num_epochs_warmup"]
        num_blocks = delta // train_hypers["scheduler_patience"]
        return 0.5 ** (num_blocks)

    scheduler = LambdaLR(optimizer, func_lr_scheduler)
    return scheduler


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = train_hypers
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.epoch = None
        self.best_metric = None
        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None

    def train(
        self,
        model: PET,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        assert dtype in PET.__supported_dtypes__
        is_distributed = self.hypers["distributed"]
        if is_distributed:
            distr_env = DistributedEnvironment(self.hypers["distributed_port"])
            torch.distributed.init_process_group(backend="nccl")
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    " If you want to run distributed training with SOAP-BPNN, please "
                    "set `device` to cuda."
                )
            # the calculation of the device number works both when GPUs on different
            # processes are not visible to each other and when they are
            device_number = distr_env.local_rank % torch.cuda.device_count()
            device = torch.device("cuda", device_number)
        else:
            device = devices[
                0
            ]  # only one device, as we don't support multi-gpu for now

        if is_distributed:
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logging.info(f"Training on device {device} with dtype {dtype}")

        # Calculate the neighbor lists in advance (in particular, this
        # needs to happen before the additive models are trained, as they
        # might need them):
        logging.info("Calculating neighbor lists for the datasets")
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        for dataset in train_datasets + val_datasets:
            # If the dataset is a disk dataset, the NLs are already attached, we will
            # just check the first system
            if _is_disk_dataset(dataset):
                system = dataset[0]["system"]
                for options in requested_neighbor_lists:
                    if options not in system.known_neighbor_lists():
                        raise ValueError(
                            "The requested neighbor lists are not attached to the "
                            f"system. Neighbor list {options} is missing from the "
                            "first system in the disk dataset. Make sure you save "
                            "the neighbor lists in the systems when saving the dataset."
                        )
            else:
                for sample in dataset:
                    system = sample["system"]
                    # The following line attaches the neighbors lists to the system,
                    # and doesn't require to reassign the system to the dataset:
                    get_system_with_neighbor_lists(system, requested_neighbor_lists)

        # Apply fine-tuning strategy if provided
        if self.hypers["finetune"]:
            model = apply_finetuning_strategy(model, self.hypers["finetune"])

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)
        # The additive models of the SOAP-BPNN are always in float64 (to avoid
        # numerical errors in the composition weights, which can be very large).
        for additive_model in model.additive_models:
            additive_model.to(dtype=torch.float64)

        # CHANGE: Do not use additive and scaler models 
        # logging.info("Calculating composition weights")
        # model.additive_models[0].train_model(  # this is the composition model
        #     train_datasets,
        #     model.additive_models[1:],
        #     self.hypers["fixed_composition_weights"],
        # )

        # if self.hypers["scale_targets"]:
        #     logging.info("Calculating scaling weights")
        #     model.scaler.train_model(
        #         train_datasets, model.additive_models, treat_as_additive=True
        #     )

        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device])

        # CHANGE: Include a permanent set that is included in every training batch
        if self.hypers['use_permanent']:
            n_samples = self.hypers['use_permanent']['n_samples']
            n_train = len(train_datasets[0])
            train_systems = []
            permanent_systems = []
            train_targets = {}
            permanent_targets = {}
            keys = ["mtt::dos", "mtt::mask", "mtt::gapdos", "mtt::gap_force"] # WARNING: Keys are hardcoded for DOS so that the mask remains identifiable
            for key in keys:
                train_targets[key] = []
                permanent_targets[key] = []
            for i in range(0, n_train, 1):
                data_i = train_datasets[0][i]
                if i < (n_train - n_samples):     
                    train_systems.append(data_i.system)
                    for key in keys:

                        train_targets[key].append(data_i[key])
                else:
                    permanent_systems.append(data_i.system)
                    for key in keys:
                        permanent_targets[key].append(data_i[key])                    
            train_datasets = [Dataset.from_dict({"system": train_systems, **train_targets})]
            permanent_datasets = [Dataset.from_dict({"system": permanent_systems, **permanent_targets})]

        logging.info("Setting up data loaders")

        if is_distributed:
            train_samplers = [
                DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                )
                for train_dataset in train_datasets
            ]
            val_samplers = [
                DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                for val_dataset in val_datasets
            ]
            # CHANGE: Include the permanent dataset but that should not be distributed
            if self.hypers['use_permanent']: # CHANGE: Include the permanent dataset
                permanent_samplers = [None] * len(permanent_datasets)
        else:
            train_samplers = [None] * len(train_datasets)
            val_samplers = [None] * len(val_datasets)
            if self.hypers['use_permanent']: # CHANGE: Include the permanent dataset
                permanent_samplers = [None] * len(permanent_datasets)

        # Create dataloader for the training datasets:
        train_dataloaders = []
        for dataset, sampler in zip(train_datasets, train_samplers):
            train_dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.hypers["batch_size"],
                    sampler=sampler,
                    shuffle=(
                        sampler is None
                    ),  # the sampler takes care of this (if present)
                    drop_last=(
                        sampler is None
                    ),  # the sampler takes care of this (if present)
                    collate_fn=collate_fn,
                )
            )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = []
        for dataset, sampler in zip(val_datasets, val_samplers):
            val_dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.hypers["batch_size"],
                    sampler=sampler,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        # CHANGE: Include the permanent dataset
        if self.hypers['use_permanent']:
            permanent_dataloaders = []
            for dataset, sampler in zip(permanent_datasets, permanent_samplers):
                permanent_dataloaders.append(
                    DataLoader(
                        dataset=dataset,
                        batch_size=self.hypers["batch_size"],
                        sampler=sampler,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=collate_fn,
                    )
                )
            permanent_dataloader = CombinedDataLoader(permanent_dataloaders, shuffle=False)
            logging.info("Setting up Permanent Dataset")  

        train_targets = (model.module if is_distributed else model).dataset_info.targets
        print ("Train targets:")
        print (train_targets)
        try:
            del train_targets["mtt::mask"] # CHANGE: Removing the mask from the targets, the mask is not a target for the model to predict
        except:
            pass
        outputs_list = []
        for target_name, target_info in train_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")

        # Create a loss weight dict:
        loss_weights_dict = {}
        for output_name in outputs_list:
            loss_weights_dict[output_name] = (
                self.hypers["loss"]["weights"][
                    to_external_name(output_name, train_targets)
                ]
                if to_external_name(output_name, train_targets)
                in self.hypers["loss"]["weights"]
                else 1.0
            )
        loss_weights_dict_external = {
            to_external_name(key, train_targets): value
            for key, value in loss_weights_dict.items()
        }
        loss_hypers = copy.deepcopy(self.hypers["loss"])
        loss_hypers["weights"] = loss_weights_dict
        logging.info(f"Training with loss weights: {loss_weights_dict_external}")

        # Create a loss function:
        loss_fn = TensorMapDictLoss(
            **loss_hypers,
        )

        if self.hypers["weight_decay"] is not None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.hypers["learning_rate"],
                weight_decay=self.hypers["weight_decay"],
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.hypers["learning_rate"]
            )
        # Disable optimizer loading
        # if self.optimizer_state_dict is not None and not self.hypers["finetune"]:
        #     # try to load the optimizer state dict, but this is only possible
        #     # if there are no new targets in the model (new parameters)
        #     if not (model.module if is_distributed else model).has_new_targets:
        #         optimizer.load_state_dict(self.optimizer_state_dict)

        lr_scheduler = get_scheduler(optimizer, self.hypers)

        if self.scheduler_state_dict is not None and not self.hypers["finetune"]:
            # same as the optimizer, try to load the scheduler state dict
            if not (model.module if is_distributed else model).has_new_targets:
                lr_scheduler.load_state_dict(self.scheduler_state_dict)

        per_structure_targets = self.hypers["per_structure_targets"]

        # Log the initial learning rate:
        old_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Base learning rate: {self.hypers['learning_rate']}")
        logging.info(f"Initial learning rate: {old_lr}")

        rotational_augmenter = RotationalAugmenter(train_targets)
        # Force restart
        start_epoch = 0 #if self.epoch is None else self.epoch + 1
        self.best_metric = float("inf")
        # CHANGE: Define the coefficients for the finite diference scheme
        interval = 0.05
        t4 = (torch.tensor([1/4, -4/3, 3., -4. , 25/12]).to(device)/interval).unsqueeze(dim = (0)).unsqueeze(dim = (0)).float()

        # Train the model:
        if self.best_metric is None:
            self.best_metric = float("inf")
        logging.info("Starting training")
        epoch = start_epoch
        train_print = True
        val_print = True
        for epoch in range(start_epoch, start_epoch + self.hypers["num_epochs"]):
            if is_distributed:
                for train_sampler in train_samplers:
                    train_sampler.set_epoch(epoch)
            # CHANGE: Not using the default Accumulators because they do not support custom loss functions yet
            # train_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            # val_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            # if self.hypers["log_mae"]:
            #     train_mae_calculator = MAEAccumulator(
            #         self.hypers["log_separate_blocks"]
            #     )
            #     val_mae_calculator = MAEAccumulator(self.hypers["log_separate_blocks"])

            train_loss = 0.0
            train_count = 0.0 # CHANGE: Added to count the number of training samples
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets = batch
                systems, targets = rotational_augmenter.apply_random_augmentations(
                    systems, targets
                )
                systems, targets = systems_and_targets_to_device(
                    systems, targets, device
                )
                # CHANGE: Remove additive and scaler models 
                # for additive_model in (
                #     model.module if is_distributed else model
                # ).additive_models:
                #     targets = remove_additive(
                #         systems, targets, additive_model, train_targets
                #     )
                # targets = remove_scale(
                #     targets, (model.module if is_distributed else model).scaler
                # )
                systems, targets = systems_and_targets_to_dtype(systems, targets, dtype)
                new_systems = []
                for system in systems:
                    new_system, strain = _prepare_system(
                        system,
                        positions_grad= True,
                        strain_grad= False,
                        check_consistency=False,
                    )
                    new_systems.append(new_system)
                systems = new_systems
                # CHANGE: Extract relevant quantities from the targets
                target_dos_batch, gap_batch, gap_force_batch = targets['mtt::gapdos'], targets['mtt::gap'], targets['mtt::gap_force']
                predictions = evaluate_model(
                    model,
                    systems,
                    # {key: train_targets[key] for key in targets.keys()},
                    {key: train_targets[key] for key in train_targets.keys()}, # CHANGE: Use the train_targets keys instead as mask is not a target
                    is_training=True,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                # targets = average_by_num_atoms(targets, systems, per_structure_targets) # By default the targets are already averaged by the number of atoms
                
                # CHANGE: DOS Training loop
                dos_predictions = predictions['mtt::gapdos'][0].values
                bandgap_predictions = model.bandgap_layer(dos_predictions)
                gap_force_predictions = torch.vstack(compute_gradient(bandgap_predictions, [system.positions for system in systems], is_training=True))
                bandgap_target = gap_batch[0].values.reshape(-1,1).to(dtype)
                bandgap_loss = torch.nn.functional.mse_loss(bandgap_predictions, bandgap_target)
                gap_force_target = torch.squeeze(gap_force_batch[0].values.to(dtype))
                gap_force_loss = torch.nn.functional.mse_loss(gap_force_predictions, gap_force_target) # Use 1:1 for now
                if train_print:
                    print ("Train print:")
                    print (gap_force_predictions.shape)
                    print (gap_force_target.shape)
                    print (gap_force_loss)
                    train_print = False

                total_loss = bandgap_loss + gap_force_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.hypers["grad_clip_norm"]
                )
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # train_loss_batch = loss_fn(predictions, targets)
                # train_loss_batch.backward()
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(), self.hypers["grad_clip_norm"]
                # )
                optimizer.step()
                total_loss = (total_loss * len(bandgap_target)).detach() # CHANGE: We need to multiply the loss by the number of samples in the batch to get the correct loss value
                train_count += len(bandgap_target) # CHANGE: Update the training count
                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(total_loss)
                train_loss += total_loss.item()
                # CHANGE: Do not use the default calculators
                # train_rmse_calculator.update(predictions, targets)
                # if self.hypers["log_mae"]:
                #     train_mae_calculator.update(predictions, targets)
            # finalized_train_info = train_rmse_calculator.finalize(
            #     not_per_atom=["positions_gradients"] + per_structure_targets,
            #     is_distributed=is_distributed,
            #     device=device,
            # )
            # if self.hypers["log_mae"]:
            #     finalized_train_info.update(
            #         train_mae_calculator.finalize(
            #             not_per_atom=["positions_gradients"] + per_structure_targets,
            #             is_distributed=is_distributed,
            #             device=device,
            #         )
            #     )
                # CHANGE: Use permanent dataset
                if self.hypers['use_permanent']:
                    pass
            train_loss /= train_count
            val_loss = 0.0
            val_force_loss = 0.0
            val_count = 0.0 # CHANGE: Added to count the number of validation samples
            # val_predictions = []
            for batch in val_dataloader:
                # CHANGE: Updated validation loop to use the new loss function 
                systems, targets = batch
                systems, targets = systems_and_targets_to_device(
                        systems, targets, device
                )
                systems, targets = systems_and_targets_to_dtype(systems, targets, dtype)
                new_systems = []
                for system in systems:
                    new_system, strain = _prepare_system(
                        system,
                        positions_grad= True,
                        strain_grad= False,
                        check_consistency=False,
                    )
                    new_systems.append(new_system)
                systems = new_systems

                target_dos_batch, gap_batch, gap_force_batch = targets['mtt::gapdos'], targets['mtt::gap'], targets['mtt::gap_force']

                predictions = evaluate_model(
                    model,
                    systems,
#                    {key: train_targets[key] for key in targets.keys()}, 
                    {key: train_targets[key] for key in train_targets.keys()},
                    is_training=False,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                # targets = average_by_num_atoms(targets, systems, per_structure_targets)
                dos_predictions = predictions['mtt::gapdos'][0].values
                bandgap_predictions = model.bandgap_layer(dos_predictions)
                gap_force_predictions = torch.vstack(compute_gradient(bandgap_predictions, [system.positions for system in systems], is_training=False))
                bandgap_target = gap_batch[0].values.reshape(-1,1).to(dtype)
                bandgap_loss = torch.nn.functional.mse_loss(bandgap_predictions, bandgap_target)
                gap_force_target = torch.squeeze(gap_force_batch[0].values.to(dtype))
                gap_force_loss = torch.nn.functional.mse_loss(gap_force_predictions, gap_force_target) # Use 1:1 for now
                if val_print:
                    print ("Val print:")
                    print (gap_force_predictions.shape)
                    print (gap_force_target.shape)
                    print (gap_force_loss)
                    val_print = False
                total_loss = bandgap_loss + gap_force_loss
                val_loss_batch = (total_loss * len(bandgap_target)).detach()# CHANGE: We need to multiply the loss by the number of samples in the batch to get the correct loss value
                val_force_loss_batch = (gap_force_loss * len(bandgap_target)).detach()
                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(val_loss_batch)
                val_loss += val_loss_batch.item()
                val_force_loss += val_force_loss_batch.item()
                val_count += len(bandgap_target) # CHANGE: Update the validation count
            val_loss /= val_count
            val_force_loss /= val_count
            # val_predictions = torch.vstack(val_predictions)
                # CHANGE: Not using the default calculators
                # val_rmse_calculator.update(predictions, targets)
                # if self.hypers["log_mae"]:
                #     val_mae_calculator.update(predictions, targets)
            # CHANGE: Not using the default calculators
            # finalized_val_info = val_rmse_calculator.finalize(
            #     not_per_atom=["positions_gradients"] + per_structure_targets,
            #     is_distributed=is_distributed,
            #     device=device,
            # )
            # if self.hypers["log_mae"]:
            #     finalized_val_info.update(
            #         val_mae_calculator.finalize(
            #             not_per_atom=["positions_gradients"] + per_structure_targets,
            #             is_distributed=is_distributed,
            #             device=device,
            #         )
            #     )

            # Now we log the information:
            # Change: Remove information other than the loss
            finalized_train_info = {"loss": train_loss} # , **finalized_train_info}
            finalized_val_info = {
                "loss": val_loss
            }
                # **finalized_val_info,
            # }

            if epoch == start_epoch:
                # CHANGE: Remove Scaler
                # scaler_scales = (
                #     model.module if is_distributed else model
                # ).scaler.get_scales_dict()
                metric_logger = MetricLogger(
                    log_obj=ROOT_LOGGER,
                    dataset_info=(
                        model.module if is_distributed else model
                    ).dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                    # scales={
                    #     key: (
                    #         scaler_scales[key.split(" ")[0]]
                    #         if ("MAE" in key or "RMSE" in key)
                    #         else 1.0
                    #     )
                    #     for key in finalized_train_info.keys()
                    # },
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                    rank=rank,
                )
                print ("Validation Force Loss:  ", val_force_loss)

            lr_scheduler.step()
            new_lr = lr_scheduler.get_last_lr()[0]
            if new_lr != old_lr:
                if new_lr < 1e-7:
                    logging.info("Learning rate is too small, stopping training")
                    break
                else:
                    if epoch >= self.hypers["num_epochs_warmup"]:
                        logging.info(
                            f"Changing learning rate from {old_lr} to {new_lr}"
                        )
                    elif epoch == self.hypers["num_epochs_warmup"] - 1:
                        logging.info(
                            "Finished warm-up. "
                            f"Now training with learning rate {new_lr}"
                        )
                    else:  # epoch < self.hypers["num_epochs_warmup"] - 1:
                        pass  # we don't clutter the log at every warm-up step
                    old_lr = new_lr

            # val_metric = get_selected_metric(
            #     finalized_val_info, self.hypers["best_model_metric"]
            # )
            val_metric = val_loss  # CHANGE: Use the validation loss as the metric
            logging.info("Current Best Validation Metric: %s", self.best_metric)
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                self.best_model_state_dict = copy.deepcopy(
                    (model.module if is_distributed else model).state_dict()
                )
                self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

            if epoch % self.hypers["checkpoint_interval"] == 0:
                if is_distributed:
                    torch.distributed.barrier()
                self.optimizer_state_dict = optimizer.state_dict()
                self.scheduler_state_dict = lr_scheduler.state_dict()
                self.epoch = epoch
                if rank == 0:
                    self.save_checkpoint(
                        (model.module if is_distributed else model),
                        Path(checkpoint_dir) / f"model_{epoch}.ckpt",
                    )
                    # torch.save(val_predictions, Path(checkpoint_dir) / f"val_predictions_{epoch}.pt") # CHANGE: Save the validation predictions

        # prepare for the checkpoint that will be saved outside the function
        self.epoch = epoch
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()

    def save_checkpoint(self, model, path: Union[str, Path]):
        checkpoint = {
            "architecture_name": "pet",
            "model_data": {
                "model_hypers": model.hypers,
                "dataset_info": model.dataset_info,
            },
            "model_state_dict": model.state_dict(),
            "train_hypers": self.hypers,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "best_metric": self.best_metric,
            "best_model_state_dict": self.best_model_state_dict,
            "best_optimizer_state_dict": self.best_optimizer_state_dict,
        }
        torch.save(
            checkpoint,
            check_file_extension(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],  # not used at the moment
        train_hypers: Dict[str, Any],
    ) -> "Trainer":
        epoch = checkpoint["epoch"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        scheduler_state_dict = checkpoint["scheduler_state_dict"]
        best_metric = checkpoint["best_metric"]
        best_model_state_dict = checkpoint["best_model_state_dict"]
        best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]

        # Create the trainer
        trainer = cls(train_hypers)
        trainer.optimizer_state_dict = optimizer_state_dict
        trainer.scheduler_state_dict = scheduler_state_dict
        trainer.epoch = epoch
        trainer.best_metric = best_metric
        trainer.best_model_state_dict = best_model_state_dict
        trainer.best_optimizer_state_dict = best_optimizer_state_dict

        return trainer
