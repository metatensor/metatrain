import copy
import logging
import os
import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
from scipy.special import roots_legendre
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataListLoader, DataLoader, DynamicBatchSampler


def get_activation(hypers):
    if hypers.ACTIVATION == "mish":
        return torch.nn.Mish()
    if hypers.ACTIVATION == "silu":
        return torch.nn.SiLU()
    raise ValueError("unknown activation")


def log_epoch_stats(epoch, total_epochs, epoch_stats, learning_rate, energies_key):
    """
    Logs the detailed training and validation statistics at the end of each epoch.

    Parameters are the same as in the previous version, with added validation metrics.
    """

    if "forces" in epoch_stats.keys():
        write_forces_error = True
    else:
        write_forces_error = False
    epoch_stats_log = f"Epoch: {epoch} | LR: {learning_rate:.3e} | "

    train_energies_mae = epoch_stats[energies_key]["train"]["mae"]
    train_energies_rmse = epoch_stats[energies_key]["train"]["rmse"]
    if write_forces_error:
        train_forces_mae = epoch_stats["forces"]["train"]["mae"]
        train_forces_rmse = epoch_stats["forces"]["train"]["rmse"]

    val_energies_mae = epoch_stats[energies_key]["val"]["mae"]
    val_energies_rmse = epoch_stats[energies_key]["val"]["rmse"]
    if write_forces_error:
        val_forces_mae = epoch_stats["forces"]["val"]["mae"]
        val_forces_rmse = epoch_stats["forces"]["val"]["rmse"]

    if energies_key == "energies per structure":
        energies_log_key = ""
    elif energies_key == "energies per atom":
        energies_log_key = "/at."
    else:
        energies_log_key = ""

    # epoch_time = epoch_stats["epoch_time"]
    total_time = epoch_stats["elapsed_time"]
    estimated_remaining_time = epoch_stats["estimated_remaining_time"]

    epoch_stats_log += f"V-E-MAE{energies_log_key} {val_energies_mae:.2e} | "
    epoch_stats_log += f"V-E-RMSE{energies_log_key} {val_energies_rmse:.2e} | "
    if write_forces_error:
        epoch_stats_log += f"V-F-MAE {val_forces_mae:.2e} | "
        epoch_stats_log += f"V-F-RMSE {val_forces_rmse:.2e} | "
    epoch_stats_log += f"T-E-MAE{energies_log_key} {train_energies_mae:.2e} | "
    epoch_stats_log += f"T-E-RMSE{energies_log_key} {train_energies_rmse:.2e} | "
    if write_forces_error:
        epoch_stats_log += f"T-F-MAE {train_forces_mae:.2e} | "
        epoch_stats_log += f"T-F-RMSE {train_forces_rmse:.2e} | "

    epoch_stats_log += f"Time/ETA: {total_time:.2f}/{estimated_remaining_time:.2f} s"
    logging.info(epoch_stats_log)


def get_calc_names(all_completed_calcs, current_name):
    name_to_load = None
    name_of_calculation = current_name
    if name_of_calculation in all_completed_calcs:
        name_to_load = name_of_calculation
        for i in range(100000):
            name_now = name_of_calculation + f"_continuation_{i}"
            if name_now not in all_completed_calcs:
                name_to_save = name_now
                break
            name_to_load = name_now
        name_of_calculation = name_to_save
    return name_to_load, name_of_calculation


def set_reproducibility(random_seed, cuda_deterministic):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    if cuda_deterministic and torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_length(delta):
    return np.sqrt(np.sum(delta * delta))


class ModelKeeper:
    def __init__(self):
        self.best_model = None
        self.best_error = None
        self.best_epoch = None
        self.additional_info = None

    def update(self, model_now, error_now, epoch_now, additional_info=None):
        if (self.best_error is None) or (error_now < self.best_error):
            self.best_error = error_now
            original_device = next(model_now.parameters()).device
            model_now.to("cpu")
            self.best_model = copy.deepcopy(model_now)
            model_now.to(original_device)
            self.best_epoch = epoch_now
            self.additional_info = additional_info


class Accumulator:
    def __init__(self):
        self.values = None

    def update(self, values_now):
        if isinstance(values_now, torch.Tensor):
            values_now = [values_now]

        if self.values is None:
            self.values = [[] for _ in range(len(values_now))]

        for index, value_now in enumerate(values_now):
            if isinstance(value_now, torch.Tensor):
                value_now = value_now.data.cpu().to(torch.float32).numpy()
            self.values[index].append(value_now)

    def consist_of_nones(self, el):
        has_none = False
        has_value = False
        for index in range(len(el)):
            if el[index] is None:
                has_none = True
            else:
                has_value = True
        if has_none and has_value:
            raise ValueError("Some values are None, some are not")
        return has_none

    def flush(self):
        result = []
        for el in self.values:
            if self.consist_of_nones(el):
                result.append(None)
            else:
                result.append(np.concatenate(el, axis=0))
        self.values = None
        return result


class Logger:
    def __init__(self, support_missing_values, use_shift_agnostic_loss, device):
        self.predictions = []
        self.targets = []
        self.support_missing_values = support_missing_values
        self.use_shift_agnostic_loss = use_shift_agnostic_loss
        self.device = device

    def update(self, predictions_now, targets_now):
        if isinstance(predictions_now, dict):
            predictions_now = predictions_now["prediction"]
        self.predictions.append(predictions_now.data.cpu().to(torch.float32).numpy())
        self.targets.append(targets_now.data.cpu().to(torch.float32).numpy())

    def flush(self):
        if not self.use_shift_agnostic_loss:
            self.predictions = np.concatenate(self.predictions, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)

            output = {}
            output["rmse"] = get_rmse(
                self.predictions,
                self.targets,
                support_missing_values=self.support_missing_values,
            )
            output["mae"] = get_mae(
                self.predictions,
                self.targets,
                support_missing_values=self.support_missing_values,
            )
            output["relative rmse"] = get_relative_rmse(
                self.predictions,
                self.targets,
                support_missing_values=self.support_missing_values,
            )

            self.predictions = []
            self.targets = []
            return output
        else:
            if self.support_missing_values:
                raise ValueError(
                    "Shift agnostic loss is not supported with missing values"
                )

            total_mse = 0.0
            total_mae = 0.0
            num_samples = 0

            with torch.no_grad():
                for prediction_batch, target_batch in zip(
                    self.predictions, self.targets
                ):
                    prediction_batch = torch.tensor(
                        prediction_batch,
                        device=self.device,
                        dtype=torch.get_default_dtype(),
                    )
                    target_batch = torch.tensor(
                        target_batch,
                        device=self.device,
                        dtype=torch.get_default_dtype(),
                    )
                    batch_size = len(target_batch)
                    current_mse = (
                        get_shift_agnostic_mse(prediction_batch, target_batch)
                        * batch_size
                    )
                    current_mae = (
                        get_shift_agnostic_mae(prediction_batch, target_batch)
                        * batch_size
                    )
                    current_mse = float(current_mse)
                    current_mae = float(current_mae)
                    total_mse += current_mse
                    total_mae += current_mae
                    num_samples += batch_size

            rmse = np.sqrt(total_mse / num_samples)
            mae = total_mae / num_samples

            output = {}
            output["rmse"] = rmse
            output["mae"] = mae

            self.predictions = []
            self.targets = []
            return output


class FullLogger:
    def __init__(self, support_missing_values, use_shift_agnostic_loss, device):
        self.train_logger = Logger(
            support_missing_values, use_shift_agnostic_loss, device
        )
        self.val_logger = Logger(
            support_missing_values, use_shift_agnostic_loss, device
        )

    def flush(self):
        return {"train": self.train_logger.flush(), "val": self.val_logger.flush()}


def get_rotations(indices, global_aug=False):
    if global_aug:
        num = np.max(indices) + 1
    else:
        num = indices.shape[0]

    rotations = Rotation.random(num).as_matrix()
    rotations[np.random.randn(rotations.shape[0]) >= 0] *= -1

    if global_aug:
        return rotations[indices]
    else:
        return rotations


def get_shift_agnostic_mse(predictions, targets):
    if predictions.shape[1] < targets.shape[1]:
        smaller = predictions
        bigger = targets
    else:
        smaller = targets
        bigger = predictions

    bigger_unfolded = bigger.unfold(1, smaller.shape[1], 1)
    smaller_expanded = smaller[:, None, :]
    delta = smaller_expanded - bigger_unfolded
    losses = torch.mean(delta * delta, dim=2)
    losses, _ = torch.min(losses, dim=1)
    result = torch.mean(losses)
    return result


def get_shift_agnostic_mae(predictions, targets):
    if predictions.shape[1] < targets.shape[1]:
        smaller = predictions
        bigger = targets
    else:
        smaller = targets
        bigger = predictions

    bigger_unfolded = bigger.unfold(1, smaller.shape[1], 1)
    smaller_expanded = smaller[:, None, :]
    delta = smaller_expanded - bigger_unfolded
    losses = torch.mean(torch.abs(delta), dim=2)
    losses, _ = torch.min(losses, dim=1)
    result = torch.mean(losses)
    return result


def get_loss(predictions, targets, support_missing_values, use_shift_agnostic_loss):
    if use_shift_agnostic_loss:
        if support_missing_values:
            raise NotImplementedError(
                "shift agnostic loss is not yet supported with missing values"
            )
        else:
            return get_shift_agnostic_mse(predictions, targets)
    else:
        if isinstance(predictions, dict):
            predictions = predictions["prediction"]
        if support_missing_values:
            delta = predictions - targets
            mask_nan = torch.isnan(targets)
            delta[mask_nan] = 0.0
            mask_not_nan = torch.logical_not(mask_nan)
            return torch.sum(delta * delta) / torch.sum(mask_not_nan)
        else:
            delta = predictions - targets
            return torch.mean(delta * delta)


def get_rmse(predictions, targets, support_missing_values=False):
    if support_missing_values:
        delta = predictions - targets
        mask_nan = np.isnan(targets)
        delta[mask_nan] = 0.0
        mask_not_nan = np.logical_not(mask_nan)
        return np.sqrt(np.sum(delta * delta) / np.sum(mask_not_nan))
    else:
        delta = predictions - targets
        return np.sqrt(np.mean(delta * delta))


def get_mae(predictions, targets, support_missing_values=False):
    if support_missing_values:
        delta = predictions - targets
        mask_nan = np.isnan(targets)
        delta[mask_nan] = 0.0
        mask_not_nan = np.logical_not(mask_nan)
        return np.sum(np.abs(delta)) / np.sum(mask_not_nan)
    else:
        delta = predictions - targets
        return np.mean(np.abs(delta))


def get_relative_rmse(predictions, targets, support_missing_values=False):
    rmse = get_rmse(predictions, targets, support_missing_values=support_missing_values)
    return rmse / get_rmse(
        np.mean(targets), targets, support_missing_values=support_missing_values
    )


def get_scheduler(optim, FITTING_SCHEME):
    def func_lr_scheduler(epoch):
        if epoch < FITTING_SCHEME.EPOCHS_WARMUP:
            return epoch / FITTING_SCHEME.EPOCHS_WARMUP
        delta = epoch - FITTING_SCHEME.EPOCHS_WARMUP
        num_blocks = delta // FITTING_SCHEME.SCHEDULER_STEP_SIZE
        return 0.5 ** (num_blocks)

    scheduler = LambdaLR(optim, func_lr_scheduler)
    return scheduler


def load_checkpoint(model, optim, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optim_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


def get_data_loaders(train_graphs, val_graphs, FITTING_SCHEME):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(FITTING_SCHEME.RANDOM_SEED)

    if FITTING_SCHEME.BALANCED_DATA_LOADER:
        train_sampler = DynamicBatchSampler(
            train_graphs,
            max_num=FITTING_SCHEME.ATOMIC_BATCH_SIZE,
            mode="node",
            shuffle=True,
        )
        val_sampler = DynamicBatchSampler(
            val_graphs,
            max_num=FITTING_SCHEME.ATOMIC_BATCH_SIZE,
            mode="node",
            shuffle=False,
        )

        if FITTING_SCHEME.MULTI_GPU:
            train_loader = DataListLoader(
                train_graphs,
                batch_sampler=train_sampler,
                worker_init_fn=seed_worker,
                generator=g,
            )
            val_loader = DataListLoader(
                val_graphs,
                batch_sampler=val_sampler,
                worker_init_fn=seed_worker,
                generator=g,
            )
        else:
            train_loader = DataLoader(
                train_graphs,
                batch_sampler=train_sampler,
                worker_init_fn=seed_worker,
                generator=g,
            )
            val_loader = DataLoader(
                val_graphs,
                batch_sampler=val_sampler,
                worker_init_fn=seed_worker,
                generator=g,
            )
    else:
        if FITTING_SCHEME.MULTI_GPU:
            train_loader = DataListLoader(
                train_graphs,
                batch_size=FITTING_SCHEME.STRUCTURAL_BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            val_loader = DataListLoader(
                val_graphs,
                batch_size=FITTING_SCHEME.STRUCTURAL_BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g,
            )
        else:
            train_loader = DataLoader(
                train_graphs,
                batch_size=FITTING_SCHEME.STRUCTURAL_BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            val_loader = DataLoader(
                val_graphs,
                batch_size=FITTING_SCHEME.STRUCTURAL_BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g,
            )

    return train_loader, val_loader


def get_optimizer(model, FITTING_SCHEME):
    if FITTING_SCHEME.USE_WEIGHT_DECAY:
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=FITTING_SCHEME.INITIAL_LR,
            weight_decay=FITTING_SCHEME.WEIGHT_DECAY,
        )
    else:
        optim = torch.optim.Adam(model.parameters(), lr=FITTING_SCHEME.INITIAL_LR)
    return optim


def get_rotational_discrepancy(all_predictions):
    predictions_mean = np.mean(all_predictions, axis=0)
    predictions_discrepancies = all_predictions - predictions_mean[np.newaxis]
    # correction for unbiased estimate
    correction = all_predictions.shape[0] / (all_predictions.shape[0] - 1)
    predictions_std = np.sqrt(np.mean(predictions_discrepancies**2) * correction)

    # biased estimate, kind of a mess with the unbiased one
    predictions_mad = np.mean(np.abs(predictions_discrepancies))
    return predictions_std, predictions_mad


def report_accuracy(
    all_predictions,
    ground_truth,
    target_name,
    verbose,
    specify_per_component,
    target_type,
    n_atoms=None,
    support_missing_values=False,
):
    predictions_mean = np.mean(all_predictions, axis=0)

    if specify_per_component:
        spec = "per component"
    else:
        spec = ""

    if ground_truth is not None:
        mae = get_mae(
            predictions_mean,
            ground_truth,
            support_missing_values=support_missing_values,
        )
        rmse = get_rmse(
            predictions_mean,
            ground_truth,
            support_missing_values=support_missing_values,
        )
        print(f"{target_name} mae {spec}: {mae}")
        print(f"{target_name} rmse {spec}: {rmse}")
    else:
        print(
            f"ground truth target for {target_name} is not provided "
            "(or is provided with a wrong key). Thus, it is impossible "
            "to estimate the error between predictions and ground truth target"
        )

    if all_predictions.shape[0] > 1:
        pred_std, pred_mad = get_rotational_discrepancy(all_predictions)
        if verbose:
            str_mae = "rotational discrepancy mad (aka mae)"
            str_rmse = "rotational discrepancy std (aka rmse)"
            print(f"{target_name} {str_mae} {spec}: {pred_mad}")

            print(f"{target_name} {str_rmse} {spec}: {pred_std} ")

    if target_type == "structural":
        if len(predictions_mean.shape) == 1:
            predictions_mean = predictions_mean[:, np.newaxis]
        predictions_mean_per_atom = predictions_mean / n_atoms[:, np.newaxis]

        if ground_truth is not None:
            if len(ground_truth.shape) == 1:
                ground_truth = ground_truth[:, np.newaxis]

            ground_truth_per_atom = ground_truth / n_atoms[:, np.newaxis]
            mae = get_mae(
                predictions_mean_per_atom,
                ground_truth_per_atom,
                support_missing_values=support_missing_values,
            )
            rmse = get_rmse(
                predictions_mean_per_atom,
                ground_truth_per_atom,
                support_missing_values=support_missing_values,
            )
            print(f"{target_name} mae per atom {spec}: " + f"{mae}")

            print(f"{target_name} rmse per atom {spec}: {rmse}")

        if all_predictions.shape[0] > 1:
            if len(all_predictions.shape) == 2:
                all_predictions = all_predictions[:, :, np.newaxis]
            all_predictions_per_atom = (
                all_predictions / n_atoms[np.newaxis, :, np.newaxis]
            )
            predictions_std_per_atom, predictions_mad_per_atom = (
                get_rotational_discrepancy(all_predictions_per_atom)
            )
            if verbose:
                str_mae = "rotational discrepancy mad (aka mae)"
                str_rmse = "rotational discrepancy std (aka rmse)"
                print(f"{target_name} {str_mae} {spec}: {predictions_mad_per_atom}")
                print(f"{target_name} {str_rmse} {spec}: {predictions_std_per_atom} ")


class NeverRun(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self):
        super(NeverRun, self).__init__()

    def forward(self, x) -> torch.Tensor:
        raise RuntimeError("This model should never be run")


def get_quadrature(L):
    matrices, weights = [], []
    for theta_index in range(0, 2 * L - 1):
        for w_index in range(0, 2 * L - 1):
            theta = 2 * np.pi * theta_index / (2 * L - 1)
            w = 2 * np.pi * w_index / (2 * L - 1)
            roots_legendre_now, weights_now = roots_legendre(L)
            all_v = np.arccos(roots_legendre_now)
            for v, weight in zip(all_v, weights_now):
                weights.append(weight)
                angles = [theta, v, w]
                rotation = R.from_euler("xyz", angles, degrees=False)
                rotation_matrix = rotation.as_matrix()
                matrices.append(rotation_matrix)

    return matrices, weights


def dtype2string(dtype):
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"

    raise ValueError("unknown dtype")


def string2dtype(string):
    if string == "float32":
        return torch.float32
    if string == "float16":
        return torch.float16
    if string == "bfloat16":
        return torch.bfloat16

    raise ValueError("unknown dtype")


def get_quadrature_predictions(batch, model, quadrature_order, dtype):
    x_initial = batch.x.clone()
    all_energies, all_forces = [], []
    rotations, weights = get_quadrature(quadrature_order)
    for rotation in rotations:
        rotation = torch.tensor(rotation, device=batch.x.device, dtype=dtype)
        batch_rotations = rotation[None, :].repeat(batch.num_nodes, 1, 1)
        batch.x = torch.bmm(x_initial, batch_rotations)
        prediction_energy, prediction_forces = model(
            batch, augmentation=False, create_graph=False
        )
        all_energies.append(prediction_energy.data.cpu().numpy())
        all_forces.append(prediction_forces.data.cpu().numpy())

    energy_mean, forces_mean, total_weight = 0.0, 0.0, 0.0
    for energy, forces, weight in zip(all_energies, all_forces, weights):
        energy_mean += energy * weight
        forces_mean += forces * weight
        total_weight += weight
    energy_mean /= total_weight
    forces_mean /= total_weight
    return energy_mean, forces_mean
