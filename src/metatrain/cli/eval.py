import argparse
import copy
import itertools
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import tqdm
from metatensor.torch import Labels, TensorBlock, TensorMap, remove_gradients
from metatomic.torch import AtomisticModel, ModelOutput, System
from metatomic.torch.o3 import SymmetrizedModel
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

from metatrain.share.base_hypers import EquivarianceHypers, EvalDatasetDictHypers
from metatrain.utils.data import (
    CollateFn,
    Dataset,
    TargetInfo,
    get_dataset,
    load_indices,
    read_systems,
    unpack_batch,
)
from metatrain.utils.data.readers import read_extra_data
from metatrain.utils.data.target_info import DEPRECATED_METATOMIC_OUTPUT_NAMES
from metatrain.utils.data.writers import (
    DiskDatasetWriter,
    MemmapWriter,
    Writer,
    get_writer,
)
from metatrain.utils.devices import pick_devices
from metatrain.utils.errors import ArchitectureError
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.io import load_model
from metatrain.utils.logging import MetricLogger
from metatrain.utils.metrics import (
    EquivarianceAccumulator,
    MAEAccumulator,
    RMSEAccumulator,
)
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.omegaconf import expand_dataset_config
from metatrain.utils.per_atom import average_by_num_atoms, divide_by_num_atoms
from metatrain.utils.pydantic import validate_eval_options
from metatrain.utils.system_data import get_system_data_transform
from metatrain.utils.transfer import batch_to


logger = logging.getLogger(__name__)


def _add_eval_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add the `eval_model` paramaters to an argparse (sub)-parser

    :param subparser: The argparse (sub)-parser to add the parameters to.
    """

    if eval_model.__doc__ is not None:
        description = eval_model.__doc__.split(r":param")[0]
    else:
        description = None

    # If you change the synopsis of these commands or add new ones adjust the completion
    # script at `src/metatrain/share/metatrain-completion.bash`.
    parser = subparser.add_parser("eval", description=description)
    parser.set_defaults(callable="eval_model")
    parser.add_argument(
        "path",
        type=str,
        help="Saved exported (.pt) model to be evaluated.",
    )
    parser.add_argument(
        "options",
        type=str,
        help="Eval options YAML file to define a dataset for evaluation.",
    )
    parser.add_argument(
        "-e",
        "--extensions-dir",
        type=str,
        required=False,
        dest="extensions_directory",
        default=None,
        help=(
            "path to a directory containing all extensions required by the exported "
            "model"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        default="output.xyz",
        help=(
            "filename of the predictions (default: %(default)s). A path ending in a "
            "path separator (e.g. 'predictions/') writes a memmap dataset directory "
            "instead of a single file."
        ),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        required=False,
        type=int,
        default=1,
        help="batch size for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--check-consistency",
        dest="check_consistency",
        action="store_true",
        help="whether to run consistency checks (default: %(default)s)",
    )
    parser.add_argument(
        "--warm-up",
        action=argparse.BooleanOptionalAction,
        dest="warm_up",
        default=True,
        help=(
            "whether to do a warm-up of the model before evaluation "
            "(default: %(default)s). The warm-up is done by running 10 "
            "model evaluations. This will give timings that are more representative "
            "of the model's performance on a large dataset. However, it will delay "
            "evaluation, which might be undesirable for expensive evaluations."
        ),
    )


def _prepare_eval_model_args(args: argparse.Namespace) -> None:
    """Prepare arguments for eval_model.

    :param args: The argparse.Namespace containing the arguments.
    """
    args.options = OmegaConf.load(args.options)
    # models for evaluation are already exported. Don't have to pass the `name` argument
    args.model = load_model(
        path=args.__dict__.pop("path"),
        extensions_directory=args.__dict__.pop("extensions_directory"),
    )


def _indexed_filename(output: Union[str, Path], idx_suffix: str) -> str:
    """Insert the index of a dataset in an output path.

    :param output: The output path given by the user.
    :param idx_suffix: Suffix identifying the dataset, empty if there is one.

    :return: The output path for this dataset.
    """
    # a trailing separator signals a memmap dataset directory, and has to be
    # detected on the raw string, since Path() silently drops it
    is_memmap = isinstance(output, str) and output.endswith(("/", os.sep))
    path = Path(output)
    return f"{path.stem}{idx_suffix}{path.suffix}" + ("/" if is_memmap else "")


def _max_angular_momentum(targets: Dict[str, TargetInfo]) -> int:
    """Largest angular momentum among the targets.

    This sizes the Wigner matrices of the symmetrized model, which has to be
    able to rotate every target back to the frame of the input system.

    :param targets: Dictionary containing the target information.

    :return: The largest angular momentum among the targets.
    """
    max_angular_momentum = 0
    for target in targets.values():
        if target.is_spherical:
            o3_lambda = target.layout.keys.column("o3_lambda")
            max_angular_momentum = max(max_angular_momentum, int(o3_lambda.max()))
        elif target.is_cartesian:
            # a Cartesian tensor of rank n decomposes into terms up to o3_lambda=n
            rank = len(target.layout.block(0).components)
            max_angular_momentum = max(max_angular_momentum, rank)
    return max_angular_momentum


class _EquivarianceEvaluation:
    """Measures the O(3) equivariance error of a model alongside its accuracy.

    Every batch is also evaluated by a metatomic ``SymmetrizedModel``, giving the
    variance of the back-rotated predictions over a rotation grid (the
    equivariance error) and their O(3) average, whose accuracy is measured too.

    :param model: The model being evaluated.
    :param targets: The targets to evaluate the model on.
    :param options: The ``EquivarianceHypers``, with their defaults.
    :param separate_blocks: Whether to report the metrics for each block.
    :param writer: Optional writer for the outputs of the symmetrized model.
    """

    def __init__(
        self,
        model: Union[AtomisticModel, torch.jit.RecursiveScriptModule],
        targets: Dict[str, TargetInfo],
        options: Dict,
        separate_blocks: bool,
        writer: Optional[Writer],
    ) -> None:
        gradients = sorted(
            {
                gradient
                for target in targets.values()
                for gradient in target.gradients  # type: ignore[union-attr]
            }
        )
        if len(gradients) > 0:
            logging.warning(
                f"the equivariance error of the {gradients} gradients is not "
                "supported and will not be reported; only the equivariance of the "
                "target values themselves is measured"
            )

        self.model = model
        self.writer = writer
        self.separate_blocks = separate_blocks
        self.symmetrized_model = SymmetrizedModel(
            model.module,
            max_angular_momentum_target=_max_angular_momentum(targets),
            max_angular_momentum_grid=options["max_angular_momentum_grid"],
            batch_size=options["batch_size"],
        )
        # atom-pair targets are excluded from the metrics, as for the accuracy
        self.names = [
            name
            for name, target in targets.items()
            if target.sample_kind != "atom_pair"
        ]
        self.outputs: Dict[str, ModelOutput] = {}
        for name in self.names:
            for key in (name, f"o3::variance::{name}"):
                self.outputs[key] = ModelOutput(
                    unit=targets[name].unit, sample_kind=targets[name].sample_kind
                )

        self.equivariance_acc = EquivarianceAccumulator(separate_blocks)
        self.averaged_rmse_acc = RMSEAccumulator(separate_blocks)
        self.averaged_mae_acc = MAEAccumulator(separate_blocks)
        self.checked_grid_convergence = False
        logging.info(
            "Equivariance error evaluation enabled: every system is additionally "
            "evaluated on many rotated and inverted copies, on a quadrature grid "
            f"of degree {self.symmetrized_model.max_angular_momentum_grid}"
        )

    def _per_atom_variances(
        self, symmetrized: Dict[str, TensorMap], systems: List[System]
    ) -> Dict[str, TensorMap]:
        """The variances of the targets, normalized per atom like the predictions.

        :param symmetrized: Outputs of the symmetrized model.
        :param systems: The systems of the batch.

        :return: The variance of each target, keyed by the name of the target.
        """
        num_atoms = torch.tensor([len(s) for s in systems], device=systems[0].device)
        # a variance scales as the square of the predictions
        return {
            name: divide_by_num_atoms(
                divide_by_num_atoms(symmetrized[f"o3::variance::{name}"], num_atoms),
                num_atoms,
            )
            for name in self.names
        }

    def update(
        self,
        systems: List[System],
        targets_per_atom: Dict[str, TensorMap],
        extra_data: Dict[str, TensorMap],
        not_per_atom: List[str],
    ) -> None:
        """Evaluate the symmetrized model on a batch and accumulate its metrics.

        :param systems: The systems of the batch.
        :param targets_per_atom: The targets of the batch, normalized per atom.
        :param extra_data: The extra data of the batch.
        :param not_per_atom: Keys containing one of these strings are not labeled
            as "(per atom)".
        """
        symmetrized = self.symmetrized_model(systems, self.outputs, None)
        self.equivariance_acc.update(self._per_atom_variances(symmetrized, systems))
        averaged = average_by_num_atoms(
            {name: symmetrized[name] for name in self.names},
            systems,
            per_structure_keys=[],
        )
        # the symmetrized model does not predict gradients, so the reference ones
        # have to go as well: the accumulators pair up the gradients of the target
        # with those of the prediction
        averaged_targets = {
            name: remove_gradients(targets_per_atom[name]) for name in self.names
        }
        self.averaged_rmse_acc.update(averaged, averaged_targets, extra_data)
        self.averaged_mae_acc.update(averaged, averaged_targets, extra_data)

        if self.writer:
            # the averaged prediction is written under the name of the target, so
            # that the result reads back as a normal dataset, and the variance next
            # to it under its metatomic name
            self.writer.write(systems, {key: symmetrized[key] for key in self.outputs})

        if not self.checked_grid_convergence:
            self._check_grid_convergence(systems, not_per_atom)
            self.checked_grid_convergence = True

    def _check_grid_convergence(
        self, systems: List[System], not_per_atom: List[str], tolerance: float = 0.01
    ) -> None:
        """Warn if re-measuring the equivariance error of a batch on a finer
        quadrature grid changes it by more than ``tolerance`` (relative).

        :param systems: The systems of the batch, the only one measured so far.
        :param not_per_atom: Keys containing one of these strings are not labeled
            as "(per atom)".
        :param tolerance: Largest acceptable relative change.
        """
        reference = self.equivariance_acc.finalize(not_per_atom=not_per_atom)
        scales = self.averaged_rmse_acc.finalize(not_per_atom=not_per_atom)
        coarse_grid = self.symmetrized_model.max_angular_momentum_grid
        finer_model = SymmetrizedModel(
            self.model.module,
            max_angular_momentum_target=self.symmetrized_model.max_angular_momentum_target,
            max_angular_momentum_grid=coarse_grid + 2,
            batch_size=self.symmetrized_model.batch_size,
        )
        finer_acc = EquivarianceAccumulator(self.separate_blocks)
        finer_acc.update(
            self._per_atom_variances(finer_model(systems, self.outputs, None), systems)
        )

        for key, finer_value in finer_acc.finalize(not_per_atom=not_per_atom).items():
            coarse_value = reference.get(key)
            if coarse_value is None:
                continue
            largest = max(abs(coarse_value), abs(finer_value))
            # ignore equivariance errors a million times smaller than the model's
            # own error
            scale = scales.get(key.replace(" equivariance RMSE", " RMSE"), 0.0)
            if largest <= 1e-6 * abs(scale):
                continue
            change = abs(finer_value - coarse_value) / largest
            if change > tolerance:
                logging.warning(
                    f"the {key} is not converged with respect to the quadrature "
                    f"grid: {coarse_value:.4g} at max_angular_momentum_grid="
                    f"{coarse_grid} against {finer_value:.4g} at "
                    f"{coarse_grid + 2} ({100 * change:.0f}% change). "
                    "The reported equivariance error is unreliable; increase "
                    "`max_angular_momentum_grid` until it stops changing."
                )

    def finalize(self, not_per_atom: List[str]) -> Dict[str, float]:
        """Finish the writer and finalize the metrics.

        The equivariance and O(3)-averaged errors are also reported combined as
        ``sqrt(equivariance^2 + averaged^2)``, the RMSE the model would have on
        randomly oriented copies of every system.

        :param not_per_atom: Keys containing one of these strings are not labeled
            as "(per atom)".

        :return: The equivariance, O(3)-averaged and oriented metrics.
        """
        if self.writer:
            self.writer.finish()

        equivariance = self.equivariance_acc.finalize(not_per_atom=not_per_atom)
        averaged_rmse = self.averaged_rmse_acc.finalize(not_per_atom=not_per_atom)
        averaged_mae = self.averaged_mae_acc.finalize(not_per_atom=not_per_atom)

        metrics = dict(equivariance)
        for key, value in averaged_rmse.items():
            metrics[key.replace(" RMSE", " O3-averaged RMSE")] = value
        for key, value in averaged_mae.items():
            metrics[key.replace(" MAE", " O3-averaged MAE")] = value

        for key, equivariance_value in equivariance.items():
            averaged_key = key.replace(" equivariance RMSE", " RMSE")
            if averaged_key in averaged_rmse:
                combined = (
                    equivariance_value**2 + averaged_rmse[averaged_key] ** 2
                ) ** 0.5
                metrics[key.replace(" equivariance RMSE", " oriented RMSE")] = combined

        return metrics


def _eval_targets(
    model: Union[AtomisticModel, torch.jit.RecursiveScriptModule],
    dataset: Dataset,
    options: Dict[str, TargetInfo] | Dict[str, ModelOutput],
    batch_size: int = 1,
    check_consistency: bool = False,
    writer: Optional[Writer] = None,
    warm_up: bool = True,
    equivariance: Optional[Dict] = None,
    equivariance_writer: Optional[Writer] = None,
    log_separate_blocks: bool = False,
) -> None:
    """
    Evaluate `model` on `dataset`, accumulate RMSE/MAE, and (if `writer` is provided)
    stream or buffer out per-sample writes.

    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate the model on.
    :param options: Dictionary containing the target information.
    :param batch_size: Batch size for evaluation.
    :param check_consistency: Whether to run consistency checks during model evaluation.
    :param writer: Optional writer to write out per-sample predictions.
    :param warm_up: Whether to do a warm-up of the model before evaluation.
    :param equivariance: If not :obj:`python:None`, also measure the O(3)
        equivariance error of the model and the accuracy of its O(3)-averaged
        predictions, with these ``EquivarianceHypers`` (defaults included).
    :param equivariance_writer: Optional writer for the outputs of the
        symmetrized model, i.e. the O(3)-averaged predictions and their
        variance over the orientations.
    :param log_separate_blocks: Report metrics separately for each block of each
        target, instead of aggregated over the whole target.
    """
    # Disable static fusion. Besides the fact that atomistic batches have variable
    # sizes, statically fused CUDA kernels cannot allocate new tensors at runtime,
    # causing "Global alloc not supported yet" errors (cuda 13+) at the time of writing
    torch.jit.set_fusion_strategy([("DYNAMIC", 10)])

    if len(dataset) == 0:
        logging.info("This dataset is empty. No evaluation will be performed.")
        return None

    # Infer device/dtype
    model_tensor = next(itertools.chain(model.parameters(), model.buffers()))
    dtype = model_tensor.dtype

    device = pick_devices(architecture_devices=model.capabilities().supported_devices)[
        0
    ]
    logging.info(f"Running on device {device} with dtype {dtype}")
    model.to(dtype=dtype, device=device)

    # DataLoader & metrics setup
    if len(dataset) % batch_size != 0:
        logging.debug(
            f"The dataset size ({len(dataset)}) is not a multiple of the batch size "
            f"({batch_size}). {len(dataset) // batch_size} batches will be "
            f"constructed with a batch size of {batch_size}, and the last batch will "
            f"have a size of {len(dataset) % batch_size}. This might lead to "
            "inaccurate average timings."
        )

    # Create a dataloader
    target_keys = list(model.capabilities().outputs.keys())
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    callables = [get_system_with_neighbor_lists_transform(requested_neighbor_lists)]

    # Attach additional per-system inputs
    requested_inputs = [
        name
        for name, output in model.requested_inputs(use_new_names=True).items()
        if output.sample_kind == "system"
    ]
    if requested_inputs:
        callables.append(get_system_data_transform(requested_inputs))
    collate_fn = CollateFn(target_keys, callables=callables)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )

    rmse_acc = RMSEAccumulator(separate_blocks=log_separate_blocks)
    mae_acc = MAEAccumulator(separate_blocks=log_separate_blocks)

    not_per_atom = ["positions_gradients"]

    equivariance_evaluation = (
        _EquivarianceEvaluation(
            model, options, equivariance, log_separate_blocks, equivariance_writer
        )
        if equivariance is not None
        else None
    )

    # Warm-up
    if warm_up:
        logging.info("Warming up the model with 10 batches...")
        cycled = itertools.cycle(dataloader)
        for _ in range(10):
            batch = unpack_batch(next(cycled))
            systems = [system.to(device=device, dtype=dtype) for system in batch[0]]
            evaluate_model(
                model,
                systems,
                options,
                is_training=False,
                check_consistency=check_consistency,
            )
    else:
        logging.info("Skipping warm-up of the model.")

    total_time = 0.0
    timings_per_atom = []

    # Main evaluation loop
    for batch in tqdm.tqdm(dataloader, ncols=100):
        systems, batch_targets, batch_extra_data = unpack_batch(batch)
        systems, batch_targets, batch_extra_data = batch_to(
            systems, batch_targets, batch_extra_data, dtype=dtype, device=device
        )

        start_time = time.time()
        batch_predictions = evaluate_model(
            model,
            systems,
            options,
            is_training=False,
            check_consistency=check_consistency,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        # Update metrics
        preds_per_atom = average_by_num_atoms(
            batch_predictions, systems, per_structure_keys=[]
        )
        targ_per_atom = average_by_num_atoms(
            batch_targets, systems, per_structure_keys=[]
        )
        # For metrics, filter out the targets that have sample_kind "atom_pair"
        # for now.
        preds_per_atom = {
            k: v
            for k, v in preds_per_atom.items()
            if options[k].sample_kind != "atom_pair"
        }
        targ_per_atom = {
            k: v
            for k, v in targ_per_atom.items()
            if options[k].sample_kind != "atom_pair"
        }
        rmse_acc.update(preds_per_atom, targ_per_atom, batch_extra_data)
        mae_acc.update(preds_per_atom, targ_per_atom, batch_extra_data)

        # Equivariance error, deliberately outside of the timed region above so
        # that it does not pollute the reported evaluation timings
        if equivariance_evaluation is not None:
            equivariance_evaluation.update(
                systems, targ_per_atom, batch_extra_data, not_per_atom
            )

        # Write out each sample if a writer is configured
        if writer:
            writer.write(systems, batch_predictions)

        # Timing
        time_taken = end_time - start_time
        total_time += time_taken
        timings_per_atom.append(time_taken / sum(len(system) for system in systems))

    # Finish writer
    if writer:
        writer.finish()

    # Finalize metrics and log
    rmse_vals = rmse_acc.finalize(not_per_atom=not_per_atom)
    mae_vals = mae_acc.finalize(not_per_atom=not_per_atom)
    metrics = {**rmse_vals, **mae_vals}
    if equivariance_evaluation is not None:
        metrics.update(equivariance_evaluation.finalize(not_per_atom))
    metric_logger = MetricLogger(
        log_obj=logger, dataset_info=model.capabilities(), initial_metrics=metrics
    )
    metric_logger.log(metrics)

    # Log timings
    timings_per_atom = np.array(timings_per_atom)
    mean_per_atom = np.mean(timings_per_atom)
    std_per_atom = np.std(timings_per_atom)
    logging.info(
        f"Evaluation time: {total_time:.2f} s "
        f"[{1000.0 * mean_per_atom:.4f} ± "
        f"{1000.0 * std_per_atom:.4f} ms per atom]"
    )


def eval_model(
    model: Union[AtomisticModel, torch.jit.RecursiveScriptModule],
    options: DictConfig,
    output: Union[Path, str] = "output.xyz",
    batch_size: int = 1,
    check_consistency: bool = False,
    append: Optional[bool] = None,
    warm_up: bool = True,
) -> None:
    """
    Evaluate an exported model on a given data set.

    If ``options`` contains a ``targets`` sub-section, RMSE values will be reported. If
    this sub-section is missing, only a xyz-file with containing the properties the
    model was trained against is written.

    :param model: Saved model to be evaluated.
    :param options: DictConfig to define a test dataset taken for the evaluation.
    :param output: Path to save the predicted values. A path ending in a path
        separator (e.g. ``"predictions/"``) writes a memmap dataset directory instead
        of a single file.
    :param batch_size: Batch size for evaluation.
    :param check_consistency: Whether to run consistency checks during model evaluation.
    :param append: If ``True``, open the output file in append mode.
    :param warm_up: Whether to do a warm-up of the model before evaluation.
    """
    logging.info("Setting up evaluation set.")
    options = validate_eval_options(OmegaConf.to_container(options))
    options = OmegaConf.create(options)
    options_list = expand_dataset_config(options)
    for i, options in enumerate(options_list):
        options = OmegaConf.merge(init_with_defaults(EvalDatasetDictHypers), options)
        idx_suffix = f"_{i}" if len(options_list) > 1 else ""
        extra_log_message = f" with index {i}" if len(options_list) > 1 else ""
        logging.info(f"Evaluating dataset{extra_log_message}")
        filename = _indexed_filename(output, idx_suffix)

        # pick the right writer
        writer = get_writer(filename, capabilities=model.capabilities(), append=append)

        # build the dataset & target-info
        if hasattr(options, "targets"):
            eval_dataset, eval_info_dict, _ = get_dataset(options)
        else:
            if isinstance(writer, (DiskDatasetWriter, MemmapWriter)):
                raise ValueError(
                    "Writing to DiskDataset or MemmapDataset is not allowed without"
                    " explicitly defining targets in the input file."
                )
            if "equivariance" in options:
                raise ValueError(
                    "measuring the equivariance error requires a `targets` section in "
                    "the eval options file"
                )
            eval_systems = read_systems(
                filename=options["systems"]["read_from"],
                reader=options["systems"]["reader"],
            )

            # FIXME: this works only for energy models
            eval_targets: Dict[str, List[TensorMap]] = {}
            eval_info_dict = copy.deepcopy(
                {
                    name: model_output
                    for name, model_output in model.capabilities().outputs.items()
                    if name not in DEPRECATED_METATOMIC_OUTPUT_NAMES
                }
            )
            for name, model_output in eval_info_dict.items():
                if "energy" in name:
                    model_output.sample_kind = "system"  # type: ignore

            extra_data: Dict[str, List[TensorMap]] = {}
            if "extra_data" in options:
                extra_data, _ = read_extra_data(conf=options["extra_data"])

            eval_dataset = Dataset.from_dict(
                {"system": eval_systems, **eval_targets, **extra_data}
            )

        if "indices" in options:
            eval_indices = options["indices"]
            if isinstance(options["indices"], str):
                eval_indices = load_indices(eval_indices)
            eval_dataset = Subset(eval_dataset, eval_indices)

        equivariance_options: Optional[Dict] = None
        if "equivariance" in options:
            equivariance_options = {
                **init_with_defaults(EquivarianceHypers),
                **OmegaConf.to_container(options["equivariance"]),  # type: ignore[dict-item]
            }
        equivariance_writer: Optional[Writer] = None
        if equivariance_options is not None and equivariance_options["output"]:
            equivariance_writer = get_writer(
                _indexed_filename(equivariance_options["output"], idx_suffix),
                capabilities=model.capabilities(),
                append=append,
            )

        # run evaluation & writing
        try:
            # we always let the writer handle I/O, so we never need return_predictions
            # here
            _eval_targets(
                model=model,
                dataset=eval_dataset,
                options=eval_info_dict,
                batch_size=batch_size,
                check_consistency=check_consistency,
                writer=writer,
                warm_up=warm_up,
                equivariance=equivariance_options,
                equivariance_writer=equivariance_writer,
                log_separate_blocks=options["log_separate_blocks"],
            )
        except Exception as e:
            raise ArchitectureError(e)

        # no post-call write_predictions necessary anymore-writer did it all


def _get_energy_layout(strain_gradient: bool) -> TensorMap:
    block = TensorBlock(
        # float64: otherwise metatensor can't serialize
        values=torch.empty(0, 1, dtype=torch.float64),
        samples=Labels(
            names=["system"],
            values=torch.empty((0, 1), dtype=torch.int32),
        ),
        components=[],
        properties=Labels.range("energy", 1),
    )
    position_gradient_block = TensorBlock(
        # float64: otherwise metatensor can't serialize
        values=torch.empty(0, 3, 1, dtype=torch.float64),
        samples=Labels(
            names=["sample", "atom"],
            values=torch.empty((0, 2), dtype=torch.int32),
        ),
        components=[
            Labels(
                names=["xyz"],
                values=torch.arange(3, dtype=torch.int32).reshape(-1, 1),
            ),
        ],
        properties=Labels.range("energy", 1),
    )
    block.add_gradient("positions", position_gradient_block)

    if strain_gradient:
        strain_gradient_block = TensorBlock(
            # float64: otherwise metatensor can't serialize
            values=torch.empty(0, 3, 3, 1, dtype=torch.float64),
            samples=Labels(
                names=["sample", "atom"],
                values=torch.empty((0, 2), dtype=torch.int32),
            ),
            components=[
                Labels(
                    names=["xyz_1"],
                    values=torch.arange(3, dtype=torch.int32).reshape(-1, 1),
                ),
                Labels(
                    names=["xyz_2"],
                    values=torch.arange(3, dtype=torch.int32).reshape(-1, 1),
                ),
            ],
            properties=Labels.range("energy", 1),
        )
        block.add_gradient("strain", strain_gradient_block)

    energy_layout = TensorMap(
        keys=Labels.single(),
        blocks=[block],
    )
    return energy_layout
