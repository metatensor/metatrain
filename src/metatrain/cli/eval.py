import argparse
import itertools
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import tqdm
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import AtomisticModel
from omegaconf import DictConfig, OmegaConf

from metatrain.cli.formatter import CustomHelpFormatter
from metatrain.utils.data import (
    CollateFn,
    Dataset,
    TargetInfo,
    get_dataset,
    read_systems,
    unpack_batch,
)
from metatrain.utils.data.writers import (
    DiskDatasetWriter,
    Writer,
    get_writer,
)
from metatrain.utils.devices import pick_devices
from metatrain.utils.errors import ArchitectureError
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.io import load_model
from metatrain.utils.logging import MetricLogger
from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.omegaconf import expand_dataset_config
from metatrain.utils.per_atom import average_by_num_atoms
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
    parser = subparser.add_parser(
        "eval",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
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
        help="filename of the predictions (default: %(default)s)",
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


def _eval_targets(
    model: Union[AtomisticModel, torch.jit.RecursiveScriptModule],
    dataset: Dataset,
    options: Dict[str, TargetInfo],
    batch_size: int = 1,
    check_consistency: bool = False,
    writer: Optional[Writer] = None,
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
    """
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
    collate_fn = CollateFn(
        target_keys,
        callables=[get_system_with_neighbor_lists_transform(requested_neighbor_lists)],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )
    rmse_acc = RMSEAccumulator()
    mae_acc = MAEAccumulator()

    # Warm-up
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
        rmse_acc.update(preds_per_atom, targ_per_atom, batch_extra_data)
        mae_acc.update(preds_per_atom, targ_per_atom, batch_extra_data)

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
    rmse_vals = rmse_acc.finalize(not_per_atom=["positions_gradients"])
    mae_vals = mae_acc.finalize(not_per_atom=["positions_gradients"])
    metrics = {**rmse_vals, **mae_vals}
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
) -> None:
    """
    Evaluate an exported model on a given data set.

    If ``options`` contains a ``targets`` sub-section, RMSE values will be reported. If
    this sub-section is missing, only a xyz-file with containing the properties the
    model was trained against is written.

    :param model: Saved model to be evaluated.
    :param options: DictConfig to define a test dataset taken for the evaluation.
    :param output: Path to save the predicted values.
    :param batch_size: Batch size for evaluation.
    :param check_consistency: Whether to run consistency checks during model evaluation.
    :param append: If ``True``, open the output file in append mode.
    """
    logging.info("Setting up evaluation set.")
    output = Path(output) if isinstance(output, str) else output

    options_list = expand_dataset_config(options)
    for i, options in enumerate(options_list):
        idx_suffix = f"_{i}" if len(options_list) > 1 else ""
        extra_log_message = f" with index {i}" if len(options_list) > 1 else ""
        logging.info(f"Evaluating dataset{extra_log_message}")
        filename = f"{output.stem}{idx_suffix}{output.suffix}"

        # pick the right writer
        writer = get_writer(filename, capabilities=model.capabilities(), append=append)

        # build the dataset & target-info
        if hasattr(options, "targets"):
            eval_dataset, eval_info_dict, _ = get_dataset(options)
            eval_systems = (
                [d.system for d in eval_dataset]
                if not isinstance(writer, DiskDatasetWriter)
                else None
            )
        else:
            if isinstance(writer, DiskDatasetWriter):
                raise ValueError(
                    "Writing to DiskDataset is not allowed without explicitly"
                    " defining targets in the input file."
                )
            eval_systems = read_systems(
                filename=options["systems"]["read_from"],
                reader=options["systems"]["reader"],
            )

            # FIXME: this works only for energy models
            eval_targets: Dict[str, TensorMap] = {}
            eval_info_dict = {}
            do_strain_grad = all(
                not torch.all(system.cell == 0) for system in eval_systems
            )
            layout = _get_energy_layout(do_strain_grad)  # TODO: layout from the user
            for key in model.capabilities().outputs.keys():
                eval_info_dict[key] = TargetInfo(
                    quantity=model.capabilities().outputs[key].quantity,
                    unit=model.capabilities().outputs[key].unit,
                    # TODO: allow the user to specify whether per-atom or not
                    layout=layout,
                )

            eval_dataset = Dataset.from_dict({"system": eval_systems, **eval_targets})

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
