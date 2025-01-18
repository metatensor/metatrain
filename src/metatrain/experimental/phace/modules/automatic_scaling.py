import torch.distributed

from ....utils.evaluate_model import evaluate_model
from ....utils.per_atom import average_by_num_atoms
from ....utils.transfer import (
    systems_and_targets_to_device,
    systems_and_targets_to_dtype,
)


def get_automatic_scaling(
    train_dataloader, scripted_model, train_targets, device, dtype, is_distributed
):
    # This routine is used to calculate a good overall scaling for the model, so that
    # its outputs are in a reasonable range (uncentered second moment of 1).
    # This works best together with target scaling, which performs a similar operation
    # on the targets. The reason why this is necessary is that the nu_scaling and
    # mp_scaling parameters in PhACE can make the scale of the outputs vary a lot.

    sum_of_squares = torch.tensor(0.0, device=device, dtype=dtype)
    num_elements = torch.tensor(0, device=device, dtype=torch.int64)
    for batch in train_dataloader:
        systems, targets = batch
        systems, targets = systems_and_targets_to_device(systems, targets, device)
        systems, targets = systems_and_targets_to_dtype(systems, targets, dtype)
        predictions = evaluate_model(
            scripted_model,
            systems,
            {key: train_targets[key] for key in targets.keys()},
            is_training=True,
        )
        # average by the number of atoms (will only happen to per-structure outputs)
        predictions = average_by_num_atoms(predictions, systems, per_structure_keys=[])
        for tensor_map in predictions.values():
            if any(block.has_gradient("positions") for block in tensor_map.blocks()):
                # if forces are there, we want to scale with respect to the forces
                do_position_gradients = True
            else:
                do_position_gradients = False
            for block in tensor_map.blocks():
                if do_position_gradients:
                    if block.has_gradient("positions"):
                        # if there is a mixture of energy-only and energy+forces
                        # targets, we ignore the energy-only targets
                        sum_of_squares += (
                            block.gradient("positions").values ** 2
                        ).sum()
                        num_elements += block.gradient("positions").values.numel()
                else:
                    sum_of_squares += (block.values**2).sum()
                    num_elements += block.values.numel()
    if is_distributed:
        torch.distributed.all_reduce(sum_of_squares)
        torch.distributed.all_reduce(num_elements)
    return 1.0 / torch.sqrt(sum_of_squares / num_elements).item()
