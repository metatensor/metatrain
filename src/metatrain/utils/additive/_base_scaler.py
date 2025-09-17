"""
Contains the ``BaseCompositionModel class. This is intended for eventual porting to
metatomic. The class ``CompositionModel`` wraps this to be compatible with
metatrain-style objects.
"""

import math
from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


class BaseScaler(torch.nn.Module):
    """
    Fits a scaler for a dict of targets. Scales are computed as the per-property (and
    therefore per-block) standard deviations.

    The :py:method:`accumulate` method is used to accumulate the necessary quantities
    based on the training data, and the :py:method:`fit` method is used to fit the model
    based on the accumulated quantities. These should both be called before the
    :py:method:`forward` method is called to compute the scales.
    """

    # Needed for torchscript compatibility
    target_names: List[str]
    scales: Dict[str, TensorMap]
    sample_kinds: Dict[str, str]
    type_to_index: torch.Tensor
    N: Dict[str, TensorMap]
    Y: Dict[str, TensorMap]
    Y2: Dict[str, TensorMap]

    def __init__(self, atomic_types, layouts: Dict[str, TensorMap]) -> None:
        """
        Initializes the composition model with the given atomic types and layouts.

        :param atomic_types: List of atomic types to use in the composition model.
        :param layouts: Dict of zero-sample layout :py:class:`TensorMap` corresponding
            to each target. The keys of the dict are the target names, and the values
            are :py:class:`TensorMap` objects with the zero-sample layout for each
            target.
        """
        super().__init__()

        self.atomic_types = torch.as_tensor(atomic_types, dtype=torch.int32)
        self.target_names = []
        self.sample_kinds = {}
        self.N = {}
        self.Y = {}
        self.Y2 = {}
        self.scales = {}

        # go from an atomic type to its position in `self.atomic_types`
        self.register_buffer(
            "type_to_index", torch.empty(max(self.atomic_types) + 1, dtype=torch.long)
        )
        for i, atomic_type in enumerate(self.atomic_types):
            self.type_to_index[atomic_type] = i

        # Add targets based on provided layouts
        for target_name, layout in layouts.items():
            self.add_output(target_name, layout)

    def add_output(self, target_name: str, layout: TensorMap) -> None:
        """
        Adds a new target to the composition model.

        :param target_name: Name of the target to add.
        :param layout: Layout of the target as a :py:class:`TensorMap`.
        """
        if target_name in self.target_names:
            raise ValueError(f"target {target_name} already exists in the model.")

        self.target_names.append(target_name)
        valid_sample_names = [
            ["system"],
            [
                "system",
                "atom",
            ],
        ]

        if layout.sample_names == valid_sample_names[0]:
            self.sample_kinds[target_name] = "per_structure"
            samples = Labels(["atomic_type"], torch.tensor([[-1]]))

        elif layout.sample_names == valid_sample_names[1]:
            self.sample_kinds[target_name] = "per_atom"
            samples = Labels(
                ["atomic_type"], torch.arange(len(self.atomic_types)).reshape(-1, 1)
            )

        else:
            raise ValueError(
                "unknown sample kind. TensorMap has sample names"
                f" {layout.sample_names} but expected one of "
                f"{valid_sample_names}."
            )

        # Initialize TensorMaps for the quantities to accumulate for this target.
        #
        self.N[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(samples),
                        1,
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
                    properties=Labels(["_"], torch.tensor([[0]])),
                )
                for block in layout
            ],
        )
        self.Y[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(samples),
                        *[len(comp) for comp in block.components],
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=block.components,
                    properties=block.properties,
                )
                for block in layout
            ],
        )
        self.Y2[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(samples),
                        *[len(comp) for comp in block.components],
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=block.components,
                    properties=block.properties,
                )
                for block in layout
            ],
        )
        self.scales[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.ones(
                        len(samples),
                        *[len(comp) for comp in block.components],
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=block.components,
                    properties=block.properties,
                )
                for block in layout
            ],
        )

    def accumulate(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> None:
        """
        Takes a batch of targets, and for each target accumulates the
        necessary quantities, i.e. the sum over samples (Y), the sum over the squared
        samples (Y2), and the number of samples overall (N).
        """

        if extra_data is None:
            extra_data = {}

        device = list(targets.values())[0][0].values.device
        dtype = list(targets.values())[0][0].values.dtype
        self._sync_device_dtype(device, dtype)

        # accumulate
        for target_name, target in targets.items():
            mask = None
            if target_name + "_mask" in extra_data:
                mask = extra_data[target_name + "_mask"]

            for key, block in target.items():
                # Get the target block values
                if (
                    target.keys.names == ["_"]
                    and len(target[0].components) == 0
                    and "positions" in target[0].gradients_list()
                ):
                    # special case for the energy with attached gradients: here we want
                    # to scale with respect to the forces rather than the energies
                    Y_block = block.gradient("positions")
                else:
                    Y_block = block

                Y_block = Y_block.to(device=device, dtype=dtype)
                assert Y_block.samples.names == [
                    "system",
                    "atom",
                ]  # only per-atom targets for now

                # Here it is assumed that the samples of the block correspond to the
                # full ordered list of atoms in the batch of systems
                Y_block_types = torch.cat([system.types for system in systems])

                for atomic_type in self.atomic_types:
                    # Slice the block to only include samples of the current atomic type
                    samples_type_mask = Y_block_types == atomic_type
                    Y = Y_block.values[samples_type_mask]

                    # Compute the number of samples in this block, account for the mask
                    # if available
                    if mask is None:
                        N = Y.shape[0]
                    else:
                        # Count N as the number of samples where the mask is True for at
                        # least one property (in other words where the mask for a given
                        # sample is not all False). This handles the case where samples
                        # are padded.
                        pad_mask_values = mask.block(key).values[samples_type_mask]
                        samples_pad_mask = pad_mask_values.any(
                            dim=list(range(1, Y.dim()))
                        )
                        N = samples_pad_mask.sum()
                        Y = Y[samples_pad_mask]

                    # Compute the Y and Y2 values and sum over samples
                    Y_values = torch.sum(Y, dim=0, keepdim=True)
                    Y2_values = torch.sum(Y**2, dim=0, keepdim=True)

                    # Repeat the along the component axes (if any) and accumulate
                    n_components: List[int] = []
                    for comp in Y.shape[1:-1]:
                        n_components.append(comp)
                    self.N[target_name][key].values[
                        self.type_to_index[atomic_type]
                    ] += N
                    self.Y[target_name][key].values[
                        self.type_to_index[atomic_type]
                    ] += Y_values.squeeze(0)
                    self.Y2[target_name][key].values[
                        self.type_to_index[atomic_type]
                    ] += Y2_values.squeeze(0)

    def fit(
        self,
        targets_to_fit: Optional[List[str]] = None,
    ) -> None:
        """
        Based on the pre-accumulated quantities from the training data, computes the
        scales for each target.
        """
        if targets_to_fit is None:
            targets_to_fit = self.target_names

        # fit
        for target_name in self.target_names:
            blocks = []
            for key in self.N[target_name].keys:
                N_block = self.N[target_name][key]
                Y_block = self.Y[target_name][key]
                Y2_block = self.Y2[target_name][key]

                N_values = N_block.values
                Y_values = Y_block.values
                Y2_values = Y2_block.values

                if self.sample_kinds[target_name] == "per_structure":  # TODO
                    assert len(Y_block.samples) == 1

                # Set a sensible default in case we don't compute a scale
                block = TensorBlock(
                    values=torch.ones_like(Y_block.values),
                    samples=Y_block.samples,
                    components=Y_block.components,
                    properties=Y_block.properties,
                )

                # Now iterate over all the atomic types in this block. For per-structure
                # targets, this is just one iteration as we do not compute
                # per-atomic-type
                for type_index in range(len(Y_block.samples)):
                    N_values_type = N_values[type_index].unsqueeze(0)
                    Y_values_type = Y_values[type_index].unsqueeze(0)
                    Y2_values_type = Y2_values[type_index].unsqueeze(0)

                    # Compute the standard deviation
                    N_type = N_values_type.item()  # (do not use Bessel's correction)

                    if N_type <= 0:
                        continue

                    # Take the norm over components and re-expand the component dims
                    Y_values_type = torch.norm(Y_values_type, dim=1, keepdim=True)
                    Y2_values_type = torch.norm(Y2_values_type, dim=1, keepdim=True)
                    Y_values_type = Y_values_type.repeat(
                        1, *[len(comp) for comp in Y_block.components], 1
                    )
                    Y2_values_type = Y2_values_type.repeat(
                        1, *[len(comp) for comp in Y_block.components], 1
                    )

                    # Divide the scales by sqrt(num components) to account for the
                    # fact that we took the norm over components when accumulating
                    # Y and Y2
                    if len(Y_values_type.shape) == 2:
                        component_factor = 1
                    else:
                        component_factor = Y_values_type.shape[1]

                    scale_vals_type = torch.sqrt(
                        (Y2_values_type / N_type) - (Y_values_type / N_type) ** 2
                    ) / math.sqrt(component_factor)

                    # If any scales are zero, set them to 1.0
                    if torch.any(scale_vals_type == 0):
                        scale_vals_type[scale_vals_type == 0] = 1.0

                    scale_vals_type = scale_vals_type.contiguous()
                    block.values[type_index][:] = scale_vals_type

                blocks.append(block)

            self.scales[target_name] = TensorMap(
                self.Y[target_name].keys.to(device=scale_vals_type.device),
                blocks,
            )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, TensorMap],
        remove: bool,
    ) -> Dict[str, TensorMap]:
        """
        Scales the targets based on the stored standard deviations.

        :param outputs: Dict of names outputs to scale. The names (keys) should be a
            subset of the target names used during fitting.
        :returns: A dictionary with the scaled outputs for each system.

        :raises ValueError: If no scales have been computed or if `outputs` keys
            contain unsupported keys.
        """

        device = list(outputs.values())[0][0].values.device
        dtype = list(outputs.values())[0][0].values.dtype
        self._sync_device_dtype(device, dtype)

        # Build the scaled outputs for each output
        predictions: Dict[str, TensorMap] = {}
        for output_name in outputs:
            if output_name not in self.target_names:  # just return output as is
                predictions[output_name] = outputs[output_name]
                continue

            output_map = outputs[output_name]
            n_blocks = len(output_map)  # TorchScript-friendly integer

            prediction_blocks: List[TensorBlock] = []
            for i in range(n_blocks):
                output_block = output_map.block(i)

                # Find the scales block and check metadata
                scales_block = self.scales[output_name].block(i)
                key = output_map.keys[i]
                assert scales_block.properties == output_block.properties, (
                    f"Properties of scales block {scales_block.properties} "
                    f"do not match output block {output_block.properties} "
                    f"for key {key}."
                )

                # Scale each atomic type separately
                output_block_types = torch.cat([system.types for system in systems])
                scaled_vals = output_block.values

                for atomic_type in self.atomic_types:
                    type_mask = output_block_types == atomic_type
                    if type_mask.sum() == 0:
                        continue

                    # Scale the values of the output block
                    if remove:  # remove the scaler
                        scaled_vals[type_mask] = (
                            output_block.values[type_mask]
                            / scales_block.values[self.type_to_index[atomic_type]]
                        )
                    else:  # apply the scaler
                        scaled_vals[type_mask] = (
                            output_block.values[type_mask]
                            * scales_block.values[self.type_to_index[atomic_type]]
                        )

                prediction_blocks.append(
                    TensorBlock(
                        values=scaled_vals,
                        samples=output_block.samples,
                        components=output_block.components,
                        properties=output_block.properties,
                    )
                )

            predictions[output_name] = TensorMap(
                outputs[output_name].keys,
                prediction_blocks,
            )

        return predictions

    def _sync_device_dtype(self, device: torch.device, dtype: torch.dtype):
        # manually move the TensorMap dicts:

        self.atomic_types = self.atomic_types.to(device=device)
        self.type_to_index = self.type_to_index.to(device=device)
        self.N = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.N.items()
        }
        self.Y = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.Y.items()
        }
        self.Y2 = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.Y2.items()
        }
        self.scales = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.scales.items()
        }
