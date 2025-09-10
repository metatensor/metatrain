"""
Contains the ``BaseCompositionModel class. This is intended for eventual porting to
metatomic. The class ``CompositionModel`` wraps this to be compatible with
metatrain-style objects.
"""

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
    atomic_types: List[int]
    target_names: List[str]
    scales: Dict[str, Dict[int, TensorMap]]
    N: Dict[str, Dict[int, TensorMap]]
    Y: Dict[str, Dict[int, TensorMap]]
    Y2: Dict[str, Dict[int, TensorMap]]

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

        self.atomic_types = atomic_types
        self.target_names = []
        self.N = {target_name: {} for target_name in layouts}
        self.Y = {target_name: {} for target_name in layouts}
        self.Y2 = {target_name: {} for target_name in layouts}
        self.scales = {target_name: {} for target_name in layouts}
        self.is_fitted: Dict[str, Dict[int, bool]] = {
            target_name: {} for target_name in layouts
        }

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

        # Initialize TensorMaps for the quantities to accumulate
        for atomic_type in self.atomic_types:
            self.is_fitted[target_name][atomic_type] = False
            self.N[target_name][atomic_type] = TensorMap(
                layout.keys,
                blocks=[
                    TensorBlock(
                        values=torch.zeros(
                            1,
                            1,
                            dtype=torch.float64,
                        ),
                        samples=Labels(["_"], torch.tensor([[0]])),
                        components=[],
                        properties=Labels(["_"], torch.tensor([[0]])),
                    )
                    for block in layout
                ],
            )
            self.Y[target_name][atomic_type] = TensorMap(
                layout.keys,
                blocks=[
                    TensorBlock(
                        values=torch.zeros(
                            1,
                            *[len(comp) for comp in block.components],
                            len(block.properties),
                            dtype=torch.float64,
                        ),
                        samples=Labels(["_"], torch.tensor([[0]])),
                        components=block.components,
                        properties=block.properties,
                    )
                    for block in layout
                ],
            )
            self.Y2[target_name][atomic_type] = TensorMap(
                layout.keys,
                blocks=[
                    TensorBlock(
                        values=torch.zeros(
                            1,
                            *[len(comp) for comp in block.components],
                            len(block.properties),
                            dtype=torch.float64,
                        ),
                        samples=Labels(["_"], torch.tensor([[0]])),
                        components=block.components,
                        properties=block.properties,
                    )
                    for block in layout
                ],
            )
            self.scales[target_name][atomic_type] = TensorMap(
                layout.keys,
                blocks=[
                    TensorBlock(
                        values=torch.ones(
                            1,
                            *[len(comp) for comp in block.components],
                            len(block.properties),
                            dtype=torch.float64,
                        ),
                        samples=Labels(["_"], torch.tensor([[0]])),
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
                    and "positions" in target.gradients
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
                    self.N[target_name][atomic_type][key].values[:] += N
                    self.Y[target_name][atomic_type][key].values[:] += Y_values
                    self.Y2[target_name][atomic_type][key].values[:] += Y2_values

    def fit(self) -> None:
        """
        Based on the pre-accumulated quantities from the training data, computes the
        scales for each target.
        """

        # fit
        for target_name in self.target_names:
            for atomic_type in self.atomic_types:
                if self.is_fitted[target_name][atomic_type]:  # already fitted
                    print("Already fitted:", target_name, atomic_type)
                    continue

                blocks = []
                for key in self.N[target_name][atomic_type].keys:
                    N_block = self.N[target_name][atomic_type][key]
                    Y_block = self.Y[target_name][atomic_type][key]
                    Y2_block = self.Y2[target_name][atomic_type][key]

                    N_values = N_block.values
                    Y_values = Y_block.values
                    Y2_values = Y2_block.values

                    # Compute the standard deviation
                    N = N_values.item()  # (do not use Bessel's correction)

                    if N <= 0:
                        # keep ones for this block (as initialized) and move on
                        blocks.append(
                            TensorBlock(
                                values=torch.ones_like(Y_block.values),
                                samples=Y_block.samples,
                                components=Y_block.components,
                                properties=Y_block.properties,
                            )
                        )
                        continue

                    # Take the norm over components and re-expand the component dims
                    Y_values = torch.norm(Y_values, dim=1, keepdim=True)
                    Y2_values = torch.norm(Y2_values, dim=1, keepdim=True)
                    Y_values = Y_values.repeat(
                        1, *[len(comp) for comp in Y_block.components], 1
                    )
                    Y2_values = Y2_values.repeat(
                        1, *[len(comp) for comp in Y_block.components], 1
                    )

                    # Divide the scales by sqrt(num components) to account for the
                    # fact that we took the norm over components when accumulating
                    # Y and Y2
                    if len(Y_values.shape) == 2:
                        component_factor = 1
                    else:
                        component_factor = Y_values.shape[1]

                    import math

                    scale_vals = torch.sqrt(
                        (Y2_values / N) - (Y_values / N) ** 2
                    ) / math.sqrt(component_factor)

                    blocks.append(
                        TensorBlock(
                            values=scale_vals.contiguous(),
                            samples=Labels(["_"], torch.tensor([[0]])).to(
                                device=scale_vals.device
                            ),
                            components=[
                                comp.to(device=scale_vals.device)
                                for comp in Y_block.components
                            ],
                            properties=Y_block.properties.to(device=scale_vals.device),
                        )
                    )

                self.scales[target_name][atomic_type] = TensorMap(
                    self.N[target_name][atomic_type].keys.to(device=scale_vals.device),
                    blocks,
                )
                self.is_fitted[target_name][atomic_type] = True

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
        scaled_outputs: Dict[str, TensorMap] = {}
        for output_name in outputs:
            # If not supported, just return the output as is
            if output_name not in self.target_names:
                scaled_outputs[output_name] = outputs[output_name]
                continue

            output_map = outputs[output_name]
            n_blocks = len(output_map)  # TorchScript-friendly integer

            scaled_blocks: List[TensorBlock] = []
            for i in range(n_blocks):
                output_block = output_map.block(i)
                output_block_types = torch.cat([system.types for system in systems])
                scaled_vals = output_block.values

                for atomic_type in self.atomic_types:
                    type_mask = output_block_types == atomic_type
                    if type_mask.sum() == 0:
                        continue

                    # Find the scales block
                    scales_block = self.scales[output_name][atomic_type].block(i)

                    assert scales_block.properties == output_block.properties, (
                        f"Properties of scales block {scales_block.properties} "
                        f"do not match output block {output_block.properties} "
                        f"for key {self.scales[output_name].keys[i]}."
                    )

                    # Scale the values of the output block
                    if remove:  # remove the scaler
                        scaled_vals[type_mask] = (
                            output_block.values[type_mask] / scales_block.values
                        )
                    else:  # apply the scaler
                        scaled_vals[type_mask] = (
                            output_block.values[type_mask] * scales_block.values
                        )

                scaled_blocks.append(
                    TensorBlock(
                        values=scaled_vals,
                        samples=output_block.samples,
                        components=output_block.components,
                        properties=output_block.properties,
                    )
                )

            scaled_outputs[output_name] = TensorMap(
                outputs[output_name].keys,
                scaled_blocks,
            )

        return scaled_outputs

    def _sync_device_dtype(self, device: torch.device, dtype: torch.dtype):
        # manually move the TensorMap dicts:

        self.N = {
            target_name: {
                atomic_type: tensor.to(device=device, dtype=dtype)
                for atomic_type, tensor in self.N[target_name].items()
            }
            for target_name in self.N.keys()
        }
        self.Y = {
            target_name: {
                atomic_type: tm.to(device=device, dtype=dtype)
                for atomic_type, tm in self.Y[target_name].items()
            }
            for target_name in self.Y.keys()
        }
        self.Y2 = {
            target_name: {
                atomic_type: tm.to(device=device, dtype=dtype)
                for atomic_type, tm in self.Y2[target_name].items()
            }
            for target_name in self.Y2.keys()
        }
        self.scales = {
            target_name: {
                atomic_type: tm.to(device=device, dtype=dtype)
                for atomic_type, tm in self.scales[target_name].items()
            }
            for target_name in self.scales.keys()
        }
