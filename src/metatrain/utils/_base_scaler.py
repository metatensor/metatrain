"""
Contains the ``BaseCompositionModel class. This is intended for eventual porting to
metatomic. The class ``CompositionModel`` wraps this to be compatible with
metatrain-style objects.
"""

from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


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
    N: Dict[str, TensorMap]
    Y: Dict[str, TensorMap]
    Y2: Dict[str, TensorMap]

    def __init__(self, layouts: Dict[str, TensorMap]) -> None:
        """
        Initializes the composition model with the given atomic types and layouts.

        :param atomic_types: List of atomic types to use in the composition model.
        :param layouts: Dict of zero-sample layout :py:class:`TensorMap` corresponding
            to each target. The keys of the dict are the target names, and the values
            are :py:class:`TensorMap` objects with the zero-sample layout for each
            target.
        """
        super().__init__()

        self.target_names = []
        self.N = {}
        self.Y = {}
        self.Y2 = {}
        self.scales = {}
        self.is_fitted: Dict[str, bool] = {}

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
        self.is_fitted[target_name] = False

        # Initialize TensorMaps for XTX and XTY for this target.
        #
        #  - N is a tensor with a single value
        #  - XTY is a matrix of shape (n_atomic_types, n_components, n_properties)
        #
        # Both are initialized with zeros, and accumulated during fitting by iterating
        # over batches in the passed dataloader.
        #
        # The weights are also a matrix of shape (n_atomic_types, n_components,
        # n_properties), and are initialized for torchscript compatibility.
        #
        # Then a linear system is solved for each target to obtain the composition
        # weights, which are stored in the `weights` attribute.
        self.N[target_name] = TensorMap(
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
        self.Y[target_name] = TensorMap(
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
        self.Y2[target_name] = TensorMap(
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
        self.scales[target_name] = TensorMap(
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
                    Y = block.gradient("positions").values
                else:
                    Y = block.values

                Y = Y.to(device=device, dtype=dtype)

                # Compute the number of samples in this block, account for the mask if
                # available
                if mask is None:
                    self.N[target_name][key].values[:] += Y.shape[0]
                else:
                    # Count N as the number of samples where the mask is True for at
                    # least one property (in other words where the mask for a given
                    # sample is not all False)
                    mask_values = mask.block(key).values
                    samples_mask = mask_values.any(dim=list(range(1, Y.dim())))
                    self.N[target_name][key].values[:] += samples_mask.sum()
                    Y = Y[samples_mask]

                # Compute the Y and Y2 values
                Y_values = torch.sum(
                    Y, dim=tuple(range(len(Y.shape) - 1)), keepdim=True
                )
                Y2_values = torch.sum(
                    Y**2, dim=list(range(len(Y.shape) - 1)), keepdim=True
                )

                # Repeat the along the component axes (if any) and accumulate
                n_components: List[int] = []
                for comp in Y.shape[1:-1]:
                    n_components.append(comp)
                self.Y[target_name][key].values[:] += Y_values.repeat(
                    1, *n_components, 1
                )
                self.Y2[target_name][key].values[:] += Y2_values.repeat(
                    1, *n_components, 1
                )

    def fit(self) -> None:
        """
        Based on the pre-accumulated quantities from the training data, computes the
        scales for each target.
        """

        # fit
        for target_name in self.target_names:
            if self.is_fitted[target_name]:  # already fitted
                continue

            blocks = []
            for key in self.N[target_name].keys:
                N_block = self.N[target_name][key]
                Y_block = self.Y[target_name][key]
                Y2_block = self.Y2[target_name][key]

                N_values = N_block.values
                Y_values = Y_block.values
                Y2_values = Y2_block.values

                # Compute the standard deviation
                N = N_values.item()  # (do not use Bessel's correction)
                scale_vals = torch.sqrt((Y2_values / N) - (Y_values / N) ** 2)

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

            self.scales[target_name] = TensorMap(
                self.N[target_name].keys.to(device=scale_vals.device),
                blocks,
            )
            self.is_fitted[target_name] = True

    def forward(
        self,
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

            scales_map = self.scales[output_name]
            output_map = outputs[output_name]
            n_blocks = len(scales_map)  # TorchScript-friendly integer

            scaled_blocks: List[TensorBlock] = []
            for i in range(n_blocks):
                scales_block = scales_map.block(i)
                output_block = output_map.block(i)

                assert scales_block.properties == output_block.properties, (
                    f"Properties of scales block {scales_block.properties} "
                    f"do not match output block {output_block.properties} "
                    f"for key {self.scales[output_name].keys[i]}."
                )

                # Scale the values of the output block
                if remove:  # remove the scaler
                    scaled_vals = output_block.values / scales_block.values
                else:  # apply the scaler
                    scaled_vals = output_block.values * scales_block.values
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
            target_name: tensor.to(device=device, dtype=dtype)
            for target_name, tensor in self.N.items()
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
