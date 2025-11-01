"""
Contains the ``BaseScaler`` class. This is intended for eventual porting to metatomic.
The class ``Scaler`` wraps this to be compatible with metatrain-style objects.
"""

import logging
from typing import Dict, List, Optional, Union

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap
from metatomic.torch import System


class BaseScaler(torch.nn.Module):
    """
    Fits a scaler for a dict of targets. Scales are computed as the per-property (and
    therefore per-block) standard deviations. By default, the scales are also computed
    per atomic type for per-atom targets.

    The :py:method:`accumulate` method is used to accumulate the necessary quantities
    based on the training data, and the :py:method:`fit` method is used to fit the model
    based on the accumulated quantities. These should both be called before the
    :py:method:`forward` method is called to compute the scales at inference time.

    :param atomic_types: List of atomic types to use in the composition model.
    :param layouts: Dict of zero-sample layout :py:class:`TensorMap` corresponding
        to each target. The keys of the dict are the target names, and the values
        are :py:class:`TensorMap` objects with the zero-sample layout for each
        target.
    """

    # Needed for torchscript compatibility
    target_names: List[str]
    scales: Dict[str, TensorMap]
    sample_kinds: Dict[str, str]
    type_to_index: torch.Tensor
    N: Dict[str, TensorMap]
    Y2: Dict[str, TensorMap]

    def __init__(self, atomic_types: List[int], layouts: Dict[str, TensorMap]) -> None:
        super().__init__()

        self.atomic_types = torch.as_tensor(atomic_types, dtype=torch.int32)
        self.target_names = []
        self.sample_kinds = {}
        self.N = {}
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
            [
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
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

        elif layout.sample_names == valid_sample_names[2]:
            self.sample_kinds[target_name] = "per_pair"
            samples = Labels(
                ["atomic_type"], torch.arange(len(self.atomic_types)).reshape(-1, 1)
            )

        else:
            raise ValueError(
                "unknown sample kind. TensorMap has sample names"
                f" {layout.sample_names} but expected one of "
                f"{valid_sample_names}."
            )

        layout = mts.filter_blocks(
            layout,
            Labels(
                layout.keys.names,
                torch.vstack([key.values for key in layout.keys if _include_key(key)]),
            ),
        )

        # Initialize TensorMaps for the quantities to accumulate for this target.
        self.N[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(samples),
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
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
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
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
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
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
        necessary quantities, i.e. the sum over the squared
        samples (Y2), and the number of samples overall (N).

        :param systems: List of systems corresponding to the targets.
        :param targets: Dict of names to targets to accumulate. The names (keys)
            should be a subset of the target names used during fitting.
        :param extra_data: Optional dict of extra data, e.g., masks for the targets
            (e.g., for padded samples).
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
                if not _include_key(key):
                    continue

                if self.sample_kinds[target_name] == "per_structure":
                    if mask is not None:
                        raise NotImplementedError(
                            "masks for per-structure targets are not yet supported"
                        )

                    Y_block = block.to(device=device, dtype=dtype)
                    Y = Y_block.values

                    if "non_conservative_stress" in target_name:
                        # For non-conservative stresses, we need to take into account
                        # the fact that we allow the user to define NaN stresses for
                        # non-fully-periodic systems.
                        valid_mask = ~torch.isnan(Y)
                        Y = Y[valid_mask]

                    # Compute sum over all axes except the property axis
                    N = Y.numel() // Y.shape[-1] if Y.numel() > 0 else 0
                    Y2_values = torch.sum(Y**2, dim=list(range(0, Y.dim() - 1)))

                    self.N[target_name][key].values[0] += N
                    self.Y2[target_name][key].values[0] += Y2_values

                else:
                    assert self.sample_kinds[target_name] in ["per_atom", "per_pair"]

                    if "n_centers" in key.names:
                        assert key["n_centers"] == 1

                    Y_block = block.to(device=device, dtype=dtype)

                    # Here it is assumed that the samples of the block correspond to the
                    # full ordered list of atoms in the batch of systems
                    Y_block_types = torch.cat([system.types for system in systems])

                    for atomic_type in self.atomic_types:
                        # Slice the block to only include samples of the current atomic
                        # type
                        samples_type_mask = Y_block_types == atomic_type
                        Y = Y_block.values[samples_type_mask]

                        # Compute the number of samples and components in this block,
                        # account for the mask if available
                        if mask is None:
                            N = Y.numel() // Y.shape[-1]
                        else:
                            # For each property, count N as the number of samples *
                            # components where the mask is True.
                            mask_values = mask.block(key).values[samples_type_mask]
                            assert Y.shape == mask_values.shape
                            N = torch.sum(mask_values, dim=list(range(0, Y.dim() - 1)))

                            # To ensure that Y^2 isn't accumulated on masked entries,
                            # mutliply Y by the mask (either 1 for real data or 0 for
                            # padded data).
                            Y = Y * mask_values.to(Y.dtype)

                        # Compute the Y2 values and sum over samples and components
                        Y2_values = torch.sum(Y**2, dim=list(range(0, Y.dim() - 1)))

                        # Repeat the along the component axes (if any) and accumulate
                        self.N[target_name][key].values[
                            self.type_to_index[atomic_type]
                        ] += N
                        self.Y2[target_name][key].values[
                            self.type_to_index[atomic_type]
                        ] += Y2_values

    def fit(
        self,
        fixed_weights: Optional[Dict[str, Union[float, Dict[int, float]]]] = None,
        targets_to_fit: Optional[List[str]] = None,
    ) -> None:
        """
        Based on the pre-accumulated quantities from the training data, computes the
        scales for each target.

        :param fixed_weights: Optional dict of fixed weights to apply to the scales
            of each target. The keys of the dict are the target names, and the values
            are either a single float value to be applied to all atomic types, or a
            dict mapping atomic type (int) to weight (float). If not provided, all
            scales will be computed based on the accumulated quantities.
        :param targets_to_fit: Optional list of target names to fit. If not provided,
            all targets will be fitted.
        """
        if targets_to_fit is None:
            targets_to_fit = self.target_names

        if fixed_weights is None:
            fixed_weights = {}

        # fit
        for target_name in targets_to_fit:
            if target_name in fixed_weights:
                self._apply_fixed_weights(target_name, fixed_weights[target_name])
                continue

            blocks = []
            for key in self.N[target_name].keys:
                N_block = self.N[target_name][key]
                Y2_block = self.Y2[target_name][key]

                N_values = N_block.values
                Y2_values = Y2_block.values

                if self.sample_kinds[target_name] == "per_structure":
                    assert len(Y2_block.samples) == 1

                # Set a sensible default in case we don't compute a scale below
                block = TensorBlock(
                    values=torch.ones_like(Y2_block.values),
                    samples=Y2_block.samples,
                    components=Y2_block.components,
                    properties=Y2_block.properties,
                )

                # Now iterate over all the atomic types in this block. For per-structure
                # targets, this is just one iteration as we do not compute
                # per-atomic-type
                for type_index in range(len(Y2_block.samples)):
                    N_values_type = N_values[type_index].unsqueeze(0)
                    Y2_values_type = Y2_values[type_index].unsqueeze(0)

                    # Compute std
                    scale_vals_type = torch.sqrt(
                        Y2_values_type / N_values_type
                    )  # (do not use Bessel's correction)

                    # Provide a warning for scales that cannot be computed. These will
                    # be NaN as N_values_type for this property will be zero.
                    if torch.isnan(
                        scale_vals_type
                    ).any():  # this can only happen for per-atom targets
                        assert self.sample_kinds[target_name] in [
                            "per_atom",
                            "per_pair",
                        ]
                        logging.info(
                            f"Per-atom target {target_name} has not enough samples in "
                            f"block {key} for atomic type"
                            f"{self.atomic_types[type_index]} to compute statistics. "
                            "The scales of one or more property cannot be computed."
                        )
                        scale_vals_type[torch.isnan(scale_vals_type)] = 1.0

                    # If any scales are zero, set them to 1.0
                    if torch.any(scale_vals_type == 0):
                        scale_vals_type[scale_vals_type == 0] = 1.0

                    scale_vals_type = scale_vals_type.contiguous()
                    block.values[type_index][:] = scale_vals_type

                blocks.append(block)

            self.scales[target_name] = TensorMap(
                self.Y2[target_name].keys.to(device=scale_vals_type.device),
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

        :param systems: List of systems corresponding to the for which the outputs
            were computed.
        :param outputs: Dict of names outputs to scale. The names (keys) should be a
            subset of the target names used during fitting.
        :param remove: If True, removes the scaling (i.e., divides by the scales). If
            False, applies the scaling (i.e., multiplies by the scales).
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
            if output_name not in self.target_names:
                # just return output as is (e.g., auxiliary outputs)
                predictions[output_name] = outputs[output_name]
                continue

            output_tmap = outputs[output_name]

            prediction_blocks: List[TensorBlock] = []
            for key, output_block in output_tmap.items():
                # TODO: Use the unscaled offsite blocks for now
                if key not in self.scales[output_name].keys:
                    # sanity check for now. TODO: handle offsite scaling
                    assert "n_centers" in key.names and key["n_centers"] == 2
                    prediction_blocks.append(output_block)
                    continue

                # Find the scales block and check metadata
                scales_block = self.scales[output_name].block(key)
                assert scales_block.properties == output_block.properties, (
                    f"Properties of scales block {scales_block.properties} "
                    f"do not match output block {output_block.properties} "
                    f"for key {key}."
                )

                # Scale each atomic type separately
                output_block_types = torch.cat([system.types for system in systems])
                scaled_vals = output_block.values

                # unsqueeze scales_block.values to make broadcasting work
                # (components are missing in scales_block)
                scales_block_values = scales_block.values
                for _ in range(scaled_vals.dim() - 2):
                    scales_block_values = scales_block_values.unsqueeze(1)

                if self.sample_kinds[output_name] == "per_structure":
                    # Scale the values of the output block
                    if remove:  # remove the scaler
                        scaled_vals = scaled_vals / scales_block_values[0]
                    else:  # apply the scaler
                        scaled_vals = scaled_vals * scales_block_values[0]

                    prediction_block = TensorBlock(
                        values=scaled_vals,
                        samples=output_block.samples,
                        components=output_block.components,
                        properties=output_block.properties,
                    )

                    # Gradients are scaled by the same factor(s) as the values
                    if len(output_block.gradients_list()) > 0:
                        for parameter, gradient in output_block.gradients():
                            if len(gradient.gradients_list()) != 0:
                                raise NotImplementedError(
                                    "gradients of gradients are not supported"
                                )

                            if remove:  # remove the scaler
                                scaled_gradient_vals = (
                                    gradient.values / scales_block_values[0]
                                )
                            else:
                                scaled_gradient_vals = (
                                    gradient.values * scales_block_values[0]
                                )

                            prediction_block.add_gradient(
                                parameter=parameter,
                                gradient=TensorBlock(
                                    values=scaled_gradient_vals,
                                    samples=gradient.samples,
                                    components=gradient.components,
                                    properties=gradient.properties,
                                ),
                            )

                else:
                    assert self.sample_kinds[output_name] in ["per_atom", "per_pair"]

                    # TODO: gradients of per-atom targets are not supported
                    if len(output_block.gradients_list()) > 0:
                        raise NotImplementedError(
                            "scaling of gradients is not implemented for per-atom "
                            "targets"
                        )

                    # Scale the values of the output block
                    if remove:  # remove the scaler
                        scaled_vals = (
                            scaled_vals
                            / scales_block_values[
                                self.type_to_index[output_block_types]
                            ]
                        )
                    else:  # apply the scaler
                        scaled_vals = (
                            scaled_vals
                            * scales_block_values[
                                self.type_to_index[output_block_types]
                            ]
                        )

                    prediction_block = TensorBlock(
                        values=scaled_vals,
                        samples=output_block.samples,
                        components=output_block.components,
                        properties=output_block.properties,
                    )
                prediction_blocks.append(prediction_block)

            predictions[output_name] = TensorMap(
                outputs[output_name].keys,
                prediction_blocks,
            )

        return predictions

    def _apply_fixed_weights(
        self, target_name: str, weights: Union[float, Dict[int, float]]
    ) -> None:
        """
        Apply fixed weights to the scales of a given target.

        :param target_name: Name of the target to which fixed weights should be applied.
        :param weights: Either a single float value to be applied to all atomic types,
            or a dict mapping atomic type (int) to weight (float).
        """
        # Error out if multiple blocks or multiple properties are present. These are
        # difficult to allow in the yaml files.
        if len(self.scales[target_name]) > 1:
            raise NotImplementedError(
                "Multiple blocks are not supported for fixed weights in `Scaler`."
            )
        if len(self.scales[target_name].block().properties) > 1:
            raise NotImplementedError(
                "Multiple properties are not supported for fixed weights in `Scaler`."
            )

        Y2_block = self.Y2[target_name].block()
        block = TensorBlock(
            values=torch.empty_like(Y2_block.values),  # [1, 1] or [n_types, 1]
            samples=Y2_block.samples,
            components=Y2_block.components,
            properties=Y2_block.properties,
        )

        if isinstance(weights, dict):
            for atomic_type in self.atomic_types.tolist():
                # Error out if `weights` is a dict but the target is per-structure
                if self.sample_kinds[target_name] == "per_structure":
                    raise ValueError(
                        "Fixed weights as a dict are not supported for per-structure "
                        "targets."
                    )
                # Error out if any atomic types are missing
                if int(atomic_type) not in weights:
                    raise ValueError(
                        f"Atomic type {atomic_type} is missing from the fixed scaling "
                        f"weights for target {target_name}."
                    )
                for atom_type, weight in weights.items():
                    block.values[self.type_to_index[atom_type], 0] = weight
        elif isinstance(weights, float):
            if self.sample_kinds[target_name] == "per_atom":
                logging.info(
                    "Fixed weights provided as a single float for a per-atom "
                    "target. The same weight will be applied to all atomic types."
                )
            block.values[:] = weights
        else:
            raise ValueError(
                "weights must be either a float or a dict of int to float."
            )

        self.scales[target_name] = TensorMap(
            self.Y2[target_name].keys.to(device=block.values.device),
            [block],
        )

    def _sync_device_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        # manually move the TensorMap dicts:

        self.atomic_types = self.atomic_types.to(device=device)
        self.type_to_index = self.type_to_index.to(device=device)
        self.N = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.N.items()
        }
        self.Y2 = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.Y2.items()
        }
        self.scales = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.scales.items()
        }


def _include_key(key: LabelsEntry) -> bool:
    """
    Determines whether a block indexed by the input ``key`` should be included in the
    scaler model. All keys except those for offsite blocks of per-pair targets are
    included.

    :param key: The key to check.

    :return: Whether the key should be included in the composition model.
    """
    valid_key_names = [
        ["_"],  # scalar
        ["o3_lambda", "o3_sigma"],  # spherical
        ["o3_lambda", "o3_sigma", "n_centers"],  # spherical per-atom
    ]
    include_key = False

    if key.names == valid_key_names[0]:
        include_key = True

    elif key.names == valid_key_names[1]:
        if key["o3_lambda"] == 0 and key["o3_sigma"] == 1:
            include_key = True

    elif key.names == valid_key_names[2]:
        if key["n_centers"] == 1:
            include_key = True
        else:
            assert key["n_centers"] == 2
            include_key = False

    else:
        raise ValueError(
            f"key names {key.names} not in valid key names {valid_key_names}"
        )

    return include_key
