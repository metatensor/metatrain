"""
Contains the ``BaseScaler`` class. This is intended for eventual porting to metatomic.
The class ``Scaler`` wraps this to be compatible with metatrain-style objects.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


FixedScalerWeights = dict[str, Union[float, dict[int, float]]]


class BaseScaler(torch.nn.Module):
    """
    Fits a scaler for a dict of targets. Scales are computed as the per-property (and
    therefore per-block) standard deviations. By default, the scales are also
    computed per atomic type for per-atom targets.

    The :py:method:`accumulate` method is used to accumulate the necessary quantities
    based on the training data, and the :py:method:`fit` method is used to fit the model
    based on the accumulated quantities. These should both be called before the
    :py:method:`forward` method is called to compute the scales at inference
    time.

    :param atomic_types: List of atomic types to use in the composition model.
    :param layouts: Dict of zero-sample layout :py:class:`TensorMap` corresponding to
        each target. The keys of the dict are the target names, and the values are
        :py:class:`TensorMap` objects with the zero-sample layout for each target.
    """

    # Needed for torchscript compatibility
    target_names: List[str]
    scales: Dict[str, TensorMap]
    sample_kinds: Dict[str, str]
    type_to_index: torch.Tensor
    N: Dict[str, TensorMap]
    Y2: Dict[str, TensorMap]
    per_property_N: Dict[str, TensorMap]
    per_property_Y2: Dict[str, TensorMap]
    per_property_scales: Dict[str, TensorMap]  # per-property scales
    per_target_scales: Dict[str, TensorMap]  # per-target scales
    multi_property_target_names: List[str]

    def __init__(self, atomic_types: List[int], layouts: Dict[str, TensorMap]) -> None:
        super().__init__()

        self.atomic_types = torch.as_tensor(atomic_types, dtype=torch.int32)
        self.target_names = []
        self.sample_kinds = {}
        self.N = {}
        self.Y2 = {}
        self.scales = {}
        self.per_property_N = {}
        self.per_property_Y2 = {}
        self.per_property_scales = {}
        self.per_target_scales = {}
        self.multi_property_target_names = []

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

        # First, the full scales. These are the multiplication of per-target and
        # per-property (if applicable) scales, stored with the same layout as the
        # targets for convenient application. For single-block, single-property targets,
        # these scales are just the per-target scales as the per-property scales are by
        # definition 1.
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

        # Store the per-target scales separately, as these are needed to be
        # applied/removed separately from the per-property scales, i.e. during training
        self.per_target_scales[target_name] = TensorMap(
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

        if len(layout.keys) > 1 or len(layout[0].properties) > 1:
            self.multi_property_target_names.append(target_name)

        # Next, per-property scales. These have a single value per-block and
        # per-property in the target, which is also separately computed for each atomic
        # type for per-atom targets. These are only computed for targets with multiple
        # blocks or multiple properties.
        self.per_property_N[target_name] = TensorMap(
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
        self.per_property_Y2[target_name] = TensorMap(
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
        self.per_property_scales[target_name] = TensorMap(
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

    def _compute_N_and_Y2(
        self,
        systems: List[System],
        target_name: str,
        target: TensorMap,
        per_property: bool,
        mask: Optional[TensorMap],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        N_list = []
        Y2_list = []
        for key, block in target.items():
            Y_block = block.to(device=self.N[target_name][0].values.device)
            Y = Y_block.values

            if per_property:
                # Compute sum over all axes except the property axis
                dim = list(range(0, Y.dim() - 1))
            else:
                # Compute sum over all axes
                dim = list(range(0, Y.dim()))

            # First get the mask
            if mask is None:  # inferred from target
                mask_vals = ~torch.isnan(Y_block.values)
            else:  # mask provided
                mask_vals = mask[key].values

            # Set any NaNs to zero so they don't contribute to the sum
            mask_vals = mask_vals.to(Y.dtype)
            Y[torch.isnan(Y)] = 0.0

            # Now handle the different cases: per-target vs per-property and
            # per-structure vs per-atom. Hnadle the mask in all cases.
            if self.sample_kinds[target_name] == "per_structure":
                N = mask_vals.sum(dim=dim)
                Y2 = torch.sum((Y * mask_vals) ** 2, dim=dim)

            else:
                assert self.sample_kinds[target_name] == "per_atom"

                block_types = torch.cat([system.types for system in systems])

                # Initialize N and Y2 tensors for this block, which will store the
                # values for each atomic type. For per-property scales, these have shape
                # (n_types, n_properties), and for per-target scales, these have shape
                # (n_types,).
                if per_property:
                    shape = [len(self.atomic_types), len(Y_block.properties)]
                else:
                    shape = [len(self.atomic_types)]
                N = torch.zeros(
                    tuple(shape),
                    dtype=torch.long,
                    device=Y.device,
                )
                Y2 = torch.zeros(
                    tuple(shape),
                    dtype=Y.dtype,
                    device=Y.device,
                )

                if "atom_type" in key.names:
                    atomic_type = key["atom_type"]
                    i = self.type_to_index[atomic_type]

                    N[i] = mask_vals.sum(dim=dim)
                    Y2[i] = torch.sum(
                        (Y * mask_vals) ** 2,
                        dim=dim,
                    )
                else:
                    for i, atomic_type in enumerate(self.atomic_types):
                        type_mask = block_types == atomic_type

                        N[i] = mask_vals[type_mask].sum(dim=dim)
                        Y2[i] = torch.sum(
                            (Y[type_mask] * mask_vals[type_mask]) ** 2,
                            dim=dim,
                        )

            N_list.append(N)
            Y2_list.append(Y2)

        return N_list, Y2_list

    def accumulate(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> None:
        """
        Takes a batch of targets, and for each target accumulates the necessary
        quantities, i.e. the sum over the squared samples (Y2), and the number of
        samples overall (N). This function computes a single scale per target for
        per-structure targets, or a scale per atomic type for per-atom quantities.

        :param systems: List of systems corresponding to the targets.
        :param targets: Dict of names to targets to accumulate. The names (keys) should
            be a subset of the target names used during fitting.
        :param extra_data: Optional dict of extra data, e.g., masks for the targets
            (e.g., for padded samples).
        """

        if extra_data is None:
            extra_data = {}

        device = list(targets.values())[0][0].values.device
        dtype = list(targets.values())[0][0].values.dtype
        self._sync_device_dtype(device, dtype)

        # accumulate per-target N and Y2 quantities
        for target_name, target in targets.items():
            mask = None
            if target_name + "_mask" in extra_data:
                mask = extra_data[target_name + "_mask"]

            N_list, Y2_list = self._compute_N_and_Y2(
                systems=systems,
                target_name=target_name,
                target=target,
                per_property=False,
                mask=mask,
            )

            # Stack N and Y2
            N = torch.stack(N_list).sum(dim=0).reshape(-1, 1)
            Y2 = torch.stack(Y2_list).sum(dim=0).reshape(-1, 1)

            # Store for each block, repeating the same values and copying along the
            # properties dimension
            for key in self.N[target_name].keys:
                self.N[target_name][key].values[:] += N.repeat(
                    1, self.N[target_name][key].values.shape[1]
                )
                self.Y2[target_name][key].values[:] += Y2.repeat(
                    1, self.Y2[target_name][key].values.shape[1]
                )

    def accumulate_per_property(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> None:
        """
        Takes a batch of targets, and for each target accumulates the necessary
        quantities, i.e. the sum over the squared samples (per_property_Y2), and the
        number of samples overall (per_property_N). This function computes per-block and
        per-property scales. If the target is per-atom, scales are computed separately
        for each atomic type.

        :param systems: List of systems corresponding to the targets.
        :param targets: Dict of names to targets to accumulate. The names (keys) should
            be a subset of the target names used during fitting.
        :param extra_data: Optional dict of extra data, e.g., masks for the targets
            (e.g., for padded samples).
        """

        if extra_data is None:
            extra_data = {}

        device = list(targets.values())[0][0].values.device
        dtype = list(targets.values())[0][0].values.dtype
        self._sync_device_dtype(device, dtype)

        # Only accumulate targets with multiple properties
        targets = {
            target_name: target
            for target_name, target in targets.items()
            if target_name in self.multi_property_target_names
        }

        # Remove the per-target scales from the targets before accumulating per-property
        # quantities, so that we only accumulate the pure per-property correction
        # factors.
        targets = self._apply_scales(
            systems,
            targets,
            remove=True,
            use_per_target_scales=True,
            use_per_property_scales=False,
        )

        # accumulate per-property quantities
        for target_name, target in targets.items():
            mask = None
            if target_name + "_mask" in extra_data:
                mask = extra_data[target_name + "_mask"]

            N_list, Y2_list = self._compute_N_and_Y2(
                systems=systems,
                target_name=target_name,
                target=target,
                per_property=True,
                mask=mask,
            )
            for key, N, Y2 in zip(target.keys, N_list, Y2_list, strict=True):
                self.per_property_N[target_name][key].values[:] += N
                self.per_property_Y2[target_name][key].values[:] += Y2

    def fit(
        self,
        fixed_weights: Optional[FixedScalerWeights] = None,
        targets_to_fit: Optional[List[str]] = None,
    ) -> None:
        """
        Based on the pre-accumulated quantities from the training data, computes the
        per-target scales.

        :param fixed_weights: Optional dict of fixed weights to apply to the scales of
            each target. The keys of the dict are the target names, and the values are
            either a single float value to be applied to all atomic types, or a dict
            mapping atomic type (int) to weight (float). If not provided, all scales
            will be computed based on the accumulated quantities.
        :param targets_to_fit: Optional list of target names to fit. If not provided,
            all targets will be fitted.
        """
        if targets_to_fit is None:
            targets_to_fit = self.target_names

        if fixed_weights is None:
            fixed_weights = {}

        # fit and store per-target scales
        for target_name in targets_to_fit:
            if target_name in fixed_weights:
                self._set_fixed_weights(target_name, fixed_weights[target_name])
            else:
                for key in self.scales[target_name].keys:
                    scales_vals = (
                        self.Y2[target_name].block(key).values
                        / self.N[target_name].block(key).values
                    ) ** 0.5
                    self.scales[target_name][key].values[:] = scales_vals
                    self.per_target_scales[target_name][key].values[:] = scales_vals

    def fit_per_property(
        self,
        targets_to_fit: Optional[List[str]] = None,
    ) -> None:
        """
        Based on the pre-accumulated quantities from the training data, computes the
        per-block, per-property scales for each target. This only applies to targets
        with multiple properties. If a target is per-atom, scales are computed for each
        atomic type separately.

        :param targets_to_fit: Optional list of target names to fit. If not provided,
            all targets will be fitted.
        """
        if targets_to_fit is None:
            targets_to_fit = self.target_names

        # Only fit and store per-property scales for targets with multiple properties
        targets_to_fit = [
            target_name
            for target_name in self.target_names
            if target_name in self.multi_property_target_names
        ]

        # fit per-block, per-property scales
        for target_name in targets_to_fit:
            blocks = []
            for key in self.per_property_N[target_name].keys:
                N_block = self.per_property_N[target_name][key]
                Y2_block = self.per_property_Y2[target_name][key]

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

                    # Compute std without Bessel's correction
                    scale_vals_type = torch.sqrt(Y2_values_type / N_values_type)

                    scale_vals_type = scale_vals_type.contiguous()
                    block.values[type_index][:] = scale_vals_type

                # Update the full scales by multiplying the per-target scales with the
                # new per-property scales
                self.scales[target_name][key].values[:] = (
                    self.per_target_scales[target_name][key].values * block.values
                )

                # If any scales are zero or NaN, set them to 1.0
                block.values[torch.isnan(block.values)] = 1.0
                self.scales[target_name][key].values[
                    torch.isnan(self.scales[target_name][key].values)
                ] = 1.0

                blocks.append(block)

            # Store the per-property scales
            self.per_property_scales[target_name] = TensorMap(
                self.per_property_Y2[target_name].keys.to(
                    device=scale_vals_type.device
                ),
                blocks,
            )

    def _apply_scales(
        self,
        systems: List[System],
        outputs: Dict[str, TensorMap],
        remove: bool,
        use_per_target_scales: bool,
        use_per_property_scales: bool,
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Applies/removes scales to/from the outputs.

        If ``per_property`` is False, applies/removes the full scales. If
        ``per_property`` is True, applies/removes the per-block, per-property scales
        only. This only applies to targets with multiple blocks or multiple properties,
        as these are the only targets for which per-block, per-property scales are
        computed and stored.

        In both cases, if the target is per-atom, scales are applied/removed separately
        for each atomic type.

        :param systems: List of systems corresponding to the outputs.
        :param outputs: Dict of names outputs to which scales should be applied/removed.
            The names (keys) should be a subset of the target names used during fitting.
            If ``per_property`` is True, only targets with multiple properties (i.e. > 1
            block or >= 1 block with > 1 property) can be included in ``outputs``.
        :param remove: If True, removes the scaling (i.e., divides by the scales). If
            False, applies the scaling (i.e., multiplies by the scales).
        :param use_per_target_scales: If True, applies/removes per-target scales.
        :param use_per_property_scales: If True, applies/removes per-block, per-property
            scales.
        :param selected_atoms: Optional labels for selected atoms. If provided, scales
            will be applied/removed only to the selected atoms, and the appropriate
            scales will be selected based on the atomic types of the selected atoms.
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

            if not use_per_target_scales and use_per_property_scales:
                if output_name not in self.multi_property_target_names:
                    # per-property scales for targets with only one property are all by
                    # definition 1, so we can skip applying them
                    predictions[output_name] = outputs[output_name]
                    continue

            output_tmap = outputs[output_name]

            prediction_blocks: List[TensorBlock] = []
            for key, output_block in output_tmap.items():
                # Find the scales block and check metadata
                if use_per_target_scales and use_per_property_scales:
                    # Apply full scales
                    scales_block = self.scales[output_name].block(key)
                elif use_per_target_scales and not use_per_property_scales:
                    # Apply per-target scales
                    scales_block = self.per_target_scales[output_name].block(key)
                elif not use_per_target_scales and use_per_property_scales:
                    # Apply per-property scales
                    scales_block = self.per_property_scales[output_name].block(key)
                else:
                    raise ValueError(
                        "At least one of `use_per_target_scales` or "
                        "`use_per_property_scales` must be True."
                    )

                assert scales_block.properties == output_block.properties, (
                    f"Properties of scales block {scales_block.properties} "
                    f"do not match output block {output_block.properties} "
                    f"for key {key}."
                )

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
                    assert self.sample_kinds[output_name] == "per_atom"

                    output_block_types = torch.cat([system.types for system in systems])
                    if "atom_type" in key.names:
                        atom_type = key["atom_type"]
                        output_block_types = torch.tensor(
                            [atom_type] * torch.sum(output_block_types == atom_type)
                        )

                    if selected_atoms is not None:
                        # Scale each atomic type separately, also handling selected
                        # atoms and/or potential reordering
                        system_indices = output_block.samples.values[:, 0]
                        atom_indices = output_block.samples.values[:, 1]
                        system_lengths = torch.tensor(
                            [len(s.types) for s in systems],
                            dtype=torch.long,
                            device=device,
                        )
                        offset = torch.cat(
                            [
                                torch.zeros(1, dtype=torch.long, device=device),
                                torch.cumsum(system_lengths[:-1], dim=0),
                            ]
                        )
                        output_block_types = output_block_types[
                            offset[system_indices] + atom_indices
                        ]

                    # TODO: gradients of per-atom targets are not supported
                    if len(output_block.gradients_list()) > 0:
                        raise NotImplementedError(
                            "scaling of gradients is not implemented for per-atom "
                            f"target '{output_name}'"
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

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, TensorMap],
        remove: bool,
        selected_atoms: Optional[Labels] = None,
        use_per_target_scales: bool = False,
        use_per_property_scales: bool = False,
    ) -> Dict[str, TensorMap]:
        """
        Scales the targets based on the stored standard deviations.

        :param systems: List of systems corresponding to the for which the outputs were
            computed.
        :param outputs: Dict of names outputs to scale. The names (keys) should be a
            subset of the target names used during fitting.
        :param remove: If True, removes the scaling (i.e., divides by the scales). If
            False, applies the scaling (i.e., multiplies by the scales).
        :param selected_atoms: Optional labels for selected atoms.
        :param use_per_target_scales: If True, applies/removes per-target scales.
        :param use_per_property_scales: If True, applies/removes per-block, per-property
            scales.
        :returns: A dictionary with the scaled outputs for each system.

        :raises ValueError: If no scales have been computed or if `outputs` keys contain
            unsupported keys.
        """

        # If removing scales, first remove per-target scales, then per-property scales.
        # Otherwise if applying scales, apply in the reverse order.
        predictions: Dict[str, TensorMap] = {
            output_name: outputs[output_name] for output_name in outputs
        }
        return self._apply_scales(
            systems,
            predictions,
            remove=remove,
            use_per_target_scales=use_per_target_scales,
            use_per_property_scales=use_per_property_scales,
            selected_atoms=selected_atoms,
        )

    def _set_fixed_weights(
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
                "Multiple blocks are not supported for fixed weights in `Scaler` "
                f"for target '{target_name}'"
            )
        if len(self.scales[target_name].block().properties) > 1:
            raise NotImplementedError(
                f"Multiple properties are not supported for fixed weights in `Scaler` "
                f"for target '{target_name}'"
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
                        f"target '{target_name}'"
                    )
                # Error out if any atomic types are missing
                if int(atomic_type) not in weights:
                    raise ValueError(
                        f"Atomic type {atomic_type} is missing from the fixed scaling "
                        f"weights for target '{target_name}'"
                    )
                for atom_type, weight in weights.items():
                    block.values[self.type_to_index[atom_type], 0] = weight
        elif isinstance(weights, float):
            if self.sample_kinds[target_name] == "per_atom":
                logging.info(
                    "Fixed weights provided as a single float for per-atom "
                    f"target '{target_name}'. The same weight will be applied to "
                    "all atomic types."
                )
            block.values[:] = weights
        else:
            raise ValueError(
                f"weights for '{target_name}' must be either a float or a dict of "
                "int to float."
            )

        self.scales[target_name] = TensorMap(
            self.Y2[target_name].keys.to(device=block.values.device),
            [block],
        )
        self.per_target_scales[target_name] = TensorMap(
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
        self.per_target_scales = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.per_target_scales.items()
        }
        self.per_property_N = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.per_property_N.items()
        }
        self.per_property_Y2 = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.per_property_Y2.items()
        }
        self.per_property_scales = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.per_property_scales.items()
        }
