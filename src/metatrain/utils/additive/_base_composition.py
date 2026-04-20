"""
Contains the ``BaseCompositionModel class. This is intended for eventual porting to
metatomic. The class ``CompositionModel`` wraps this to be compatible with
metatrain-style objects.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System


FixedCompositionWeights = dict[str, float | dict[int, float]]


class BaseCompositionModel(torch.nn.Module):
    """
    Fits a composition model for a dict of targets.

    A composition model is a model that predicts the target values based on the
    composition of the system, i.e., the chemical identity of atoms in the system.

    Only invariant blocks of the specified targets are fitted, i.e. those indexed by
    keys with a single name "_" (for scalars) or keys where "o3_lambda=0" and
    "o3_sigma=1" (for spherical targets).

    The :py:method:`accumulate` method is used to accumulate the necessary quantities
    based on the training data, and the :py:method:`fit` method is used to
    fit the model based on the accumulated quantities. These should both be called
    before the :py:method:`forward` method is called to compute the predictions.

    .. attribute :: atomic_types

        List of atomic types used in the model.

    .. attribute :: target_names

        List of target names in the model.

    .. attribute :: weights

        Dict of :py:class:`TensorMap` containing the fitted weights for each target.

    .. attribute :: sample_kinds

        Dict of sample kinds for each target. The sample kind can be either
        "per_structure" or "per_atom".

    .. attribute :: type_to_index

        Tensor mapping atomic types to their index in ``atomic_types``.

    .. attribute :: XTX

        Dict of :py:class:`TensorMap` containing the accumulated X^T * X for each
        target.

    .. attribute :: XTY

        Dict of :py:class:`TensorMap` containing the accumulated X^T * Y for each
        target.

    :param atomic_types: List of atomic types to use in the composition model.
    :param layouts: Dict of zero-sample layout :py:class:`TensorMap` corresponding
        to each target. The keys of the dict are the target names, and the values
        are :py:class:`TensorMap` objects with the zero-sample layout for each
        target.
    """

    # Needed for torchscript compatibility
    atomic_types: torch.Tensor
    target_names: List[str]
    weights: Dict[str, TensorMap]
    sample_kinds: Dict[str, str]
    type_to_index: torch.Tensor
    XTX: Dict[str, TensorMap]
    XTY: Dict[str, TensorMap]

    def __init__(
        self,
        atomic_types: Union[List[int], torch.Tensor],
        layouts: Dict[str, TensorMap],
    ) -> None:
        super().__init__()

        self.atomic_types = torch.as_tensor(atomic_types, dtype=torch.int32)
        self.target_names = []
        self.sample_kinds = {}
        self.XTX = {}
        self.XTY = {}
        self.weights = {}

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

        elif layout.sample_names == valid_sample_names[1]:
            self.sample_kinds[target_name] = "per_atom"

        else:
            raise ValueError(
                "unknown sample kind. TensorMap has sample names"
                f" {layout.sample_names} but expected one of "
                f"{valid_sample_names}."
            )

        # First slice the layout to only include the keys that the composition
        # model applies to. This is done by filtering the keys that have a single name
        # "_" (for scalars) or keys where "o3_lambda=0" and "o3_sigma=1"
        # (for spherical targets).
        layout = mts.filter_blocks(
            layout,
            Labels(
                layout.keys.names,
                torch.vstack([key.values for key in layout.keys if _include_key(key)]),
            ),
        )

        # Initialize TensorMaps for XTX and XTY for this target.
        #
        #  - XTX is a square matrix of shape (n_atomic_types, n_atomic_types)
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
        self.XTX[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(self.atomic_types),
                        len(self.atomic_types),
                        dtype=torch.float64,
                    ),
                    samples=Labels(["center_type"], self.atomic_types.reshape(-1, 1)),
                    components=[],
                    properties=Labels(
                        ["center_type"], self.atomic_types.reshape(-1, 1)
                    ),
                )
                for _ in layout
            ],
        )
        self.XTY[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(self.atomic_types),
                        *[len(c) for c in block.components],
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=Labels(["center_type"], self.atomic_types.reshape(-1, 1)),
                    components=block.components,
                    properties=block.properties,
                )
                for block in layout
            ],
        )
        self.weights[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(self.atomic_types),
                        *[len(c) for c in block.components],
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=Labels(["center_type"], self.atomic_types.reshape(-1, 1)),
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
    ) -> None:
        """
        Takes a batch of systems and targets, and for each target accumulates the
        necessary quantities (XTX and XTY).

        :param systems: List of systems in the batch.
        :param targets: Dict of target names to :py:class:`TensorMap` containing
            the target values for each system in the batch.
        """

        device = systems[0].positions.device
        dtype = systems[0].positions.dtype
        self._sync_device_dtype(device, dtype)

        # check that the systems contain no unexpected atom types
        for system in systems:
            if not torch.all(torch.isin(system.types, self.atomic_types)):
                raise ValueError(
                    "system contains unexpected atom types. "
                    f"Expected atomic types: {self.atomic_types}, "
                    f"found: {torch.unique(system.types)}"
                )

        # Compute the Xs that we will need.
        Xs: dict[str | int, torch.Tensor] = {}
        for target_name, sample_kind in self.sample_kinds.items():
            if sample_kind in Xs:
                continue
            if sample_kind == "per_structure":
                X = self._compute_X_per_structure(systems)
            elif sample_kind == "per_atom":
                X = self._compute_X_per_atom(systems, self.atomic_types)
            else:
                raise ValueError(
                    f"unknown sample kind: {sample_kind} for target {target_name}"
                )
            X = X.to(device=device, dtype=dtype)
            Xs[sample_kind] = (X, X.T @ X)

        if any("atom_type" in target.keys.names for target in targets.values()):
            # For atomic basis targets, blocks only contain the atoms of a
            # given type, so we need to subsample X to get only the atoms
            # of that type. Reminder: X has shape (n_atoms, n_atomic_types).
            # We are assuming here that targets with "atom_type" are per-atom!
            per_atom_X = Xs["per_atom"][0]
            for atom_type in self.atomic_types:
                type_index = self.type_to_index[atom_type]
                atom_type_mask = per_atom_X[:, type_index].bool()
                X = per_atom_X[atom_type_mask]
                Xs[int(atom_type)] = (X, X.T @ X)

        # accumulate
        for target_name, target in targets.items():
            for key, block in target.items():
                if not _include_key(key):
                    continue

                # Get X and XTX for this block.
                if "atom_type" in key.names:
                    X, XTX = Xs[int(key["atom_type"])]
                else:
                    X, XTX = Xs[self.sample_kinds[target_name]]

                # Get the target block values
                Y = block.values

                # Accumulate "XTX", i.e. X.T @ X
                # TODO: store XTX by sample kind instead, saving memory
                self.XTX[target_name][key].values[:] += XTX

                # Compute and accummulate "XTY", i.e. X.T @ Y
                XTY = self.XTY[target_name][key]
                if self.sample_kinds[target_name] != "per_atom":
                    # Explicitly compute X.T @ Y.
                    XTY.values[:] += torch.tensordot(X, Y, dims=([0], [0]))
                else:
                    # X in this case is a one hot encoding of the atom types,
                    # so we just need to sum Y over atoms of each type to get X.T @ Y.
                    # This avoids NaNs getting leaked from one atom type to another.
                    type_indices = X.argmax(dim=1)
                    # scatter_add_ does not broadcast, so we have to expand type_indices
                    idx = type_indices.reshape(-1, *[1] * len(Y.shape[1:])).expand_as(Y)
                    XTY.values.scatter_add_(dim=0, index=idx, src=Y)

    def _sanitize_fixed_weights(
        self,
        fixed_weights: Optional[FixedCompositionWeights],
    ) -> dict[str, dict[int, float]]:
        """Sanitizes the input fixed composition weights to ensure that all targets
        contain a dict of atomic types to weights.

        This function converts something like `{"energy": 1.0}` to
        `{"energy": {1: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}}` if the atomic types are
        `[1, 6, 7, 8]`.

        :param fixed_weights: The raw fixed weights provided by the user.
        :return: The sanitized fixed weights.
        """
        if fixed_weights is None:
            return {}

        atomic_types = self.atomic_types.tolist()

        sanitized_fixed_weights = {}
        for target_name, weights in fixed_weights.items():
            if target_name not in self.target_names:
                logging.warning(
                    f"Fixed weights provided for unknown target '{target_name}'. "
                    f"Available targets are: {self.target_names}"
                )
                continue

            if isinstance(weights, float):
                # A float is provided for this target, which means that the same
                # weight should be used for all atomic types.
                weights = {
                    int(atomic_type): float(weights) for atomic_type in atomic_types
                }
            elif missing_types := set(atomic_types) - set(weights):
                # The user provided a dict, check that all atomic types are present.
                raise ValueError(
                    f"Fixed weights for target '{target_name}' are missing "
                    f"the following atomic types: {missing_types}"
                )

            sanitized_fixed_weights[target_name] = weights

        return sanitized_fixed_weights

    def fit(
        self,
        fixed_weights: Optional[FixedCompositionWeights] = None,
        targets_to_fit: Optional[List[str]] = None,
    ) -> None:
        """
        Based on the pre-accumulated quantities from the training data, fits the
        compositions for each target.

        :param fixed_weights: Optional dict of target names to either (1) a single
            weight for all atomic_types or (2) a dict of atomic types to weights.
            If provided, these weights will be fixed instead of being fitted.
            If a dict of weights is provided for a target, all atomic types handled
            by the model must have a weight specified.
        :param targets_to_fit: List of target names to fit. If `None`,
            all targets in the model will be fitted.
        """
        if targets_to_fit is None:
            targets_to_fit = self.target_names

        sanitized_fixed_weights = self._sanitize_fixed_weights(fixed_weights)

        # fit
        for target_name in targets_to_fit:
            blocks = []
            for key in self.XTX[target_name].keys:
                XTX_block = self.XTX[target_name][key]
                XTY_block = self.XTY[target_name][key]

                XTX_values = XTX_block.values
                XTY_values = XTY_block.values

                if target_name in sanitized_fixed_weights:
                    weight_vals = torch.vstack(
                        [
                            torch.full(
                                (
                                    1,
                                    *[len(c) for c in XTY_block.components],
                                    len(XTY_block.properties),
                                ),
                                sanitized_fixed_weights[target_name][int(atomic_type)],
                                dtype=XTY_values.dtype,
                                device=XTY_values.device,
                            )
                            for atomic_type in self.atomic_types
                        ]
                    )
                else:
                    XTY_shape = XTY_values.shape
                    if len(XTY_values.shape) != 2:
                        XTY_values = XTY_values.reshape(XTY_values.shape[0], -1)

                    if torch.all(XTX_values == 0):
                        # If XTX is all zeros, it means that this key was not present in
                        # the training data, so we set the weights to zero to avoid
                        # numerical issues when solving the linear system.
                        # This can happen when we are handling an atomic-basis target.
                        logging.warning(
                            f"The composition model has not seen any block for {key}."
                            f" Setting composition weights to zero for this block."
                        )
                        weight_vals = torch.zeros(
                            XTY_shape,
                            dtype=XTY_values.dtype,
                            device=XTY_values.device,
                        )

                    else:
                        if self.sample_kinds[target_name] != "per_atom":
                            # Solve linear system explicitly.
                            weight_vals = _solve_linear_system(XTX_values, XTY_values)
                        else:
                            # XTX in this case is a diagonal matrix (the counts of atoms
                            # of each type), so we can solve it faster. This also avoids
                            # NaNs getting leaked from one atom type to another.
                            weight_vals = XTY_values / torch.diag(XTX_values).unsqueeze(
                                1
                            )
                        weight_vals = weight_vals.reshape(*XTY_shape)

                blocks.append(
                    TensorBlock(
                        values=weight_vals.contiguous(),
                        samples=XTY_block.samples.to(device=weight_vals.device),
                        components=XTY_block.components,
                        properties=XTY_block.properties.to(device=weight_vals.device),
                    )
                )

            self.weights[target_name] = TensorMap(
                self.XTX[target_name].keys.to(device=weight_vals.device),
                blocks,
            )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Compute the targets for each system based on the composition weights.

        :param systems: List of systems to calculate the energy.
        :param outputs: Dict of named outputs to compute. These names (in the keys)
            should be a subset of the target names used during fitting, and the values
            are the corresponding :py:class:`ModelOutput` objects.
        :param selected_atoms: Optional selection of atoms for which to compute the
            predictions.
        :return: A dictionary with the computed predictions for each system.

        :raises ValueError: If no weights have been computed or if `outputs` keys
            contain unsupported keys.
        """

        device = systems[0].positions.device
        dtype = systems[0].positions.dtype
        self._sync_device_dtype(device, dtype)

        # Build the sample labels that are required
        _, sample_labels = _get_system_indices_and_labels(systems, device)

        # Compute the X tensor
        X = self._compute_X_per_atom(systems, self.atomic_types)

        # Build the predictions for each output
        predictions: Dict[str, TensorMap] = {}
        for output_name in outputs:
            if output_name not in self.target_names:
                raise ValueError(
                    f"output {output_name} is not supported by this composition model."
                )
            weights = self.weights[output_name]

            prediction_key_vals = []
            prediction_blocks: List[TensorBlock] = []
            for key, weight_block in weights.items():
                sample_labels_block = sample_labels

                # If selected_atoms is provided, slice the samples labels and the X
                # tensor
                if selected_atoms is None:
                    X_block = X
                else:
                    sample_indices = sample_labels_block.select(selected_atoms)
                    sample_labels_block = Labels(
                        sample_labels_block.names,
                        sample_labels_block.values[sample_indices],
                    ).to(device=device)
                    X_block = X[sample_indices]

                # Handle the case of atomic basis targets where there is a subset of
                # atoms in the samples of each block
                if "atom_type" in key.names:
                    type_index = self.type_to_index[int(key["atom_type"])]
                    atom_type_mask = X_block[:, type_index].to(dtype=torch.bool)
                    sample_labels_block = Labels(
                        sample_labels_block.names,
                        sample_labels_block.values[atom_type_mask],
                    ).to(device=device)
                    X_block = X_block[atom_type_mask]

                # Compute X.T @ W
                if self.sample_kinds[output_name] != "per_atom":
                    out_vals = torch.tensordot(
                        X_block, weight_block.values, dims=([1], [0])
                    )
                else:
                    # No multiplication is needed for per-atom targets, we just need to
                    # broadcast the weights according to the atom types in the samples.
                    # This also avoids NaN getting leaked from one atom type to another.
                    out_vals = weight_block.values[X_block.argmax(dim=1)]

                prediction_blocks.append(
                    TensorBlock(
                        values=out_vals,
                        samples=sample_labels_block,
                        components=weight_block.components,
                        properties=weight_block.properties,
                    )
                )
                prediction_key_vals.append(key.values)

            prediction = TensorMap(
                Labels(
                    self.weights[output_name].keys.names,
                    torch.vstack(prediction_key_vals),
                    assume_unique=True,
                ),
                prediction_blocks,
            )

            # If a per-structure output is requested, sum over the sample dimensions
            # that aren't "system".
            if not outputs[output_name].per_atom:
                prediction = mts.sum_over_samples(prediction, "atom")
            predictions[output_name] = prediction

        return predictions

    def _compute_X_per_structure(self, systems: List[System]) -> torch.Tensor:
        """
        Computes the one-hot encoding of the atomic types for the atoms in the
        provided systems.

        :param systems: List of systems to compute the one-hot encoding for.

        :return: Tensor of shape ``(n_systems, n_atomic_types)``, where each row
            corresponds to a system and each column corresponds to an atomic type. The
            value is the number of atoms of that type in the system.
        """
        dtype = systems[0].positions.dtype

        counts = []
        for system in systems:
            bincount = torch.bincount(
                self.type_to_index[system.types], minlength=len(self.atomic_types)
            )
            counts.append(bincount.to(dtype=dtype))
        return torch.vstack(counts)

    def _compute_X_per_atom(
        self, systems: List[System], center_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the one-hot encoding of the atomic types for the atoms in the provided
        systems, but only for the specified center types.

        :param systems: List of systems to compute the one-hot encoding for.
        :param center_types: Tensor of atomic types to include in the one-hot encoding.

        :return: A tensor of shape ``(n_atoms, n_atomic_types)``, where each row
            corresponds to an atom in the systems and each column corresponds to an
            atomic type. The value is 1 if the atom's type matches the atomic type,
            and 0 otherwise.
        """
        dtype = systems[0].positions.dtype
        all_types = torch.concatenate([system.types for system in systems])
        all_types_as_indices = self.type_to_index[all_types]
        one_hot_encoding = torch.nn.functional.one_hot(
            all_types_as_indices, num_classes=len(center_types)
        )
        return one_hot_encoding.to(dtype)

    def _sync_device_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        # manually move the TensorMap dicts:

        self.atomic_types = self.atomic_types.to(device=device)
        self.type_to_index = self.type_to_index.to(device=device)
        self.XTX = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.XTX.items()
        }
        self.XTY = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.XTY.items()
        }
        self.weights = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.weights.items()
        }


def _include_key(key: LabelsEntry) -> bool:
    """
    Determines whether a block indexed by the input ``key`` should be included in the
    composition model.

    The rules are as follows:
        - If the key has a single name "_" (indicating a scalar), it is included.
        - If the key has names ["o3_lambda", "o3_sigma"] it is included if values are 0
          and 1 respectively (indicating an invariant block of a spherical target).
        - If the key has names ["o3_lambda", "o3_sigma", "atom_type"] it is included if
          values are 0, 1, and any value respectively (indicating an invariant block of
          an atomic-basis spherical target).
    :param key: The key to check.

    :return: Whether the key should be included in the composition model.
    """
    valid_key_names = [
        ["_"],  # scalar
        ["o3_lambda", "o3_sigma"],  # spherical
        ["o3_lambda", "o3_sigma", "atom_type"],  # spherical atomic basis
    ]
    include_key = False

    if key.names == valid_key_names[0]:
        include_key = True

    elif key.names == valid_key_names[1]:
        if key["o3_lambda"] == 0 and key["o3_sigma"] == 1:
            include_key = True

    elif key.names == valid_key_names[2]:
        if key["o3_lambda"] == 0 and key["o3_sigma"] == 1:
            include_key = True

    else:
        raise ValueError(
            f"key names {key.names} not in valid key names {valid_key_names}"
        )

    return include_key


def _solve_linear_system(
    XTX_vals: torch.Tensor, XTY_vals: torch.Tensor
) -> torch.Tensor:
    """
    Solves the linear system XTX * W = XTY for the weights W, where XTX is a square
    matrix of shape (n_atomic_types, n_atomic_types) and XTY is a matrix
    of shape (n_atomic_types, n_components, n_properties).

    :py:func:`metatensor.torch.solve` is not used due to numerical stability issues
    when the matrix is ill-conditioned. Instead, a regularization term is added to the
    diagonal of XTX to improve stability.

    :param XTX_vals: Values of the XTX matrix.
    :param XTY_vals: Values of the XTY matrix.

    :return: The weights W, of shape (n_atomic_types, n_components, n_properties).
    """
    trace_magnitude = float(torch.diag(XTX_vals).abs().mean())
    regularizer = 1e-14 * trace_magnitude
    shape = (XTX_vals.shape[0], *XTY_vals.shape[1:])
    return torch.linalg.solve(
        XTX_vals
        + regularizer
        * torch.eye(
            XTX_vals.shape[1],
            dtype=XTX_vals.dtype,
            device=XTX_vals.device,
        ),
        XTY_vals.reshape(XTY_vals.shape[0], -1),
    ).reshape(shape)


def _get_system_indices_and_labels(
    systems: List[System], device: torch.device
) -> Tuple[torch.Tensor, Labels]:
    system_indices = torch.concatenate(
        [
            torch.full(
                (len(system),),
                i_system,
                device=device,
            )
            for i_system, system in enumerate(systems)
        ],
    )

    sample_values = torch.stack(
        [
            system_indices,
            torch.concatenate(
                [
                    torch.arange(
                        len(system),
                        device=device,
                    )
                    for system in systems
                ],
            ),
        ],
        dim=1,
    )
    sample_labels = Labels(
        names=["system", "atom"],
        values=sample_values,
        assume_unique=True,
    )
    return system_indices, sample_labels
