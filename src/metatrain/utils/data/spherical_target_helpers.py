import itertools
from collections import defaultdict
from typing import List, Literal, Optional

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def _build_spherical_target_block(
    sample_names: List[str],
    target_name: str,
    irreps: list[tuple[int, int, int]],
    properties: Optional[Labels] = None,
) -> TensorBlock:
    """Builds an empty `TensorBlock` for some given spherical irreps.

    :param sample_names: The names of the sample labels of the block.
    :param target_name: The name of the target (used for the properties labels).
    :param irreps: A list of irreps, where each irrep is a tuple of the form
      (num_subtargets, o3_lambda, o3_sigma).
    :param properties: An optional ``Labels`` object containing the properties
        labels of the block. If `None`, the properties labels will be generated
        automatically with names ``n`` for rank 1 blocks and ``n_1``,
        ``n_2``, ... for higher rank blocks.
    :return: an empty `TensorBlock` with the appropriate samples, components
      and properties labels for the given irreps
    """
    rank = len(irreps)

    if rank == 1:
        lambd = irreps[0][1]
        n_properties = irreps[0][0]
        components = [
            Labels(
                names=["o3_mu"],
                values=torch.arange(-lambd, lambd + 1, dtype=torch.int32).reshape(
                    -1, 1
                ),
            )
        ]
        if properties is None:
            properties = Labels.range("n", n_properties)
    else:
        n_properties = torch.tensor(irreps)[:, 0].prod()
        components = []
        for component in range(1, rank + 1):
            lambd = irreps[component - 1][1]
            components.append(
                Labels(
                    names=[f"o3_mu_{component}"],
                    values=torch.arange(-lambd, lambd + 1, dtype=torch.int32).reshape(
                        -1, 1
                    ),
                )
            )

        if properties is None:
            properties_values = list(
                itertools.product(*[np.arange(irrep[0]) for irrep in irreps])
            )
            properties_values = torch.tensor(properties_values, dtype=torch.int32)

            properties = Labels(
                names=[f"n_{component}" for component in range(1, rank + 1)],
                values=properties_values,
            )

    block = TensorBlock(
        # float64: otherwise metatensor can't serialize
        values=torch.empty(
            0,
            *[len(component) for component in components],
            n_properties,
            dtype=torch.float64,
        ),
        samples=Labels(
            names=sample_names,
            values=torch.empty((0, len(sample_names)), dtype=torch.int32),
        ),
        components=components,
        properties=properties,
    )
    return block


def _get_spherical_irreps_iter(
    irreps: list[dict],
    target: dict,
    product: Optional[Literal["cartesian", "coupled"]] = None,
) -> tuple[list[list[tuple[int, int, int]]], list[Labels]]:
    if product is not None and product not in ["cartesian", "coupled"]:
        raise ValueError(
            f"Product '{product}' is not supported. Supported products are"
            " 'cartesian' and 'coupled'."
        )

    irreps_iter = [
        [
            (
                irrep.get("num", target["num_subtargets"]),
                irrep["o3_lambda"],
                irrep["o3_sigma"],
            )
        ]
        for irrep in irreps
    ]
    if product in ["cartesian", "coupled"]:
        irreps_iter = [
            i_irrep + j_irrep
            for i_irrep, j_irrep in itertools.product(irreps_iter, repeat=2)
        ]
    if product == "coupled":
        coupled_blocks: dict[tuple[int, int], int] = defaultdict(int)
        coupled_properties = defaultdict(list)
        for (num1, l1, sig1), (num2, l2, sig2) in irreps_iter:
            for lam in range(abs(l1 - l2), l1 + l2 + 1):
                sig = sig1 * sig2 * (-1) ** (l1 + l2 + lam)

                # If the two spherical harmonics are in the same center,
                # the coupled values for sigma = -1 are zero. This is true
                # for the Hamiltonian, density matrix and overlap, but might
                # be not true for other targets. In that case, we will need to
                # add an input to specify if the target has this property.
                # For per-pair targets (not introduced yet), this will not be
                # the case.
                if sig == -1:
                    continue

                coupled_blocks[(lam, sig)] += num1 * num2

                # Build properties
                added_properties = []
                for i in range(num1):
                    for j in range(num2):
                        added_properties.append((l1, l2, i, j))
                coupled_properties[(lam, sig)].extend(added_properties)

        irreps_iter = [[(num, lam, sig)] for (lam, sig), num in coupled_blocks.items()]
        properties = [
            Labels(
                names=["l_1", "l_2", "n_1", "n_2"],
                values=torch.tensor(props, dtype=torch.int32),
            )
            for props in coupled_properties.values()
        ]
    else:
        properties = [None] * len(irreps_iter)

    return irreps_iter, properties


def match_predictions_to_targets(
    targets: dict[str, TensorMap],
    predictions: dict[str, TensorMap],
    extra_data: dict[str, TensorMap],
    target_keys: Optional[List[str]] = None,
) -> tuple[dict[str, TensorMap], dict[str, TensorMap]]:
    """Function to make sure that the samples of predictions and targets match.

    :param targets: A dictionary mapping target names to `TensorMap`s containing
      the target data.
    :param predictions: A dictionary mapping target names to `TensorMap`s containing
      the predictions data.
    :param extra_data: A dictionary mapping extra data names to `TensorMap`s
      containing the extra data.
    :param target_keys: A list of target keys on which to apply the matching.
      If ``None``, the matching is applied on all the target keys.
    :return: The matched targets and predictions dictionaries.
    """
    if target_keys is None:
        target_keys = list(targets.keys())
    if len(target_keys) == 0:
        return targets, predictions

    targets = {**targets}
    predictions = {**predictions}

    system_indices = extra_data["mtt::aux::system_index"].block().samples["system"]
    for target_key in target_keys:
        blocks = []
        for key in predictions[target_key].keys:
            prediction_block = predictions[target_key].block(key)

            pred_systems = prediction_block.samples["system"]
            correct_systems = system_indices[pred_systems]

            samples_vals = prediction_block.samples.values.clone()
            samples_vals[:, 0] = correct_systems

            pred_samples = Labels(
                names=prediction_block.samples.names,
                values=samples_vals,
            )

            target_samples = targets[target_key].block(key).samples
            selected_pred_samples = pred_samples.select(target_samples)

            blocks.append(
                TensorBlock(
                    values=prediction_block.values[selected_pred_samples],
                    samples=target_samples,
                    components=prediction_block.components,
                    properties=prediction_block.properties,
                )
            )

        predictions[target_key] = TensorMap(
            keys=predictions[target_key].keys,
            blocks=blocks,
        )

    return targets, predictions
