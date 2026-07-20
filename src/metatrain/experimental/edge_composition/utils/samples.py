from typing import Literal, Optional
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def match_samples(
    predictions: dict[str, TensorMap],
    targets: dict[str, TensorMap],
    extra_data: dict[str, TensorMap],
    which_samples: Literal["predictions", "targets"] = "targets",
) -> dict[str, TensorMap]:
    if len(predictions) == 0:
        return {}

    system_indices = extra_data["mtt::aux::system_index"].block().values.ravel()
    matched = {}

    for target_name, predictions_tmap in predictions.items():
        target_tmap = targets[target_name]
        new_blocks = []

        ref_tmap = predictions_tmap if which_samples == "predictions" else target_tmap
        other_tmap = target_tmap if which_samples == "predictions" else predictions_tmap

        for key, ref_block in ref_tmap.items():
            try:
                other_block = other_tmap.block(key)
            except ValueError:
                other_block = TensorBlock(
                    values=torch.empty(
                        (0, *ref_block.values.shape[1:]),
                        dtype=ref_block.values.dtype,
                        device=ref_block.values.device,
                    ),
                    samples=Labels(
                        names=ref_block.samples.names,
                        values=torch.empty((0, ref_block.samples.values.shape[1]), dtype=torch.long, device=ref_block.samples.values.device),
                    ),
                    components=ref_block.components,
                    properties=ref_block.properties,
                )

            pred_block = ref_block if which_samples == "predictions" else other_block

            pred_samples_values = pred_block.samples.values.clone()

            pred_samples_values[:, 0] = system_indices[pred_block.samples.values[:, 0]]
            pred_samples = Labels(
                names=pred_block.samples.names,
                values=pred_samples_values,
            )

            if which_samples == "predictions":
                # We want to keep the samples from the predictions, so we will sample
                # the other block to match the predictions.
                new_block = sample_from_tensorblock(
                    other_block, pred_samples, missing_value=0.0
                )
            else:
                pred_block = TensorBlock(
                    values=pred_block.values,
                    samples=pred_samples,
                    components=pred_block.components,
                    properties=pred_block.properties,
                )

                new_block = sample_from_tensorblock(
                    pred_block, ref_block.samples, missing_value=0.0
                )

            new_blocks.append(new_block)

        matched[target_name] = TensorMap(
            keys=ref_tmap.keys, blocks=new_blocks
        )

    return matched

def _labels_select(
    requested: Labels, available: Labels
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the indices to select the requested labels from the available labels.

    The function finds the intersection between both and returns arrays to do the
    transformation: available -> intersection -> requested.

    :param requested: The requested labels.
    :param available: The available labels.
    :return: A tuple of two tensors:
        - Indices to go from the intersection to the requested labels.
        - Indices to go from the available labels to the intersection labels.
    """
    # Find the intersection of the requested samples and the block's samples
    intersec_samples, req_select, select = requested.intersection_and_mapping(
        available
    )

    # "select" gives you the index of the intersection where each available label
    # entry landed. We want the opposite: for each intersection label entry, we
    # want to know how to find it in the available labels. So we create an inverse
    # mapping.
    # We also have to take into account that "select" contains -1 when the label
    # entry is not in the intersection, that's why we use select >= 0 as a mask.
    inv_select = torch.zeros(
        intersec_samples.values.shape[0],
        device=available.values.device,
        dtype=torch.long,
    )
    arange = torch.arange(available.values.shape[0], device=available.values.device)
    mask = select >= 0
    inv_select[select[mask]] = arange[mask]

    return req_select, inv_select

def sample_from_tensorblock(
    block: TensorBlock,
    samples: Labels,
    properties: Optional[Labels] = None,
    missing_value: float = 0.0,
) -> TensorBlock:
    """Selects the values of the tensor block corresponding to the given samples.

    :param block: The tensor block to select from.
    :param samples: The samples to select.
    :param properties: The properties to select.
      If None, the properties of the block will be used.
    :param missing_value: The value to use for samples that are not in the block.
    :return: A new tensor block with the selected values. This block's samples are
        the same as the requested samples, and components and properties are the
        same as the original block.
    """
    properties = properties or block.properties
    # Initialize the returned block's values
    values = torch.full(
        (samples.values.shape[0], *block.values.shape[1:-1], properties.values.shape[0]),
        missing_value,
        dtype=block.values.dtype,
        device=block.values.device,
    )

    samples_select, inv_samples_select = _labels_select(samples, block.samples)
    props_select, inv_props_select = _labels_select(properties, block.properties)

    # Values for the samples and properties that were found in the block.
    existing_values = block.values[inv_samples_select][..., inv_props_select]

    # Get the masks that tell us whether a given request is present.
    is_sample_present = (samples_select >= 0).nonzero().ravel()
    is_prop_present = (props_select >= 0).nonzero().ravel()
    
    # Reorder the existing values to match the order of the requested samples and properties.
    reordered = existing_values[samples_select[is_sample_present]][..., props_select[is_prop_present]]

    # Set them. It is a bit tricky to set values using the two masks,
    # that's why the following looks hacky, but in essence we are
    # just doing values[is_sample_present][..., is_prop_present] = reordered
    tmp = values.index_select(0, is_sample_present)
    tmp[..., is_prop_present] = reordered
    values.index_copy_(0, is_sample_present, tmp)

    return TensorBlock(
        values=values,
        samples=samples,
        components=block.components,
        properties=properties,
    )


def match_samples_to_neighborlist(
    tmap: TensorMap,
    neighborlist: TensorMap,
    extra_data: dict[str, TensorMap],
    fill_value: float = 0.0,
) -> dict[str, TensorMap]:
    system_indices = extra_data["mtt::aux::system_index"].block().values.ravel()

    atom_type_samples = {}
    for key, block in neighborlist.items():
        nl_samples_values = block.samples.values.clone()

        nl_samples_values[:, 0] = system_indices[block.samples.values[:, 0]]
        nl_samples = Labels(
            names=block.samples.names,
            values=nl_samples_values,
        )
        atom_type_samples[int(key[0]), int(key[1])] = nl_samples

    new_keys = []
    new_blocks = []
    for key, block in tmap.items():
        type1, type2 = int(key["first_atom_type"]), int(key["second_atom_type"])
        nl_samples = atom_type_samples.get((type1, type2), None)
        if nl_samples is None or nl_samples.values.shape[0] == 0:
            continue

        new_block = sample_from_tensorblock(block, nl_samples, missing_value=fill_value)

        new_blocks.append(new_block)
        new_keys.append(key.values)

    return TensorMap(
        keys=Labels(names=tmap.keys.names, values=torch.stack(new_keys)),
        blocks=new_blocks,
    )


def match_samples_to_neighborlist_and_layout(
    tmap: TensorMap,
    neighborlist: TensorMap,
    layout: TensorMap,
    extra_data: dict[str, TensorMap],
    fill_value: float = 0.0,
) -> dict[str, TensorMap]:
    type_keys = {}
    for block_key, layout_block in layout.items():
        type1, type2 = block_key["first_atom_type"], block_key["second_atom_type"]
        if (type1, type2) not in type_keys:
            type_keys[(type1, type2)] = []
        type_keys[(type1, type2)].append(block_key)

    system_indices = extra_data["mtt::aux::system_index"].block().values.ravel()

    new_keys = []
    new_blocks = []
    for key, nl_block in neighborlist.items():
        if nl_block.samples.values.shape[0] == 0:
            continue

        # Move system indices to dataset indices instead of batch indices
        nl_samples_values = nl_block.samples.values.clone()
        nl_samples_values[:, 0] = system_indices[nl_block.samples.values[:, 0]]
        nl_samples = Labels(
            names=nl_block.samples.names,
            values=nl_samples_values,
        )

        type1, type2 = key["first_atom_type"], key["second_atom_type"]

        for block_key in type_keys.get((type1, type2), []):
            layout_block = layout.block(block_key)
            try:
                block = tmap.block(block_key)
            except ValueError:
                block = layout_block

            new_block = sample_from_tensorblock(block, nl_samples, layout_block.properties, missing_value=fill_value)

            new_blocks.append(new_block)
            new_keys.append(block_key.values)        

    return TensorMap(
        keys=Labels(names=tmap.keys.names, values=torch.stack(new_keys)),
        blocks=new_blocks,
    )