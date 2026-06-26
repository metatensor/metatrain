import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def match_samples(
    predictions: dict[str, TensorMap],
    targets: dict[str, TensorMap],
    extra_data: dict[str, TensorMap],
    keep_samples: bool = True
) -> dict[str, TensorMap]:
    if len(predictions) == 0:
        return {}

    system_indices = extra_data["mtt::aux::system_index"].block().values.ravel()
    matched_targets = {}

    for target_name, predictions_tmap in predictions.items():
        target_tmap = targets[target_name]
        new_blocks = []

        for key, block in predictions_tmap.items():
            target_block = target_tmap.block(key)

            pred_samples_values = block.samples.values.clone()

            pred_samples_values[:, 0] = system_indices[block.samples.values[:, 0]]
            pred_samples = Labels(
                names=target_block.samples.names,
                values=pred_samples_values,
            )

            selector = target_block.samples.select(pred_samples)

            if keep_samples:
                new_samples = pred_samples
            else:
                new_samples = block.samples

            new_block = TensorBlock(
                values=target_block.values[selector],
                samples=new_samples,
                components=target_block.components,
                properties=target_block.properties,
            )

            new_blocks.append(new_block)

        matched_targets[target_name] = TensorMap(
            keys=predictions_tmap.keys, blocks=new_blocks
        )

    return matched_targets


def match_samples_to_neighborlist(
    tmap: TensorMap,
    neighborlist: TensorMap,
    extra_data: dict[str, TensorMap],
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

        selector = block.samples.select(nl_samples)

        new_block = TensorBlock(
            values=block.values[selector],
            samples=nl_samples,
            components=block.components,
            properties=block.properties,
        )

        new_blocks.append(new_block)
        new_keys.append(key.values)

    return TensorMap(
        keys=Labels(names=tmap.keys.names, values=torch.stack(new_keys)),
        blocks=new_blocks,
    )
