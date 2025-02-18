from metatensor.learn.data import DataLoader

from ...utils.additive import remove_additive
from ...utils.data import collate_fn
from ...utils.data.system_to_ase import system_to_ase
from ...utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


# dummy dataloaders due to https://github.com/metatensor/metatensor/issues/521
def dataset_to_ase(dataset, model, do_forces=True, target_name="energy"):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    ase_dataset = []
    for (system,), targets in dataloader:
        # remove additive model (e.g. ZBL) contributions
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        system = get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for additive_model in model.additive_models:
            targets = remove_additive(
                [system], targets, additive_model, model.dataset_info.targets
            )
        # transform to ase atoms
        ase_atoms = system_to_ase(system)
        ase_atoms.info["energy"] = float(
            targets[target_name].block().values.squeeze(-1).detach().cpu().numpy()
        )
        if do_forces:
            ase_atoms.arrays["forces"] = (
                -targets[target_name]
                .block()
                .gradient("positions")
                .values.squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )
        ase_dataset.append(ase_atoms)
    return ase_dataset
