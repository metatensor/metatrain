from metatensor.learn.data import DataLoader

from ....utils.data import collate_fn
from ....utils.data.system_to_ase import system_to_ase


# dummy dataloaders due to https://github.com/lab-cosmo/metatensor/issues/521
def dataset_to_ase(dataset, do_forces=True, target_name="energy"):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    ase_dataset = []
    for (system,), targets in dataloader:
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
