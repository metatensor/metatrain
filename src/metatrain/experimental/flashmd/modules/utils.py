import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


def verify_masses(systems: list[System], masses: torch.Tensor):
    """Attach masses to systems that don't have them yet."""
    for system in systems:
        if "masses" not in system.known_data():
            # obtain the masses from the atomic types
            values = masses[system.types].unsqueeze(-1)

            # wrap everything in a tensor map and attach to the system
            label_values = torch.column_stack(
                [
                    torch.zeros(len(system), dtype=torch.int32, device=values.device),
                    torch.arange(len(system), device=values.device),
                ]
            )

            masses_map = TensorMap(
                keys=Labels(
                    names="_",
                    values=torch.zeros((1, 1), dtype=torch.int32, device=values.device),
                ),
                blocks=[
                    TensorBlock(
                        values=values,
                        samples=Labels(
                            names=["system", "atom"],
                            values=label_values,
                        ),
                        components=[],
                        properties=Labels(
                            names="mass",
                            values=torch.arange(1, device=values.device).unsqueeze(-1),
                        ),
                    )
                ],
            )
            system.add_data("masses", masses_map)
        else:
            # verify that the masses are correct
            # (compare them to the ones stored in the model)
            system_masses = system.get_data("masses").block(0).values
            for atom_index in range(len(system)):
                # get the (model's expected) mass for this atom
                atomic_number = system.types[atom_index]
                model_mass = masses[atomic_number]

                # get the mass stored in the system
                system_mass = system_masses[atom_index, 0]
                if model_mass != system_mass:
                    print("MASSES:", masses)
                    raise ValueError(
                        f"The mass of atom {atom_index} in a system is {model_mass}, "
                        f"while the expected mass for atomic number {atomic_number} "
                        f"is {system_mass}."
                    )
