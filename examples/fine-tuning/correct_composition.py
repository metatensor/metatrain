#!/usr/bin/env python

from metatrain.pet import PET
import ase.io
import numpy as np

from pet.data_preparation import get_self_contributions, get_compositional_features
import os


def get_self_energies(
    structures: list[ase.Atoms], all_species: list[int], self_contributions: np.array
) -> np.array:
    """Get self (composition) energies of structures."""

    compositional_features = get_compositional_features(structures, all_species)
    self_contributions_energies = []
    for i in range(len(structures)):
        self_contributions_energies.append(
            np.dot(compositional_features[i], self_contributions)
        )
    return np.array(self_contributions_energies)


def correct_dataset_self_energy(
    dataset_path: str,
    pet_checkpoint_file: str,
    energy_key: str = "energy",
    forces_key: str = "forces",
) -> None:
    """Correct the self energy of a dataset using a trained PET model.

    The corrected energies are computed as:

    E_corected = E_dataset + E_comp,dataset - E_comp,model

    The corrected energies are written to a new file with the suffix "_corrected"
    """

    # Load dataset and model
    frames = ase.io.read(dataset_path, ":")
    model = PET.load_checkpoint(pet_checkpoint_file)

    # Get all atomic types in dataset
    dataset_atomic_types = sorted(
        np.unique([atoms.get_atomic_numbers() for atoms in frames])
    )

    # Compute self (composition) weights
    dataset_self_contributions = get_self_contributions(
        energy_key=energy_key,
        train_structures=frames,
        all_species=dataset_atomic_types,
    )

    # Compute self (composition) energies
    self_energies_dataset = get_self_energies(
        structures=frames,
        all_species=dataset_atomic_types,
        self_contributions=dataset_self_contributions,
    )
    self_energies_model = get_self_energies(
        structures=frames,
        all_species=model.atomic_types,
        self_contributions=model.pet.self_contributions,
    )

    # Correct energies
    frames_correced = []
    for i_frame, atoms in enumerate(frames):
        atoms_corrected = atoms.copy()
        atoms_corrected.info[energy_key] = (
            atoms.info[energy_key]
            - self_energies_dataset[i_frame]
            + self_energies_model[i_frame]
        )

        frames_correced.append(atoms_corrected)

    # Write corrected dataset
    base, ext = os.path.splitext(dataset_path)
    ase.io.write(f"{base}_corrected{ext}", frames_correced)


if __name__ == "__main__":

    dataset_path = "../../../../pbesol_alloys_MAD_compatible.xyz"
    pet_checkpoint_file = "pet-mad-v1.1.0.ckpt"
    energy_key = "REF_energy"
    forces_key = "REF_forces"

    correct_dataset_self_energy(
        dataset_path=dataset_path,
        pet_checkpoint_file=pet_checkpoint_file,
        energy_key=energy_key,
        forces_key=forces_key,
    )
Collapse




