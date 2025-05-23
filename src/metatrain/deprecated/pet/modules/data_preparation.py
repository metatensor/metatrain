import numpy as np
import torch
from sklearn.linear_model import Ridge

from .molecule import Molecule


def get_all_species(structures):
    all_species = []
    for structure in structures:
        all_species.append(np.array(structure.get_atomic_numbers()))
    all_species = np.concatenate(all_species, axis=0)
    all_species = np.sort(np.unique(all_species))
    return all_species


def get_forces(structures, FORCES_KEY):
    forces = []
    for structure in structures:
        forces.append(
            torch.tensor(structure.arrays[FORCES_KEY], dtype=torch.get_default_dtype())
        )
    return forces


def update_pyg_graphs(pyg_graphs, key, values):
    for index in range(len(pyg_graphs)):
        pyg_graphs[index].update({key: values[index]})


def get_pyg_graphs(
    structures,
    all_species,
    R_CUT,
    USE_ADDITIONAL_SCALAR_ATTRIBUTES,
    USE_LONG_RANGE,
    K_CUT,
    MULTI_TARGET,
    TARGET_INDEX_KEY,
):
    molecules = [
        Molecule(
            structure,
            R_CUT,
            USE_ADDITIONAL_SCALAR_ATTRIBUTES,
            USE_LONG_RANGE,
            K_CUT,
            MULTI_TARGET,
            TARGET_INDEX_KEY,
        )
        for structure in structures
    ]

    max_nums = [molecule.get_max_num() for molecule in molecules]
    max_num = np.max(max_nums)

    if USE_LONG_RANGE:
        k_nums = [molecule.get_num_k() for molecule in molecules]
        max_k_num = np.max(k_nums)
    else:
        max_k_num = None

    pyg_graphs = [
        molecule.get_graph(max_num, all_species, max_k_num) for molecule in molecules
    ]
    return pyg_graphs


def get_compositional_features(structures, all_species):
    result = np.zeros([len(structures), len(all_species)])
    for i, structure in enumerate(structures):
        species_now = structure.get_atomic_numbers()
        for j, specie in enumerate(all_species):
            num = np.sum(species_now == specie)
            result[i, j] = num
    return result


def get_targets(structures, GENERAL_TARGET_SETTINGS):
    """Get general targets from structures"""

    targets = []
    if GENERAL_TARGET_SETTINGS.TARGET_TYPE not in ["structural", "atomic"]:
        raise ValueError("unknown target type")

    for structure in structures:
        if GENERAL_TARGET_SETTINGS.TARGET_TYPE == "structural":
            target_now = structure.info[GENERAL_TARGET_SETTINGS.TARGET_KEY]
            if not isinstance(target_now, np.ndarray):
                raise ValueError("target must be numpy array")
            if len(target_now.shape) != 1:
                raise ValueError("structural target must be 1D array")

            target_now = target_now[np.newaxis]

        elif GENERAL_TARGET_SETTINGS.TARGET_TYPE == "atomic":
            target_now = structure.arrays[GENERAL_TARGET_SETTINGS.TARGET_KEY]
            if not isinstance(target_now, np.ndarray):
                raise ValueError("target must be numpy array")
            if len(target_now.shape) != 2:
                raise ValueError("atomic target must be 2D array")

        targets.append(torch.tensor(target_now, dtype=torch.get_default_dtype()))
    return targets


def get_self_contributions(energy_key, train_structures, all_species):
    train_energies = np.array(
        [structure.info[energy_key] for structure in train_structures]
    )
    train_c_feat = get_compositional_features(train_structures, all_species)
    rgr = Ridge(alpha=1e-10, fit_intercept=False)
    rgr.fit(train_c_feat, train_energies)
    return rgr.coef_


def get_corrected_energies(energy_key, structures, all_species, self_contributions):
    energies = np.array([structure.info[energy_key] for structure in structures])

    compositional_features = get_compositional_features(structures, all_species)
    self_contributions_energies = []
    for i in range(len(structures)):
        self_contributions_energies.append(
            np.dot(compositional_features[i], self_contributions)
        )
    self_contributions_energies = np.array(self_contributions_energies)
    return energies - self_contributions_energies
