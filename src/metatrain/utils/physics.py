from typing import Dict, List, Optional

import torch
from metatensor.torch import TensorBlock, TensorMap
from metatomic.torch import System


def enforce_physical_constraints(
    systems: List[System],
    predictions: Dict[str, TensorMap],
    timestep: Optional[float] = None,
) -> Dict[str, TensorMap]:
    """Enforces physical constraints in the predictions of a model.

    Only global constraints are currently implemented by this function. These need
    to live outside of the model in a post-processing step, as implementing them in the
    model would lead to failures in domain-decomposed engines, where the model is only
    aware of a subset of the full system.

    This function searches for specific keys in the predictions and applies the
    relevant physical constraints. Currently, it acts on the following keys:

    - "non_conservative_forces": net zero force is enforced on each structure
    - "momenta": conservation of momentum of the center of mass is enforced on each
      structure
    - "positions": uniform linear motion is enforced on the center of mass, based on the
      initial positions and momenta

    Note that constraints on "non_conservative_stress" are not affected by domain
    decomposition, as therefore each architecture should process it internally.

    :param systems: The systems used to compute the predictions.
    :param predictions: A dictionary of ``TensorMap`` objects.
    """
    new_predictions = {}
    for key, prediction_tmap in predictions.items():
        if key == "non_conservative_forces":
            # net zero force
            system_sizes = [len(s) for s in systems]
            nc_forces_tensor = prediction_tmap.block().values
            nc_forces_by_system = torch.split(nc_forces_tensor, system_sizes)
            nc_forces_by_system = [
                f - f.mean(dim=0, keepdim=True) for f in nc_forces_by_system
            ]
            new_predictions[key] = TensorMap(
                prediction_tmap.keys,
                [
                    TensorBlock(
                        values=torch.concatenate(nc_forces_by_system),
                        samples=prediction_tmap.block().samples,
                        components=prediction_tmap.block().components,
                        properties=prediction_tmap.block().properties,
                    )
                ],
            )
        elif key == "momenta":
            # conservation of momentum of the center of mass
            system_sizes = [len(s) for s in systems]
            masses = [s.get_data("masses") for s in systems]
            total_masses = [m.sum() for m in masses]
            momenta_before = [s.get_data("momenta") for s in systems]
            momenta_now = torch.split(prediction_tmap.block().values, system_sizes)
            velocities_now = [p / m[:, None] for p, m in zip(momenta_now, masses)]
            velocities_com_before = [
                torch.sum(p, dim=0) / M for p, M in zip(momenta_before, total_masses)
            ]
            velocities_com_now = [
                torch.sum(p, dim=0) / M for p, M in zip(momenta_now, total_masses)
            ]
            velocities_now = [
                v - v_com_now_i + v_com_before_i
                for v, v_com_before_i, v_com_now_i in zip(
                    velocities_now, velocities_com_before, velocities_com_now
                )
            ]
            momenta_now = [v * m[:, None] for v, m in zip(velocities_now, masses)]
            new_predictions[key] = TensorMap(
                prediction_tmap.keys,
                [
                    TensorBlock(
                        values=torch.concatenate(momenta_now),
                        samples=prediction_tmap.block().samples,
                        components=prediction_tmap.block().components,
                        properties=prediction_tmap.block().properties,
                    )
                ],
            )
        elif key == "positions":
            # uniform linear motion of the center of mass
            # we need a timestep for this
            if timestep is None:
                raise ValueError(
                    "A timestep must be provided to enforce physical constraints on "
                    "the positions."
                )
            system_sizes = [len(s) for s in systems]
            masses = [s.get_data("masses") for s in systems]
            total_masses = [m.sum() for m in masses]
            positions_before = [s.positions.unsqueeze(-1) for s in systems]
            momenta = [s.get_data("momenta") for s in systems]
            positions_now = torch.split(prediction_tmap.block().values, system_sizes)
            velocities_com = [
                torch.sum(p, dim=0) / M for p, M in zip(momenta, total_masses)
            ]
            positions_com_before = [
                torch.sum(q * m[:, None], dim=0) / M
                for q, m, M in zip(positions_before, masses, total_masses)
            ]
            positions_com_now = [
                torch.sum(q * m[:, None], dim=0) / M
                for q, m, M in zip(positions_now, masses, total_masses)
            ]
            positions_now = [
                q - q_com_now_i + q_com_before_i + v_com_i * timestep
                for q, q_com_now_i, q_com_before_i, v_com_i in zip(
                    positions_now,
                    positions_com_now,
                    positions_com_before,
                    velocities_com,
                )
            ]
            new_predictions[key] = TensorMap(
                prediction_tmap.keys,
                [
                    TensorBlock(
                        values=torch.concatenate(positions_now),
                        samples=prediction_tmap.block().samples,
                        components=prediction_tmap.block().components,
                        properties=prediction_tmap.block().properties,
                    )
                ],
            )
        else:
            new_predictions[key] = prediction_tmap
    return new_predictions
