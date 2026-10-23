"""Measuring the O(3) equivariance error of a model, for ``mtt eval``."""

import logging
from typing import Dict, List, Union

import torch
from metatensor.torch import TensorMap, remove_gradients
from metatomic.torch import AtomisticModel, ModelOutput, System
from metatomic.torch.o3 import SymmetrizedModel

from .data import TargetInfo
from .metrics import EquivarianceAccumulator, MAEAccumulator, RMSEAccumulator
from .per_atom import average_by_num_atoms, divide_by_num_atoms


def _max_angular_momentum(targets: Dict[str, TargetInfo]) -> int:
    """Largest angular momentum among the targets.

    This sizes the Wigner matrices of the symmetrized model, which has to be
    able to rotate every target back to the frame of the input system.

    :param targets: Dictionary containing the target information.

    :return: The largest angular momentum among the targets.
    """
    max_angular_momentum = 0
    for target in targets.values():
        if target.is_spherical:
            o3_lambda = target.layout.keys.column("o3_lambda")
            max_angular_momentum = max(max_angular_momentum, int(o3_lambda.max()))
        elif target.is_cartesian:
            # a Cartesian tensor of rank n decomposes into terms up to o3_lambda=n
            rank = len(target.layout.block(0).components)
            max_angular_momentum = max(max_angular_momentum, rank)
    return max_angular_momentum


class EquivarianceEvaluation:
    """Measures the O(3) equivariance error of a model alongside its accuracy.

    Every batch is also evaluated by a metatomic ``SymmetrizedModel``, giving the
    variance of the back-rotated predictions over a rotation grid (the
    equivariance error) and their O(3) average, whose accuracy is measured too.

    :param model: The model being evaluated.
    :param targets: The targets to evaluate the model on.
    :param options: The ``EquivarianceHypers``, with their defaults.
    :param batch_size: Number of rotated copies of a system evaluated in a single
        call to the model.
    :param separate_blocks: Whether to report the metrics for each block.
    """

    def __init__(
        self,
        model: Union[AtomisticModel, torch.jit.RecursiveScriptModule],
        targets: Dict[str, TargetInfo],
        options: Dict,
        batch_size: int,
        separate_blocks: bool,
    ) -> None:
        gradients = sorted(
            {
                gradient
                for target in targets.values()
                for gradient in target.gradients  # type: ignore[union-attr]
            }
        )
        if len(gradients) > 0:
            logging.warning(
                f"the equivariance error of the {gradients} gradients is not "
                "supported and will not be reported; only the equivariance of the "
                "target values themselves is measured"
            )

        self.model = model
        self.separate_blocks = separate_blocks
        self.symmetrized_model = SymmetrizedModel(
            model.module,
            max_angular_momentum_target=_max_angular_momentum(targets),
            max_angular_momentum_grid=options["max_angular_momentum_grid"],
            batch_size=batch_size,
        )
        # atom-pair targets are excluded from the metrics, as for the accuracy
        self.names = [
            name
            for name, target in targets.items()
            if target.sample_kind != "atom_pair"
        ]
        self.outputs: Dict[str, ModelOutput] = {}
        for name in self.names:
            for key in (name, f"o3::variance::{name}"):
                self.outputs[key] = ModelOutput(
                    unit=targets[name].unit, sample_kind=targets[name].sample_kind
                )

        self.equivariance_acc = EquivarianceAccumulator(separate_blocks)
        self.averaged_rmse_acc = RMSEAccumulator(separate_blocks)
        self.averaged_mae_acc = MAEAccumulator(separate_blocks)
        self.checked_grid_convergence = False
        # every rotation of the grid is evaluated both with and without inversion
        n_copies = 2 * len(self.symmetrized_model._rotation_weights)
        logging.info(
            "Equivariance error evaluation enabled: every system is additionally "
            f"evaluated on {n_copies} rotated and inverted copies, from a quadrature "
            f"grid of degree {self.symmetrized_model.max_angular_momentum_grid}"
        )

    def _per_atom_variances(
        self, symmetrized: Dict[str, TensorMap], systems: List[System]
    ) -> Dict[str, TensorMap]:
        """The variances of the targets, normalized per atom like the predictions.

        :param symmetrized: Outputs of the symmetrized model.
        :param systems: The systems of the batch.

        :return: The variance of each target, keyed by the name of the target.
        """
        num_atoms = torch.tensor([len(s) for s in systems], device=systems[0].device)
        # a variance scales as the square of the predictions
        return {
            name: divide_by_num_atoms(
                divide_by_num_atoms(symmetrized[f"o3::variance::{name}"], num_atoms),
                num_atoms,
            )
            for name in self.names
        }

    def update(
        self,
        systems: List[System],
        targets_per_atom: Dict[str, TensorMap],
        extra_data: Dict[str, TensorMap],
        not_per_atom: List[str],
    ) -> Dict[str, TensorMap]:
        """Evaluate the symmetrized model on a batch and accumulate its metrics.

        :param systems: The systems of the batch.
        :param targets_per_atom: The targets of the batch, normalized per atom.
        :param extra_data: The extra data of the batch.
        :param not_per_atom: Keys containing one of these strings are not labeled
            as "(per atom)".

        :return: The O(3)-averaged predictions, as ``o3::average::<target>``, and
            their variance over the orientations, as ``o3::variance::<target>``, to
            be written alongside the predictions of the model.
        """
        symmetrized = self.symmetrized_model(systems, self.outputs, None)
        self.equivariance_acc.update(self._per_atom_variances(symmetrized, systems))
        averaged = average_by_num_atoms(
            {name: symmetrized[name] for name in self.names},
            systems,
            per_structure_keys=[],
        )
        # the symmetrized model does not predict gradients, so the reference ones
        # have to go as well: the accumulators pair up the gradients of the target
        # with those of the prediction
        averaged_targets = {
            name: remove_gradients(targets_per_atom[name]) for name in self.names
        }
        self.averaged_rmse_acc.update(averaged, averaged_targets, extra_data)
        self.averaged_mae_acc.update(averaged, averaged_targets, extra_data)

        if not self.checked_grid_convergence:
            self._check_grid_convergence(systems, not_per_atom)
            self.checked_grid_convergence = True

        # the averaged predictions are renamed, so as not to clash with the
        # predictions of the model itself
        return {
            **{f"o3::average::{name}": symmetrized[name] for name in self.names},
            **{
                f"o3::variance::{name}": symmetrized[f"o3::variance::{name}"]
                for name in self.names
            },
        }

    def _check_grid_convergence(
        self, systems: List[System], not_per_atom: List[str], tolerance: float = 0.01
    ) -> None:
        """Warn if re-measuring the equivariance error of a batch on a finer
        quadrature grid changes it by more than ``tolerance`` (relative).

        :param systems: The systems of the batch, the only one measured so far.
        :param not_per_atom: Keys containing one of these strings are not labeled
            as "(per atom)".
        :param tolerance: Largest acceptable relative change.
        """
        reference = self.equivariance_acc.finalize(not_per_atom=not_per_atom)
        scales = self.averaged_rmse_acc.finalize(not_per_atom=not_per_atom)
        coarse_grid = self.symmetrized_model.max_angular_momentum_grid
        finer_model = SymmetrizedModel(
            self.model.module,
            max_angular_momentum_target=self.symmetrized_model.max_angular_momentum_target,
            max_angular_momentum_grid=coarse_grid + 2,
            batch_size=self.symmetrized_model.batch_size,
        )
        finer_acc = EquivarianceAccumulator(self.separate_blocks)
        finer_acc.update(
            self._per_atom_variances(finer_model(systems, self.outputs, None), systems)
        )

        for key, finer_value in finer_acc.finalize(not_per_atom=not_per_atom).items():
            coarse_value = reference.get(key)
            if coarse_value is None:
                continue
            largest = max(abs(coarse_value), abs(finer_value))
            # ignore equivariance errors a million times smaller than the model's
            # own error
            scale = scales.get(key.replace(" equivariance RMSE", " RMSE"), 0.0)
            if largest <= 1e-6 * abs(scale):
                continue
            change = abs(finer_value - coarse_value) / largest
            if change > tolerance:
                logging.warning(
                    f"the {key} is not converged with respect to the quadrature "
                    f"grid: {coarse_value:.4g} at max_angular_momentum_grid="
                    f"{coarse_grid} against {finer_value:.4g} at "
                    f"{coarse_grid + 2} ({100 * change:.0f}% change). "
                    "The reported equivariance error is unreliable; increase "
                    "`max_angular_momentum_grid` until it stops changing."
                )

    def finalize(self, not_per_atom: List[str]) -> Dict[str, float]:
        """Finalize the metrics.

        The equivariance and O(3)-averaged errors are also reported combined as
        ``sqrt(equivariance^2 + averaged^2)``, the RMSE the model would have on
        randomly oriented copies of every system.

        :param not_per_atom: Keys containing one of these strings are not labeled
            as "(per atom)".

        :return: The equivariance, O(3)-averaged and oriented metrics.
        """
        equivariance = self.equivariance_acc.finalize(not_per_atom=not_per_atom)
        averaged_rmse = self.averaged_rmse_acc.finalize(not_per_atom=not_per_atom)
        averaged_mae = self.averaged_mae_acc.finalize(not_per_atom=not_per_atom)

        metrics = dict(equivariance)
        for key, value in averaged_rmse.items():
            metrics[key.replace(" RMSE", " O3-averaged RMSE")] = value
        for key, value in averaged_mae.items():
            metrics[key.replace(" MAE", " O3-averaged MAE")] = value

        for key, equivariance_value in equivariance.items():
            averaged_key = key.replace(" equivariance RMSE", " RMSE")
            if averaged_key in averaged_rmse:
                combined = (
                    equivariance_value**2 + averaged_rmse[averaged_key] ** 2
                ) ** 0.5
                metrics[key.replace(" equivariance RMSE", " oriented RMSE")] = combined

        return metrics
