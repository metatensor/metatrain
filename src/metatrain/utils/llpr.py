from typing import Callable, Dict, List, Optional

import metatensor.torch
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelOutput,
    System,
)
from torch.utils.data import DataLoader

from metatrain.utils.data.target_info import is_auxiliary_output

from .data import DatasetInfo, TargetInfo, get_atomic_types
from .evaluate_model import evaluate_model
from .per_atom import average_by_num_atoms


class LLPRUncertaintyModel(torch.nn.Module):
    """A wrapper that adds LLPR uncertainties to a model.

    In order to be compatible with this class, a model needs to have the last-layer
    feature size available as an attribute (with the ``last_layer_feature_size`` name)
    and be capable of returning last-layer features (see auxiliary outputs in
    metatrain), optionally per atom to calculate LPRs with the LLPR method.

    :param model: The model to wrap.
    """

    def __init__(
        self,
        model: torch.jit.RecursiveScriptModule,
    ) -> None:
        super().__init__()

        self.model = model
        self.ll_feat_size = self.model.module.last_layer_feature_size

        # update capabilities: now we have additional outputs for the uncertainty
        old_capabilities = self.model.capabilities()
        additional_capabilities = {}
        self.uncertainty_multipliers = {}
        for name, output in old_capabilities.outputs.items():
            if is_auxiliary_output(name):
                continue  # auxiliary output
            uncertainty_name = f"mtt::aux::{name.replace('mtt::', '')}_uncertainty"
            additional_capabilities[uncertainty_name] = ModelOutput(
                quantity="",
                unit=f"({output.unit})^2",
                per_atom=True,
            )
            self.uncertainty_multipliers[uncertainty_name] = 1.0
        self.capabilities = ModelCapabilities(
            outputs={**old_capabilities.outputs, **additional_capabilities},
            atomic_types=old_capabilities.atomic_types,
            interaction_range=old_capabilities.interaction_range,
            length_unit=old_capabilities.length_unit,
            supported_devices=old_capabilities.supported_devices,
            dtype=old_capabilities.dtype,
        )

        # register covariance and inverse covariance buffers
        device = next(self.model.parameters()).device
        dtype = getattr(torch, old_capabilities.dtype)
        self.covariances = {
            uncertainty_name: torch.zeros(
                (self.ll_feat_size, self.ll_feat_size),
                device=device,
                dtype=dtype,
            )
            for uncertainty_name in self.uncertainty_multipliers.keys()
        }
        self.inv_covariances = {
            uncertainty_name: torch.zeros(
                (self.ll_feat_size, self.ll_feat_size),
                device=device,
                dtype=dtype,
            )
            for uncertainty_name in self.uncertainty_multipliers.keys()
        }

        # flags
        self.covariance_computed = False
        self.inv_covariance_computed = False
        self.is_calibrated = False

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        device = systems[0].positions.device
        if list(self.covariances.values())[0].device != device:
            for name in self.covariances.keys():
                self.covariances[name] = self.covariances[name].to(device=device)
                self.inv_covariances[name] = self.inv_covariances[name].to(
                    device=device
                )

        if not self.inv_covariance_computed:
            raise ValueError(
                "Trying to predict with LLPR, but inverse covariance has not "
                "been computed yet."
            )

        if all("_uncertainty" not in output for output in outputs):
            # no uncertainties requested
            options = ModelEvaluationOptions(
                length_unit="",
                outputs=outputs,
                selected_atoms=selected_atoms,
            )
            return self.model(systems, options, check_consistency=False)

        per_atom_all_targets = [output.per_atom for output in outputs.values()]
        # impose either all per atom or all not per atom
        if not all(per_atom_all_targets) and any(per_atom_all_targets):
            raise ValueError(
                "All output uncertainties must be either be requested per "
                "atom or not per atom with LLPR."
            )
        per_atom = per_atom_all_targets[0]
        outputs_for_model: Dict[str, ModelOutput] = {}
        for name in outputs.keys():
            if name.endswith("_uncertainty"):
                base_name = name.replace("_uncertainty", "").replace("mtt::aux::", "")
                if base_name not in outputs and f"mtt::{base_name}" not in outputs:
                    raise ValueError(
                        f"Requested uncertainty '{name}' without corresponding "
                        f"output `{base_name}` (or `mtt::{base_name}`)."
                    )
                # request corresponding features
                target_name = name.replace("mtt::aux::", "").replace("_uncertainty", "")
                outputs_for_model[f"mtt::aux::{target_name}_last_layer_features"] = (
                    ModelOutput(
                        quantity="",
                        unit="",
                        per_atom=per_atom,
                    )
                )
        for name, output in outputs.items():
            # remove uncertainties from the requested outputs for the
            # wrapped model
            if name.startswith("mtt::aux") and name.endswith("_uncertainty"):
                continue
            if name.endswith("_ensemble"):
                continue
            outputs_for_model[name] = output

        options = ModelEvaluationOptions(
            length_unit="",
            outputs=outputs_for_model,
            selected_atoms=selected_atoms,
        )
        return_dict = self.model(systems, options, check_consistency=False)

        requested_uncertainties: List[str] = []
        for name in outputs.keys():
            if name.startswith("mtt::aux::") and name.endswith("_uncertainty"):
                requested_uncertainties.append(name)

        for name in requested_uncertainties:
            ll_features = return_dict[
                name.replace("_uncertainty", "_last_layer_features")
            ]

            # compute PRs
            # the code is the same for PR and LPR
            one_over_pr_values = torch.einsum(
                "ij, jk, ik -> i",
                ll_features.block().values,
                self.inv_covariances[name],
                ll_features.block().values,
            ).unsqueeze(1)
            one_over_pr = TensorMap(
                keys=Labels(
                    names=["_"],
                    values=torch.tensor(
                        [[0]], device=ll_features.block().values.device
                    ),
                ),
                blocks=[
                    TensorBlock(
                        values=one_over_pr_values,
                        samples=ll_features.block().samples,
                        components=ll_features.block().components,
                        properties=Labels(
                            names=["_"],
                            values=torch.tensor(
                                [[0]], device=ll_features.block().values.device
                            ),
                        ),
                    )
                ],
            )

            return_dict[name] = metatensor.torch.multiply(
                one_over_pr, self.uncertainty_multipliers[name]
            )

        # now deal with potential ensembles (see generate_ensemble method)
        requested_ensembles: List[str] = []
        for name in outputs.keys():
            if name.endswith("_ensemble"):
                requested_ensembles.append(name)

        for name in requested_ensembles:
            ll_features_name = name.replace("_ensemble", "_last_layer_features")
            if ll_features_name == "energy_last_layer_features":
                # special case for energy_ensemble
                ll_features_name = "mtt::aux::energy_last_layer_features"
            ll_features = return_dict[ll_features_name]
            # get the ensemble weights (getattr not supported by torchscript)
            ensemble_weights = torch.tensor(0.0)
            for buffer_name, buffer in self.named_buffers():
                if buffer_name == name + "_weights":
                    ensemble_weights = buffer
            # the ensemble weights should always be found (checks are performed
            # in the generate_ensemble method and in the metatensor wrapper)
            ensemble_values = torch.einsum(
                "ij, jk -> ik",
                ll_features.block().values,
                ensemble_weights,
            )

            # since we know the exact mean of the ensemble from the model's prediction,
            # it should be mathematically correct to use it to re-center the ensemble.
            # Besides making sure that the average is always correct (so that results
            # will always be consistent between LLPR ensembles and the original model),
            # this also takes care of additive contributions that are not present in the
            # last layer, which can be composition, short-range models, a bias in the
            # last layer, etc.
            original_name = (
                name.replace("_ensemble", "").replace("aux::", "")
                if name.replace("_ensemble", "").replace("aux::", "") in outputs
                else name.replace("_ensemble", "").replace("mtt::aux::", "")
            )
            ensemble_values = (
                ensemble_values
                - ensemble_values.mean(dim=1, keepdim=True)
                + return_dict[original_name].block().values
            )

            property_name = "energy" if name == "energy_ensemble" else "ensemble_member"
            ensemble = TensorMap(
                keys=Labels(
                    names=["_"],
                    values=torch.tensor(
                        [[0]], device=ll_features.block().values.device
                    ),
                ),
                blocks=[
                    TensorBlock(
                        values=ensemble_values,
                        samples=ll_features.block().samples,
                        components=ll_features.block().components,
                        properties=Labels(
                            names=[property_name],
                            values=torch.arange(
                                ensemble_values.shape[1], device=ensemble_values.device
                            ).unsqueeze(1),
                        ),
                    )
                ],
            )
            return_dict[name] = ensemble

        # remove the last-layer features from return_dict if they were not requested
        for key in list(return_dict.keys()):
            if key.endswith("_last_layer_features"):
                if key not in outputs:
                    return_dict.pop(key)

        return return_dict

    def compute_covariance(self, train_loader: DataLoader) -> None:
        """A function to compute the covariance matrix for a training set.

        The covariance is stored as a buffer in the model.

        :param train_loader: A PyTorch DataLoader with the training data.
            The individual samples need to be compatible with the ``Dataset``
            class in ``metatrain``.
        """
        device = next(iter(self.covariances.values())).device
        dtype = next(iter(self.covariances.values())).dtype
        for batch in train_loader:
            systems, targets, extra_data = batch
            n_atoms = torch.tensor(
                [len(system.positions) for system in systems], device=device
            )
            systems = [system.to(device=device, dtype=dtype) for system in systems]
            options_for_targets = {
                name: ModelOutput(
                    quantity="",
                    unit="",
                    per_atom=False,
                )
                for name in targets.keys()
            }
            outputs_for_features = {
                f"mtt::aux::{name.replace('mtt::', '')}_last"
                "_layer_features": ModelOutput(
                    quantity="",
                    unit="",
                    per_atom=False,
                )
                for name in targets.keys()
            }
            options = ModelEvaluationOptions(
                length_unit="",
                outputs={**options_for_targets, **outputs_for_features},
            )
            output = self.model(systems, options, check_consistency=False)
            for name in targets.keys():
                ll_feat_tmap = output[
                    f"mtt::aux::{name.replace('mtt::', '')}_last_layer_features"
                ]
                ll_feats = ll_feat_tmap.block().values.detach() / n_atoms.unsqueeze(1)
                self.covariances[
                    f"mtt::aux::{name.replace('mtt::', '')}_uncertainty"
                ] += ll_feats.T @ ll_feats

        self.covariance_computed = True

    def compute_covariance_as_pseudo_hessian(
        self,
        train_loader: DataLoader,
        target_infos: Dict[str, TargetInfo],
        loss_fn: Callable,
        parameters: Dict[str, List[torch.nn.Parameter]],
    ) -> None:
        """A function to compute the covariance matrix for a training set
        as the pseudo-Hessian of the loss function.

        The covariance/pseudo-Hessian is stored as a buffer in the model. The
        loss function must be compatible with the Dataloader (i.e., it should
        have the same structure as the outputs of the model and as the targets
        in the dataset). All contributions to the loss functions are assumed to
        be per-atom, except for quantities that are already per-atom (e.g.,
        forces).

        :param train_loader: A PyTorch DataLoader with the training data.
            The individual samples need to be compatible with the ``Dataset``
            class in ``metatrain``.
        :param loss_fn: A loss function that takes the model outputs and the
            targets and returns a scalar loss.
        :param parameters: A list of model parameters for which the pseudo-Hessian
            should be computed. This is often necessary as models can have very
            large numbers of parameters, and the pseudo-Hessian's number of
            elements grows quadratically with the number of parameters. For this
            reason, only a subset of the parameters of the model is usually used
            in the calculation. This list allows the user to feed the parameters
            of interest directly to the function. In order to function correctly,
            the model's parameters should be those corresponding to the last
            layer(s) of the model, such that their concatenation corresponds to the
            last-layer features, in the same order as those are returned by the
            base model.
        """
        self.model = self.model.train()  # we need gradients w.r.t. parameters
        # disable gradients for all parameters that are not in the list
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        all_parameters_that_require_grad = []
        all_output_names = []
        for output_name, output_parameters in parameters.items():
            for parameter in output_parameters:
                all_parameters_that_require_grad.append(parameter)
                all_output_names.append(output_name)
                parameter.requires_grad = True

        dataset = train_loader.dataset
        dataset_info = DatasetInfo(
            length_unit=self.capabilities.length_unit,  # TODO: check
            atomic_types=get_atomic_types(dataset),
            targets=target_infos,
        )
        train_targets = dataset_info.targets
        device = next(iter(self.covariances.values())).device
        dtype = next(iter(self.covariances.values())).dtype
        for batch in train_loader:
            systems, targets, extra_data = batch
            systems = [system.to(device=device, dtype=dtype) for system in systems]
            targets = {
                name: tmap.to(device=device, dtype=dtype)
                for name, tmap in targets.items()
            }
            predictions = evaluate_model(
                self.model,
                systems,
                {key: train_targets[key] for key in targets.keys()},
                is_training=True,  # keep the computational graph
            )

            # average by the number of atoms
            predictions = average_by_num_atoms(predictions, systems, [])
            targets = average_by_num_atoms(targets, systems, [])

            loss = loss_fn(predictions, targets)

            grads = torch.autograd.grad(
                loss,
                all_parameters_that_require_grad,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,  # if there are multiple last-layers
                materialize_grads=True,  # avoid Nones
            )

            for output_name in np.unique(all_output_names):
                grads = [
                    grad
                    for grad, name in zip(grads, all_output_names)
                    if name == output_name
                ]
                grads = torch.cat(grads, dim=1)
                self.covariances[
                    "mtt::aux::" + output_name.replace("mtt::", "") + "_uncertainty"
                ] += grads.T @ grads

            for parameter in all_parameters_that_require_grad:
                parameter.grad = None  # reset the gradients

        self.covariance_computed = True

        for parameter in self.model.parameters():
            parameter.requires_grad = True

        self.model = self.model.eval()  # restore the model to evaluation mode

    def compute_inverse_covariance(self, regularizer: Optional[float] = None):
        """A function to compute the inverse covariance matrix.

        The inverse covariance is stored as a buffer in the model.

        :param regularizer: A regularization parameter to ensure the matrix is
            invertible. If not provided, the function will try to compute the
            inverse without regularization and increase the regularization
            parameter until the matrix is invertible.
        """
        for name in self.covariances.keys():
            if regularizer is not None:
                self.inv_covariances[name] = torch.inverse(
                    self.covariances[name]
                    + regularizer
                    * torch.eye(self.ll_feat_size, device=self.covariances[name].device)
                )
            else:
                # Try with an increasingly high regularization parameter until
                # the matrix is invertible
                def is_psd(x):
                    return torch.all(torch.linalg.eigvalsh(x) >= 0.0)

                for log10_sigma_squared in torch.linspace(-20.0, 16.0, 33):
                    if not is_psd(
                        self.covariances[name]
                        + 10**log10_sigma_squared
                        * torch.eye(
                            self.ll_feat_size, device=self.covariances[name].device
                        )
                    ):
                        continue
                    else:
                        inverse = torch.inverse(
                            self.covariances[name]
                            + 10 ** (log10_sigma_squared + 0.0)
                            * torch.eye(
                                self.ll_feat_size, device=self.covariances[name].device
                            )
                        )
                        self.inv_covariances[name] = (inverse + inverse.T) / 2.0
                        break

        self.inv_covariance_computed = True

    def calibrate(self, valid_loader: DataLoader):
        """
        Calibrate the LLPR model.

        This function computes the calibration constants (one for each output)
        that are used to scale the uncertainties in the LLPR model. The
        calibration is performed in a simple way by computing the calibration
        constant as the mean of the squared residuals divided by the mean of
        the non-calibrated uncertainties.

        :param valid_loader: A data loader with the validation data.
            This data loader should be generated from a dataset from the
            ``Dataset`` class in ``metatrain.utils.data``.
        """
        # calibrate the LLPR
        # TODO: in the future, we might want to have one calibration factor per
        # property for outputs with multiple properties
        device = next(iter(self.covariances.values())).device
        dtype = next(iter(self.covariances.values())).dtype
        all_predictions = {}  # type: ignore
        all_targets = {}  # type: ignore
        all_uncertainties = {}  # type: ignore
        for batch in valid_loader:
            systems, targets, extra_data = batch
            systems = [system.to(device=device, dtype=dtype) for system in systems]
            targets = {
                name: target.to(device=device, dtype=dtype)
                for name, target in targets.items()
            }
            # evaluate the targets and their uncertainties, not per atom
            requested_outputs = {}
            for name in targets:
                requested_outputs[name] = ModelOutput(
                    quantity="",
                    unit="",
                    per_atom=False,
                )
                uncertainty_name = f"mtt::aux::{name.replace('mtt::', '')}_uncertainty"
                requested_outputs[uncertainty_name] = ModelOutput(
                    quantity="",
                    unit="",
                    per_atom=False,
                )
            outputs = self.forward(systems, requested_outputs)
            for name, target in targets.items():
                uncertainty_name = f"mtt::aux::{name.replace('mtt::', '')}_uncertainty"
                if name not in all_predictions:
                    all_predictions[name] = []
                    all_targets[name] = []
                    all_uncertainties[uncertainty_name] = []
                all_predictions[name].append(outputs[name].block().values.detach())
                all_targets[name].append(target.block().values)
                all_uncertainties[uncertainty_name].append(
                    outputs[uncertainty_name].block().values.detach()
                )

        for name in all_predictions:
            all_predictions[name] = torch.cat(all_predictions[name], dim=0)
            all_targets[name] = torch.cat(all_targets[name], dim=0)
            uncertainty_name = f"mtt::aux::{name.replace('mtt::', '')}_uncertainty"
            all_uncertainties[uncertainty_name] = torch.cat(
                all_uncertainties[uncertainty_name], dim=0
            )

        for name in all_predictions:
            # compute the uncertainty multiplier
            residuals = all_predictions[name] - all_targets[name]
            uncertainty_name = f"mtt::aux::{name.replace('mtt::', '')}_uncertainty"
            uncertainties = all_uncertainties[uncertainty_name]
            self.uncertainty_multipliers[uncertainty_name] = torch.mean(
                residuals**2 / uncertainties
            ).item()

        self.is_calibrated = True

    def generate_ensemble(
        self, weight_tensors: Dict[str, torch.Tensor], n_members: int
    ) -> None:
        """Generate an ensemble of weights for the model.

        The ensemble is generated by sampling from a multivariate normal
        distribution with mean given by the input weights and covariance given
        by the inverse covariance matrix.

        :param weight_tensors: A dictionary with the weights for the ensemble.
            The keys should be the names of the weights in the model and the
            values should be 1D PyTorch tensors.
        :param n_members: The number of members in the ensemble.
        """
        # note: we could also allow n_members to be different for each output

        # basic checks
        if not self.is_calibrated:
            raise ValueError(
                "LLPR model needs to be calibrated before generating ensembles"
            )
        for key in weight_tensors:
            if key not in self.capabilities.outputs.keys():
                raise ValueError(f"Output '{key}' not supported by model")
            if len(weight_tensors[key].shape) != 1:
                raise ValueError("All weights must be 1D tensors")

        # sampling; each member is sampled from a multivariate normal distribution
        # with mean given by the input weights and covariance given by the inverse
        # covariance matrix
        for name, weights in weight_tensors.items():
            uncertainty_name = "mtt::aux::" + name.replace("mtt::", "") + "_uncertainty"
            device = self.inv_covariances[uncertainty_name].device
            dtype = self.inv_covariances[uncertainty_name].dtype
            rng = np.random.default_rng()
            ensemble_weights = rng.multivariate_normal(
                weights.clone().detach().cpu().numpy(),
                self.inv_covariances[uncertainty_name].clone().detach().cpu().numpy()
                * self.uncertainty_multipliers[uncertainty_name],
                size=n_members,
                method="svd",
            ).T
            ensemble_weights = torch.tensor(
                ensemble_weights, device=device, dtype=dtype
            )
            ensemble_weights_name = (
                "mtt::aux::" + name.replace("mtt::", "") + "_ensemble_weights"
            )
            if ensemble_weights_name == "mtt::aux::energy_ensemble_weights":
                ensemble_weights_name = "energy_ensemble_weights"
            self.register_buffer(
                ensemble_weights_name,
                ensemble_weights,
            )

        # add the ensembles to the capabilities
        old_outputs = self.capabilities.outputs
        new_outputs = {}
        for name in weight_tensors.keys():
            ensemble_name = "mtt::aux::" + name.replace("mtt::", "") + "_ensemble"
            if ensemble_name == "mtt::aux::energy_ensemble":
                ensemble_name = "energy_ensemble"
            new_outputs[ensemble_name] = ModelOutput(
                quantity=old_outputs[name].quantity,
                unit=old_outputs[name].unit,
                per_atom=old_outputs[name].per_atom,
            )
        self.capabilities = ModelCapabilities(
            outputs={**old_outputs, **new_outputs},
            atomic_types=self.capabilities.atomic_types,
            interaction_range=self.capabilities.interaction_range,
            length_unit=self.capabilities.length_unit,
            supported_devices=self.capabilities.supported_devices,
            dtype=self.capabilities.dtype,
        )
