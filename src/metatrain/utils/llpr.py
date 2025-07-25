from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)
from torch.utils.data import DataLoader

from metatrain.utils.data.target_info import is_auxiliary_output
from metatrain.utils.io import check_file_extension, model_from_checkpoint
from metatrain.utils.metadata import merge_metadata


class LLPRUncertaintyModel(torch.nn.Module):
    __checkpoint_version__ = 1
    __default_metadata__ = ModelMetadata(
        references={
            "architecture": [
                "LLPR (uncertainty method): https://iopscience.iop.org/article/10.1088/2632-2153/ad805f",  # noqa: E501
                "LPR (if using per-atom uncertainty): https://pubs.acs.org/doi/10.1021/acs.jctc.3c00704",  # noqa: E501
            ],
        }
    )

    """A wrapper that adds LLPR uncertainties to a model.

    In order to be compatible with this class, a model needs to have the last-layer
    feature size available as an attribute (with the ``last_layer_feature_size`` name)
    and be capable of returning last-layer features (see auxiliary outputs in
    metatrain), optionally per atom to calculate LPRs with the LLPR method.

    All uncertainties provided by this class are standard deviations (as opposed to
    variances). Prediction rigidities (local and total) can be calculated as the inverse
    of the square of the standard deviation.

    :param model: The model to wrap.
    :param ensemble_weight_sizes: The sizes of the ensemble weights, only used
        internally when reloading checkpoints.
    """

    def __init__(
        self, model, ensemble_weight_sizes: Optional[Dict[str, List[int]]] = None
    ) -> None:
        super().__init__()

        self.model = model
        self.ll_feat_size = self.model.last_layer_feature_size

        # we need the capabilities of the model to be able to infer the capabilities
        # of the LLPR model. Here, we do a trick: we call export on the model to to make
        # it handle the conversion from dataset_info to capabilities
        old_capabilities = self.model.export().capabilities()
        dtype = getattr(torch, old_capabilities.dtype)

        # update capabilities: now we have additional outputs for the uncertainty
        additional_capabilities = {}
        self.outputs_list = []
        for name, output in old_capabilities.outputs.items():
            if is_auxiliary_output(name):
                continue  # auxiliary output
            self.outputs_list.append(name)
            uncertainty_name = _get_uncertainty_name(name)
            additional_capabilities[uncertainty_name] = ModelOutput(
                quantity=output.quantity,
                unit=output.unit,
                per_atom=output.per_atom,
            )
        self.capabilities = ModelCapabilities(
            outputs={**old_capabilities.outputs, **additional_capabilities},
            atomic_types=old_capabilities.atomic_types,
            interaction_range=old_capabilities.interaction_range,
            length_unit=old_capabilities.length_unit,
            supported_devices=old_capabilities.supported_devices,
            dtype=old_capabilities.dtype,
        )

        # register covariance, inverse covariance and multiplier buffers
        for name in self.outputs_list:
            uncertainty_name = _get_uncertainty_name(name)
            self.register_buffer(
                f"covariance_{uncertainty_name}",
                torch.zeros(
                    (self.ll_feat_size, self.ll_feat_size),
                    dtype=dtype,
                ),
            )
            self.register_buffer(
                f"inv_covariance_{uncertainty_name}",
                torch.zeros(
                    (self.ll_feat_size, self.ll_feat_size),
                    dtype=dtype,
                ),
            )
            self.register_buffer(
                f"multiplier_{uncertainty_name}",
                torch.tensor([1.0], dtype=dtype),
            )

        if ensemble_weight_sizes is None:
            ensemble_weight_sizes = {}

        # register buffers for ensemble weights and ensemble outputs
        ensemble_outputs = {}
        for name in self.outputs_list:
            ensemble_weights_name = (
                "mtt::aux::" + name.replace("mtt::", "") + "_ensemble_weights"
            )
            if ensemble_weights_name == "mtt::aux::energy_ensemble_weights":
                ensemble_weights_name = "energy_ensemble_weights"
            if ensemble_weights_name not in ensemble_weight_sizes:
                continue
            self.register_buffer(
                ensemble_weights_name,
                torch.zeros(ensemble_weight_sizes[ensemble_weights_name], dtype=dtype),
            )
            ensemble_output_name = (
                "mtt::aux::" + name.replace("mtt::", "") + "_ensemble"
            )
            if ensemble_output_name == "mtt::aux::energy_ensemble":
                ensemble_output_name = "energy_ensemble"
            ensemble_outputs[ensemble_output_name] = ModelOutput(
                quantity=old_capabilities.outputs[name].quantity,
                unit=old_capabilities.outputs[name].unit,
                per_atom=old_capabilities.outputs[name].per_atom,
            )
        self.capabilities = ModelCapabilities(
            outputs={**self.capabilities.outputs, **ensemble_outputs},
            atomic_types=self.capabilities.atomic_types,
            interaction_range=self.capabilities.interaction_range,
            length_unit=self.capabilities.length_unit,
            supported_devices=self.capabilities.supported_devices,
            dtype=self.capabilities.dtype,
        )

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
        if all("_uncertainty" not in output for output in outputs):
            # no uncertainties requested
            return self.model(systems, outputs, selected_atoms)

        if not self.inv_covariance_computed:
            raise ValueError(
                "Trying to predict with LLPR, but inverse covariance has not "
                "been computed yet."
            )

        outputs_for_model: Dict[str, ModelOutput] = {}
        for name, output in outputs.items():
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
                        per_atom=output.per_atom,
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

        return_dict = self.model(systems, outputs_for_model, selected_atoms)

        requested_uncertainties: List[str] = []
        for name in outputs.keys():
            if (name.startswith("mtt::aux::") and name.endswith("_uncertainty")) or (
                name == "energy_uncertainty"
            ):
                requested_uncertainties.append(name)

        for uncertainty_name in requested_uncertainties:
            ll_features_name = uncertainty_name.replace(
                "_uncertainty", "_last_layer_features"
            )
            if ll_features_name == "energy_last_layer_features":
                # special case for energy_ensemble
                ll_features_name = "mtt::aux::energy_last_layer_features"
            ll_features = return_dict[ll_features_name]
            property_name = (
                "energy" if uncertainty_name == "energy_uncertainty" else "_"
            )

            # compute PRs
            # the code is the same for PR and LPR
            one_over_pr_values = torch.einsum(
                "ij, jk, ik -> i",
                ll_features.block().values,
                self._get_inv_covariance(uncertainty_name),
                ll_features.block().values,
            ).unsqueeze(1)
            uncertainty = TensorMap(
                keys=Labels(
                    names=["_"],
                    values=torch.tensor(
                        [[0]], device=ll_features.block().values.device
                    ),
                ),
                blocks=[
                    TensorBlock(
                        # the output is a standard deviation (not a variance)
                        values=torch.sqrt(one_over_pr_values),
                        samples=ll_features.block().samples,
                        components=ll_features.block().components,
                        properties=Labels(
                            names=[property_name],
                            values=torch.tensor(
                                [[0]], device=ll_features.block().values.device
                            ),
                        ),
                    )
                ],
            )

            return_dict[uncertainty_name] = mts.multiply(
                uncertainty, float(self._get_multiplier(uncertainty_name).item())
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
        device = next(iter(self.buffers())).device
        dtype = next(iter(self.buffers())).dtype
        for batch in train_loader:
            systems, targets, extra_data = batch
            n_atoms = torch.tensor(
                [len(system.positions) for system in systems], device=device
            )
            systems = [system.to(device=device, dtype=dtype) for system in systems]
            outputs_for_targets = {
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
            output = self.forward(
                systems, {**outputs_for_targets, **outputs_for_features}
            )
            for name in targets.keys():
                ll_feat_tmap = output[
                    f"mtt::aux::{name.replace('mtt::', '')}_last_layer_features"
                ]
                ll_feats = ll_feat_tmap.block().values.detach() / n_atoms.unsqueeze(1)
                uncertainty_name = _get_uncertainty_name(name)
                covariance = self._get_covariance(uncertainty_name)
                covariance += ll_feats.T @ ll_feats

        self.covariance_computed = True

    def compute_inverse_covariance(self, regularizer: Optional[float] = None):
        """A function to compute the inverse covariance matrix.

        The inverse covariance is stored as a buffer in the model.

        :param regularizer: A regularization parameter to ensure the matrix is
            invertible. If not provided, the function will try to compute the
            inverse without regularization and increase the regularization
            parameter until the matrix is invertible.
        """
        if not self.covariance_computed:
            raise ValueError(
                "Trying to compute inverse covariance, but covariance has not "
                "been computed yet."
            )

        for name in self.outputs_list:
            uncertainty_name = _get_uncertainty_name(name)
            covariance = self._get_covariance(uncertainty_name)
            inv_covariance = self._get_inv_covariance(uncertainty_name)
            if regularizer is not None:
                inv_covariance[:] = torch.inverse(
                    covariance
                    + regularizer
                    * torch.eye(self.ll_feat_size, device=covariance.device)
                )
            else:
                # Try with an increasingly high regularization parameter until
                # the matrix is invertible
                def is_psd(x):
                    return torch.all(torch.linalg.eigvalsh(x) >= 0.0)

                for log10_sigma_squared in torch.linspace(-20.0, 16.0, 33):
                    if not is_psd(
                        covariance
                        + 10**log10_sigma_squared
                        * torch.eye(self.ll_feat_size, device=covariance.device)
                    ):
                        continue
                    else:
                        inverse = torch.inverse(
                            covariance
                            + 10 ** (log10_sigma_squared + 2.0)  # for good conditioning
                            * torch.eye(self.ll_feat_size, device=covariance.device)
                        )
                        inv_covariance[:] = (inverse + inverse.T) / 2.0
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
        device = next(iter(self.buffers())).device
        dtype = next(iter(self.buffers())).dtype
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
                uncertainty_name = _get_uncertainty_name(name)
                requested_outputs[uncertainty_name] = ModelOutput(
                    quantity="",
                    unit="",
                    per_atom=False,
                )
            outputs = self.forward(systems, requested_outputs)
            for name, target in targets.items():
                uncertainty_name = _get_uncertainty_name(name)
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
            uncertainty_name = _get_uncertainty_name(name)
            all_uncertainties[uncertainty_name] = torch.cat(
                all_uncertainties[uncertainty_name], dim=0
            )

        for name in all_predictions:
            # compute the uncertainty multiplier
            residuals = all_predictions[name] - all_targets[name]
            uncertainty_name = _get_uncertainty_name(name)
            uncertainties = all_uncertainties[uncertainty_name]
            multiplier = self._get_multiplier(uncertainty_name)
            multiplier[:] = torch.sqrt(torch.mean(residuals**2 / uncertainties**2))

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
        device = next(iter(self.buffers())).device
        dtype = next(iter(self.buffers())).dtype
        for name, weights in weight_tensors.items():
            uncertainty_name = _get_uncertainty_name(name)
            rng = np.random.default_rng()
            ensemble_weights = rng.multivariate_normal(
                weights.clone().detach().cpu().numpy(),
                self._get_inv_covariance(uncertainty_name)
                .clone()
                .detach()
                .cpu()
                .numpy()
                * self._get_multiplier(uncertainty_name).item() ** 2,
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

    def save_checkpoint(self, path: Union[str, Path]):
        wrapped_model_checkpoint = self.model.get_checkpoint()
        state_dict = {
            k: v for k, v in self.state_dict().items() if not k.startswith("model.")
        }
        state_dict["covariance_computed"] = self.covariance_computed
        state_dict["inv_covariance_computed"] = self.inv_covariance_computed
        state_dict["is_calibrated"] = self.is_calibrated

        checkpoint = {
            "architecture_name": "llpr",
            "model_ckpt_version": self.__checkpoint_version__,
            "wrapped_model_checkpoint": wrapped_model_checkpoint,
            "state_dict": state_dict,
        }
        torch.save(checkpoint, check_file_extension(path, ".ckpt"))

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "LLPRUncertaintyModel":
        model = model_from_checkpoint(checkpoint["wrapped_model_checkpoint"], context)
        if context == "finetune":
            return model
        elif context == "restart":
            raise NotImplementedError(
                "Restarting from the LLPR checkpoint is not supported. "
                "Please consider finetuning the model, or just export it "
                "in the TorchScript format for final usage."
            )
        elif context == "export":
            # Find the size of the ensemble weights, if any:
            ensemble_weight_sizes = {}
            for name, tensor in checkpoint["state_dict"].items():
                if name.endswith("_ensemble_weights"):
                    ensemble_weight_sizes[name] = list(tensor.shape)

            # Create the model
            wrapped_model = cls(model, ensemble_weight_sizes)
            dtype = next(model.parameters()).dtype
            wrapped_model.covariance_computed = checkpoint["state_dict"].pop(
                "covariance_computed"
            )
            wrapped_model.inv_covariance_computed = checkpoint["state_dict"].pop(
                "inv_covariance_computed"
            )
            wrapped_model.is_calibrated = checkpoint["state_dict"].pop("is_calibrated")
            wrapped_model.to(dtype).load_state_dict(
                checkpoint["state_dict"], strict=False
            )
            return wrapped_model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This function moves them:
        try:
            self.model.additive_models[0]._move_weights_to_device_and_dtype(
                torch.device("cpu"), torch.float64
            )
        except Exception:
            # no weights to move
            pass

        metadata = merge_metadata(
            merge_metadata(self.__default_metadata__, metadata),
            self.model.export().metadata(),
        )

        return AtomisticModel(self.eval(), metadata, self.capabilities)

    def _get_covariance(self, name: str):
        name = "covariance_" + name
        requested_buffer = torch.tensor(0)
        for n, buffer in self.named_buffers():
            if n == name:
                requested_buffer = buffer
        return requested_buffer

    def _get_inv_covariance(self, name: str):
        name = "inv_covariance_" + name
        requested_buffer = torch.tensor(0)
        for n, buffer in self.named_buffers():
            if n == name:
                requested_buffer = buffer
        if requested_buffer.shape == torch.Size([]):
            raise ValueError(f"Inverse covariance for {name} not found.")
        return requested_buffer

    def _get_multiplier(self, name: str):
        name = "multiplier_" + name
        requested_buffer = torch.tensor(0)
        for n, buffer in self.named_buffers():
            if n == name:
                requested_buffer = buffer
        return requested_buffer


def _get_uncertainty_name(name: str):
    if name == "energy":
        uncertainty_name = "energy_uncertainty"
    else:
        uncertainty_name = f"mtt::aux::{name.replace('mtt::', '')}_uncertainty"
    return uncertainty_name


__model__ = LLPRUncertaintyModel
