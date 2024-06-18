from typing import Dict, List, Optional

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelOutput,
    System,
)
from torch.utils.data import DataLoader


class LLPRModel(torch.nn.Module):
    # In order to do LLPRs: LL features (per atom), LL feat size
    def __init__(
        self,
        model: torch.jit._script.RecursiveScriptModule,
    ) -> None:
        super().__init__()

        self.model = model
        self.ll_feat_size = self.model._module.last_layer_feature_size

        # update capabilities: now we have additional outputs for the uncertainty
        old_capabilities = self.model.capabilities()
        additional_capabilities = {}
        self.uncertainty_multipliers = {}
        for name, output in old_capabilities.outputs.items():
            if name.startswith("mtt::aux"):
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
        self.register_buffer(
            "covariance",
            torch.zeros(
                (self.ll_feat_size, self.ll_feat_size),
                device=next(self.model.parameters()).device,
                dtype=next(self.model.parameters()).dtype,
            ),
        )
        self.register_buffer(
            "inv_covariance",
            torch.zeros(
                (self.ll_feat_size, self.ll_feat_size),
                device=next(self.model.parameters()).device,
                dtype=next(self.model.parameters()).dtype,
            ),
        )

        # flags
        self.covariance_computed = False
        self.inv_covariance_computed = False

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:

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
            return self.model(systems, options, check_consistency=True)

        per_atom_all_targets = [output.per_atom for output in outputs.values()]
        # impose either all per atom or all not per atom
        if not all(per_atom_all_targets) and any(per_atom_all_targets):
            raise ValueError(
                "All output uncertainties must be either be requested per "
                "atom or not per atom with LLPR."
            )
        per_atom = per_atom_all_targets[0]
        outputs_for_model = {
            "mtt::aux::last_layer_features": ModelOutput(
                quantity="",
                unit="",
                per_atom=per_atom,
            ),
        }
        for name, output in outputs.items():
            # remove uncertainties from the requested outputs for the
            # wrapped model
            if name.startswith("mtt::aux") and name.endswith("_uncertainty"):
                continue
            outputs_for_model[name] = output

        options = ModelEvaluationOptions(
            length_unit="",
            outputs=outputs_for_model,
            selected_atoms=selected_atoms,
        )
        return_dict = self.model(
            systems, options, check_consistency=True
        )  # TODO: True or False here?

        ll_features = return_dict["mtt::aux::last_layer_features"]

        # the code is the same for PR and LPR
        one_over_pr_values = torch.einsum(
            "ij, jk, ik -> i",
            ll_features.block().values,
            self.inv_covariance,
            ll_features.block().values,
        ).unsqueeze(1)
        one_over_pr = TensorMap(
            keys=Labels(
                names=["_"],
                values=torch.tensor([[0]], device=ll_features.block().values.device),
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

        requested_uncertainties: List[str] = []
        for name in outputs.keys():
            if name.startswith("mtt::aux") and name.endswith("_uncertainty"):
                requested_uncertainties.append(name)

        for name in requested_uncertainties:
            return_dict[name] = metatensor.torch.multiply(
                one_over_pr, self.uncertainty_multipliers[name]
            )

        return return_dict

    def compute_covariance(self, train_loader: DataLoader) -> None:
        # Utility function to compute the covariance matrix for a training set.
        device = self.covariance.device
        for batch in train_loader:
            systems, _ = batch
            n_atoms = torch.tensor(
                [len(system.positions) for system in systems], device=device
            )
            systems = [system.to(device=device) for system in systems]
            outputs = {
                "mtt::aux::last_layer_features": ModelOutput(
                    quantity="",
                    unit="",
                    per_atom=False,
                )
            }
            options = ModelEvaluationOptions(
                length_unit="",
                outputs=outputs,
            )
            output = self.model(
                systems, options, check_consistency=True
            )  # TODO: True or False here?
            ll_feat_tmap = output["mtt::aux::last_layer_features"]
            ll_feats = ll_feat_tmap.block().values / n_atoms.unsqueeze(1)
            self.covariance += ll_feats.T @ ll_feats
        self.covariance_computed = True

    def compute_inverse_covariance(self, regularizer: Optional[float] = None):
        if regularizer is not None:
            self.inv_covariance = torch.inverse(
                self.covariance
                + regularizer
                * torch.eye(self.ll_feat_size, device=self.covariance.device)
            )
            self.inv_covariance_computed = True
        else:
            # Try with an increasingly high regularization parameter until
            # the matrix is invertible
            for log10_sigma_squared in torch.linspace(-16.0, 16.0, 33):
                try:
                    self.inv_covariance = torch.inverse(
                        self.covariance
                        + 10**log10_sigma_squared
                        * torch.eye(self.ll_feat_size, device=self.covariance.device)
                    )
                    break
                except torch.linalg.LinAlgError:
                    continue
            self.inv_covariance_computed = True

    def calibrate(self, valid_loader: DataLoader):
        # calibrate the LLPR
        device = self.covariance.device
        all_predictions = {}  # type: ignore
        all_targets = {}  # type: ignore
        all_uncertainties = {}  # type: ignore
        for batch in valid_loader:
            systems, targets = batch
            systems = [system.to(device=device) for system in systems]
            targets = {
                name: target.to(device=device) for name, target in targets.items()
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
                all_predictions[name].append(outputs[name].block().values)
                all_targets[name].append(target.block().values)
                all_uncertainties[uncertainty_name].append(
                    outputs[uncertainty_name].block().values
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
