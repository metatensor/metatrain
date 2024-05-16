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
        self.ll_feat_size = self.model.wrapped_module().last_layer_feature_size

        # update capabilities: now we have additional outputs for the uncertainty
        old_capabilities = self.model.capabilities()
        additional_capabilities = {}
        self.uncertainty_multipliers = {}
        for name, output in old_capabilities.outputs.items():
            if name.startswith("mtm::aux"):
                continue  # auxiliary output
            new_name = name.replace("mtm::", "")  # remove mtm:: prefix
            additional_capabilities[new_name] = ModelOutput(
                f"mtm::aux::{name}_uncertainty",
                quantity="",
                unit=f"({output.unit})^2",
                per_atom=True,
            )
            self.uncertainty_multipliers[new_name] = 1.0
        self.capabilities = ModelCapabilities(
            old_capabilities.length_unit,
            outputs={**old_capabilities.outputs, **additional_capabilities},
        )

        # register covariance and inverse covariance buffers
        self.register_buffer(
            "covariance",
            torch.zeros(
                (self.ll_feat_size, self.ll_feat_size),
                device=next(self.model.parameters()).device,
            ),
        )
        self.register_buffer(
            "inv_covariance",
            torch.zeros(
                (self.ll_feat_size, self.ll_feat_size),
                device=next(self.model.parameters()).device,
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
            return self.model(systems, outputs, selected_atoms)

        per_atom_all_targets = [output.per_atom for output in outputs.values()]
        # impose either all per atom or all not per atom
        if not all(per_atom_all_targets) and any(per_atom_all_targets):
            raise ValueError(
                "All output uncertainties must be either be requested per "
                "atom or not per atom with LLPR."
            )
        per_atom = per_atom_all_targets[0]
        outputs_with_last_layer = {
            "mtm::aux::last_layer_features": ModelOutput(
                quantity="",
                unit="",
                per_atom=per_atom,
            ),
        }
        for name, output in outputs.items():
            outputs_with_last_layer[name] = output

        options = ModelEvaluationOptions(
            length_unit="",
            outputs=outputs_with_last_layer,
        )
        return_dict = self.model(
            systems, options, check_consistency=True
        )  # TODO: True or False here?

        ll_features = return_dict["mtm::aux::last_layer_features"]

        # the code is the same for PR and LPR
        one_over_pr_values = torch.einsum(
            "ij, jk, ik -> i",
            ll_features.block().values,
            self.inv_covariance,
            ll_features.block().values,
        ).unsqueeze(1)
        one_over_pr = TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=one_over_pr_values,
                    samples=ll_features.block().samples,
                    components=ll_features.block().components,
                    properties=Labels.single(),
                )
            ],
        )

        requested_uncertainties = [
            name
            for name in outputs
            if name.startswith("mtm::aux") and name.endswith("_uncertainty")
        ]

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
                "mtm::aux::last_layer_features": ModelOutput(
                    quantity="",
                    unit="",
                    per_atom=False,
                )
            }
            output = self.forward(systems, outputs)
            ll_feat_tmap = output["mtm::aux::last_layer_features"]
            ll_feats = ll_feat_tmap.block().values / n_atoms.unsqueeze(1)
            self.covariance += ll_feats.T @ ll_feats
        self.covariance_computed = True

    def compute_inverse_covariance(self):
        # Try with an increasingly high regularization parameter until
        # the matrix is invertible

        for sigma_squared in torch.geomspace(1e-16, 1e16, 33):
            try:
                self.inv_covariance = torch.inverse(
                    self.covariance
                    + sigma_squared
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
                uncertainty_name = name.replace("mtm::", "")
                uncertainty_name = f"mtm::aux::{uncertainty_name}_uncertainty"
                requested_outputs[uncertainty_name] = ModelOutput(
                    quantity="",
                    unit="",
                    per_atom=False,
                )
            outputs = self.forward(systems, requested_outputs)
            for name, target in targets.items():
                uncertainty_name = name.replace("mtm::", "")
                uncertainty_name = f"mtm::aux::{uncertainty_name}_uncertainty"
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
            uncertainty_name = name.replace("mtm::", "")
            uncertainty_name = f"mtm::aux::{uncertainty_name}_uncertainty"
            all_uncertainties[uncertainty_name] = torch.cat(
                all_uncertainties[uncertainty_name], dim=0
            )

        for name in all_predictions:
            # compute the uncertainty multiplier
            residuals = all_predictions[name] - all_targets[name]
            uncertainty_name = name.replace("mtm::", "")
            uncertainty_name = f"mtm::aux::{uncertainty_name}_uncertainty"
            uncertainties = all_uncertainties[uncertainty_name]
            self.uncertainty_multipliers[name] = torch.mean(
                residuals**2 / uncertainties
            )
