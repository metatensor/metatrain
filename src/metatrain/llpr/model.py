import logging
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

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

from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    DatasetInfo,
    unpack_batch,
)
from metatrain.utils.data.target_info import is_auxiliary_output
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)

from . import checkpoints
from .documentation import ModelHypers


class LLPRUncertaintyModel(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 3

    # all torch devices and dtypes are supported, if they are supported by the wrapped
    # the check is performed in the trainer
    __supported_devices__ = ["cuda", "cpu", "mps"]
    __supported_dtypes__ = [torch.float32, torch.float64, torch.bfloat16, torch.float16]
    # more to be added if needed

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
    metatrain), optionally per atom to calculate LPRs (per-atom uncertainties)
    with the LLPR method.

    Optionally, in order to be compatible with the LLPR ensemble capabilities of this
    class, the wrapped model also needs to have last-layer weights accessible for each
    target. These should be provided as a dictionary mapping target names to lists of
    parameter names in the ``last_layer_parameter_names`` attribute of the wrapped
    model. If multiple parameters constitute the last-layer weights for a target, then
    these should be provided in the same order that corresponds to the order of the
    last-layer features.

    All uncertainties provided by this class are standard deviations (as opposed to
    variances). Prediction rigidities (local and total) can be calculated, according to
    their definition, as the inverse of the square of the standard deviations returned
    by this class.

    :param model: The model to wrap.
    :param ensemble_weight_sizes: The sizes of the ensemble weights, only used
        internally when reloading checkpoints.
    """

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        self.hypers = hypers
        self.dataset_info = dataset_info

    def set_wrapped_model(self, model: ModelInterface) -> None:
        # this function is called after initialization, as well as

        hypers = self.hypers
        dataset_info = self.dataset_info

        # ensemble weight sizes need to be extracted from the hypers

        self.model = model
        self.ll_feat_size = self.model.last_layer_feature_size

        # we need the capabilities of the model to be able to infer the capabilities
        # of the LLPR model. Here, we do a trick: we call export on the model to to make
        # it handle the conversion from dataset_info to capabilities, as well as to
        # get its dtype
        old_capabilities = self.model.export().capabilities()
        dtype = getattr(torch, old_capabilities.dtype)

        # checks between dataset_info and model outputs
        if dataset_info.length_unit != old_capabilities.length_unit:
            raise ValueError(
                "The length unit in the dataset info is different from the "
                "length unit of the wrapped model"
            )
        for atomic_type in dataset_info.atomic_types:
            if atomic_type not in old_capabilities.atomic_types:
                raise ValueError(
                    f"Atomic type {atomic_type} not supported by the wrapped model"
                )
        for target_name, target in dataset_info.targets.items():
            if target_name not in old_capabilities.outputs:
                raise ValueError(
                    f"Target {target_name} not supported by the wrapped model"
                )
            if target.unit != old_capabilities.outputs[target_name].unit:
                raise ValueError(
                    f"Target {target_name} has unit {target.unit}, but the "
                    f"wrapped model has unit "
                    f"{old_capabilities.outputs[target_name].unit}"
                )

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
                description=output.description,
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

        self.ensemble_weight_sizes = hypers["num_ensemble_members"]

        # register buffers for ensemble weights and ensemble outputs
        ensemble_outputs = {}
        for name in self.ensemble_weight_sizes:
            if name not in self.outputs_list:
                raise ValueError(
                    f"Output '{name}' in ensembles section is not supported by "
                    "the model"
                )
            ensemble_weights_name = (
                "mtt::aux::" + name.replace("mtt::", "") + "_ensemble_weights"
            )
            if ensemble_weights_name == "mtt::aux::energy_ensemble_weights":
                ensemble_weights_name = "energy_ensemble_weights"
            ensemble_output_name = (
                "mtt::aux::" + name.replace("mtt::", "") + "_ensemble"
            )
            if ensemble_output_name == "mtt::aux::energy_ensemble":
                ensemble_output_name = "energy_ensemble"
            ensemble_outputs[ensemble_output_name] = ModelOutput(
                quantity=old_capabilities.outputs[name].quantity,
                unit=old_capabilities.outputs[name].unit,
                per_atom=old_capabilities.outputs[name].per_atom,
                description=f"ensemble of '{name}'",
            )
        self.capabilities = ModelCapabilities(
            outputs={**self.capabilities.outputs, **ensemble_outputs},
            atomic_types=self.capabilities.atomic_types,
            interaction_range=self.capabilities.interaction_range,
            length_unit=self.capabilities.length_unit,
            supported_devices=self.capabilities.supported_devices,
            dtype=self.capabilities.dtype,
        )
        self.llpr_ensemble_layers = torch.nn.ModuleDict()
        for name, value in self.ensemble_weight_sizes.items():
            # create the linear layer for ensemble members
            self.llpr_ensemble_layers[name] = torch.nn.Linear(
                self.ll_feat_size,
                value,
                bias=False,
            )

    def restart(self, dataset_info: DatasetInfo) -> "LLPRUncertaintyModel":
        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.model.atomic_types
        ]
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0

        if self.has_new_targets:
            raise ValueError(
                f"New targets found in the dataset: {new_targets}. "
                "The LLPR ensemble calibration does not support adding new targets."
            )
        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The LLPR ensemble calibration does not support adding new atomic "
                "types."
            )

        self.dataset_info = merged_info

        # invoke restart routine for the wrapped model
        self.model.restart(dataset_info)

        return self

    def _get_dataloader(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        batch_size: int,
        is_distributed: bool,
    ) -> DataLoader:
        """
        Create a DataLoader for the provided datasets. As the dataloader is only used to
        accumulate the quantities needed for LLPR calibration, there is no need to
        shuffle or drop the last non-full batch. Distributed sampling can be used or
        not, based on the `is_distributed` argument, and training with double
        precision is enforced.

        :param datasets: List of datasets to create the dataloader from.
        :param batch_size: Batch size to use for the dataloader.
        :param is_distributed: Whether to use distributed sampling or not.
        :return: The created DataLoader.
        """
        # Create the collate function
        targets_keys = list(self.dataset_info.targets.keys())
        requested_neighbor_lists = get_requested_neighbor_lists(self)
        collate_fn = CollateFn(
            target_keys=targets_keys,
            callables=[
                get_system_with_neighbor_lists_transform(requested_neighbor_lists)
            ],
        )

        # Validate dtype from datasets
        if len(datasets) == 0:
            raise ValueError(
                "Cannot create dataloader from empty datasets list. "
                "Please provide non-empty datasets for LLPR calibration."
            )
        if len(datasets[0]) == 0:
            raise ValueError(
                "Cannot create dataloader from empty dataset. "
                "Please provide non-empty datasets for LLPR calibration."
            )

        # Build the dataloaders
        samplers: List[torch.utils.data.Sampler | None]
        if is_distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            samplers = [
                NoPadDistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    seed=0,
                )
                for dataset in datasets
            ]
        else:
            samplers = [None] * len(datasets)

        dataloaders = []
        for dataset, sampler in zip(datasets, samplers, strict=True):
            if len(dataset) < batch_size:
                raise ValueError(
                    f"A dataset has fewer samples "
                    f"({len(dataset)}) than the batch size "
                    f"({batch_size}). "
                    "Please reduce the batch size."
                )
            dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            )

        return CombinedDataLoader(dataloaders, shuffle=True)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if all(output.endswith("_uncertainty") for output in outputs) and all(
            output.endswith("_ensemble") for output in outputs
        ):
            # no uncertainties requested
            return self.model(systems, outputs, selected_atoms)

        outputs_for_model: Dict[str, ModelOutput] = {}
        for name, output in outputs.items():
            if name.endswith("_uncertainty") or name.endswith("_ensemble"):
                # request corresponding features
                target_name = (
                    name.replace("mtt::aux::", "")
                    .replace("_uncertainty", "")
                    .replace("_ensemble", "")
                )
                outputs_for_model[f"mtt::aux::{target_name}_last_layer_features"] = (
                    ModelOutput(per_atom=output.per_atom)
                )
                # for both uncertainties and ensembles, we need the original output,
                # so we request it as well
                if name.endswith("_ensemble") or name.endswith("_uncertainty"):
                    original_name = self._get_original_name(name)
                    outputs_for_model[original_name] = output
                # (will be removed at the end if not requested by the user)

        for name, output in outputs.items():
            # remove uncertainties and ensembles from the requested outputs for the
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

            # compute PRs
            # the code is the same for PR and LPR
            one_over_pr_values = torch.einsum(
                "ij, jk, ik -> i",
                ll_features.block().values,
                self._get_inv_covariance(uncertainty_name),
                ll_features.block().values,
            ).unsqueeze(1)

            original_name = self._get_original_name(uncertainty_name)

            # create labels for properties
            cur_prop = return_dict[original_name].block().properties
            num_prop = len(cur_prop.values)

            # uncertainty TensorMap (values expanded into shape (num_samples, num_prop),
            # with expansion targeting num_prop
            # Note that we take the square root here (just below) to convert variance to
            # standard deviation
            uncertainty = TensorMap(
                keys=Labels(
                    names=["_"],
                    values=torch.tensor(
                        [[0]], device=ll_features.block().values.device
                    ),
                ),
                blocks=[
                    TensorBlock(
                        values=torch.sqrt(one_over_pr_values.expand((-1, num_prop))),
                        samples=ll_features.block().samples,
                        components=ll_features.block().components,
                        properties=cur_prop,
                    )
                ],
            )

            # calibrated multiplier TensorMaps (values expanded into shape (num_samples,
            # num_prop), with expansion targeting num_samples
            multipliers = TensorMap(
                keys=Labels(
                    names=["_"],
                    values=torch.tensor(
                        [[0]], device=ll_features.block().values.device
                    ),
                ),
                blocks=[
                    TensorBlock(
                        values=self._get_multiplier(uncertainty_name).expand(
                            one_over_pr_values.shape[0], num_prop
                        ),
                        samples=ll_features.block().samples,
                        components=ll_features.block().components,
                        properties=cur_prop,
                    )
                ],
            )

            # two TensorMaps of same shape in values are multiplied together here
            return_dict[uncertainty_name] = mts.multiply(uncertainty, multipliers)

        # now deal with potential ensembles (see generate_ensemble method)
        requested_ensembles: List[str] = []
        for name in outputs.keys():
            if name.endswith("_ensemble"):
                requested_ensembles.append(name)

        for ens_name in requested_ensembles:
            original_name = self._get_original_name(ens_name)

            ll_features_name = ens_name.replace("_ensemble", "_last_layer_features")
            if ll_features_name == "energy_last_layer_features":
                # special case for energy_ensemble
                ll_features_name = "mtt::aux::energy_last_layer_features"
            ll_features = return_dict[ll_features_name]

            # Loop needed due to torchscript limitations
            ensemble_values = torch.tensor([0])
            for lin_layer_name, module in self.llpr_ensemble_layers.items():
                if lin_layer_name == original_name:
                    # raw ens output shape is (samples, (num_ens * num_prop))
                    ensemble_values = module(ll_features.block().values)

            # extract property labels and shape
            cur_prop = return_dict[original_name].block().properties
            num_prop = len(cur_prop.values)

            # reshape values accordingly
            ensemble_values = ensemble_values.reshape(
                ensemble_values.shape[0],
                -1,  # num_ens
                num_prop,
            )  # shape: samples, num_ens, num_prop

            # since we know the exact mean of the ensemble from the model's prediction,
            # it should be mathematically correct to use it to re-center the ensemble.
            # Besides making sure that the average is always correct (so that results
            # will always be consistent between LLPR ensembles and the original model),
            # this also takes care of additive contributions that are not present in the
            # last layer, which can be composition, short-range models, a bias in the
            # last layer, etc.
            ensemble_values = (
                ensemble_values
                - ensemble_values.mean(dim=1, keepdim=True)
                + return_dict[original_name].block().values.unsqueeze(1)  # ens_dim
            )

            ensemble_values = ensemble_values.reshape(
                ensemble_values.shape[0],
                -1,
            )  # shape: (samples, (num_ens * num_prop))

            # prepare the properties Labels object for ensemble output, i.e. account
            # for the num_ens dimension
            old_prop_val = return_dict[original_name].block().properties.values
            num_ens = ensemble_values.shape[1]
            num_samples = old_prop_val.shape[0]
            exp_prop_val = old_prop_val.repeat(num_ens, 1)
            ens_idxs = torch.arange(
                num_ens,
                device=old_prop_val.device,
                dtype=old_prop_val.dtype,
            )
            ens_idxs = ens_idxs.repeat_interleave(num_samples).unsqueeze(1)
            new_prop_val = torch.cat([ens_idxs, exp_prop_val], dim=-1)
            ens_prop = Labels(
                names=["ensemble_member"]
                + return_dict[original_name].block().properties.names,
                values=new_prop_val,
            )

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
                        properties=ens_prop,
                    ),
                ],
            )

            return_dict[ens_name] = ensemble

        # Remove any keys if they were not requested. This can happen for last-layer
        # features needed for uncertainty/ensemble calculation as well as for
        # the original outputs when only uncertainties/ensembles were requested
        for key in list(return_dict.keys()):
            if key not in outputs:
                return_dict.pop(key)

        return return_dict

    def compute_covariance(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        batch_size: int,
        is_distributed: bool,
    ) -> None:
        """A function to compute the covariance matrix for a training set.

        The covariance is stored as a buffer in the model.

        :param datasets: List of datasets to use for covariance calculation.
        :param batch_size: Batch size to use for the dataloader.
        :param is_distributed: Whether to use distributed sampling or not.
        """
        # Create dataloader for the training datasets
        train_loader = self._get_dataloader(
            datasets, batch_size, is_distributed=is_distributed
        )

        device = next(iter(self.buffers())).device
        dtype = next(iter(self.buffers())).dtype
        for batch in train_loader:
            systems, targets, extra_data = unpack_batch(batch)
            n_atoms = torch.tensor(
                [len(system.positions) for system in systems], device=device
            )
            systems = [system.to(device=device, dtype=dtype) for system in systems]
            outputs_for_targets = {
                name: ModelOutput(per_atom="atom" in target.block(0).samples.names)
                for name, target in targets.items()
            }
            outputs_for_features = {
                f"mtt::aux::{n.replace('mtt::', '')}_last_layer_features": o
                for n, o in outputs_for_targets.items()
            }
            output = self.forward(
                systems, {**outputs_for_targets, **outputs_for_features}
            )
            for name in targets.keys():
                ll_feat_tmap = output[
                    f"mtt::aux::{name.replace('mtt::', '')}_last_layer_features"
                ]
                # TODO: interface ll_feat calculation with the loss function,
                # paying attention to normalization w.r.t. n_atoms
                if not outputs_for_targets[name].per_atom:
                    ll_feats = ll_feat_tmap.block().values.detach() / n_atoms.unsqueeze(
                        1
                    )
                else:
                    # For per-atom targets, use the features directly
                    ll_feats = ll_feat_tmap.block().values.detach()
                uncertainty_name = _get_uncertainty_name(name)
                covariance = self._get_covariance(uncertainty_name)
                covariance += ll_feats.T @ ll_feats

        if is_distributed:
            torch.distributed.barrier()
            # All-reduce the covariance matrices across all processes
            for name in self.outputs_list:
                uncertainty_name = _get_uncertainty_name(name)
                covariance = self._get_covariance(uncertainty_name)
                torch.distributed.all_reduce(covariance)

    def compute_inverse_covariance(self, regularizer: Optional[float] = None) -> None:
        """A function to compute the inverse covariance matrix.

        The inverse covariance is stored as a buffer in the model.

        :param regularizer: A regularization parameter to ensure the matrix is
            invertible. If not provided, the function will try to compute the
            inverse without regularization and increase the regularization
            parameter until the matrix is invertible.
        """

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
                def is_psd(x: torch.Tensor) -> torch.Tensor:
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

    def calibrate(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        batch_size: int,
        is_distributed: bool,
    ) -> None:
        """
        Calibrate the LLPR model.

        This function computes the calibration constants (one for each output)
        that are used to scale the uncertainties in the LLPR model. The
        calibration is performed in a simple way by computing the calibration
        constant as the mean of the squared residuals divided by the mean of
        the non-calibrated uncertainties.

        :param datasets: List of datasets to use for calibration.
        :param batch_size: Batch size to use for the dataloader.
        :param is_distributed: Whether to use distributed sampling or not.
        """
        # Create dataloader for the validation datasets
        valid_loader = self._get_dataloader(
            datasets, batch_size, is_distributed=is_distributed
        )

        # calibrate the LLPR
        device = next(iter(self.buffers())).device
        dtype = next(iter(self.buffers())).dtype

        sums = {}  # type: ignore
        counts = {}  # type: ignore

        with torch.no_grad():
            for batch in valid_loader:
                systems, targets, extra_data = unpack_batch(batch)
                systems = [system.to(device=device, dtype=dtype) for system in systems]
                targets = {
                    name: target.to(device=device, dtype=dtype)
                    for name, target in targets.items()
                }
                requested_outputs = {}
                for name in targets:
                    per_atom = "atom" in targets[name].block(0).samples.names
                    requested_outputs[name] = ModelOutput(per_atom=per_atom)
                    uncertainty_name = _get_uncertainty_name(name)
                    requested_outputs[uncertainty_name] = ModelOutput(per_atom=per_atom)

                outputs = self.forward(systems, requested_outputs)

                for name, target in targets.items():
                    uncertainty_name = _get_uncertainty_name(name)

                    pred = outputs[name].block().values.detach()
                    targ = target.block().values
                    unc = outputs[uncertainty_name].block().values.detach()

                    # compute the uncertainty multiplier
                    residuals = pred - targ
                    squared_residuals = residuals**2
                    if squared_residuals.ndim > 2:
                        # squared residuals need to be summed over component dimensions,
                        # i.e., all but the first and last dimensions
                        squared_residuals = torch.sum(
                            squared_residuals,
                            dim=tuple(range(1, squared_residuals.ndim - 1)),
                        )

                    ratios = squared_residuals / unc**2  # can be multi-dimensional

                    ratios_sum64 = torch.sum(ratios.to(torch.float64), dim=0)
                    count = torch.tensor(
                        ratios.shape[0], dtype=torch.long, device=device
                    )

                    if uncertainty_name not in sums:
                        sums[uncertainty_name] = ratios_sum64
                        counts[uncertainty_name] = count
                    else:
                        sums[uncertainty_name] = sums[uncertainty_name] + ratios_sum64
                        counts[uncertainty_name] = counts[uncertainty_name] + count

        if is_distributed:
            # All-reduce the accumulated statistics across all processes
            for uncertainty_name in sums:
                torch.distributed.all_reduce(
                    sums[uncertainty_name], op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    counts[uncertainty_name], op=torch.distributed.ReduceOp.SUM
                )

        for uncertainty_name in sums:
            global_mean64 = sums[uncertainty_name] / counts[uncertainty_name].to(
                torch.float64
            )
            multiplier = self._get_multiplier(uncertainty_name)
            multiplier[:] = torch.sqrt(global_mean64).to(multiplier.dtype)

    def generate_ensemble(self) -> None:
        """Generate an ensemble of weights for the model.

        The ensemble is generated by sampling from a multivariate normal
        distribution with mean given by the input weights and covariance given
        by the inverse covariance matrix.
        """
        # concatenate the provided weight tensors
        # (necessary if there are multiple, as in the case of PET)
        # weight tensor is of shape (num_subtarget, concat_llfeat)
        weight_tensors = {}  # type: ignore
        for name in self.ensemble_weight_sizes:
            tensor_names = self.model.last_layer_parameter_names[name]
            weight_tensors[name] = torch.concatenate(
                [self.model.state_dict()[tn] for tn in tensor_names],
                axis=-1,
            )  # type: ignore

        # sampling; each member is sampled from a multivariate normal distribution
        # with mean given by the input weights and covariance given by the inverse
        # covariance matrix
        device = next(iter(self.buffers())).device
        dtype = next(iter(self.buffers())).dtype

        for name, weights in weight_tensors.items():
            uncertainty_name = _get_uncertainty_name(name)
            cur_multiplier = self._get_multiplier(uncertainty_name)
            cur_inv_covariance = (
                self._get_inv_covariance(uncertainty_name)
                .clone()
                .detach()
                .cpu()
                .numpy()
            )
            rng = np.random.default_rng()

            ensemble_weights = []

            for ii in range(weights.shape[0]):
                cur_ensemble_weights = rng.multivariate_normal(
                    weights[ii].clone().detach().cpu().numpy(),
                    cur_inv_covariance * cur_multiplier[ii].item() ** 2,
                    size=self.ensemble_weight_sizes[name],
                    method="svd",
                ).T
                cur_ensemble_weights = torch.tensor(
                    cur_ensemble_weights, device=device, dtype=dtype
                )
                ensemble_weights.append(cur_ensemble_weights)

            ensemble_weights = torch.stack(
                ensemble_weights,
                axis=-1,
            )  # shape: (ll_feat, n_ens, n_subtarget)
            ensemble_weights = ensemble_weights.reshape(
                ensemble_weights.shape[0],
                -1,
            )  # shape: (ll_feat, n_ens * n_subtarget)
            # assign the generated weights
            with torch.no_grad():
                self.llpr_ensemble_layers[name].weight.copy_(ensemble_weights.T)

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
                description=f"ensemble of {name}",
            )
        self.capabilities = ModelCapabilities(
            outputs={**old_outputs, **new_outputs},
            atomic_types=self.capabilities.atomic_types,
            interaction_range=self.capabilities.interaction_range,
            length_unit=self.capabilities.length_unit,
            supported_devices=self.capabilities.supported_devices,
            dtype=self.capabilities.dtype,
        )

    def get_checkpoint(self) -> Dict[str, Any]:
        wrapped_model_checkpoint = self.model.get_checkpoint()
        state_dict = {
            k: v for k, v in self.state_dict().items() if not k.startswith("model.")
        }
        checkpoint = {
            "architecture_name": "llpr",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": state_dict,
            "best_model_state_dict": state_dict,
            "wrapped_model_checkpoint": wrapped_model_checkpoint,
        }
        return checkpoint

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "LLPRUncertaintyModel":
        model = model_from_checkpoint(checkpoint["wrapped_model_checkpoint"], context)
        if context == "finetune":
            # In this case, we want to allow fine-tuning of the underlying model by
            # extracting it and returning it directly
            return model
        elif context == "restart":
            logging.info(
                "Restart for LLPRUncertaintyModel will attempt continuation of "
                "ensemble calibration"
            )
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context == "export":
            # TODO: other models print the best epoch here; consider doing the same
            # Here, it depends on whether we are exporting a model whose ensemble was
            # also trained by backpropagation or not
            model_state_dict = checkpoint["best_model_state_dict"]
            # this is None if the ensemble was not trained by backpropagation
            if model_state_dict is None:
                model_state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        llpr_model = cls(**checkpoint["model_data"])
        llpr_model.set_wrapped_model(model)

        state_dict_iter = iter(model_state_dict.values())
        next(state_dict_iter)
        dtype = next(state_dict_iter).dtype
        # TODO: find a way to refactor this to avoid strict=False
        llpr_model.to(dtype).load_state_dict(model_state_dict, strict=False)
        return llpr_model

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

    def _get_covariance(self, name: str) -> torch.Tensor:
        name = "covariance_" + name
        requested_buffer = torch.tensor(0)
        for n, buffer in self.named_buffers():
            if n == name:
                requested_buffer = buffer
        return requested_buffer

    def _get_inv_covariance(self, name: str) -> torch.Tensor:
        name = "inv_covariance_" + name
        requested_buffer = torch.tensor(0)
        for n, buffer in self.named_buffers():
            if n == name:
                requested_buffer = buffer
        if requested_buffer.shape == torch.Size([]):
            raise ValueError(f"Inverse covariance for {name} not found.")
        return requested_buffer

    def _get_multiplier(self, name: str) -> torch.Tensor:
        name = "multiplier_" + name
        requested_buffer = torch.tensor(0)
        for n, buffer in self.named_buffers():
            if n == name:
                requested_buffer = buffer
        return requested_buffer

    def _get_original_name(self, name: str) -> str:
        # hopefully a bulletproof way to get the original output name from an
        # uncertainty or ensemble name
        if name.endswith("_uncertainty"):
            original_name = name.replace("_uncertainty", "")
        elif name.endswith("_ensemble"):
            original_name = name.replace("_ensemble", "")
        else:
            raise ValueError(f"Output name {name} is neither uncertainty nor ensemble.")
        if original_name.startswith("mtt::aux::"):
            # original name could be either mtt::output or output
            # try the former, return the latter if not found
            # TODO: not sure what happens if both mtt::output and output are there
            original_name = original_name.replace("aux::", "")
            if original_name not in self.capabilities.outputs:
                original_name = original_name.replace("mtt::", "")
        return original_name

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["model_ckpt_version"] == v:
                update = getattr(checkpoints, f"model_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["model_ckpt_version"] = v + 1

        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )

        return checkpoint

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.capabilities.outputs


def _get_uncertainty_name(name: str) -> str:
    if name == "energy":
        uncertainty_name = "energy_uncertainty"
    else:
        uncertainty_name = f"mtt::aux::{name.replace('mtt::', '')}_uncertainty"
    return uncertainty_name


class NoPadDistributedSampler(torch.utils.data.Sampler[int]):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        indices = torch.arange(n, dtype=torch.long)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = indices[torch.randperm(n, generator=g)]
        # Key property: no padding, no dropping
        return iter(indices[self.rank :: self.num_replicas].tolist())

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n - self.rank + self.num_replicas - 1) // self.num_replicas
