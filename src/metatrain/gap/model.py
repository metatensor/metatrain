from typing import Dict, List, Optional, Tuple, Type, Union

import metatensor.torch
import numpy as np
import rascaline
import rascaline.torch
import scipy
import skmatter
import torch
from metatensor import Labels, TensorBlock, TensorMap
from metatensor.torch import Labels as TorchLabels
from metatensor.torch import TensorBlock as TorchTensorBlock
from metatensor.torch import TensorMap as TorchTensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)
from skmatter._selection import _FPS

from metatrain.utils.data.dataset import DatasetInfo

from ..utils.additive import ZBL, CompositionModel
from ..utils.metadata import append_metadata_references


class GAP(torch.nn.Module):
    __supported_devices__ = ["cpu"]
    __supported_dtypes__ = [torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "implementation": [
                "rascaline: https://github.com/Luthaf/rascaline",
            ],
            "architecture": [
                "SOAP: https://doi.org/10.1002/qua.24927",
                "GAP: https://doi.org/10.1103/PhysRevB.87.184115",
            ],
        }
    )

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()

        if len(dataset_info.targets) > 1:
            raise NotImplementedError("GAP only supports a single output")

        # Check capabilities
        for target in dataset_info.targets.values():
            if not (
                target.is_scalar
                and target.quantity == "energy"
                and len(target.layout.block(0).properties) == 1
            ):
                raise ValueError(
                    "GAP only supports total-energy-like outputs, "
                    f"but a {target.quantity} was provided"
                )
            if target.per_atom:
                raise ValueError(
                    "GAP only supports per-structure outputs, "
                    "but a per-atom output was provided"
                )
        target_name = next(iter(dataset_info.targets.keys()))
        if dataset_info.targets[target_name].quantity != "energy":
            raise ValueError("GAP only supports energies as target")
        if dataset_info.targets[target_name].per_atom:
            raise ValueError("GAP does not support per-atom energies")

        self.dataset_info = dataset_info

        self.outputs = {
            key: ModelOutput(
                quantity=value.quantity,
                unit=value.unit,
                per_atom=False,
            )
            for key, value in dataset_info.targets.items()
        }

        self.atomic_types = dataset_info.atomic_types
        self.hypers = model_hypers

        # creates a composition weight tensor that can be directly indexed by species,
        # this can be left as a tensor of zero or set from the outside using
        # set_composition_weights (recommended for better accuracy)
        n_outputs = len(dataset_info.targets)
        self.register_buffer(
            "composition_weights",
            torch.zeros(
                (n_outputs, max(self.atomic_types) + 1),
                dtype=torch.float64,  # we only support float64 for now
            ),
        )
        # buffers cannot be indexed by strings (torchscript), so we create a single
        # tensor for all output. Due to this, we need to slice the tensor when we use
        # it and use the output name to select the correct slice via a dictionary
        self.output_to_index = {
            output_name: i for i, output_name in enumerate(dataset_info.targets.keys())
        }

        self.register_buffer(
            "kernel_weights",
            torch.zeros(
                model_hypers["krr"]["num_sparse_points"],
                dtype=torch.float64,  # we only support float64 for now
            ),
        )
        self._soap_torch_calculator = rascaline.torch.SoapPowerSpectrum(
            **model_hypers["soap"]
        )

        kernel_kwargs = {
            "degree": model_hypers["krr"]["degree"],
            "aggregate_names": ["atom", "center_type"],
        }
        self._subset_of_regressors = SubsetOfRegressors(
            kernel_type=model_hypers["krr"].get("kernel", "polynomial"),
            kernel_kwargs=kernel_kwargs,
        )

        self._sampler = FPS(n_to_select=model_hypers["krr"]["num_sparse_points"])

        # set it do dummy keys, these are properly set during training
        self._keys = TorchLabels.empty("_")

        dummy_weights = TorchTensorMap(
            TorchLabels(["_"], torch.tensor([[0]])),
            [metatensor.torch.block_from_array(torch.empty(1, 1))],
        )
        dummy_X_pseudo = TorchTensorMap(
            TorchLabels(["_"], torch.tensor([[0]])),
            [metatensor.torch.block_from_array(torch.empty(1, 1))],
        )
        self._subset_of_regressors_torch = TorchSubsetofRegressors(
            dummy_weights,
            dummy_X_pseudo,
            kernel_kwargs={
                "aggregate_names": ["atom", "center_type"],
            },
        )
        self._species_labels: TorchLabels = TorchLabels.empty("_")

        # additive models: these are handled by the trainer at training
        # time, and they are added to the output at evaluation time
        composition_model = CompositionModel(
            model_hypers={},
            dataset_info=dataset_info,
        )
        additive_models = [composition_model]
        if self.hypers["zbl"]:
            additive_models.append(
                ZBL(
                    {},
                    dataset_info=DatasetInfo(
                        length_unit=dataset_info.length_unit,
                        atomic_types=self.atomic_types,
                        targets={
                            target_name: target_info
                            for target_name, target_info in dataset_info.targets.items()
                            if ZBL.is_valid_target(target_name, target_info)
                        },
                    ),
                )
            )
        self.additive_models = torch.nn.ModuleList(additive_models)

    def restart(self, dataset_info: DatasetInfo) -> "GAP":
        raise ValueError("GAP does not allow restarting training")

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[TorchLabels] = None,
    ) -> Dict[str, TorchTensorMap]:
        soap_features = self._soap_torch_calculator(
            systems, selected_samples=selected_atoms
        )
        # move keys and species labels to device
        self._keys = self._keys.to(systems[0].device)
        self._species_labels = self._species_labels.to(systems[0].device)

        new_blocks: List[TorchTensorBlock] = []
        # HACK: to add a block of zeros if there are missing species
        # which were present at training time
        # (with samples "system", "atom" = 0, 0)
        # given the values are all zeros, it does not introduce an error
        dummyblock: TorchTensorBlock = TorchTensorBlock(
            values=torch.zeros(
                (1, len(soap_features[0].properties)),
                dtype=systems[0].positions.dtype,
                device=systems[0].device,
            ),
            samples=TorchLabels(
                ["system", "atom"],
                torch.tensor([[0, 0]], dtype=torch.int, device=systems[0].device),
            ),
            properties=soap_features[0].properties,
            components=[],
        )
        if len(soap_features[0].gradients_list()) > 0:
            for idx, grad in enumerate(soap_features[0].gradients_list()):
                dummyblock_grad: TorchTensorBlock = TorchTensorBlock(
                    values=torch.zeros(
                        (
                            1,
                            soap_features[0].gradient(grad).values.shape[1 + idx],
                            len(soap_features[0].gradient(grad).properties),
                        ),
                        dtype=systems[0].positions.dtype,
                        device=systems[0].device,
                    ),
                    samples=TorchLabels(
                        ["sample", "system", "atom"],
                        torch.tensor(
                            [[0, 0, 0]], dtype=torch.int, device=systems[0].device
                        ),
                    ),
                    components=soap_features[0].gradient(grad).components,
                    properties=soap_features[0].gradient(grad).properties,
                )
                dummyblock.add_gradient(grad, dummyblock_grad)

        for idx_key in range(len(self._species_labels)):
            key = self._species_labels.entry(idx_key)
            if soap_features.keys.position(key) is not None:
                new_blocks.append(soap_features.block(key))
            else:
                new_blocks.append(dummyblock)
        soap_features = TorchTensorMap(keys=self._species_labels, blocks=new_blocks)
        soap_features = soap_features.keys_to_samples("center_type")
        # here, we move to properties to use metatensor operations to aggregate
        # later on. Perhaps we could retain the sparsity all the way to the kernels
        # of the soap features with a lot more implementation effort
        soap_features = soap_features.keys_to_properties(
            ["neighbor_1_type", "neighbor_2_type"]
        )
        soap_features = TorchTensorMap(self._keys, soap_features.blocks())
        output_key = list(outputs.keys())[0]
        energies = self._subset_of_regressors_torch(soap_features)
        return_dict = {output_key: energies}

        if not self.training:
            # at evaluation, we also add the additive contributions
            for additive_model in self.additive_models:
                outputs_for_additive_model: Dict[str, ModelOutput] = {}
                for name, output in outputs.items():
                    if name in additive_model.outputs:
                        outputs_for_additive_model[name] = output
                additive_contributions = additive_model(
                    systems,
                    outputs_for_additive_model,
                    selected_atoms,
                )
                for name in additive_contributions:
                    return_dict[name] = metatensor.torch.add(
                        return_dict[name],
                        additive_contributions[name],
                    )

        return return_dict

    def export(
        self, metadata: Optional[ModelMetadata] = None
    ) -> MetatensorAtomisticModel:
        interaction_ranges = [self.hypers["soap"]["cutoff"]]
        for additive_model in self.additive_models:
            if hasattr(additive_model, "cutoff_radius"):
                interaction_ranges.append(additive_model.cutoff_radius)
        interaction_range = max(interaction_ranges)

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This funciton moves them:
        self.additive_models[0]._move_weights_to_device_and_dtype(
            torch.device("cpu"), torch.float64
        )

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=sorted(self.dataset_info.atomic_types),
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=["cuda", "cpu"],
            dtype="float64",
        )

        # we export a torch scriptable regressor TorchSubsetofRegressors
        # that is used in the forward path
        self._subset_of_regressors_torch = (
            self._subset_of_regressors.export_torch_script_model()
        )

        if metadata is None:
            metadata = ModelMetadata()

        append_metadata_references(metadata, self.__default_metadata__)

        return MetatensorAtomisticModel(self.eval(), metadata, capabilities)

    def set_composition_weights(
        self,
        output_name: str,
        input_composition_weights: torch.Tensor,
        species: List[int],
    ) -> None:
        """Set the composition weights for a given output."""
        # all species that are not present retain their weight of zero
        self.composition_weights[self.output_to_index[output_name]][  # type: ignore
            species
        ] = input_composition_weights.to(
            dtype=self.composition_weights.dtype,  # type: ignore
            device=self.composition_weights.device,  # type: ignore
        )

    def apply_composition_weights(
        self, systems: List[System], energies: TorchTensorMap
    ) -> TorchTensorMap:
        """Apply the composition weights to the energies."""
        new_blocks: List[TorchTensorBlock] = []
        for block in energies.blocks():
            atomic_species = [system.types for system in systems]
            new_values = block.values
            for i in range(len(new_values)):
                for s in atomic_species[i]:
                    new_values[i] += self.composition_weights[0, s]
            new_blocks.append(
                TorchTensorBlock(
                    values=new_values,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )

        return TorchTensorMap(energies.keys, new_blocks)


########################################################################################
# All classes and methods will be moved to metatensor-operations and metatensor-learn] #
########################################################################################


class _SorKernelSolver:
    """
    A few quick implementation notes, docs to be done.

    This is meant to solve the subset of regressors (SoR) problem::

    .. math::

        w = (KNM.T@KNM + reg*KMM)^-1 @ KNM.T@y

    The inverse needs to be stabilized with application of a numerical jitter,
    that is expressed as a fraction of the largest eigenvalue of KMM

    :param KMM:
        KNM matrix
    :param regularizer:
        regularizer
    :param jitter:
        numerical jitter to stabilize fit
    :param solver:
        Method to solve the sparse KRR equations.

        * RKHS-QR: TBD
        * RKHS: Compute first the reproducing kernel features by diagonalizing K_MM and
          computing `P_NM = K_NM @ U_MM @ Lam_MM^(-1.2)` and then solves the linear
          problem for those (which is usually better conditioned)::

              (P_NM.T@P_NM + 1)^(-1) P_NM.T@Y

        * QR: TBD
        * solve: Uses `scipy.linalg.solve` for the normal equations::

              (K_NM.T@K_NM + K_MM)^(-1) K_NM.T@Y

        * lstsq: require rcond value. Use `numpy.linalg.solve(rcond=rcond)` for the
          normal equations::

             (K_NM.T@K_NM + K_MM)^(-1) K_NM.T@Y

    Reference
    ---------
    Foster, L., Waagen, A., Aijaz, N., Hurley, M., Luis, A., Rinsky, J., ... &
    Srivastava, A. (2009). Stable and Efficient Gaussian Process Calculations. Journal
    of Machine Learning Research, 10(4).
    """

    def __init__(
        self,
        KMM: np.ndarray,
        regularizer: float = 1.0,
        jitter: float = 0.0,
        solver: str = "RKHS",
        relative_jitter: bool = True,
    ):
        self.solver = solver
        self.KMM = KMM
        self.relative_jitter = relative_jitter

        self._nM = len(KMM)
        if self.solver == "RKHS" or self.solver == "RKHS-QR":
            self._vk, self._Uk = scipy.linalg.eigh(KMM)
            self._vk = self._vk[::-1]
            self._Uk = self._Uk[:, ::-1]
        elif self.solver == "QR" or self.solver == "solve" or self.solver == "lstsq":
            # gets maximum eigenvalue of KMM to scale the numerical jitter
            self._KMM_maxeva = scipy.sparse.linalg.eigsh(
                KMM, k=1, return_eigenvectors=False
            )[0]
        else:
            raise ValueError(
                f"Solver {solver} not supported. Possible values "
                "are 'RKHS', 'RKHS-QR', 'QR', 'solve' or lstsq."
            )
        if relative_jitter:
            if self.solver == "RKHS" or self.solver == "RKHS-QR":
                self._jitter_scale = self._vk[0]
            elif (
                self.solver == "QR" or self.solver == "solve" or self.solver == "lstsq"
            ):
                self._jitter_scale = self._KMM_maxeva
        else:
            self._jitter_scale = 1.0
        self.set_regularizers(regularizer, jitter)

    def set_regularizers(self, regularizer=1.0, jitter=0.0):
        self.regularizer = regularizer
        self.jitter = jitter
        if self.solver == "RKHS" or self.solver == "RKHS-QR":
            self._nM = len(np.where(self._vk > self.jitter * self._jitter_scale)[0])
            self._PKPhi = self._Uk[:, : self._nM] * 1 / np.sqrt(self._vk[: self._nM])
        elif self.solver == "QR":
            self._VMM = scipy.linalg.cholesky(
                self.regularizer * self.KMM
                + np.eye(self._nM) * self._jitter_scale * self.jitter
            )
        self._Cov = np.zeros((self._nM, self._nM))
        self._KY = None

    def fit(self, KNM, Y, rcond=None):
        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]
        if self.solver == "RKHS":
            Phi = KNM @ self._PKPhi
            self._weights = self._PKPhi @ scipy.linalg.solve(
                Phi.T @ Phi + np.eye(self._nM) * self.regularizer,
                Phi.T @ Y,
                assume_a="pos",
            )
        elif self.solver == "RKHS-QR":
            A = np.vstack(
                [KNM @ self._PKPhi, np.sqrt(self.regularizer) * np.eye(self._nM)]
            )
            Q, R = np.linalg.qr(A)
            self._weights = self._PKPhi @ scipy.linalg.solve_triangular(
                R, Q.T @ np.vstack([Y, np.zeros((self._nM, Y.shape[1]))])
            )
        elif self.solver == "QR":
            A = np.vstack([KNM, self._VMM])
            Q, R = np.linalg.qr(A)
            self._weights = scipy.linalg.solve_triangular(
                R, Q.T @ np.vstack([Y, np.zeros((KNM.shape[1], Y.shape[1]))])
            )
        elif self.solver == "solve":
            self._weights = scipy.linalg.solve(
                KNM.T @ KNM
                + self.regularizer * self.KMM
                + np.eye(self._nM) * self.jitter * self._jitter_scale,
                KNM.T @ Y,
                assume_a="pos",
            )
        elif self.solver == "lstsq":
            self._weights = np.linalg.lstsq(
                KNM.T @ KNM
                + self.regularizer * self.KMM
                + np.eye(self._nM) * self.jitter * self._jitter_scale,
                KNM.T @ Y,
                rcond=rcond,
            )[0]
        else:
            ValueError("solver not implemented")

    def partial_fit(self, KNM, Y, accumulate_only=False, rcond=None):
        if len(Y) > 0:
            # only accumulate if we are passing data
            if len(Y.shape) == 1:
                Y = Y[:, np.newaxis]
            if self.solver == "RKHS":
                Phi = KNM @ self._PKPhi
            elif self.solver == "solve" or self.solver == "lstsq":
                Phi = KNM
            else:
                raise ValueError(
                    "Partial fit can only be realized with solver = 'RKHS' or 'solve'"
                )
            if self._KY is None:
                self._KY = np.zeros((self._nM, Y.shape[1]))

            self._Cov += Phi.T @ Phi
            self._KY += Phi.T @ Y

        # do actual fit if called with empty array or if asked
        if len(Y) == 0 or (not accumulate_only):
            if self.solver == "RKHS":
                self._weights = self._PKPhi @ scipy.linalg.solve(
                    self._Cov + np.eye(self._nM) * self.regularizer,
                    self._KY,
                    assume_a="pos",
                )
            elif self.solver == "solve":
                self._weights = scipy.linalg.solve(
                    self._Cov
                    + self.regularizer * self.KMM
                    + np.eye(self.KMM.shape[0]) * self.jitter * self._jitter_scale,
                    self._KY,
                    assume_a="pos",
                )
            elif self.solver == "lstsq":
                self._weights = np.linalg.lstsq(
                    self._Cov
                    + self.regularizer * self.KMM
                    + np.eye(self.KMM.shape[0]) * self.jitter * self._jitter_scale,
                    self._KY,
                    rcond=rcond,
                )[0]

    @property
    def weights(self):
        return self._weights

    def predict(self, KTM):
        return KTM @ self._weights


class AggregateKernel(torch.nn.Module):
    """
    A kernel that aggregates values in a kernel over :param aggregate_names: using
    a aggregaten function given by :param aggregate_type:

    :param aggregate_names:

    :param aggregate_type:
    """

    def __init__(
        self,
        aggregate_names: Union[str, List[str]],
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
    ):
        super().__init__()
        val_aggregate_types = ["sum", "mean"]
        if aggregate_type not in val_aggregate_types:
            raise ValueError(
                f"Given aggregate_type {aggregate_type!r} but only "
                f"{aggregate_type!r} are supported."
            )
        if structurewise_aggregate:
            raise NotImplementedError(
                "structurewise aggregation has not been implemented."
            )

        self._aggregate_names = aggregate_names
        self._aggregate_type = aggregate_type
        self._structurewise_aggregate = structurewise_aggregate

    def aggregate_features(self, tensor: TensorMap) -> TensorMap:
        if self._aggregate_type == "sum":
            return metatensor.sum_over_samples(
                tensor, sample_names=self._aggregate_names
            )
        elif self._aggregate_type == "mean":
            return metatensor.mean_over_samples(
                tensor, sample_names=self._aggregate_names
            )
        else:
            raise NotImplementedError(
                f"aggregate_type {self._aggregate_type!r} has not been implemented."
            )

    def aggregate_kernel(
        self, kernel: TensorMap, are_pseudo_points: Tuple[bool, bool] = (False, False)
    ) -> TensorMap:
        if self._aggregate_type == "sum":
            if not are_pseudo_points[0]:
                kernel = metatensor.sum_over_samples(kernel, self._aggregate_names)
            if not are_pseudo_points[1]:
                raise NotImplementedError(
                    "properties dimenson cannot be aggregated for the moment"
                )
            return kernel
        elif self._aggregate_type == "mean":
            if not are_pseudo_points[0]:
                kernel = metatensor.mean_over_samples(kernel, self._aggregate_names)
            if not are_pseudo_points[1]:
                raise NotImplementedError(
                    "properties dimenson cannot be aggregated for the moment"
                )
            return kernel
        else:
            raise NotImplementedError(
                f"aggregate_type {self._aggregate_type!r} has not been implemented."
            )

    def forward(
        self,
        tensor1: TensorMap,
        tensor2: TensorMap,
        are_pseudo_points: Tuple[bool, bool] = (False, False),
    ) -> TensorMap:
        return self.aggregate_kernel(
            self.compute_kernel(tensor1, tensor2), are_pseudo_points
        )

    def compute_kernel(self, tensor1: TensorMap, tensor2: TensorMap) -> TensorMap:
        raise NotImplementedError("compute_kernel needs to be implemented.")


class AggregateLinear(AggregateKernel):
    def __init__(
        self,
        aggregate_names: Union[str, List[str]],
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
    ):
        super().__init__(aggregate_names, aggregate_type, structurewise_aggregate)

    def forward(
        self,
        tensor1: TensorMap,
        tensor2: TensorMap,
        are_pseudo_points: Tuple[bool, bool] = (False, False),
    ) -> TensorMap:
        # we overwrite default behavior because for linear kernels we can do it more
        # memory efficient
        if not are_pseudo_points[0]:
            tensor1 = self.aggregate_features(tensor1)
        if not are_pseudo_points[1]:
            tensor2 = self.aggregate_features(tensor2)
        return self.compute_kernel(tensor1, tensor2)

    def compute_kernel(self, tensor1: TensorMap, tensor2: TensorMap) -> TensorMap:
        return metatensor.dot(tensor1, tensor2)


class AggregatePolynomial(AggregateKernel):
    def __init__(
        self,
        aggregate_names: Union[str, List[str]],
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
        degree: int = 2,
    ):
        super().__init__(aggregate_names, aggregate_type, structurewise_aggregate)
        self._degree = degree

    def compute_kernel(self, tensor1: TensorMap, tensor2: TensorMap):
        return metatensor.pow(metatensor.dot(tensor1, tensor2), self._degree)


class TorchAggregateKernel(torch.nn.Module):
    """
    A kernel that aggregates values in a kernel over :param aggregate_names: using
    a aggregaten function given by :param aggregate_type:

    :param aggregate_names:

    :param aggregate_type:
    """

    def __init__(
        self,
        aggregate_names: Union[str, List[str]],
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
    ):
        super().__init__()
        val_aggregate_types = ["sum", "mean"]
        if aggregate_type not in val_aggregate_types:
            raise ValueError(
                f"Given aggregate_type {aggregate_type} but only "
                f"{aggregate_type} are supported."
            )
        if structurewise_aggregate:
            raise NotImplementedError(
                "structurewise aggregation has not been implemented."
            )

        self._aggregate_names = aggregate_names
        self._aggregate_type = aggregate_type
        self._structurewise_aggregate = structurewise_aggregate

    def aggregate_features(self, tensor: TorchTensorMap) -> TorchTensorMap:
        if self._aggregate_type == "sum":
            return metatensor.torch.sum_over_samples(
                tensor, sample_names=self._aggregate_names
            )
        elif self._aggregate_type == "mean":
            return metatensor.torch.mean_over_samples(
                tensor, sample_names=self._aggregate_names
            )
        else:
            raise NotImplementedError(
                f"aggregate_type {self._aggregate_type} has not been implemented."
            )

    def aggregate_kernel(
        self,
        kernel: TorchTensorMap,
        are_pseudo_points: Tuple[bool, bool] = (False, False),
    ) -> TorchTensorMap:
        if self._aggregate_type == "sum":
            if not are_pseudo_points[0]:
                kernel = metatensor.torch.sum_over_samples(
                    kernel, self._aggregate_names
                )
            if not are_pseudo_points[1]:
                raise NotImplementedError(
                    "properties dimenson cannot be aggregated for the moment"
                )
            return kernel
        elif self._aggregate_type == "mean":
            if not are_pseudo_points[0]:
                kernel = metatensor.torch.mean_over_samples(
                    kernel, self._aggregate_names
                )
            if not are_pseudo_points[1]:
                raise NotImplementedError(
                    "properties dimenson cannot be aggregated for the moment"
                )
            return kernel
        else:
            raise NotImplementedError(
                f"aggregate_type {self._aggregate_type} has not been implemented."
            )

    def forward(
        self,
        tensor1: TorchTensorMap,
        tensor2: TorchTensorMap,
        are_pseudo_points: Tuple[bool, bool] = (False, False),
    ) -> TorchTensorMap:
        return self.aggregate_kernel(
            self.compute_kernel(tensor1, tensor2), are_pseudo_points
        )

    def compute_kernel(
        self, tensor1: TorchTensorMap, tensor2: TorchTensorMap
    ) -> TorchTensorMap:
        raise NotImplementedError("compute_kernel needs to be implemented.")


class TorchAggregateLinear(TorchAggregateKernel):
    def __init__(
        self,
        aggregate_names: Union[str, List[str]],
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
    ):
        super().__init__(aggregate_names, aggregate_type, structurewise_aggregate)

    def forward(
        self,
        tensor1: TorchTensorMap,
        tensor2: TorchTensorMap,
        are_pseudo_points: Tuple[bool, bool] = (False, False),
    ) -> TorchTensorMap:
        # we overwrite default behavior because for linear kernels we can do it more
        # memory efficient
        if not are_pseudo_points[0]:
            tensor1 = self.aggregate_features(tensor1)
        if not are_pseudo_points[1]:
            tensor2 = self.aggregate_features(tensor2)
        return self.compute_kernel(tensor1, tensor2)

    def compute_kernel(
        self, tensor1: TorchTensorMap, tensor2: TorchTensorMap
    ) -> TorchTensorMap:
        return metatensor.torch.dot(tensor1, tensor2)


class TorchAggregatePolynomial(TorchAggregateKernel):
    def __init__(
        self,
        aggregate_names: Union[str, List[str]],
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
        degree: int = 2,
    ):
        super().__init__(aggregate_names, aggregate_type, structurewise_aggregate)
        self._degree = degree

    def compute_kernel(self, tensor1: TorchTensorMap, tensor2: TorchTensorMap):
        return metatensor.torch.pow(
            metatensor.torch.dot(tensor1, tensor2), self._degree
        )


class GreedySelector:
    """Wraps :py:class:`skmatter._selection.GreedySelector` for a TensorMap.

    The class creates a selector for each block. The selection will be done based
    the values of each :py:class:`TensorBlock`. Gradients will not be considered for
    the selection.
    """

    def __init__(
        self,
        selector_class: Type[skmatter._selection.GreedySelector],
        selection_type: str,
        **selector_arguments,
    ) -> None:
        self._selector_class = selector_class
        self._selection_type = selection_type
        self._selector_arguments = selector_arguments

        self._selector_arguments["selection_type"] = self._selection_type
        self._support = None

    @property
    def selector_class(self) -> Type[skmatter._selection.GreedySelector]:
        """The class to perform the selection."""
        return self._selector_class

    @property
    def selection_type(self) -> str:
        """Whether to choose a subset of columns ('feature') or rows ('sample')."""
        return self._selection_type

    @property
    def selector_arguments(self) -> dict:
        """Arguments passed to the ``selector_class``."""
        return self._selector_arguments

    @property
    def support(self) -> TensorMap:
        """TensorMap containing the support."""
        if self._support is None:
            raise ValueError("No selections. Call fit method first.")

        return self._support

    def fit(self, X: TensorMap, warm_start: bool = False):  # -> GreedySelector:
        """Learn the features to select.

        :param X:
            Training vectors.
        :param warm_start:
            Whether the fit should continue after having already run, after increasing
            `n_to_select`. Assumes it is called with the same X.
        """
        if len(X.component_names) != 0:
            raise ValueError("Only blocks with no components are supported.")

        blocks = []
        for _, block in X.items():
            selector = self.selector_class(**self.selector_arguments)
            selector.fit(block.values, warm_start=warm_start)
            mask = selector.get_support()

            if self._selection_type == "feature":
                samples = Labels.single()
                properties = Labels(
                    names=block.properties.names, values=block.properties.values[mask]
                )
            elif self._selection_type == "sample":
                samples = Labels(
                    names=block.samples.names, values=block.samples.values[mask]
                )
                properties = Labels.single()

            blocks.append(
                TensorBlock(
                    values=np.zeros([len(samples), len(properties)], dtype=np.int32),
                    samples=samples,
                    components=[],
                    properties=properties,
                )
            )

        self._support = TensorMap(X.keys, blocks)

        return self

    def transform(self, X: TensorMap) -> TensorMap:
        """Reduce X to the selected features.

        :param X:
            The input tensor.
        :returns:
            The selected subset of the input.
        """
        blocks = []
        for key, block in X.items():
            block_support = self.support.block(key)

            if self._selection_type == "feature":
                new_block = metatensor.slice_block(
                    block, "properties", block_support.properties
                )
            elif self._selection_type == "sample":
                new_block = metatensor.slice_block(
                    block, "samples", block_support.samples
                )
            blocks.append(new_block)

        return TensorMap(X.keys, blocks)

    def fit_transform(self, X: TensorMap, warm_start: bool = False) -> TensorMap:
        """Fit to data, then transform it.

        :param X:
            Training vectors.
        :param warm_start:
            Whether the fit should continue after having already run, after increasing
            `n_to_select`. Assumes it is called with the same X.
        """
        return self.fit(X, warm_start=warm_start).transform(X)


class FPS(GreedySelector):
    """
    Transformer that performs Greedy Sample Selection using Farthest Point Sampling.

    Refer to :py:class:`skmatter.sample_selection.FPS` for full documentation.
    """

    def __init__(
        self,
        initialize=0,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selector_class=_FPS,
            selection_type="sample",
            initialize=initialize,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )
        self._n_to_select = n_to_select


def torch_tensor_map_to_core(torch_tensor: TorchTensorMap):
    torch_blocks = []
    for _, torch_block in torch_tensor.items():
        torch_blocks.append(torch_tensor_block_to_core(torch_block))
    torch_keys = torch_labels_to_core(torch_tensor.keys)
    return TensorMap(torch_keys, torch_blocks)


def torch_tensor_block_to_core(torch_block: TorchTensorBlock):
    """Transforms a tensor block from metatensor-torch to metatensor-torch
    :param torch_block:
        tensor block from metatensor-torch
    :returns torch_block:
        tensor block from metatensor-torch
    """
    block = TensorBlock(
        values=torch_block.values.detach().cpu().numpy(),
        samples=torch_labels_to_core(torch_block.samples),
        components=[
            torch_labels_to_core(component) for component in torch_block.components
        ],
        properties=torch_labels_to_core(torch_block.properties),
    )
    for parameter, gradient in torch_block.gradients():
        block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient.values.detach().cpu().numpy(),
                samples=torch_labels_to_core(gradient.samples),
                components=[
                    torch_labels_to_core(component) for component in gradient.components
                ],
                properties=torch_labels_to_core(gradient.properties),
            ),
        )
    return block


def torch_labels_to_core(torch_labels: TorchLabels):
    """Transforms labels from metatensor-torch to metatensor-torch
    :param torch_block:
        tensor block from metatensor-torch
    :returns torch_block:
        labels from metatensor-torch
    """
    return Labels(torch_labels.names, torch_labels.values.detach().cpu().numpy())


###


def core_tensor_map_to_torch(core_tensor: TensorMap):
    """Transforms a tensor map from metatensor-core to metatensor-torch
    :param core_tensor:
        tensor map from metatensor-core
    :returns torch_tensor:
        tensor map from metatensor-torch
    """

    torch_blocks = []
    for _, core_block in core_tensor.items():
        torch_blocks.append(core_tensor_block_to_torch(core_block))
    torch_keys = core_labels_to_torch(core_tensor.keys)
    return TorchTensorMap(torch_keys, torch_blocks)


def core_tensor_block_to_torch(core_block: TensorBlock):
    """Transforms a tensor block from metatensor-core to metatensor-torch
    :param core_block:
        tensor block from metatensor-core
    :returns torch_block:
        tensor block from metatensor-torch
    """
    block = TorchTensorBlock(
        values=torch.tensor(core_block.values),
        samples=core_labels_to_torch(core_block.samples),
        components=[
            core_labels_to_torch(component) for component in core_block.components
        ],
        properties=core_labels_to_torch(core_block.properties),
    )
    for parameter, gradient in core_block.gradients():
        block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient.values.detach().cpu().numpy(),
                samples=torch_labels_to_core(gradient.samples),
                components=[
                    torch_labels_to_core(component) for component in gradient.components
                ],
                properties=torch_labels_to_core(gradient.properties),
            ),
        )
    return block


def core_labels_to_torch(core_labels: Labels):
    """Transforms labels from metatensor-core to metatensor-torch
    :param core_block:
        tensor block from metatensor-core
    :returns torch_block:
        labels from metatensor-torch
    """
    return TorchLabels(core_labels.names, torch.tensor(core_labels.values))


class SubsetOfRegressors:
    def __init__(
        self,
        kernel_type: Union[str, AggregateKernel] = "linear",
        kernel_kwargs: Optional[dict] = None,
    ):
        if kernel_kwargs is None:
            kernel_kwargs = {}
        self._set_kernel(kernel_type, **kernel_kwargs)
        self._kernel_type = kernel_type
        self._kernel_kwargs = kernel_kwargs
        self._X_pseudo = None
        self._weights = None

    def _set_kernel(self, kernel: Union[str, AggregateKernel], **kernel_kwargs):
        val_kernels = ["linear", "polynomial", "precomputed"]
        aggregate_type = kernel_kwargs.get("aggregate_type", "sum")
        if aggregate_type != "sum":
            raise ValueError(
                f'aggregate_type={aggregate_type!r} found but must be "sum"'
            )
        self._kernel: Union[AggregateKernel, None] = None
        if kernel == "linear":
            self._kernel = AggregateLinear(aggregate_type="sum", **kernel_kwargs)
        elif kernel == "polynomial":
            self._kernel = AggregatePolynomial(aggregate_type="sum", **kernel_kwargs)
        elif kernel == "precomputed":
            self._kernel = None
        elif isinstance(kernel, AggregateKernel):
            self._kernel = kernel
        else:
            raise ValueError(
                f"kernel type {kernel!r} is not supported. Please use one "
                f"of the valid kernels {val_kernels!r}"
            )

    def fit(
        self,
        X: TensorMap,
        X_pseudo: TensorMap,
        y: TensorMap,
        alpha: float = 1.0,
        alpha_forces: Optional[float] = None,
        solver: str = "RKHS-QR",
        rcond: Optional[float] = None,
    ):
        r"""
        :param X:
            features
            if kernel type "precomputed" is used, the kernel k_nm is assumed
        :param X_pseudo:
            pseudo points
            if kernel type "precomputed" is used, the kernel k_mm is assumed
        :param y:
            targets
        :param alpha:
            regularizationfor the energies, it must be a float
        :param alpha_forces:
            regularization for the forces, it must be a float. If None is set
            equal to alpha
        :param solver:
            determines which solver to use
        :param rcond:
            argument for the solver lstsq


        Derivation
        ----------

        We take equation the mean expression

        .. math::

            \sigma^{-2} K_{tm}\Sigma K_{MN}y

        we put in the $\Sigma$

        .. math::

            \sigma^{-2} K_{tm}(\sigma^{-2}K_{mn}K_{mn}+K_{mm})^{-1} K_{mn}y

        We can move around the $\sigma's$

        .. math::

             K_{tm}((K_{mn}\sigma^{-1})(\sigma^{-1}K_{mn)}+K_{mm})^{-1}
                            (K_{mn}\sigma^{-1})(y\sigma^{-1})

        you can see the building blocks in the code are $K_{mn}\sigma^{-1}$ and
        $y\sigma^{-1}$
        """
        if isinstance(alpha, float):
            alpha_energy = alpha
        else:
            raise ValueError("alpha must either be a float")

        if alpha_forces is None:
            alpha_forces = alpha_energy
        else:
            if not isinstance(alpha_forces, float):
                raise ValueError("alpha must either be a float")

        X = X.to(arrays="numpy")
        X_pseudo = X_pseudo.to(arrays="numpy")
        y = y.to(arrays="numpy")

        if self._kernel is None:
            # _set_kernel only returns None if kernel type is precomputed
            k_nm = X
            k_mm = X_pseudo
        else:
            k_mm = self._kernel(X_pseudo, X_pseudo, are_pseudo_points=(True, True))
            k_nm = self._kernel(X, X_pseudo, are_pseudo_points=(False, True))

        # solve
        # TODO: allow for different regularizer for energies and forces
        weight_blocks = []
        for key, y_block in y.items():
            k_nm_block = k_nm.block(key)
            k_mm_block = k_mm.block(key)
            X_block = X.block(key)
            structures = metatensor.operations._dispatch.unique(
                k_nm_block.samples["system"]
            )
            n_atoms_per_structure = []
            for structure in structures:
                n_atoms = np.sum(X_block.samples["system"] == structure)
                n_atoms_per_structure.append(float(n_atoms))

            n_atoms_per_structure = np.array(n_atoms_per_structure)
            normalization = metatensor.operations._dispatch.sqrt(n_atoms_per_structure)

            if not (np.allclose(alpha_energy, 0.0)):
                normalization /= alpha_energy
            normalization = normalization[:, None]

            k_nm_reg = k_nm_block.values * normalization
            y_reg = (y_block.values) * normalization
            if len(k_nm_block.gradients_list()) > 0:
                grad_shape = k_nm_block.gradient("positions").values.shape
                k_nm_reg = np.vstack(
                    [
                        k_nm_reg,
                        k_nm_block.gradient("positions").values.reshape(
                            grad_shape[0] * grad_shape[1],
                            grad_shape[2],
                        )
                        / alpha_forces,
                    ]
                )
                grad_shape = y_block.gradient("positions").values.shape
                y_reg = np.vstack(
                    [
                        y_reg,
                        y_block.gradient("positions").values.reshape(
                            grad_shape[0] * grad_shape[1],
                            grad_shape[2],
                        )
                        / alpha_forces,
                    ]
                )
            self._solver = _SorKernelSolver(
                k_mm_block.values, regularizer=1, jitter=0, solver=solver
            )

            if rcond is None:
                rcond_ = max(k_nm_reg.shape) * np.finfo(k_nm_reg.dtype.char.lower()).eps
            else:
                rcond_ = rcond
            self._solver.fit(k_nm_reg, y_reg, rcond=rcond_)

            weight_block = TensorBlock(
                values=self._solver.weights.T,
                samples=y_block.properties,
                components=k_nm_block.components,
                properties=k_nm_block.properties,
            )
            weight_blocks.append(weight_block)

        self._weights = TensorMap(y.keys, weight_blocks)

        self._X_pseudo = X_pseudo.copy()

    def predict(self, T: TensorMap) -> TensorMap:
        """
        :param T:
            features
            if kernel type "precomputed" is used, the kernel k_tm is assumed
        """
        if self._weights is None:
            raise ValueError(
                "The weights are not defined. Are you sure you"
                " have run the `fit` method?"
            )
        if self._kernel_type == "precomputed":
            k_tm = T
        else:
            k_tm = self._kernel(T, self._X_pseudo, are_pseudo_points=(False, True))
        return metatensor.dot(k_tm, self._weights)

    def export_torch_script_model(self):
        return TorchSubsetofRegressors(
            core_tensor_map_to_torch(self._weights),
            core_tensor_map_to_torch(self._X_pseudo),
            self._kernel_type,
            self._kernel_kwargs,
        )


class TorchSubsetofRegressors(torch.nn.Module):
    def __init__(
        self,
        weights: TorchTensorMap,
        X_pseudo: TorchTensorMap,
        kernel_type: Union[str, AggregateKernel] = "linear",
        kernel_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self._weights = weights
        self._X_pseudo = X_pseudo
        if kernel_kwargs is None:
            kernel_kwargs = {}
        self._set_kernel(kernel_type, **kernel_kwargs)

    def forward(self, T: TorchTensorMap) -> TorchTensorMap:
        """
        :param T:
            features
            if kernel type "precomputed" is used, the kernel k_tm is assumed
        """
        # move weights and X_pseudo to the same device as T
        self._weights = self._weights.to(T.device)
        self._X_pseudo = self._X_pseudo.to(T.device)

        k_tm = self._kernel(T, self._X_pseudo, are_pseudo_points=(False, True))
        return metatensor.torch.dot(k_tm, self._weights)

    def _set_kernel(self, kernel: Union[str, TorchAggregateKernel], **kernel_kwargs):
        val_kernels = ["linear", "polynomial", "precomputed"]
        aggregate_type = kernel_kwargs.get("aggregate_type", "sum")
        if aggregate_type != "sum":
            raise ValueError(
                f'aggregate_type={aggregate_type!r} found but must be "sum"'
            )
        if kernel == "linear":
            self._kernel = TorchAggregateLinear(aggregate_type="sum", **kernel_kwargs)
        elif kernel == "polynomial":
            self._kernel = TorchAggregatePolynomial(
                aggregate_type="sum", **kernel_kwargs
            )
        elif kernel == "precomputed":
            raise NotImplementedError(
                "precomputed kernels are note supported in torch"
                "version of subset of regressor."
            )
        elif isinstance(kernel, TorchAggregateKernel):
            self._kernel = kernel
        else:
            raise ValueError(
                f"kernel type {kernel!r} is not supported. Please use one "
                f"of the valid kernels {val_kernels!r}"
            )
