# postpones evaluation of annotations
# see https://stackoverflow.com/a/33533514
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union

import metatensor.torch
import numpy as np
import rascaline
import rascaline.torch
import scipy
import skmatter
import torch
from metatensor.torch import Labels as TorchLabels
from metatensor.torch import TensorBlock as TorchTensorBlock
from metatensor.torch import TensorMap as TorchTensorMap
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, System
from omegaconf import OmegaConf
from skmatter._selection import _FPS

import metatensor
from metatensor import Labels, TensorBlock, TensorMap

from .. import ARCHITECTURE_CONFIG_PATH


# TODO needed when we apply composition features
# from ..utils.composition import apply_composition_contribution


DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / "sparse_gap.yaml")
)

DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["model"]

ARCHITECTURE_NAME = "sparse_gap"


class Model(torch.nn.Module):
    def __init__(
        self, capabilities: ModelCapabilities, hypers: Dict = DEFAULT_MODEL_HYPERS
    ) -> None:
        super().__init__()
        self.name = ARCHITECTURE_NAME

        if len(capabilities.outputs) > 1:
            # PR COMMENT
            raise NotImplementedError("I don't understand how it is used in SOAP-BPNN")

        # Check capabilities
        for output in capabilities.outputs.values():
            if output.quantity != "energy":
                raise ValueError(
                    "SparseGAP only supports energy-like outputs, "
                    f"but a {output.quantity} was provided"
                )
            if output.per_atom:
                raise ValueError(
                    "SparseGAP only supports per-structure outputs, "
                    "but a per-atom output was provided"
                )

        self.capabilities = capabilities
        self.all_species = capabilities.species
        self.hypers = hypers

        # creates a composition weight tensor that can be directly indexed by species,
        # this can be left as a tensor of zero or set from the outside using
        # set_composition_weights (recommended for better accuracy)
        n_outputs = len(capabilities.outputs)
        self.register_buffer(
            "composition_weights", torch.zeros((n_outputs, max(self.all_species) + 1))
        )
        # buffers cannot be indexed by strings (torchscript), so we create a single
        # tensor for all output. Due to this, we need to slice the tensor when we use
        # it and use the output name to select the correct slice via a dictionary
        self.output_to_index = {
            output_name: i for i, output_name in enumerate(capabilities.outputs.keys())
        }

        self.register_buffer(
            "kernel_weights", torch.zeros((hypers["sparse_points"]["points"]))
        )
        self._soap_torch_calculator = rascaline.torch.SoapPowerSpectrum(
            **hypers["soap"]
        )
        self._soap_calculator = rascaline.SoapPowerSpectrum(**hypers["soap"])

        # TODO is always sum energy kernel, keep in mind for property
        kernel_kwargs = {
            "degree": hypers["krr"]["degree"],
            "aggregate_names": ["center", "species_center"],
        }
        self._subset_of_regressors = SubsetOfRegressors(
            kernel_type=hypers["krr"].get("kernel", "polynomial"),
            kernel_kwargs=kernel_kwargs,
        )

        self._sampler = FPS(n_to_select=hypers["sparse_points"]["points"])
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
            dummy_weights, dummy_X_pseudo
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[TorchLabels] = None,
    ) -> Dict[str, TorchTensorMap]:
        if selected_atoms is not None:
            # change metatensor names to match rascaline
            selected_atoms = selected_atoms.rename("system", "structure")
            selected_atoms = selected_atoms.rename("atom", "center")

        soap_features = self._soap_torch_calculator(
            systems, selected_samples=selected_atoms
        )
        # TODO implement accumulate_key_names so we do not loose sparsity
        soap_features = soap_features.keys_to_samples("species_center")
        soap_features = soap_features.keys_to_properties(
            ["species_neighbor_1", "species_neighbor_2"]
        )
        soap_features = TorchTensorMap(self._keys, soap_features.blocks())
        # TODO we assume only one output for the moment, ask Filippo what he has done in
        #      SOAP-BPNN
        output_key = list(outputs.keys())[0]
        # TODO does not work if different species are present in systems
        #      should work if we implement kernels properly
        return {output_key: self._subset_of_regressors_torch(soap_features)}


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
                    "Partial fit can only be realized with "
                    "solver = 'RKHS' or 'solve'"
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
        aggregate_names: Union[str, List[str]] = "aggregate",
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
    ):
        super().__init__()
        valid_aggregate_types = ["sum", "mean"]
        if aggregate_type not in valid_aggregate_types:
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
                # TODO {sum,mean}_over_properties does not exist
                raise NotImplementedError(
                    "properties dimenson cannot be aggregated for the moment"
                )
                kernel = metatensor.sum_over_properties(kernel, self._aggregate_names)
            return kernel
        elif self._aggregate_type == "mean":
            if not are_pseudo_points[0]:
                kernel = metatensor.mean_over_samples(kernel, self._aggregate_names)
            if not are_pseudo_points[1]:
                # TODO {sum,mean}_over_properties does not exist
                raise NotImplementedError(
                    "properties dimenson cannot be aggregated for the moment"
                )
                kernel = metatensor.mean_over_properties(kernel, self._aggregate_names)
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
        aggregate_names: Union[str, List[str]] = "aggregate",
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
        aggregate_names: Union[str, List[str]] = "aggregate",
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
        degree: int = 2,
    ):
        super().__init__(aggregate_names, aggregate_type, structurewise_aggregate)
        self._degree = 2

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
        aggregate_names: Union[str, List[str]] = "aggregate",
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
    ):
        super().__init__()
        valid_aggregate_types = ["sum", "mean"]
        if aggregate_type not in valid_aggregate_types:
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
                # TODO {sum,mean}_over_properties does not exist
                raise NotImplementedError(
                    "properties dimenson cannot be aggregated for the moment"
                )
                kernel = metatensor.torch.sum_over_properties(
                    kernel, self._aggregate_names
                )
            return kernel
        elif self._aggregate_type == "mean":
            if not are_pseudo_points[0]:
                kernel = metatensor.torch.mean_over_samples(
                    kernel, self._aggregate_names
                )
            if not are_pseudo_points[1]:
                # TODO {sum,mean}_over_properties does not exist
                raise NotImplementedError(
                    "properties dimenson cannot be aggregated for the moment"
                )
                kernel = metatensor.torch.mean_over_properties(
                    kernel, self._aggregate_names
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
        aggregate_names: Union[str, List[str]] = "aggregate",
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
        aggregate_names: Union[str, List[str]] = "aggregate",
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
        degree: int = 2,
    ):
        super().__init__(aggregate_names, aggregate_type, structurewise_aggregate)
        self._degree = 2

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

    def fit(self, X: TensorMap, warm_start: bool = False) -> GreedySelector:
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
    return TensorBlock(
        values=torch_block.values.detach().cpu().numpy(),
        samples=torch_labels_to_core(torch_block.samples),
        components=[
            torch_labels_to_core(component) for component in torch_block.components
        ],
        properties=torch_labels_to_core(torch_block.properties),
    )


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
    return TorchTensorBlock(
        values=torch.tensor(core_block.values),
        samples=core_labels_to_torch(core_block.samples),
        components=[
            core_labels_to_torch(component) for component in core_block.components
        ],
        properties=core_labels_to_torch(core_block.properties),
    )


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
        valid_kernels = ["linear", "polynomial", "precomputed"]
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
                f"of the valid kernels {valid_kernels!r}"
            )

    def fit(
        self,
        X: TensorMap,
        X_pseudo: TensorMap,
        y: TensorMap,
        alpha: Union[float, TensorMap] = 1.0,
        accumulate_key_names: Optional[List[str]] = None,  # TODO implementation
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
        :param kernel_type:
            type of kernel used
        :param kernel_kwargs:
            additional keyword argument for specific kernel
            - **linear** None
            - **polynomial** degree
        :param accumulate_key_names:
            a string or list of strings that specify which key names should be
            accumulate to one kernel. This is intended for key columns inducing sparsity
            in the properties (e.g. neighbour species)
        :param alpha:
            regularization
        :param solver:
            determines which solver to use
        :param rcond:
            argument for the solver lstsq


        Derivation
        ----------

        We take equation (16b) (the mean expression)

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
            alpha_tensor = metatensor.ones_like(y)

            samples = Labels(
                names=y.sample_names,
                values=np.zeros([1, len(y.sample_names)], dtype=int),
            )

            alpha_tensor = metatensor.slice(
                alpha_tensor, axis="samples", labels=samples
            )
            self._alpha = metatensor.multiply(alpha_tensor, alpha)
        elif isinstance(alpha, TensorMap):
            raise NotImplementedError("TensorMaps are not yet supported")
        else:
            raise ValueError("alpha must either be a float or a TensorMap")

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
        weight_blocks = []
        for key, y_block in y.items():
            k_nm_block = k_nm.block(key)
            k_mm_block = k_mm.block(key)
            X_block = X.block(key)
            structures = metatensor.operations._dispatch.unique(
                k_nm_block.samples["structure"]
            )
            n_atoms_per_structure = []
            for structure in structures:
                n_atoms = np.sum(X_block.samples["structure"] == structure)
                n_atoms_per_structure.append(float(n_atoms))

            # PR COMMENT removed delta because would say this is part of the standardizr
            n_atoms_per_structure = np.array(n_atoms_per_structure)
            normalization = metatensor.operations._dispatch.sqrt(n_atoms_per_structure)
            alpha_values = self._alpha.block(key).values
            if not (np.allclose(alpha_values[0, 0], 0.0)):
                normalization /= alpha_values[0, 0]
            normalization = normalization[:, None]

            k_nm_reg = k_nm_block.values * normalization
            y_reg = y_block.values * normalization

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
        k_tm = self._kernel(T, self._X_pseudo, are_pseudo_points=(False, True))
        return metatensor.torch.dot(k_tm, self._weights)

    def _set_kernel(self, kernel: Union[str, TorchAggregateKernel], **kernel_kwargs):
        valid_kernels = ["linear", "polynomial", "precomputed"]
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
                f"of the valid kernels {valid_kernels!r}"
            )
