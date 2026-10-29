import copy
import os
import random
from typing import Literal, Optional

import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatomic.torch import ModelOutput, System, systems_to_torch
from metatomic.torch.o3 import random_transformations

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.readers import (
    read_systems,
)
from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.readers.metatensor import _check_tensor_map_metadata
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from .architectures import ArchitectureTests


def set_random_seed(seed: int) -> None:
    """
    Set the random seed of everything for reproducibility.

    :param seed: The random seed to set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class OutputTests(ArchitectureTests):
    """Test suite to check that the model can produce different types of outputs.

    If a model does not support a given type of output, set the corresponding
    ``supports_*_outputs`` attribute to ``False`` to skip the corresponding tests.
    By default, they are all set to ``True`` to avoid supported outputs from
    being untested accidentally.
    """

    supports_scalar_outputs: bool = True
    """Whether the model supports scalar outputs."""
    supports_multiscalar_outputs: bool = True
    """Whether the model supports outputs with multiple scalar subtargets."""
    supports_vector_outputs: bool = True
    """Whether the model supports vector outputs."""
    supports_spherical_outputs: bool = True
    """Whether the model supports spherical tensor outputs."""
    supports_spherical_rank2_outputs: bool = True
    """Whether the model supports spherical tensor outputs of rank 2."""
    supports_spherical_atomic_basis_outputs: bool = True
    """Whether the model supports spherical atomic basis outputs."""
    supports_selected_atoms: bool = True
    """Whether the model supports the ``selected_atoms`` argument in the
    ``forward()`` method.
    """
    supports_features: bool = True
    """Whether the model supports returning features."""
    supports_last_layer_features: bool | dict[str, bool | None] = {
        "scalar": None,
        "multiscalar": None,
        "vector": None,
        "spherical": None,
        "spherical_rank2": None,
        "spherical_atomic_basis": None,
    }
    """Whether the model supports returning last-layer features for each
    target type. If None, or the target is not present in the dictionary,
    it will be assumed to be ``True`` when the target type is supported
    and ``False`` otherwise."""
    last_layer_features_are_affine: bool = True
    """Whether each target is an affine function of its last-layer features."""
    is_equivariant_rotations: bool = True
    """Whether the model is equivariant (i.e. produces outputs that
    transform correctly under rotations by architecture's design)."""
    is_equivariant_reflections: bool = True
    """Whether the model is equivariant (i.e. produces outputs that
    transform correctly under reflections by architecture's design)."""
    equivariance_error_tolerance: float = 1e-5
    """Tolerance for equivariance tests."""

    monomer_equal_dimer: bool = False
    """Whether the model is expected to produce the same output for
    a dimer and two monomers at a large distance."""

    atoms_per_system: int = 3
    """Number of atoms per system to use in the tests."""
    batch_size: int = 40
    """Batch size to use in the tests."""

    @pytest.fixture
    def n_features(self) -> Optional[int | list[int]]:
        """Fixture that returns the number of features produced by the model.

        By default this is set to ``None``, which skips checking the number
        of features in the output. Override this fixture for your architecture
        if you want the test suite to check that the number of features in the
        output is correct.

        :return: The number of features produced by the model.
        """
        return None

    @pytest.fixture
    def n_last_layer_features(self) -> Optional[int]:
        """Fixture that returns the number of last-layer features produced
        by the model.

        By default this is set to ``None``, which skips checking the number
        of last-layer features in the output. Override this fixture for your
        architecture if you want the test suite to check that the number of
        last-layer features in the output is correct.

        THIS FIXTURE IS CURRENTLY NOT USED, we used to check this in the past
        because we only had last layer features for scalar outputs so it was
        easy to check. We will reintroduce checks for the number of last
        layer features for all outputs in the future.

        :return: The number of last-layer features produced by the model.
        """
        return None

    @pytest.fixture
    def single_atom_energy(self) -> Optional[float]:
        """Fixture that returns the single atom energy value.

        By default this is set to ``None``, which skips checking the single
        atom energy value in the output. Override this fixture for your
        architecture if you want the test suite to check that the single atom
        energy value in the output is correct.

        :return: The single atom energy value.
        """
        return None

    def _get_output(
        self,
        model_hypers: dict,
        dataset_info: DatasetInfo,
        sample_kind: Literal["atom", "system"],
        outputs: Optional[list[str]] = None,
        last_layer_features: bool = False,
        check_metadata: bool = True,
    ) -> dict[str, mts.TensorMap]:
        """Helper function to get the model output for different types of outputs.

        It initializes the model and runs a forward pass with a simple system.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        :param sample_kind: The sample kind of the requested outputs.
        :param outputs: List of output names to request. If ``None``, all outputs
            defined in the model are requested.
        :param last_layer_features: Whether to request and check last-layer
            features for the requested outputs.
        :param check_metadata: Whether to check that the metadata of the returned
            outputs matches that of the targets in the dataset info.

        :return: The model outputs.
        """
        set_random_seed(0)

        model = self.model_cls(model_hypers, dataset_info)
        supported_outs = model.supported_outputs()

        # Check the layout of the last layer features.
        if last_layer_features:
            for out_name, target_info in supported_outs.items():
                if out_name.endswith("_last_layer_features"):
                    target_name = out_name.replace("mtt::aux::", "").replace(
                        "_last_layer_features", ""
                    )
                    for block in target_info.layout.blocks():
                        assert block.properties.names == ["feature"], (
                            f"Last layer features for output '{target_name}' should "
                            f"have only one property named 'feature', but got "
                            f"{block.properties.names}."
                        )

        # Check that the layouts of the supported outputs match the
        # dataset info targets.
        if check_metadata:
            for target_name, target_info in dataset_info.targets.items():
                assert target_name in supported_outs, (
                    f"Target '{target_name}' is not present in the model's "
                    "supported outputs."
                )
                _check_tensor_map_metadata(
                    supported_outs[target_name].layout, target_info.layout
                )

        # Build distinct systems with the requested number of atoms, by shrinking
        # and jittering a chain of atoms (needed to check the last-layer features)
        positions = torch.zeros(self.atoms_per_system, 3)
        Z = ([1, 6, 7, 8] * (self.atoms_per_system // 4 + 1))[: self.atoms_per_system]
        positions[:, -1] = torch.linspace(0, 8, positions.shape[0])
        generator = torch.Generator().manual_seed(0)
        systems = [
            get_system_with_neighbor_lists(
                System(
                    types=torch.tensor(Z),
                    positions=positions
                    * (0.25 + 0.25 * torch.rand(1, generator=generator))
                    + 0.1 * torch.rand(positions.shape, generator=generator),
                    cell=torch.zeros(3, 3),
                    pbc=torch.tensor([False, False, False]),
                ),
                model.requested_neighbor_lists(),
            )
            for _ in range(self.batch_size)
        ]

        # Outputs that we will request to the model
        if outputs is None:
            outputs = list(model.outputs.keys())
        if last_layer_features:
            outputs += [
                f"mtt::aux::{target_name}_last_layer_features"
                for target_name in outputs
            ]

        # Evaluate the model.
        model = model.to(systems[0].positions.dtype)
        model.eval()
        out = model(
            systems,
            {k: ModelOutput(sample_kind=sample_kind) for k in outputs},
        )

        # Check that all requested outputs are present.
        for key in outputs:
            assert key in out, (
                f"Output '{key}' was requested, but the model did not return it."
            )

        # Check that the outputs have the declared layout.
        if check_metadata:
            for k, v in out.items():
                try:
                    _check_tensor_map_metadata(v, supported_outs[k].layout)
                except Exception as e:
                    if (
                        isinstance(e, ValueError)
                        and "['system', 'atom']" in str(e)
                        and sample_kind == "system"
                    ):
                        # Sometimes supported outputs are declared with sample kind
                        # "atom", but the model can also produce them with sample
                        # kind "system". For now we accept this.
                        pass
                    else:
                        raise ValueError(
                            f"Output '{k}' does not match the expected layout: \n\n{e}"
                        ) from e

                expected_samples = None
                if sample_kind == "atom":
                    expected_samples = self.atoms_per_system * self.batch_size
                    if "atom_type" in v.keys.names or "first_atom_type" in v.keys.names:
                        # Blocks have different number of samples depending on the
                        # atom type so avoid checking the number of samples in
                        # this case.
                        expected_samples = None
                else:
                    expected_samples = self.batch_size

                for block_key, block in v.items():
                    # Check that samples are correct
                    if expected_samples is not None:
                        assert block.samples.values.shape[0] == expected_samples

                    # Check that properties are correct
                    expected_props = (
                        supported_outs[k].layout.block(block_key).properties
                    )
                    assert block.properties == expected_props

        if last_layer_features and self.last_layer_features_are_affine:
            for k, v in out.items():
                if k.endswith("_last_layer_features"):
                    out_name = k.replace("mtt::aux::", "").replace(
                        "_last_layer_features", ""
                    )
                    self._check_llfs_affine(out[out_name], v, out_name)

        return out

    def test_no_output(
        self,
        model_hypers: dict,
        dataset_info: DatasetInfo,
        sample_kind: Literal["atom", "system"],
    ) -> None:
        """Tests that forward pass works for no output.

        It also tests that there are no outputs returned.

        If this test is failing, it means that your model always produces (or wants to
        produce) outputs, even if none are requested.
        If this is the expected behavior for your model, we need to introduce
        a variable to skip this test, contact the ``metatrain`` developers.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information.
        :param sample_kind: The kind of sample for the requested outputs.
        """
        outputs = self._get_output(model_hypers, dataset_info, sample_kind, [])

        assert len(outputs) == 0

    def test_output_scalar(
        self,
        model_hypers: dict,
        dataset_info_scalar: DatasetInfo,
        num_subtargets: int,
        sample_kind: Literal["atom", "system"],
    ) -> None:
        """Tests that forward pass works for scalar outputs.

        It also tests that the returned outputs have the expected samples
        and values shape.

        This test is skipped if the model does not support scalar outputs,
        i.e., if ``supports_scalar_outputs`` is set to ``False``. The test
        is run twice, once for single scalar outputs, and once for a scalar
        output with multiple subtargets. If the model does not support
        multiple scalar subtargets, set ``supports_multiscalar_outputs``
        to ``False``, which will skip the test for multiple subtargets.

        If this test is failing, your model might:

        - not be producing scalar outputs when requested.
        - not be taking into account correctly the ``sample_kind`` field of the
          outputs passed to the ``outputs`` argument of the ``forward()`` method.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_scalar: Dataset information with scalar outputs.
        :param num_subtargets: The number of scalars that the target contains.
        :param sample_kind: The sample kind of the requested scalar output.
        """
        if not self.supports_scalar_outputs:
            pytest.skip(f"{self.architecture} does not support scalar outputs.")

        if num_subtargets > 1 and not self.supports_multiscalar_outputs:
            pytest.skip(
                f"{self.architecture} does not support multiple scalar subtargets."
            )

        target_type = "scalar" if num_subtargets == 1 else "multiscalar"
        last_layer_features = self._are_llfs_supported(target_type)

        outputs = self._get_output(
            model_hypers,
            dataset_info_scalar,
            sample_kind,
            ["scalar"],
            last_layer_features=last_layer_features,
        )

        if sample_kind == "atom":
            assert outputs["scalar"].block().values.shape[1] == num_subtargets
        else:
            assert outputs["scalar"].block().values.shape[1] == num_subtargets

    def test_output_vector(
        self,
        model_hypers: dict,
        dataset_info_vector: DatasetInfo,
        sample_kind: Literal["atom", "system"],
    ) -> None:
        """Tests that forward pass works for vector outputs.

        It also tests that the returned outputs have the expected samples
        and values shape.

        This test is skipped if the model does not support vector outputs,
        i.e., if ``supports_vector_outputs`` is set to ``False``.

        If this test is failing, your model might:
        - not be producing vector outputs when requested.
        - not be taking into account correctly the ``sample_kind`` field of the
        outputs passed to the ``outputs`` argument of the ``forward()`` method.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_vector: Dataset information with vector outputs.
        :param sample_kind: The sample kind of the requested vector output.
        """
        if not self.supports_vector_outputs:
            pytest.skip(f"{self.architecture} does not support vector outputs.")

        last_layer_features = self._are_llfs_supported("vector")

        outputs = self._get_output(
            model_hypers,
            dataset_info_vector,
            sample_kind,
            ["vector"],
            last_layer_features=last_layer_features,
        )

        if sample_kind == "atom":
            assert outputs["vector"].block().values.shape[1:] == (3, 5)
        else:
            assert outputs["vector"].block().values.shape[1:] == (3, 5)

    def test_output_spherical(
        self,
        model_hypers: dict,
        dataset_info_spherical: DatasetInfo,
        sample_kind: Literal["atom", "system"],
    ) -> None:
        """Tests that forward pass works for spherical outputs.

        It also tests that the returned outputs have the expected samples
        and values shape.

        This test is skipped if the model does not support spherical outputs,
        i.e., if ``supports_spherical_outputs`` is set to ``False``.

        If this test is failing, your model might:
        - not be producing spherical outputs when requested.
        - not be taking into account correctly the ``sample_kind`` field of the
        outputs passed to the ``outputs`` argument of the ``forward()`` method.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_spherical: Dataset information with spherical outputs.
        :param sample_kind: The sample kind of the requested spherical output.
        """
        if not self.supports_spherical_outputs:
            pytest.skip(f"{self.architecture} does not support spherical outputs.")

        last_layer_features = self._are_llfs_supported("spherical")

        self._get_output(
            model_hypers,
            dataset_info_spherical,
            sample_kind,
            ["spherical_target"],
            last_layer_features=last_layer_features,
        )

    def test_output_multispherical(
        self,
        model_hypers: dict,
        dataset_info_multispherical: DatasetInfo,
        sample_kind: Literal["atom", "system"],
    ) -> None:
        """Tests that forward pass works for spherical tensor outputs
        with multiple irreps.

        It also tests that the returned outputs have the expected samples
        and values shape.

        This test is skipped if the model does not support spherical outputs,
        i.e., if ``supports_spherical_outputs`` is set to ``False``.

        If this test is failing and ``test_output_spherical`` is passing, your model
        probably is not handling the possibility that spherical outputs can have
        multiple irreps.

        If ``test_output_spherical`` is also failing, fix that test first.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_multispherical: Dataset information with multiple
          spherical outputs.
        :param sample_kind: The sample kind of the requested spherical output.
        """
        if not self.supports_spherical_outputs:
            pytest.skip(f"{self.architecture} does not support spherical outputs.")

        last_layer_features = self._are_llfs_supported("spherical")

        outputs = self._get_output(
            model_hypers,
            dataset_info_multispherical,
            sample_kind,
            ["spherical_tensor"],
            last_layer_features=last_layer_features,
        )

        assert len(outputs["spherical_tensor"]) == 3

    def test_output_spherical_rank2(
        self,
        model_hypers: dict,
        dataset_info_spherical_rank2: DatasetInfo,
        sample_kind: Literal["atom", "system"],
    ) -> None:
        """Tests that forward pass works for spherical outputs of rank 2.

        It also tests that the returned outputs have the expected samples
        and values shape.

        This test is skipped if the model does not support spherical outputs,
        i.e., if ``supports_spherical_outputs`` is set to ``False``, or it does
        not support spherical rank 2 outputs, i.e., if
        ``supports_spherical_rank2_outputs`` is set to ``False``.

        If this test is failing, your model might:
        - not be producing spherical outputs of rank 2 when requested.
        - not be taking into account correctly the ``sample_kind`` field of the
        outputs passed to the ``outputs`` argument of the ``forward()`` method.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_spherical_rank2: Dataset information with spherical
          outputs of rank 2.
        :param sample_kind: The sample kind of the requested spherical output.
        """
        if not self.supports_spherical_outputs:
            pytest.skip(f"{self.architecture} does not support spherical outputs.")
        if not self.supports_spherical_rank2_outputs:
            pytest.skip(
                f"{self.architecture} does not support spherical rank 2 outputs."
            )

        last_layer_features = self._are_llfs_supported("spherical_rank2")

        self._get_output(
            model_hypers,
            dataset_info_spherical_rank2,
            sample_kind,
            ["spherical_tensor_rank2"],
            last_layer_features=last_layer_features,
        )

    def test_output_spherical_atomic_basis(
        self,
        model_hypers: dict,
        dataset_info_spherical_atomic_basis: DatasetInfo,
    ) -> None:
        """Tests that forward pass works for spherical outputs that
        use an atomic basis.

        It also tests that the returned outputs have the expected samples
        and values shape.

        This test is skipped if the model does not support spherical outputs,
        i.e., if ``supports_spherical_outputs`` is set to ``False`` or it does
        not support spherical atomic basis outputs, i.e., if
        ``supports_spherical_atomic_basis_outputs`` is set to ``False``.

        If this test is failing and ``test_output_spherical`` is passing, your model
        probably is not handling the possibility that spherical outputs can be in an
        atomic basis.

        If ``test_output_spherical`` is also failing, fix that test first.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_spherical_atomic_basis: Dataset information with spherical
            outputs using an atomic basis.
        """
        if not self.supports_spherical_outputs:
            pytest.skip(f"{self.architecture} does not support spherical outputs.")
        if not self.supports_spherical_atomic_basis_outputs:
            pytest.skip(
                f"{self.architecture} does not support spherical atomic basis outputs."
            )

        last_layer_features = self._are_llfs_supported("spherical_atomic_basis")

        self._get_output(
            model_hypers,
            dataset_info_spherical_atomic_basis,
            sample_kind="atom",
            outputs=["spherical_tensor_atomic_basis"],
            last_layer_features=last_layer_features,
        )

    def test_output_spherical_atomic_basis_rank2(
        self,
        model_hypers: dict,
        dataset_info_spherical_atomic_basis_rank2: DatasetInfo,
    ) -> None:
        """Tests that forward pass works for spherical outputs that
        use an atomic basis and are of rank 2.

        It also tests that the returned outputs have the expected samples
        and values shape.

        This test is skipped if one of the following is False:

        - ``supports_spherical_outputs``
        - ``supports_spherical_atomic_basis_outputs``
        - ``supports_spherical_rank2_outputs``

        If this test is failing and

        - ``test_output_spherical``, ``test_output_spherical_atomic_basis`` and
          ``test_output_spherical_rank2`` are passing: your model is not handling
          the possibility that outputs can be spherical, in an atomic basis and
          of rank 2 at the same time.
        - Any of the previously mentioned tests is failing: fix those tests first.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_spherical_atomic_basis_rank2: Dataset information with
            spherical outputs of rank 2 using an atomic basis.
        """
        if not self.supports_spherical_outputs:
            pytest.skip(f"{self.architecture} does not support spherical outputs.")
        if not self.supports_spherical_atomic_basis_outputs:
            pytest.skip(
                f"{self.architecture} does not support spherical atomic basis outputs."
            )
        if not self.supports_spherical_rank2_outputs:
            pytest.skip(
                f"{self.architecture} does not support spherical rank 2 outputs."
            )

        self._get_output(
            model_hypers,
            dataset_info_spherical_atomic_basis_rank2,
            sample_kind="atom",
            outputs=["spherical_tensor_atomic_basis_rank2"],
        )

    def test_prediction_energy_subset_elements(
        self, model_hypers: dict, dataset_info: DatasetInfo
    ) -> None:
        """Tests that the model can predict on a subset of the elements it was trained
        on.

        If this test is failing, it means that your model needs each system
        to contain all the elements that are present in the dataset.
        If this is the expected behavior for your model, we need to introduce
        a variable to skip this test, contact the ``metatrain`` developers.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        """
        model = self.model_cls(model_hypers, dataset_info)

        system = System(
            types=torch.tensor([6, 6]),
            positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            cell=torch.zeros(3, 3),
            pbc=torch.tensor([False, False, False]),
        )
        system = get_system_with_neighbor_lists(
            system, model.requested_neighbor_lists()
        )
        model = model.to(system.positions.dtype)
        model(
            [system],
            {"energy": ModelOutput(sample_kind="atom")},
        )

    def test_prediction_energy_subset_atoms(
        self, model_hypers: dict, dataset_info: DatasetInfo
    ) -> None:
        """Tests that the model can predict on a subset
        of the atoms in a system.

        This test checks that the model supports the ``selected_atoms``
        argument of the ``forward()`` method, and it handles it correctly.
        That is, the model only returns outputs for the selected atoms.

        The test runs the model on a dimer system and compares the per-atom
        energies to those of a system where the monomers are far apart. If
        these two energies are meant to be the same in the model, this test
        will fail unless ``monomer_equal_dimer`` is set to ``True``.

        This test is skipped if the model does not support the ``selected_atoms``
        argument of the ``forward()`` method, i.e., if ``supports_selected_atoms``
        is set to ``False``.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        """

        if not self.supports_selected_atoms:
            pytest.skip(
                f"{self.architecture} does not support selected atom predictions."
            )
        # we need float64 for this test, then we will change it back at the end
        default_dtype_before = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

        try:
            model = self.model_cls(model_hypers, dataset_info).to(torch.float64)

            # Since we don't yet support atomic predictions, we will test this by
            # predicting on a system with two monomers at a large distance

            system_monomer = System(
                types=torch.tensor([7, 8, 8]),
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
                ),
                cell=torch.zeros(3, 3),
                pbc=torch.tensor([False, False, False]),
            )
            system_monomer = get_system_with_neighbor_lists(
                system_monomer, model.requested_neighbor_lists()
            )

            energy_monomer = model(
                [system_monomer],
                {"energy": ModelOutput(sample_kind="system")},
            )

            system_far_away_dimer = System(
                types=torch.tensor([7, 7, 8, 8, 8, 8]),
                positions=torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 50.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 2.0],
                        [0.0, 51.0, 0.0],
                        [0.0, 42.0, 0.0],
                    ],
                ),
                cell=torch.zeros(3, 3),
                pbc=torch.tensor([False, False, False]),
            )
            system_far_away_dimer = get_system_with_neighbor_lists(
                system_far_away_dimer, model.requested_neighbor_lists()
            )

            selection_labels = mts.Labels(
                names=["system", "atom"],
                values=torch.tensor([[0, 0], [0, 2], [0, 3]]),
            )

            energy_dimer = model(
                [system_far_away_dimer],
                {"energy": ModelOutput(sample_kind="system")},
            )

            energy_monomer_in_dimer = model(
                [system_far_away_dimer],
                {"energy": ModelOutput(sample_kind="system")},
                selected_atoms=selection_labels,
            )

            if self.monomer_equal_dimer:
                assert mts.allclose(energy_monomer["energy"], energy_dimer["energy"])
            else:
                assert not mts.allclose(
                    energy_monomer["energy"], energy_dimer["energy"]
                )

            assert mts.allclose(
                energy_monomer["energy"], energy_monomer_in_dimer["energy"]
            )
        except Exception as e:
            # make sure to set back the default dtype before raising
            torch.set_default_dtype(default_dtype_before)
            raise e

        torch.set_default_dtype(default_dtype_before)

    @pytest.mark.parametrize("sample_kind", ["atom", "system"])
    def test_output_features(
        self,
        model_hypers: dict,
        dataset_info: DatasetInfo,
        sample_kind: Literal["atom", "system"],
        n_features: Optional[int | list[int]],
    ) -> None:
        """Tests that the model can output its learned features.

        If this test is failing you are probably not exposing correctly
        the features output in your model.

        This test is skipped if the model does not support features output,
        i.e., if ``supports_features`` is set to ``False``.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        :param sample_kind: The kind of samples for which to request features.
        :param n_features: Expected number of features. If ``None``, the number
          of features is not checked.
        """

        if not self.supports_features:
            pytest.skip(f"{self.architecture} does not support features output.")

        model = self.model_cls(model_hypers, dataset_info)

        system = System(
            types=torch.tensor([6, 1, 8, 7]),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            ),
            cell=torch.zeros(3, 3),
            pbc=torch.tensor([False, False, False]),
        )
        system = get_system_with_neighbor_lists(
            system, model.requested_neighbor_lists()
        )

        features_output_options = ModelOutput(
            unit="",
            sample_kind=sample_kind,
        )
        model = model.to(system.positions.dtype)
        outputs = model(
            [system],
            {
                "energy": ModelOutput(sample_kind="atom"),
                "feature": features_output_options,
            },
        )
        assert "energy" in outputs
        assert "feature" in outputs

        features_outputs = outputs["feature"]
        for i in range(len(features_outputs)):
            features = features_outputs.block(i)

            if sample_kind == "atom":
                expected_samples = ["system", "atom"]
            else:
                expected_samples = ["system"]

            per_atom = sample_kind == "atom"

            assert features.samples.names == expected_samples
            assert features.properties.names == ["feature"]
            assert features.values.shape[0] == (4 if per_atom else 1)
            if isinstance(n_features, int):
                assert features.values.shape[-1] == n_features, (
                    f"Block {i}, expected {n_features} features "
                    f"but got {features.values.shape[-1]}"
                )
            elif isinstance(n_features, list):
                assert features.values.shape[-1] == n_features[i], (
                    f"Block {i}, expected {n_features[i]} features "
                    f"but got {features.values.shape[-1]}"
                )

    @pytest.mark.parametrize("select_atoms", [[0, 2]])
    def test_output_last_layer_features_selected_atoms(
        self,
        model_hypers: dict,
        dataset_info: DatasetInfo,
        dataset_path: str,
        select_atoms: list[int],
    ) -> None:
        """Tests that the model can output its last layer features for selected atoms.

        This test is skipped if the model does not support last-layer features
        or the model does not support the ``selected_atoms`` argument of the
        ``forward()`` method, i.e. if either ``supports_last_layer_features``
        or ``supports_selected_atoms`` is set to ``False``.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        :param dataset_path: Path to a dataset file to read systems from.
        :param select_atoms: List of atom indices to select for the output.
        """
        if not self.supports_last_layer_features:
            pytest.skip(
                f"{self.architecture} does not support last-layer features output."
            )
        if not self.supports_selected_atoms:
            pytest.skip(
                f"{self.architecture} does not support selected atom predictions."
            )

        systems = read_systems(dataset_path)
        systems = [system.to(torch.float32) for system in systems]

        hypers = copy.deepcopy(model_hypers)
        model = self.model_cls(hypers, dataset_info)
        systems = [
            get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
            for system in systems
        ]

        output_label = "mtt::aux::energy_last_layer_features"
        outputs = {output_label: ModelOutput(sample_kind="atom")}
        selected_atoms = mts.Labels(
            names=["system", "atom"],
            values=torch.tensor(
                [(n, i) for n in range(len(systems)) for i in select_atoms]
            ),
        )
        model = model.to(systems[0].positions.dtype)
        out = model(systems, outputs, selected_atoms=selected_atoms)
        features = out[output_label].block(0).samples.values
        assert features.shape == selected_atoms.values.shape

    def _are_llfs_supported(self, target_type: str) -> bool:
        if isinstance(self.supports_last_layer_features, bool):
            return self.supports_last_layer_features

        are_supported = self.supports_last_layer_features.get(target_type, None)
        if are_supported is None:
            are_supported = getattr(self, f"supports_{target_type}_outputs")
        return are_supported

    def _check_llfs_affine(
        self,
        output: mts.TensorMap,
        llf: mts.TensorMap,
        output_name: Optional[str] = None,
    ) -> None:
        shared = len(llf.block(0).components) == 0
        for key, block in output.items():
            feature_block = llf.block(0 if shared else key)
            if feature_block.properties.values.shape[0] == 0:
                # There are no last layer features for this block.
                # Before calling this function we always check that the layout of
                # the last layer features is fine, so this means that truly there
                # are no last layer features for this block. Skip it.
                continue
            rows = feature_block.samples.select(block.samples)
            x = feature_block.values[rows]
            if shared:
                y = block.values.reshape(block.values.shape[0], -1)
            else:
                x = x.reshape(-1, x.shape[-1])
                y = block.values.reshape(-1, block.values.shape[-1])
            # constant feature for the biases, fitted in float64 because the
            # features can span several orders of magnitude
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1).to(torch.float64)
            y = y.to(torch.float64)
            # more distinct samples than features, otherwise any target fits
            assert torch.unique(x, dim=0).shape[0] > x.shape[1]
            weights = torch.linalg.lstsq(x, y).solution
            if not torch.allclose(x @ weights, y, atol=1e-3):
                if output_name is None:
                    raise ValueError(
                        "Last-layer features are not affine with respect to the output."
                    )
                else:
                    raise ValueError(
                        f"Last-layer features for output '{output_name}' are not "
                        "affine with respect to the output."
                    )

    def test_single_atom(
        self,
        model_hypers: dict,
        dataset_info: DatasetInfo,
        single_atom_energy: Optional[float],
    ) -> None:
        """Tests that the model runs fine on a single atom system.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        :param single_atom_energy: Expected single atom energy value. If ``None``,
          the single atom energy value is not checked.
        """
        model = self.model_cls(model_hypers, dataset_info)

        system = System(
            types=torch.tensor([6]),
            positions=torch.tensor([[0.0, 0.0, 1.0]]),
            cell=torch.zeros(3, 3),
            pbc=torch.tensor([False, False, False]),
        )
        system = get_system_with_neighbor_lists(
            system, model.requested_neighbor_lists()
        )
        model = model.to(system.positions.dtype)
        outputs = model([system], {"energy": ModelOutput(sample_kind="system")})
        if single_atom_energy is not None:
            assert outputs["energy"].block().values.item() == single_atom_energy

    def test_output_scalar_invariant(
        self, model_hypers: dict, dataset_info: DatasetInfo, dataset_path: str
    ) -> None:
        """Tests that scalar outputs are invariant to rotation.

        This test is skipped if the model does not support scalar outputs,
        or if the model is not equivariant by design, i.e., if either
        ``supports_scalar_outputs`` or ``is_equivariant_rotations`` is set to
        ``False``.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        :param dataset_path: Path to a dataset file to read systems from.
        """
        if not self.supports_scalar_outputs or not self.is_equivariant_rotations:
            pytest.skip(
                f"{self.architecture} does not produce invariant scalar outputs."
            )

        model = self.model_cls(model_hypers, dataset_info)

        system: System = read(dataset_path)
        original_system = copy.deepcopy(system)
        system.rotate(48, "y")
        original_system = systems_to_torch(original_system)
        system = systems_to_torch(system)

        requested_neighbor_lists = get_requested_neighbor_lists(model)

        model = model.to(original_system.positions.dtype)

        original_output = model(
            [get_system_with_neighbor_lists(original_system, requested_neighbor_lists)],
            {"energy": ModelOutput(sample_kind="atom")},
        )
        rotated_output = model(
            [get_system_with_neighbor_lists(system, requested_neighbor_lists)],
            {"energy": ModelOutput(sample_kind="atom")},
        )

        torch.testing.assert_close(
            original_output["energy"].block().values,
            rotated_output["energy"].block().values,
            atol=self.equivariance_error_tolerance,
            rtol=self.equivariance_error_tolerance,
        )

    def test_output_spherical_equivariant_rotations(
        self, model_hypers: dict, dataset_info_spherical: DatasetInfo, dataset_path: str
    ) -> None:
        """Tests that the model is rotationally equivariant when predicting
        spherical tensors.

        This test is skipped if the model does not support spherical outputs,
        or if the model is not equivariant by design, i.e., if either
        ``supports_spherical_outputs`` or ``is_equivariant_rotations`` is set to
        ``False``.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_spherical: Dataset information with spherical outputs.
        :param dataset_path: Path to a dataset file to read systems from.
        """

        if not self.supports_spherical_outputs or not self.is_equivariant_rotations:
            pytest.skip(
                f"{self.architecture} does not produce equivariant spherical outputs."
            )

        model = self.model_cls(model_hypers, dataset_info_spherical)

        system = read(dataset_path)
        original_system = systems_to_torch(system)
        target_layout = dataset_info_spherical.targets["spherical_target"].layout
        ell = (len(target_layout.block(0).components[0]) - 1) // 2
        transformation = random_transformations(
            1,
            ell,
            device=original_system.positions.device,
            dtype=original_system.positions.dtype,
        )
        rotated_system = transformation.transform_systems([original_system])[0]

        requested_neighbor_lists = get_requested_neighbor_lists(model)
        original_system = get_system_with_neighbor_lists(
            original_system, requested_neighbor_lists
        )
        rotated_system = get_system_with_neighbor_lists(
            rotated_system, requested_neighbor_lists
        )

        model = model.to(original_system.positions.dtype)

        outputs = {"spherical_target": ModelOutput(sample_kind="atom")}
        # the last-layer features rotate with the target
        features_name = "mtt::aux::spherical_target_last_layer_features"
        if features_name in model.supported_outputs():
            outputs[features_name] = ModelOutput(sample_kind="atom")
        original_output = model([original_system], outputs)
        rotated_output = model([rotated_system], outputs)

        for name in outputs:
            original_values = original_output[name].block().values.detach()
            expected_values = (
                original_values.swapaxes(-1, -2)
                @ transformation.wigner_D_matrices(ell)[0].T
            ).swapaxes(-1, -2)
            torch.testing.assert_close(
                rotated_output[name].block().values.detach(),
                expected_values,
                atol=self.equivariance_error_tolerance,
                rtol=self.equivariance_error_tolerance,
            )

    def test_output_spherical_equivariant_inversion(
        self,
        model_hypers: dict,
        dataset_info_spherical: DatasetInfo,
        dataset_path: str,
        o3_lambda: int,
        o3_sigma: int,
    ) -> None:
        """Tests that the model is equivariant with respect to inversions.

        This test is done on spherical targets (not scalar targets).

        This test is skipped if the model does not support spherical outputs,
        or if the model is not equivariant by design, i.e., if either
        ``supports_spherical_outputs`` or ``is_equivariant_reflections`` is set to
        ``False``.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_spherical: Dataset information with spherical outputs.
        :param dataset_path: Path to a dataset file to read systems from.
        :param o3_lambda: The O(3) lambda of the spherical output to test.
        :param o3_sigma: The O(3) sigma of the spherical output to test.
        """

        if not self.supports_spherical_outputs or not self.is_equivariant_reflections:
            pytest.skip(
                f"{self.architecture} does not produce equivariant spherical outputs."
            )

        model = self.model_cls(model_hypers, dataset_info_spherical)

        system = read(dataset_path)
        original_system = systems_to_torch(system)
        inverted_system = System(
            positions=original_system.positions * (-1),
            cell=original_system.cell * (-1),
            types=original_system.types,
            pbc=original_system.pbc,
        )

        requested_neighbor_lists = get_requested_neighbor_lists(model)
        original_system = get_system_with_neighbor_lists(
            original_system, requested_neighbor_lists
        )
        inverted_system = get_system_with_neighbor_lists(
            inverted_system, requested_neighbor_lists
        )

        model = model.to(original_system.positions.dtype)

        original_output = model(
            [original_system],
            {"spherical_target": ModelOutput(sample_kind="atom")},
        )
        inverted_output = model(
            [inverted_system],
            {"spherical_target": ModelOutput(sample_kind="atom")},
        )

        torch.testing.assert_close(
            original_output["spherical_target"].block().values
            * (-1) ** o3_lambda
            * (-1 if o3_sigma == -1 else 1),
            inverted_output["spherical_target"].block().values,
            atol=self.equivariance_error_tolerance,
            rtol=self.equivariance_error_tolerance,
        )
