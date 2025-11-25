import copy
from typing import Optional

import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatomic.torch import ModelOutput, System, systems_to_torch

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.readers import (
    read_systems,
)
from metatrain.utils.data.readers.ase import read
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from .architectures import ArchitectureTests
from .equivariance import (
    get_random_rotation,
    rotate_spherical_tensor,
    rotate_system,
)


class OutputTests(ArchitectureTests):
    """Test suite to check that the model can produce different types of outputs.

    If a model does not support a given type of output, set the corresponding
    ``supports_*_outputs`` attribute to ``False`` to skip the corresponding tests.
    By default, they are all set to ``True`` to avoid supported outputs from
    being untested accidentally.
    """

    supports_scalar_outputs: bool = True
    """Whether the model supports scalar outputs."""
    supports_vector_outputs: bool = True
    """Whether the model supports vector outputs."""
    supports_spherical_outputs: bool = True
    """Whether the model supports spherical tensor outputs."""
    supports_selected_atoms: bool = True
    """Whether the model supports the ``selected_atoms`` argument in the
    ``forward()`` method.
    """
    supports_features: bool = True
    """Whether the model supports returning features."""
    supports_last_layer_features: bool = True
    """Whether the model supports returning last-layer features."""
    is_equivariant_model: bool = True
    """Whether the model is equivariant (i.e. produces outputs that
    transform correctly under rotations by architecture's design)."""

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
    def n_last_layer_features(self) -> Optional[int | list[int]]:
        """Fixture that returns the number of last-layer features produced
        by the model.

        By default this is set to ``None``, which skips checking the number
        of last-layer features in the output. Override this fixture for your
        architecture if you want the test suite to check that the number of
        last-layer features in the output is correct.

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
        per_atom: bool,
        outputs: Optional[list[str]] = None,
    ) -> dict[str, mts.TensorMap]:
        """Helper function to get the model output for different types of outputs.

        It initializes the model and runs a forward pass with a simple system.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        :param per_atom: Whether the requested outputs are per-atom or not.
        :param outputs: List of output names to request. If ``None``, all outputs
            defined in the model are requested.

        :return: The model outputs.
        """
        model = self.model_cls(model_hypers, dataset_info)

        system = System(
            types=torch.tensor([1, 6, 7, 8]),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
            ),
            cell=torch.zeros(3, 3),
            pbc=torch.tensor([False, False, False]),
        )
        system = get_system_with_neighbor_lists(
            system, model.requested_neighbor_lists()
        )

        if outputs is None:
            outputs = list(model.outputs.keys())

        model = model.to(system.positions.dtype)
        return model([system], {k: ModelOutput(per_atom=per_atom) for k in outputs})

    def test_output_scalar(
        self, model_hypers: dict, dataset_info_scalar: DatasetInfo, per_atom: bool
    ) -> None:
        """Tests that forward pass works for scalar outputs.

        It also tests that the returned outputs have the expected samples
        and values shape.

        This test is skipped if the model does not support scalar outputs,
        i.e., if ``supports_scalar_outputs`` is set to ``False``.

        If this test is failing, your model might:

        - not be producing scalar outputs when requested.
        - not be taking into account correctly the ``per_atom`` field of the
          outputs passed to the ``outputs`` argument of the ``forward()`` method.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_scalar: Dataset information with scalar outputs.
        :param per_atom: Whether the requested outputs are per-atom or not.
        """
        if not self.supports_scalar_outputs:
            pytest.skip(f"{self.architecture} does not support scalar outputs.")

        outputs = self._get_output(
            model_hypers, dataset_info_scalar, per_atom, ["scalar"]
        )

        if per_atom:
            assert outputs["scalar"].block().samples.names == ["system", "atom"]
            assert outputs["scalar"].block().values.shape == (4, 5)
        else:
            assert outputs["scalar"].block().samples.names == ["system"], (
                outputs["scalar"].block().samples.names
            )
            assert outputs["scalar"].block().values.shape == (1, 5)

    def test_output_vector(
        self, model_hypers: dict, dataset_info_vector: DatasetInfo, per_atom: bool
    ) -> None:
        """Tests that forward pass works for vector outputs.

        It also tests that the returned outputs have the expected samples
        and values shape.

        This test is skipped if the model does not support vector outputs,
        i.e., if ``supports_vector_outputs`` is set to ``False``.

        If this test is failing, your model might:
        - not be producing vector outputs when requested.
        - not be taking into account correctly the ``per_atom`` field of the
        outputs passed to the ``outputs`` argument of the ``forward()`` method.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_vector: Dataset information with vector outputs.
        :param per_atom: Whether the requested outputs are per-atom or not.
        """
        if not self.supports_vector_outputs:
            pytest.skip(f"{self.architecture} does not support vector outputs.")
        outputs = self._get_output(
            model_hypers, dataset_info_vector, per_atom, ["vector"]
        )

        if per_atom:
            assert outputs["vector"].block().samples.names == ["system", "atom"]
            assert outputs["vector"].block().values.shape == (4, 3, 5)
        else:
            assert outputs["vector"].block().samples.names == ["system"]
            assert outputs["vector"].block().values.shape == (1, 3, 5)

    def test_output_spherical(
        self, model_hypers: dict, dataset_info_spherical: DatasetInfo, per_atom: bool
    ) -> None:
        """Tests that forward pass works for spherical outputs.

        It also tests that the returned outputs have the expected samples
        and values shape.

        This test is skipped if the model does not support spherical outputs,
        i.e., if ``supports_spherical_outputs`` is set to ``False``.

        If this test is failing, your model might:
        - not be producing spherical outputs when requested.
        - not be taking into account correctly the ``per_atom`` field of the
        outputs passed to the ``outputs`` argument of the ``forward()`` method.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_spherical: Dataset information with spherical outputs.
        :param per_atom: Whether the requested outputs are per-atom or not.
        """
        if not self.supports_spherical_outputs:
            pytest.skip(f"{self.architecture} does not support spherical outputs.")

        outputs = self._get_output(
            model_hypers, dataset_info_spherical, per_atom, ["spherical_target"]
        )

        if per_atom:
            assert outputs["spherical_target"].block().samples.names == [
                "system",
                "atom",
            ]
            assert outputs["spherical_target"].block().values.shape[0] == 4
        else:
            assert outputs["spherical_target"].block().samples.names == ["system"]
            assert outputs["spherical_target"].block().values.shape[0] == 1

    def test_output_multispherical(
        self,
        model_hypers: dict,
        dataset_info_multispherical: DatasetInfo,
        per_atom: bool,
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
        :param per_atom: Whether the requested outputs are per-atom or not.
        """
        if not self.supports_spherical_outputs:
            pytest.skip(f"{self.architecture} does not support spherical outputs.")

        outputs = self._get_output(
            model_hypers, dataset_info_multispherical, per_atom, ["spherical_tensor"]
        )

        assert len(outputs["spherical_tensor"]) == 3

        for i in range(len(outputs["spherical_tensor"])):

            spherical_target_block = outputs["spherical_tensor"].block(i)

            if per_atom:
                assert spherical_target_block.samples.names == [
                    "system",
                    "atom",
                ]
                assert spherical_target_block.values.shape[0] == 4
            else:
                assert spherical_target_block.samples.names == ["system"]
                assert spherical_target_block.values.shape[0] == 1

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
        model(
            [system],
            {"energy": model.outputs["energy"]},
        )

    def test_prediction_energy_subset_atoms(
        self, model_hypers: dict, dataset_info: DatasetInfo
    ) -> None:
        """Tests that the model can predict on a subset
        of the atoms in a system.

        This test checks that the model supports the ``selected_atoms``
        argument of the ``forward()`` method, and it handles it correctly.
        That is, the model only returns outputs for the selected atoms.

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
            model = self.model_cls(model_hypers, dataset_info)

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
                {"energy": ModelOutput(per_atom=False)},
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
                {"energy": ModelOutput(per_atom=False)},
            )

            energy_monomer_in_dimer = model(
                [system_far_away_dimer],
                {"energy": ModelOutput(per_atom=False)},
                selected_atoms=selection_labels,
            )

            assert not mts.allclose(energy_monomer["energy"], energy_dimer["energy"])

            assert mts.allclose(
                energy_monomer["energy"], energy_monomer_in_dimer["energy"]
            )
        except Exception as e:
            # make sure to set back the default dtype before raising
            torch.set_default_dtype(default_dtype_before)
            raise e

        torch.set_default_dtype(default_dtype_before)

    @pytest.mark.parametrize("per_atom", [True, False])
    def test_output_features(
        self,
        model_hypers: dict,
        dataset_info: DatasetInfo,
        per_atom: bool,
        n_features: Optional[int | list[int]],
    ) -> None:
        """Tests that the model can output its learned features.

        If this test is failing you are probably not exposing correctly
        the features output in your model.

        This test is skipped if the model does not support features output,
        i.e., if ``supports_features`` is set to ``False``.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        :param per_atom: Whether to request per-atom features or not.
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
            quantity="",
            unit="unitless",
            per_atom=per_atom,
        )
        outputs = model(
            [system],
            {
                "energy": model.outputs["energy"],
                "features": features_output_options,
            },
        )
        assert "energy" in outputs
        assert "features" in outputs

        features_outputs = outputs["features"]
        for i in range(len(features_outputs)):
            features = features_outputs.block(i)

            expected_samples = ["system", "atom"] if per_atom else ["system"]
            assert features.samples.names == expected_samples
            assert features.properties.names == ["feature"]
            assert features.values.shape[0] == (4 if per_atom else 1)
            if isinstance(n_features, int):
                assert features.values.shape[-1] == n_features, f"Block {i}, expected {n_features} features but got {features.values.shape[-1]}"
            elif isinstance(n_features, list):
                assert features.values.shape[-1] == n_features[i], f"Block {i}, expected {n_features[i]} features but got {features.values.shape[-1]}"

    @pytest.mark.parametrize("per_atom", [True, False])
    def test_output_last_layer_features(
        self,
        model_hypers: dict,
        dataset_info: DatasetInfo,
        per_atom: bool,
        n_last_layer_features: Optional[int | list[int]],
    ) -> None:
        """Tests that the model can output its last layer features.

        If this test is failing you are probably not exposing correctly
        the last-layer features output in your model.

        This test is skipped if the model does not support last-layer features
        output, i.e., if ``supports_last_layer_features`` is set to ``False``.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        :param per_atom: Whether to request per-atom last-layer features or not.
        :param n_last_layer_features: Expected number of last-layer features.
          If ``None``, the number of last-layer features is not checked.
        """

        if not self.supports_last_layer_features:
            pytest.skip(
                f"{self.architecture} does not support last-layer features output."
            )

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

        # last-layer features per atom:
        ll_output_options = ModelOutput(
            quantity="",
            unit="unitless",
            per_atom=per_atom,
        )
        outputs = model(
            [system],
            {
                "energy": model.outputs["energy"],
                "mtt::aux::energy_last_layer_features": ll_output_options,
            },
        )
        assert "energy" in outputs
        assert "mtt::aux::energy_last_layer_features" in outputs

        last_layer_features = outputs["mtt::aux::energy_last_layer_features"].block()
        expected_samples = ["system", "atom"] if per_atom else ["system"]
        assert last_layer_features.samples.names == expected_samples
        assert last_layer_features.properties.names == ["feature"]
        assert last_layer_features.values.shape[0] == (4 if per_atom else 1)
        if n_last_layer_features is not None:
            assert last_layer_features.values.shape[-1] == n_last_layer_features

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
        modeloutput = model.outputs[output_label]
        modeloutput.per_atom = True
        outputs = {output_label: modeloutput}
        selected_atoms = mts.Labels(
            names=["system", "atom"],
            values=torch.tensor(
                [(n, i) for n in range(len(systems)) for i in select_atoms]
            ),
        )
        out = model(systems, outputs, selected_atoms=selected_atoms)
        features = out[output_label].block().samples.values
        assert features.shape == selected_atoms.values.shape

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
        outputs = model([system], {"energy": ModelOutput(per_atom=False)})
        if single_atom_energy is not None:
            assert outputs["energy"].block().values.item() == single_atom_energy

    def test_output_scalar_invariant(
        self, model_hypers: dict, dataset_info: DatasetInfo, dataset_path: str
    ) -> None:
        """Tests that scalar outputs are invariant to rotation.

        This test is skipped if the model does not support scalar outputs,
        or if the model is not equivariant by design, i.e., if either
        ``supports_scalar_outputs`` or ``is_equivariant_model`` is set to
        ``False``.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        :param dataset_path: Path to a dataset file to read systems from.
        """
        if not self.supports_scalar_outputs or not self.is_equivariant_model:
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
            [
                get_system_with_neighbor_lists(
                    original_system, requested_neighbor_lists
                )
            ],
            {"energy": model.outputs["energy"]},
        )
        rotated_output = model(
            [
                get_system_with_neighbor_lists(
                    system, requested_neighbor_lists
                )
            ],
            {"energy": model.outputs["energy"]},
        )

        torch.testing.assert_close(
            original_output["energy"].block().values,
            rotated_output["energy"].block().values,
        )

    def test_output_spherical_equivariant_rotations(
        self, model_hypers: dict, dataset_info_spherical: DatasetInfo, dataset_path: str
    ) -> None:
        """Tests that the model is rotationally equivariant when predicting
        spherical tensors.

        This test is skipped if the model does not support spherical outputs,
        or if the model is not equivariant by design, i.e., if either
        ``supports_spherical_outputs`` or ``is_equivariant_model`` is set to
        ``False``.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_spherical: Dataset information with spherical outputs.
        :param dataset_path: Path to a dataset file to read systems from.
        """

        if not self.supports_spherical_outputs or not self.is_equivariant_model:
            pytest.skip(
                f"{self.architecture} does not produce equivariant spherical outputs."
            )

        model = self.model_cls(model_hypers, dataset_info_spherical)

        system = read(dataset_path)
        original_system = systems_to_torch(system)
        rotation = get_random_rotation()
        rotated_system = rotate_system(original_system, rotation)

        requested_neighbor_lists = get_requested_neighbor_lists(model)
        original_system = get_system_with_neighbor_lists(
            original_system, requested_neighbor_lists
        )
        rotated_system = get_system_with_neighbor_lists(
            rotated_system, requested_neighbor_lists
        )

        model = model.to(original_system.positions.dtype)

        original_output = model(
            [original_system],
            {"spherical_target": model.outputs["spherical_target"]},
        )
        rotated_output = model(
            [rotated_system],
            {"spherical_target": model.outputs["spherical_target"]},
        )

        np.testing.assert_allclose(
            rotate_spherical_tensor(
                original_output["spherical_target"].block().values.detach().numpy(),
                rotation,
            ),
            rotated_output["spherical_target"].block().values.detach().numpy(),
            atol=1e-5,
            rtol=1e-5,
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
        ``supports_spherical_outputs`` or ``is_equivariant_model`` is set to
        ``False``.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_spherical: Dataset information with spherical outputs.
        :param dataset_path: Path to a dataset file to read systems from.
        :param o3_lambda: The O(3) lambda of the spherical output to test.
        :param o3_sigma: The O(3) sigma of the spherical output to test.
        """

        if not self.supports_spherical_outputs or not self.is_equivariant_model:
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
            {"spherical_target": model.outputs["spherical_target"]},
        )
        inverted_output = model(
            [inverted_system],
            {"spherical_target": model.outputs["spherical_target"]},
        )

        torch.testing.assert_close(
            original_output["spherical_target"].block().values
            * (-1) ** o3_lambda
            * (-1 if o3_sigma == -1 else 1),
            inverted_output["spherical_target"].block().values,
            atol=1e-5,
            rtol=1e-5,
        )
