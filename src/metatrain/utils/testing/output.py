import copy

import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatomic.torch import ModelOutput, System, systems_to_torch

from metatrain.utils.data.readers import (
    read_systems,
)
from metatrain.utils.data.readers.ase import read
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from .base import ArchitectureTests
from .equivariance import (
    get_random_rotation,
    rotate_spherical_tensor,
    rotate_system,
)


class OutputTests(ArchitectureTests):
    supports_scalar_outputs: bool = True
    supports_vector_outputs: bool = True
    supports_spherical_outputs: bool = True
    supports_selected_atoms: bool = True
    supports_features: bool = True
    supports_last_layer_features: bool = True

    is_equivariant_model: bool = True

    @pytest.fixture
    def n_features(self):
        """Override this fixture if you want to check the number of features."""
        return None

    @pytest.fixture
    def n_last_layer_features(self):
        """Override this fixture if you want to check the number of
        last-layer features."""
        return None

    @pytest.fixture
    def single_atom_energy(self):
        """Override this fixture if you want to check the single atom energy value."""
        return None

    def _get_output(self, model_hypers, dataset_info, per_atom):
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

        return model(
            [system], {k: ModelOutput(per_atom=per_atom) for k in model.outputs}
        )

    def test_output_scalar(self, model_hypers, dataset_info_scalar, per_atom):
        """Tests that forward pass works for scalar outputs."""
        if not self.supports_scalar_outputs:
            pytest.skip(f"{self.architecture} does not support scalar outputs.")

        outputs = self._get_output(model_hypers, dataset_info_scalar, per_atom)

        if per_atom:
            assert outputs["scalar"].block().samples.names == ["system", "atom"]
            assert outputs["scalar"].block().values.shape == (4, 5)
        else:
            assert outputs["scalar"].block().samples.names == ["system"], (
                outputs["scalar"].block().samples.names
            )
            assert outputs["scalar"].block().values.shape == (1, 5)

    def test_output_vector(self, model_hypers, dataset_info_vector, per_atom):
        """Tests that forward pass works for vector outputs."""
        if not self.supports_vector_outputs:
            pytest.skip(f"{self.architecture} does not support vector outputs.")
        outputs = self._get_output(model_hypers, dataset_info_vector, per_atom)

        if per_atom:
            assert outputs["vector"].block().samples.names == ["system", "atom"]
            assert outputs["vector"].block().values.shape == (4, 3, 5)
        else:
            assert outputs["vector"].block().samples.names == ["system"]
            assert outputs["vector"].block().values.shape == (1, 3, 5)

    def test_output_spherical(self, model_hypers, dataset_info_spherical, per_atom):
        """Tests that forward pass works for spherical outputs."""
        if not self.supports_spherical_outputs:
            pytest.skip(f"{self.architecture} does not support spherical outputs.")

        outputs = self._get_output(model_hypers, dataset_info_spherical, per_atom)

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
        self, model_hypers, dataset_info_multispherical, per_atom
    ):
        """Tests that forward pass works for spherical tensor outputs
        with multiple irreps."""
        if not self.supports_spherical_outputs:
            pytest.skip(f"{self.architecture} does not support spherical outputs.")

        outputs = self._get_output(model_hypers, dataset_info_multispherical, per_atom)

        assert len(outputs["spherical_tensor"]) == 3

        if per_atom:
            assert outputs["spherical_tensor"].block(0).samples.names == [
                "system",
                "atom",
            ]
            assert outputs["spherical_tensor"].block(0).values.shape[0] == 4
        else:
            assert outputs["spherical_tensor"].block(0).samples.names == ["system"]
            assert outputs["spherical_tensor"].block(0).values.shape[0] == 1

    def test_prediction_energy_subset_elements(self, model_hypers, dataset_info):
        """Tests that the model can predict on a subset of the elements it was trained
        on."""

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

    def test_prediction_energy_subset_atoms(self, model_hypers, dataset_info):
        """Tests that the model can predict on a subset
        of the atoms in a system."""

        if not self.supports_selected_atoms:
            pytest.skip(
                f"{self.architecure} does not support selected atom predictions."
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
    def test_output_features(self, model_hypers, dataset_info, per_atom, n_features):
        """Tests that the model can output its learned features."""

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

        features = outputs["features"].block()

        expected_samples = ["system", "atom"] if per_atom else ["system"]
        assert features.samples.names == expected_samples
        assert features.properties.names == ["feature"]
        assert features.values.shape[0] == (4 if per_atom else 1)
        if n_features is not None:
            assert features.values.shape[1] == n_features

    @pytest.mark.parametrize("per_atom", [True, False])
    def test_output_last_layer_features(
        self, model_hypers, dataset_info, per_atom, n_last_layer_features
    ):
        """Tests that the model can output its last layer features."""

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
            assert last_layer_features.values.shape[1] == n_last_layer_features

    @pytest.mark.parametrize("select_atoms", [[0, 2]])
    def test_output_last_layer_features_selected_atoms(
        self, model_hypers, dataset_info, DATASET_PATH, select_atoms
    ):
        if not self.supports_last_layer_features:
            pytest.skip(
                f"{self.architecture} does not support last-layer features output."
            )
        if not self.supports_selected_atoms:
            pytest.skip(
                f"{self.architecture} does not support selected atom predictions."
            )

        systems = read_systems(DATASET_PATH)
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

    def test_single_atom(self, model_hypers, dataset_info, single_atom_energy):
        """Tests that the model runs fine on a single atom system."""
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

    def test_output_scalar_invariant(self, model_hypers, dataset_info, DATASET_PATH):
        """Tests that scalar outputs are invariant to rotation"""

        if not self.supports_scalar_outputs or not self.is_equivariant_model:
            pytest.skip(
                f"{self.architecture} does not produce invariant scalar outputs."
            )

        model = self.model_cls(model_hypers, dataset_info)

        system = read(DATASET_PATH)
        original_system = copy.deepcopy(system)
        system.rotate(48, "y")

        requested_neighbor_lists = get_requested_neighbor_lists(model)
        original_output = model(
            [
                get_system_with_neighbor_lists(
                    systems_to_torch(original_system), requested_neighbor_lists
                )
            ],
            {"energy": model.outputs["energy"]},
        )
        rotated_output = model(
            [
                get_system_with_neighbor_lists(
                    systems_to_torch(system), requested_neighbor_lists
                )
            ],
            {"energy": model.outputs["energy"]},
        )

        torch.testing.assert_close(
            original_output["energy"].block().values,
            rotated_output["energy"].block().values,
        )

    def test_output_spherical_equivariant_rotations(
        self, model_hypers, dataset_info_spherical, DATASET_PATH
    ):
        """Tests that the model is rotationally equivariant when predicting
        spherical tensors."""

        if not self.supports_spherical_outputs or not self.is_equivariant_model:
            pytest.skip(
                f"{self.architecture} does not produce equivariant spherical outputs."
            )

        model = self.model_cls(model_hypers, dataset_info_spherical)

        system = read(DATASET_PATH)
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
        self, model_hypers, dataset_info_spherical, DATASET_PATH, o3_lambda, o3_sigma
    ):
        """Tests that the model is equivariant with respect to inversions."""

        if not self.supports_spherical_outputs or not self.is_equivariant_model:
            pytest.skip(
                f"{self.architecture} does not produce equivariant spherical outputs."
            )

        model = self.model_cls(model_hypers, dataset_info_spherical)

        system = read(DATASET_PATH)
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
