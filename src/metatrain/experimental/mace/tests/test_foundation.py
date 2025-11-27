import copy
from pathlib import Path

import ase.io
import torch
from mace.calculators import MACECalculator
from metatomic.torch.ase_calculator import MetatomicCalculator

from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo

from .test_basic import MACETests


class TestFoundation(MACETests):
    """Tests for MACE foundational models."""

    def test_mace_equals_metatomic(
        self, device: torch.device, mace_model_path: Path, dataset_info: DatasetInfo
    ) -> None:
        """Tests that the energy and forces computed with the MACE foundational model
        are the same when using the native MACE calculator and when using the
        Metatomic calculator.

        :param device: Device to run the metatomic model on.
        :param mace_model_path: Path to the MACE foundational model file.
        :param dataset_info: Dataset information.
        """

        # Get an atoms object to test
        periodic_water_file = (
            Path(__file__).parents[5] / "tests" / "resources" / "periodic_water.lmp"
        )
        atoms = ase.io.read(periodic_water_file, format="lammps-data")

        # Get an atomistic model from MACE foundational model file
        model_hypers = copy.deepcopy(get_default_hypers(self.architecture)["model"])

        model_hypers["mace_model"] = mace_model_path
        model_hypers["mace_model_remove_scale_shift"] = False

        model = self.model_cls(model_hypers, dataset_info)

        # Compute the energy and forces with the metatomic calculator
        atoms.calc = MetatomicCalculator(model.export(), device=device)

        mta_energy = atoms.get_potential_energy()
        mta_forces = atoms.get_forces()

        # Compute the energy and forces with the native MACE calculator
        atoms.calc = MACECalculator(mace_model_path)
        mace_energy = atoms.get_potential_energy()
        mace_forces = atoms.get_forces()

        assert abs(mta_energy - mace_energy) < 1e-9
        assert ((mta_forces - mace_forces) ** 2).sum() < 1e-20
