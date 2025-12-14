from pathlib import Path

import ase.io
import torch
from mace.calculators import MACECalculator
from metatomic.torch.ase_calculator import MetatomicCalculator

from metatrain.experimental.mace.utils._load_model_file import load_mace_model_file

from .test_basic import MACETests


class TestFoundation(MACETests):
    """Tests for MACE foundational models."""

    def test_mace_equals_metatomic(
        self,
        device: torch.device,
        mace_model_path: Path,
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
            Path(__file__).parents[5] / "tests" / "resources" / "periodic_water.data"
        )
        atoms = ase.io.read(periodic_water_file, format="lammps-data")

        model = load_mace_model_file(
            mace_model_path,
            mace_head_target="energy",
            device=device,
        )

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
