# tools to run the simulation
import ase.build
import ase.md
import ase.md.velocitydistribution
import ase.units

# Integration with ASE calculator for metatensor atomistic models
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator

# The SOAP-BPNN model contains compiled extensions from rascaline.torch
import rascaline.torch


# Initial positions (reading them from a file):
atoms = ase.io.read("../../tests/resources/ethanol_reduced_100.xyz")

# Initialize the velocities:
ase.md.velocitydistribution.MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Load the model and register it as the energy calculator for these ``atoms``:
atoms.calc = MetatensorCalculator("exported-model.pt")

integrator = ase.md.Langevin(
    atoms,
    timestep=1.0 * ase.units.fs,
    temperature_K=300,
    friction=0.1 / ase.units.fs,
)

integrator.run(10)
