"""
Running molecular dynamics with ASE
===================================

This tutorial shows how to use an exported model to run an ASE simulation.
"""

# %%
#
# First, we import the necessary libraries:

# Tools to run the simulation
import ase.md
import ase.md.velocitydistribution
import ase.units
import ase.visualize.plot

# Plotting
import matplotlib.pyplot as plt

# NumPy
import numpy as np

# Integration with ASE calculator for metatensor atomistic models
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator

# The SOAP-BPNN model contains compiled extensions from rascaline.torch
import rascaline.torch  # noqa


# %%
#
# Next, we initialize the simulation. We first obatin the initial positions based
# on the dataset file which we trained the model on. You can obtain the
# dataset used in this example from our :download:`website
# <../../../../static/ethanol_reduced_100.xyz>`.


atoms = ase.io.read("ethanol_reduced_100.xyz")

# %%
#

ase.visualize.plot.plot_atoms(atoms)
plt.show()

# %%
#
# Our initial coordinates do not include velocities. We Iiitialize the velocities
# according to a Maxwell Boltzman Distribution at 300 K.

ase.md.velocitydistribution.MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# %%
#
# We add or register a exported model as the energy calculator. The model was trained
# using the following training options.
#
# .. literalinclude:: ../../static/options.yaml
#    :language: yaml
#
# As step by step introduction to train this model with these options is given in the
# :ref:`label_basic_usage` tutorial.

atoms.calc = MetatensorCalculator("exported-model.pt")

# %%
#
# Finally we define the integrator which we use to obtain new positions and velocities based on
# our energy calculator. We use a common timestep of 0.5 fs.

integrator = ase.md.VelocityVerlet(atoms, timestep=0.5 * ase.units.fs)


# %%
#
# Run a short simulation:

n_steps = 100

potential_energy = np.zeros(n_steps)
kinetic_energy = np.zeros(n_steps)
total_energy = np.zeros(n_steps)
trajectory = []

for step in range(n_steps):
    # run a single simulation step
    integrator.run(1)

    # collect data about the simulation
    potential_energy[step] = atoms.get_potential_energy()
    kinetic_energy[step] = atoms.get_kinetic_energy()
    total_energy[step] = atoms.get_total_energy()
    trajectory.append(atoms.copy())

# Plot the final configuration:
ase.visualize.plot.plot_atoms(trajectory[-1])
plt.show()

# %%
#
# Plot the evolution of kinetic, potential, and total energy.
# The total energy should approximately be conserved:

plt.plot(potential_energy, label="potential energy")
plt.plot(kinetic_energy, label="kinetic energy")
plt.plot(total_energy, label="total energy")

plt.legend()
plt.show()
