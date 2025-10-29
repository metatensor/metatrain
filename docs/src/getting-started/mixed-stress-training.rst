.. _mixed_stress_training:

Training with Mixed Stress Structures
======================================

Overview
--------

In many practical scenarios, you may need to train a model on datasets containing
different types of structures:

- **Bulk materials** with well-defined stress tensors
- **Molecules** where stress is not physically meaningful
- **Slabs** (surfaces) with non-periodic boundary conditions where stress is undefined

``metatrain`` supports training on such mixed datasets by allowing structures with and
without stress information in the same training set. Structures without stress should
have their stress values set to ``NaN`` (Not a Number).

Creating a Mixed Dataset
-------------------------

Here's a complete example showing how to create a dataset containing bulk structures,
molecules, and slabs:

.. code-block:: python

    import warnings
    import ase.build
    import ase.io
    import numpy as np
    from ase.calculators.emt import EMT

    calculator = EMT()
    structures = []

    # Create bulk structures with valid stress
    for _ in range(5):
        bulk = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
        bulk.rattle(0.01)  # Small perturbation to make structures different
        bulk.calc = calculator
        bulk.info["energy"] = bulk.get_potential_energy()
        bulk.arrays["forces"] = bulk.get_forces()
        bulk.info["stress"] = bulk.get_stress(voigt=False)
        bulk.calc = None
        structures.append(bulk)

    # Create molecules with NaN stress (stress not defined for molecules)
    for i in range(5):
        molecule = ase.Atoms("Cu2", positions=[[0, 0, 0], [2.5 + 0.1 * i, 2.5, 2.5]])
        molecule.calc = calculator
        molecule.info["energy"] = molecule.get_potential_energy()
        molecule.arrays["forces"] = molecule.get_forces()
        molecule.info["stress"] = np.full((3, 3), np.nan)  # Use NaN for undefined stress
        molecule.calc = None
        structures.append(molecule)

    # Create slabs with NaN stress (stress not defined for slabs with non-periodic BC)
    for _ in range(5):
        slab = ase.build.fcc111("Cu", size=(2, 2, 4), vacuum=10.0)
        slab.pbc = (True, True, False)  # Periodic in xy, non-periodic in z
        slab.rattle(0.01)  # Small perturbation
        slab.calc = calculator
        slab.info["energy"] = slab.get_potential_energy()
        slab.arrays["forces"] = slab.get_forces()
        slab.info["stress"] = np.full((3, 3), np.nan)  # Use NaN for undefined stress
        slab.calc = None
        structures.append(slab)

    # Write structures to file
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ase.io.write("mixed_structures.xyz", structures)

Configuration for Mixed Stress Training
----------------------------------------

Once you have created your mixed dataset, you can configure your training with the
following ``options.yaml`` file:

.. code-block:: yaml

    architecture:
        name: soap_bpnn  # or any other architecture
        model:
            cutoff: 5.0
        training:
            batch_size: 4
            num_epochs: 100

    training_set:
        systems:
            read_from: mixed_structures.xyz
            length_unit: angstrom
        targets:
            energy:
                key: energy
                unit: eV
                forces:
                    key: forces
                stress:
                    key: stress

    validation_set: 0.1
    test_set: 0.1

Key Points
----------

1. **Use NaN for undefined stress**: When stress is not defined for a structure (e.g.,
   molecules or slabs with non-periodic boundaries), set the stress tensor to
   ``np.full((3, 3), np.nan)``.

2. **Mixed training is automatic**: ``metatrain`` will automatically handle the mixed
   dataset during training, only computing stress loss for structures where stress is
   defined (not NaN).

3. **All structures need stress field**: Even if stress is not defined for some
   structures, the stress field must be present in the dataset. Use NaN values for
   structures where stress is undefined.

4. **Force training works normally**: Forces can be trained for all structure types
   (bulk, molecules, and slabs) without any special handling.

Common Use Cases
----------------

This feature is particularly useful when:

- Training models that need to work across different system types (bulk + molecules)
- Working with surface calculations where stress is only meaningful in periodic
  directions
- Combining datasets from different sources where stress information may not be
  available for all structures
- Training on mixed experimental and simulation data where stress measurements may not
  always be available

.. note::

   The training will proceed normally, with stress contributions to the loss only
   coming from structures where stress is well-defined (non-NaN values).
