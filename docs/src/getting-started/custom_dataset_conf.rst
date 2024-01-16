Customize a Dataset Configuration
=================================

Overview
--------
The main task in setting up a training procedure with `metatensor-models` is to provide
files for training, validation, and testing datasets. Our system allows flexibility in
parsing data for training. Mandatory sections in the `options.yaml` file include:

- ``training_set``
- ``test_set``
- ``validation_set``

Each section can follow a similar structure, with shorthand methods available to
simplify dataset definitions.

Minimal Configuration Example
-----------------------------
Below is the simplest form of these sections:

.. code-block:: yaml

    training_set: "dataset.xyz"
    test_set: 0.1
    validation_set: 0.1

This configuration parses all information from ``dataset.xyz``, with 20% of the training
set randomly selected for testing and validation (10% each).

Expanded Configuration Format
-----------------------------
The train script automatically expands the ``training_set`` section into the following
format, which is also valid for initial input:

.. code-block:: yaml

    training_set:
        structures:
            read_from: dataset.xyz
            file_format: .xyz
            unit: null
        targets:
            energy:
                quantity: energy
                read_from: dataset.xyz
                file_format: .xyz
                key: energy
                unit: null
                forces:
                    read_from: dataset.xyz
                    file_format: .xyz
                    key: forces
                stress:
                    read_from: dataset.xyz
                    file_format: .xyz
                    key: stress
                virial: false
    test_set: 0.1
    validation_set: 0.1

Understanding the YAML Block
----------------------------
The ``training_set`` is divided into sections ``structures`` and ``targets``:

Structures Section
------------------
Describes the structure data like positions and cell information.

:param read_from: The file containing structure data.
:param file_format: The file format, guessed from the suffix if ``null`` or not
    provided.
:param unit: The unit of lengths, optional but recommended for simulations.

A single string in this section automatically expands, using the string as the
``read_from`` parameter.

.. note::

   Metatensor-models does not convert units during training or evaluation. Units are
   necessary for MD simulations.

Targets Section
---------------
Allows defining multiple target sections, each with a unique name.

- Commonly, a section named ``energy`` is defined, which is essential for MD
  simulations.
- For other target sections, gradients are disabled by default.

Target section parameters include:

:param quantity: The target's quantity (e.g., energy, dipole).
:param read_from: The file for target data, defaults to the ``structures.read_from``
  file if not provided.
:param file_format: The file format, guessed from the suffix if not provided.
:param key: The key for reading from the file, defaulting to the target section's name
  if not provided.
:param unit: The unit of the target.
:param forces: Gradient sections. See :ref:`gradient-section` for parameters.
:param stress: Gradient sections. See :ref:`gradient-section` for parameters.
:param virial: Gradient sections. See :ref:`gradient-section` for parameters.

A single string in a target section automatically expands, using the string as the
``read_from`` parameter.

.. _gradient-section:

Gradient Sections
-----------------
Each gradient section (like ``forces`` or ``stress``) has similar parameters:

:param read_from: The file for gradient data.
:param file_format: The file format, guessed from the suffix if not provided.
:param key: The key for reading from the file.

Sections set to ``true`` or ``on`` automatically expand with default parameters.

Energy Section
--------------
The ``energy`` section is mandatory for MD simulations, with forces and stresses enabled
by default.

- A warning is raised if requisite data is missing, but training proceeds without them.
- Setting a ``virial`` section automatically disables the ``stress`` section in the
  ``energy`` target.

.. note::

   Metatensor-models ignores unknown keys in these sections during dataset parsing.
