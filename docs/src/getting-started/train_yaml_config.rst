.. _train_yaml_config:

Training YAML Reference
=======================

Overview
--------

``metatrain`` uses a YAML file to configure model training. YAML is a human-readable text
format for configuration - think of it as a more user-friendly alternative to JSON or XML.
You run training with ``mtt train options.yaml``.

**New to YAML?** Don't worry! It's straightforward:
- Use indentation (spaces) to show structure
- Use ``key: value`` pairs to specify settings
- Use ``#`` for comments

This page provides a complete reference for all available parameters. **For your first
training**, start with the minimal example in the :ref:`Quickstart <label_quickstart>`
section, then come back here to understand what each option does.

The YAML file has five main sections:

- :ref:`computational-parameters-section` - Where and how to run computations
- :ref:`architecture-section` - Which machine learning model to use
- :ref:`loss-section` - What to optimize during training
- :ref:`data-section` - Your training data and targets
- :ref:`wandb-integration-section` - Optional logging to Weights & Biases

.. _computational-parameters-section:

Computational Parameters
------------------------

These parameters control **where and how** your model is trained. All are **optional** -
metatrain will choose sensible defaults if you don't specify them.

.. code-block:: yaml

    device: cuda      # Use GPU (optional - auto-detected by default)
    precision: 32     # Use 32-bit floating point numbers (optional - default is 32)
    seed: 0           # Random seed for reproducibility (optional)

Device (where to run training)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:param device [optional]: Controls whether to use CPU or GPU for training.

**Recommended values:**

- ``cpu`` - Use the CPU (slower but works everywhere)
- ``gpu`` - Use GPU if available (much faster, especially for large models like PET)
- ``multi-gpu`` - Use multiple GPUs (for advanced users with large datasets)

If not specified, metatrain automatically detects and uses the best available option.

.. note::
   GPU training is **much faster** (often 10-100x) than CPU for large models. If you have
   a CUDA-compatible NVIDIA GPU or Apple Silicon Mac, the ``gpu`` option will
   automatically use it.

Precision (numerical accuracy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:param precision [optional]: Controls the numerical precision used during training.

**Options:**

- ``64`` - Double precision (``float64``) - most accurate but slowest
- ``32`` - Single precision (``float32``) - good balance (default and recommended)
- ``16`` - Half precision (``float16``) - fastest but less accurate

**Recommendation:** Keep the default (32-bit). Only change if you have specific reasons:

- Use ``64`` if you encounter numerical stability issues
- Use ``16`` only if you're experienced and need maximum speed with large models

Seed (reproducibility)
^^^^^^^^^^^^^^^^^^^^^^

:param seed [optional]: A number that controls random operations, ensuring reproducible
    results.

Setting the seed means you'll get the **same results every time** you run training with
the same data and options. This is useful for:

- Comparing different hyperparameters fairly
- Reproducing results from papers
- Debugging

**Example:** ``seed: 42`` (any non-negative integer works)

If not specified, a random seed is generated and saved in your log files, so you can still
reproduce the training later if needed.

.. _architecture-section:

Architecture
------------

The next section of the YAML file would focus on options pertaining the architecture. As
these options, along with their default values, are highly specific to the model
architecture. It is better to instead consult the respective :ref:`architecture
documentation <available-architectures>` page for further details.

.. _loss-section:

Loss
----

Within the architecture section, there is a parameter dedicated to the loss. Due to the
plethora of loss functions used in different ML workflows, it is best to refer to the
page on :ref:`loss functions <loss-functions>` page for further details.

.. _data-section:

Data
----

This is where you tell metatrain about your **training data** - the atomic structures and
properties that the model will learn from.

Understanding the Three Data Splits
-----------------------------------

Your data is divided into three parts, each with a specific purpose:

- **Training set**: Examples the model learns from (typically 70-80% of your data). The
  model adjusts itself to match these examples.

- **Validation set**: Used during training to check if the model is improving or
  overfitting (typically 10-15%). Think of this as a "practice test" - the model doesn't
  learn from it directly, but we use it to monitor progress.

- **Test set**: Used **only after training** to evaluate final performance on completely
  unseen data (typically 10-15%). This is the "final exam" that tells you how well your
  model really works.

.. tip::
   **Easy splitting:** You can let metatrain automatically split your data! Just provide
   your full dataset as ``training_set`` and specify fractions for the other sets:

   .. code-block:: yaml

       training_set: "my_data.xyz"
       validation_set: 0.1  # Use 10% of data for validation
       test_set: 0.1        # Use 10% of data for testing

   Metatrain will randomly select structures for each set and save the indices.

Configuring Your Training Data
------------------------------

Each data split (training, validation, test) uses the same configuration format. Here's a
detailed example for the training set:

.. code-block:: yaml

    training_set:
        systems:
            read_from: dataset.xyz
            reader: ase
            length_unit: null
        targets:
            energy:
                quantity: energy
                read_from: dataset.xyz
                reader: ase
                key: energy
                unit: null
                per_atom: True
                type: scalar
                num_subtargets: 1
                forces:
                    read_from: dataset.xyz
                    reader: ase
                    key: forces
                stress:
                    read_from: dataset.xyz
                    reader: ase
                    key: stress
            non_conservative_forces:
                quantity: null
                read_from: nonconservative_force.mts
                reader: metatensor
                key: forces
                unit: null
                per_atom: True
                type:
                    cartesian:
                        rank: 1
                num_subtargets: 1
            mtt::dos:
                quantity: null
                read_from: DOS.mts
                reader: metatensor
                key: dos
                unit: null
                per_atom: False
                type: scalar
                num_subtargets: 4000
        extra_data:
            mtt::dos_mask:
                quantity: null
                read_from: dataset.xyz
                reader: ase
                key: dos_mask
                unit: null
                per_atom: False
                type: scalar
                num_subtargets: 4000

A training dataset consists of three components:

- **systems**: The atomic structures (positions, elements, cell) - the "inputs"
- **targets**: The properties to predict (energies, forces, etc.) - the "outputs"
- **extra_data**: Additional information needed by some advanced loss functions (optional)

Systems YAML (Atomic Structures)
--------------------------------

The ``systems`` section tells metatrain where to find your atomic structures.

.. code-block:: yaml

    systems:
        read_from: "my_structures.xyz"  # Path to your file
        reader: ase                     # How to read it (optional)
        length_unit: angstrom           # Unit of atomic positions (optional but recommended)

Parameters:

:param read_from: **Required.** Path to the file containing atomic structures. Can be:

    - XYZ file (``structures.xyz``) - most common
    - Extended XYZ file (``structures.extxyz``)
    - ASE database (``database.db``)
    - Metatensor file (``data.mts``)

:param reader [optional]: Which library to use for reading the file. Options:

    - ``ase`` - For XYZ, extended XYZ, and ASE databases (recommended for beginners)
    - ``metatensor`` - For metatensor files (``.mts``)

    If not specified, metatrain guesses based on file extension (.xyz → ase, .mts →
    metatensor).

:param length_unit [optional]: The unit of atomic positions in your file. Common values:

    - ``angstrom`` - Most common (Ångströms)
    - ``bohr`` - Atomic units

    **Highly recommended** to specify this, especially if you'll run molecular dynamics
    simulations!

**Shorthand notation:** If you only need to specify the filename, you can write:

.. code-block:: yaml

    systems: "my_structures.xyz"

This is equivalent to the full form above with defaults.

Targets YAML (Properties to Predict)
------------------------------------

The ``targets`` section defines what properties you want the model to predict. Each target
has a name and configuration. The most common target is ``energy``.

.. code-block:: yaml

    targets:
        energy:                         # Standard target name
            quantity: energy            # What physical quantity this is
            read_from: "my_data.xyz"    # Where to find the values (optional if same as systems)
            key: "U0"                   # Column/key name in the file
            unit: "eV"                  # Energy unit
            forces: on                  # Also train on forces (optional)
            stress: on                  # Also train on stress (optional)

**Target names:** Use standard names (``energy``, ``dipole``) or custom names starting
with ``mtt::`` (e.g., ``mtt::dos`` for density of states). See
https://docs.metatensor.org/metatomic/latest/outputs/index.html for standard names.

Parameters for each target:

:param quantity [optional]: The physical quantity (``energy`` is most common and best
    supported). Tells the model how to handle the target.

:param read_from [optional]: File containing target values. If not specified, uses the
    same file as ``systems``. This is convenient when all your data is in one XYZ file.

:param reader [optional]: How to read the file (``ase`` or ``metatensor``). Auto-detected
    from file extension if not provided.

:param key [optional]: The name of the property in your data file. For XYZ files, this is
    the name in the ``info`` dictionary (for scalars) or ``arrays`` dictionary (for
    per-atom properties). If not specified, uses the target name (e.g., ``energy``).

:param unit [optional]: The unit of your target values. Common values:

    - For energy: ``eV``, ``kcal/mol``, ``hartree``
    - For forces: ``eV/angstrom``, ``kcal/mol/angstrom``
    - For stress: ``eV/angstrom^3``, ``GPa``

    **Highly recommended** to specify! Critical for using models in simulations.

:param per_atom [optional]: Set to ``true`` if the target is **extensive** (scales with
    system size).

    - ``true``: Total energy divided by number of atoms (common for ML)
    - ``false``: Total energy (default)

    Example: If your data has total energies but you want to predict energy per atom, set
    ``per_atom: true``.

:param type [optional]: The mathematical type of target (``scalar``, ``cartesian``, or
    ``spherical``). Usually you don't need to specify this. See :ref:`Fitting Generic
    Targets <fitting-generic-targets>` for advanced usage.

:param num_subtargets [optional]: Number of related properties to predict together
    (advanced). Example: 4000 energy grid points for density of states. Defaults to 1.

:param forces: Include force training (see :ref:`gradient-subsection` below)
:param stress: Include stress training (see :ref:`gradient-subsection` below)
:param virial: Include virial training (see :ref:`gradient-subsection` below)

**Shorthand notation:** For simple cases, just specify the file:

.. code-block:: yaml

    targets:
        energy: "my_data.xyz"

This reads the ``energy`` key from the file with default settings.

.. _gradient-subsection:

Gradient Subsection (Forces, Stress, Virial)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**What are gradients?** Forces are the negative gradient of energy with respect to atomic
positions. Stress is related to gradients with respect to cell dimensions. Training on
gradients usually improves model accuracy significantly!

Each gradient subsection (``forces``, ``stress``, ``virial``) has the same parameters:

:param read_from [optional]: File containing gradient data. If not specified, uses the
    same file as the target (usually your XYZ file).

:param reader [optional]: How to read the file (``ase`` or ``metatensor``). Auto-detected
    if not specified.

:param key [optional]: Property name in your file. Defaults to the gradient name
    (``forces``, ``stress``, or ``virial``).

**Shorthand notations:**

1. **Enable with default settings** (most common):

   .. code-block:: yaml

       energy:
           forces: on    # or 'true'
           stress: on

   This tells metatrain to look for forces and stress in your data file using standard
   names.

2. **Specify file explicitly** (if in a different file):

   .. code-block:: yaml

       energy:
           forces: "my_forces.xyz"
           stress: "my_stress.xyz"

**What happens if data is missing?** If you specify ``forces: on`` but metatrain doesn't
find force data, it will show a warning and continue training without forces. This lets
you use the same configuration file for different datasets.

Example of common usage:

.. code-block:: yaml

        targets:
            energy:
                quantity: energy
                read_from: dataset.xyz
                reader: ase
                key: energy
                unit: null
                forces:
                    read_from: dataset.xyz
                    reader: ase
                    key: forces
                stress:
                    read_from: dataset.xyz
                    reader: ase
                    key: stress

can be condensed into

.. code-block:: yaml

        targets:
            energy:
                quantity: energy
                read_from: dataset.xyz
                reader: ase
                key: energy
                unit: null
                forces: on
                stress: on

.. note::

   Unknown keys are ignored and not deleted in all sections during dataset parsing.

Datasets requiring additional data
----------------------------------

Some targets require additional data to be passed to the loss function for training. In
the example above, we included the mask for the density of states, which defines the
regions of the DOS that are well-defined based on the eigenvalues of the underlying
electronic structure calculation. This is important when the DOS is computed over a
finite energy range, as the DOS near the edges of this range may be inaccurate due to
the lack of states computed beyond this range. ``metatrain`` supports passing additional
data in the ``options.yaml`` file. This can be seen in the ``extra_data`` section of the
full example above.

As another example, training a model to predict the polarization for extended systems
under periodic boundary conditions might require the quantum of polarization to be
provided for each system in the dataset. For this, you can add the following section to
your ``options.yaml`` file:

.. code-block:: yaml

    training_set:
        systems:
            read_from: dataset_0.xyz
            length_unit: angstrom
        targets:
            mtt::polarization:
                read_from: polarization.mts
        extra_data:
            polarization_quantum:
                read_from: polarization_quantum.mts

.. warning::

   While the ``extra_data`` section can always be present, it will typically be ignored
   unless using specific loss functions. If the loss function you picked does not
   support the extra data, it will be ignored.

The ``extra_data`` section supports the same parameters as the target sections. In this
case, we have also read the targets and extra data from files other than the systems
file.

.. _validation-and-test-systems:

Validation and Test Systems
---------------------------

The validation and test set sections have the same structure as the training set
section. However, instead of specifying the ``systems`` and ``targets`` subsections, one
can simply provide a float between 0 and 1, which indicates the fraction of the training
set to be randomly selected for validation and testing respectively. For example,
setting ``validation_set: 0.1`` will randomly select 10% of the training set for
validation. The selected indices for the training, validation and test subset will be
available in the ``outputs`` directory.

As an example, the following configuration would use 10% of the training set for
validation and 20 % for testing:

.. code-block:: yaml

    training_set: "dataset.xyz" validation_set: 0.1 test_set: 0.2

Using Multiple Files for Training
---------------------------------

For some applications, it is simpler to provide more than one dataset for model
training. ``metatrain`` supports stacking several datasets together using the ``YAML``
list syntax, which consists of lines beginning at the same indentation level starting
with a ``"- "`` (a dash and a space)

.. code-block:: yaml

    training_set:
        - systems:
              read_from: dataset_0.xyz
              length_unit: angstrom
          targets:
              energy:
                  quantity: energy
                  key: my_energy_label0
                  unit: eV
        - systems:
              read_from: dataset_1.xyz
              length_unit: angstrom
          targets:
              energy:
                  quantity: energy
                  key: my_energy_label1
                  unit: eV
              free-energy:
                  quantity: energy
                  key: my_free_energy
                  unit: hartree
    test_set: 0.1
    validation_set: 0.1

The required test and validation splits are performed consistently for each element
element in ``training_set`` .

The ``length_unit`` has to be the same for each element of the list. If target section
names are the same for different elements of the list, their unit also has to be the
same. In the the example above the target section ``energy`` exists in both list
elements and therefore has the the same unit ``eV``. The target section ``free-energy``
only exists in the second element and its unit does not have to be the same as in the
first element of the list.

Typically the global atomic types the the model is defined for are inferred from the
training and validation datasets. Sometimes, due to shuffling of datasets with low
representation of some types, these datasets may not contain all atomic types that you
want to use in your model. To explicitly control the atomic types the model is defined
for, specify the ``atomic_types`` key in the ``architecture`` section of the options
file:

.. code-block:: yaml

    architecture:
        name: pet
        model:
            cutoff: 5.0
        training:
            batch_size: 32
            epochs: 100
        atomic_types: [1, 6, 7, 8, 16]  # i.e. for H, C, N, O, S

.. warning::

   Even though parsing several datasets is supported by the library, it may not work
   with every architecture. Check your :ref:`desired architecture
   <available-architectures>` if they **support multiple datasets**.

.. _wandb-integration-section:

WandB Integration
-----------------

Optional section dealing with integration with `Weights and Biases (wandb)
<https://wandb.ai>`_ logging. Leaving this section blank will simply disable wandb
integration. The parameters for this section is the same as that in `wandb.init
<https://docs.wandb.ai/models/ref/python/functions/init>`_. Here we provide a minimal example for the
YAML input

.. code-block:: yaml

    wandb:
        project: my_project
        name: my_run_name
        tags:
          - tag1
          - tag2
        notes: This is a test run

All parameters of your ``options.yaml`` file will be automatically added to the wandb
run so you don't have to set the ``config`` parameter.

.. important::

    You need to install wandb with ``pip install wandb`` if you want to use this logger.
    **Before** running also set up your credentials with ``wandb login`` from the
    command line. See `wandb login documentation
    <https://docs.wandb.ai/ref/cli/wandb-login>`_ for details on the setup.
