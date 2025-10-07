.. _train_yaml_config:

Training YAML Reference
************************************
Overview
===================
``metatrain`` uses a YAML file to specify the parameters for model training, 
accessed via ``mtt train options.yaml``. In this section, we provide a complete reference 
for the parameters provided by the training YAML input.

The YAML input file can be divided into five sections: 
- Computational Parameters
- wandb integration
- ``architecture``
- ``loss``
- Data

Computational Parameters
======================================
The computational parameters define the computational device, precision and seed. These parameters are optional.

.. code-block:: yaml

    device: 'cuda'
    precision: 32
    seed: 0

:param device [optional]: The computational device used for model training. The script automatically 
    chooses the best option by default. The possible devices that can be used, and the best device option, 
    depend on the model architecture. The easiest way to use this parameter is to use either either ``cpu``, ``gpu``,
    ``multi-gpu``. Internally, under the choice ``gpu``, the script will automatically choose between ``cuda`` or ``mps``.
:param precision [optional]: The base precision for all floats in model training. This impacts the datatype used. Possible 
    options are the integers ``64``, ``32`` and ``16``, resulting in the datatype used to be ``float64``, ``float32`` and 
    ``float16`` respectively. The datatypes that can be supported also depends on the model architecture used.
:param seed [optional]: The seed used for non-deterministic operations and is used to set the seed for ``numpy.random``, 
    ``random``, ``torch`` and ``torch.cuda``. The input must be a non-negative integer. This parameter is important for ensuring 
    reproducibility. If not specified, the seed is generated randomly. 

wandb integration
===================
The next set of parameters are also optional and deals with integration with Weights and Biases (wandb) logging. Leaving this 
section blank will simply disable wandb integration. The parameters for this section is the same as that in 
`wandb.init <https://docs.wandb.ai/ref/python/init/>`_. Here we provide a minimal example for the YAML input

.. code-block:: yaml

    wandb:
        project: my_project
        name: my_run_name
        tags:
        - tag1
        - tag2
        notes: This is a test run

All parameters of your options file will be automatically added to the wandb run so
you don't have to set the ``config`` parameter.

.. important::

    You need to install wandb with ``pip install wandb`` if you want to use this
    logger. **Before** running also set up your credentials with ``wandb login``
    from the command line. See `wandb login
    documentation <https://docs.wandb.ai/ref/cli/wandb-login/>`_ for details on the
    setup.

Architecture
===================
The next section of the YAML file would focus on options pertaining the architecture. As these options, along with 
their default values, are highly specific to the model architecture. It is better to instead consult the respective 
:ref:`architecture documentation <available-architectures>` page for further details.

Loss
===================
Within the architecture section, there is a parameter dedicated to the loss. Due to the plethora of loss functions 
used in different ML workflows, it is best to refer to the page on :ref:`loss functions <loss-functions>` page for further details.

Data
===================
The final section of the YAML file focuses on options regarding the data used in model training. This secion can be broken 
down into three subsections:

- ``training_set``
- ``validation_set``
- ``test_set``

The training set is the data that will be used for model training, the validation set is the data that will be used to 
track the generalizability of the model during trainingand is used to decide on the best model. The test set is only used after 
training and it is used to evaluate the model's performance on an unseen dataset after training. Each subsection has the same 
parameter configuration. As an example, the configuration of the training set is as follows:

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
                read_from: dataset.xyz
                reader: ase
                key: forces
                unit: null
        extra_data:

            blah


The options for ``training set`` is divided into two categories, ``systems`` and ``targets``. ``systems`` refer to the molecular/crystal structures, 
which are the inputs to the model. ``targets`` refer to the output that is predicted by the model. 

Systems YAML
----------------
For the ``systems`` category:

:param read_from: The path to the file containing system data
:param reader [optional]: The reader library to use for parsing, currently supports ``ase`` and ``metatensor``. If ``null`` or not provided, 
    the reader will be guessed from the file extension, ``.xyz`` and ``.extxyz`` will be read by ``ase`` and ``.mts`` will be read by 
    ``metatensor``.
:param length_unit  [optional]: The unit of lengths in the system file, optional but highly recommended for running simulations.

A single string in this section automatically expands, using the string as the ``read_from`` parameter. This means that

.. code-block:: yaml
        systems:
            read_from: dataset.xyz
            reader: null
            length_unit: null

can be condensed into 
.. code-block:: yaml
        systems: dataset.xyz

Targets YAML
----------------
In the ``targets`` category, one can define any number of target sections, each with a unique name. The name of the target should either 
be a standard output of ``metatomic`` (see https://docs.metatensor.org/metatomic/latest/outputs/index.html) or begin with ``mtt::``.

The parameters for each target section are as follows:
:param quantity: The quantity the target represents(e.g., ``energy``, ``dipole``). Currently only
    ``energy`` is supported.
:param read_from: The path to the file containing the target data, ``systems.read_from``
    path if not provided.
:param reader: The reader library to use for parsing, behaves the same way as ``systems.reader``
:param key: The key for reading from the file, defaulting to the target section's name
    if not provided.
:param unit: The unit of the target, optional but highly recommended for running
    simulations.
:param forces: Gradient subsections. See :ref:`gradient-subsection` for parameters.
:param stress: Gradient subsections. See :ref:`gradient-subsection` for parameters.
:param virial: Gradient subsections. See :ref:`gradient-subsection` for parameters.

A single string in a target section automatically expands, using the string as the
``read_from`` parameter.

Gradient Subsection
^^^^^^^^^^^^^^^^^^^^
Each gradient subsection (like ``forces`` or ``stress``) has similar parameters:

:param read_from: The path to the file for gradient data.
:param reader: The reader library to use for parsing, behaves the same way as ``systems.reader``
:param key: The key for reading from the file.

A single string in a gradient section automatically expands, using the string as the
``read_from`` parameter.

Sections set to ``true`` or ``on`` automatically expand with default parameters. A
warning is raised if requisite data for a gradient is missing, but training proceeds
without them.

.. note::

   Unknown keys are ignored and not deleted in all sections during dataset parsing.


Minimal Configuration Example
-----------------------------
Below is the simplest form of these sections:

.. code-block:: yaml

    training_set: "dataset.xyz"
    test_set: 0.1
    validation_set: 0.1

This configuration parses all information from ``dataset.xyz``, with 20% of the training
set randomly selected for testing and validation (10% each). The selected indices for
the training, validation and test subset will be available in the ``outputs`` directory.


Understanding the YAML Block
----------------------------
The ``training_set`` is divided into sections ``systems`` and ``targets``:

Systems Section
^^^^^^^^^^^^^^^
Describes the system data like positions and cell information.

:param read_from: The file containing system data.
:param reader: The reader library to use for parsing, guessed from the file extension if
    ``null`` or not provided.
:param length_unit: The unit of lengths, optional but highly recommended for running
    simulations.

A single string in this section automatically expands, using the string as the
``read_from`` parameter.

.. note::

   ``metatrain`` does not convert units during training or evaluation. Units are
   only required if model should be used to run MD simulations.

Targets Section
^^^^^^^^^^^^^^^
Allows defining multiple target sections, each with a unique name.

- Commonly, a section named ``energy`` should be defined, which is essential for running
  molecular dynamics simulations. For the ``energy`` section gradients like ``forces``
  and ``stress`` are enabled by default.
- Other target sections can also be defined, as long as they are prefixed by
  ``mtt::``. For example, ``mtt::free_energy``. In general, all targets that are
  not standard outputs of ``metatomic`` (see
  https://docs.metatensor.org/metatomic/latest/outputs/index.html) should be
  prefixed by ``mtt::``.

Target section parameters include:

:param quantity: The target's quantity (e.g., ``energy``, ``dipole``). Currently only
    ``energy`` is supported.
:param read_from: The file for target data, defaults to the ``systems.read_from``
  file if not provided.
:param reader: The reader library to use for parsing, guessed from the file extension if
    ``null`` or not provided.
:param key: The key for reading from the file, defaulting to the target section's name
  if not provided.
:param unit: The unit of the target, optional but highly recommended for running
    simulations.
:param forces: Gradient sections. See :ref:`gradient-section` for parameters.
:param stress: Gradient sections. See :ref:`gradient-section` for parameters.
:param virial: Gradient sections. See :ref:`gradient-section` for parameters.

A single string in a target section automatically expands, using the string as the
``read_from`` parameter.

.. _gradient-saasection:

Gradient Section
^^^^^^^^^^^^^^^^
Each gradient section (like ``forces`` or ``stress``) has similar parameters:

:param read_from: The file for gradient data.
:param reader: The reader library to use for parsing, guessed from the file extension if
    ``null`` or not provided.:param key: The key for reading from the file.

A single string in a gradient section automatically expands, using the string as the
``read_from`` parameter.

Sections set to ``true`` or ``on`` automatically expand with default parameters. A
warning is raised if requisite data for a gradient is missing, but training proceeds
without them.

.. note::

   Unknown keys are ignored and not deleted in all sections during dataset parsing.

Multiple Datasets
-----------------
For some applications, it is required to provide more than one dataset for model
training. ``metatrain`` supports stacking several datasets together using the
``YAML`` list syntax, which consists of lines beginning at the same indentation level
starting with a ``"- "`` (a dash and a space)


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
element in ``training_set``

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

   Even though parsing several datasets is supported by the library, it may not
   work with every architecture. Check your :ref:`desired architecture
   <available-architectures>` if they **support multiple datasets**.

In the next tutorials we explain and show how to set some advanced global training
parameters.

Datasets requiring additional data
----------------------------------
Some targets require additional data to be passed to the loss function for training.
For example, training a model to predict the polarization for extended systems under
periodic boundary conditions might require the quantum of polarization to be provided
for each system in the dataset.

``metatrain`` supports passing additional data in the ``options.yaml`` file.
For example, if you want to train a polarization model, you can add the following
section to your ``options.yaml`` file:

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
