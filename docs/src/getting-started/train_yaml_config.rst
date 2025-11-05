.. _train_yaml_config:

Training YAML Reference
***********************

Overview
========

``metatrain`` uses a YAML file to specify the parameters for model training, accessed
via ``mtt train options.yaml``. In this section, we provide a complete reference for the
parameters provided by the training YAML input. For a minimal example of a YAML input
file, suitable to start a first training, we refer the viewer to the sample YAML file in
the :ref:`Quickstart <label_quickstart>` section.

The YAML input file can be divided into five sections:

- :ref:`Computational Parameters``
- :ref:`Architecture`
- :ref:`Loss`
- :ref:`Data`
- :ref:`WandB Integration`

Computational Parameters
========================

The computational parameters define the computational ``device``, ``precision`` and
``seed``. These parameters are optional.

.. code-block:: yaml

    device: 'cuda' precision: 32 seed: 0

:param device [optional]: The computational device used for model training. The
    metatrain automatically chooses the best option by default. The possible devices
    that can be used, and the best device option, depend on the model architecture. The
    easiest way to use this parameter is to use either either ``cpu``, ``gpu``,
    ``multi-gpu``. Internally, under the choice ``gpu``, the script will automatically
    choose between ``cuda`` or ``mps``.
:param precision [optional]: The base precision for all floats in model training. This
    impacts the datatype used. Possible options are the integers ``64``, ``32`` and
    ``16``, resulting in the datatype used to be ``float64``, ``float32`` and
    ``float16`` respectively. The datatypes that can be supported also depends on the
    model architecture used.
:param seed [optional]: The seed used for non-deterministic operations and is used to
    set the seed for ``numpy.random``, ``random``, ``torch`` and ``torch.cuda``. The
    input must be a non-negative integer. This parameter is important for ensuring
    reproducibility. If not specified, the seed is generated randomly and reported in
    the log.

Architecture
============

The next section of the YAML file would focus on options pertaining the architecture. As
these options, along with their default values, are highly specific to the model
architecture. It is better to instead consult the respective :ref:`architecture
documentation <available-architectures>` page for further details.

Loss
====

Within the architecture section, there is a parameter dedicated to the loss. Due to the
plethora of loss functions used in different ML workflows, it is best to refer to the
page on :ref:`loss functions <loss-functions>` page for further details.

.. _data-section:

Data
====

The final section of the YAML file focuses on options regarding the data used in model
training. This secion can be broken down into three subsections:

- ``training_set``
- ``validation_set``
- ``test_set``

The training set is the data that will be used for model training, the validation set is
the data that will be used to track the generalizability of the model during trainingand
is used to decide on the best model. The test set is only used after training and it is
used to evaluate the model's performance on an unseen dataset after training. Each
subsection has the same parameter configuration. As an example, the configuration of the
training set is as follows:

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



The options for ``training set`` is divided into two categories, ``systems``,
``targets`` and ``extra_data``. ``systems`` refer to the molecular/crystal structures,
which are the inputs to the model. ``targets`` refer to the output that is predicted by
the model. ``extra_data`` refer to any additional data that is required by the loss
function during training. One can also get ``metatrain`` to automatically split the
training data into training and validation sets by providing a float between 0 and 1 for
the ``validation_set`` and ``test_set`` parameters. This float indicates the fraction of
the training data to be used for validation and testing respectively. See the
:ref:`Validation and Test Systems <validation-and-test-systems>` section for more
details.

Systems YAML
------------

For the ``systems`` category:

:param read_from: The path to the file containing system data
:param reader [optional]: The reader library to use for parsing, currently supports
    ``ase`` and ``metatensor``. If ``null`` or not provided, the reader will be guessed
    from the file extension, ``.xyz`` and ``.extxyz`` will be read by ``ase`` and
    ``.mts`` will be read by ``metatensor``.
:param length_unit  [optional]: The unit of lengths in the system file, optional but
    highly recommended for running simulations.

A single string in this section automatically expands, using the string as the
``read_from`` parameter. This means that

.. code-block:: yaml

        systems:
            read_from: dataset.xyz
            reader: null
            length_unit: null

can be condensed into

.. code-block:: yaml

        systems: dataset.xyz

Targets YAML
------------

In the ``targets`` category, one can define any number of target sections, each with a
unique name. The name of the target should either be a standard output of ``metatomic``
(see https://docs.metatensor.org/metatomic/latest/outputs/index.html) or begin with
``mtt::``, for instance ``mtt::dos`` for the electronic density of states in the full
example above.

The parameters for each target section are as follows:

:param quantity [optional]: The quantity the target represents(e.g., ``energy``,
    ``dipole``). Currently only ``energy`` is supported. Defaults to ``""``.
:param read_from [optional]: The path to the file containing the target data, defaults
    to ``systems.read_from`` path if not provided.
:param reader [optional]: The reader library to use for parsing, behaves the same way as
    ``systems.reader``
:param key [optional]: The key for reading from the file, defaulting to the target
    section's name if not provided.
:param unit [optional]: The unit of the target, optional but highly recommended for
    running simulations. Defaults to ``""``.
:param per_atom [optional]: Whether the target is extensive (i.e., scales with the
    number of atoms). If ``true``, the target value will be divided by the number of
    atoms in the system. Defaults to ``false``.
:param type [optional]: This field specifies the type of the target. Possible values are
    ``scalar``, ``cartesian``, and ``spherical``. For detailed information on the
    ``type`` field, see the following page on :ref:`Fitting Generic Targets
    <fitting-generic-targets>`.
:param num_subtargets [optional]: This field specifies the number of sub-targets that
    need to be learned as part of this target. They are treated as entirely equivalent
    by models in metatrain and will often be represented as outputs of the same neural
    network layer. A common use case for this field is when you are learning a
    discretization of a continuous target, such as the grid points of a function. In the
    example above, there are 4000 sub-targets for the density of states (DOS). In
    metatensor, these correspond to the number of properties of the target. Defaults to
    1
:param forces: Gradient subsections. See the following :ref:`gradient-subsection` for
    parameters.
:param stress: Gradient subsections. See the following :ref:`gradient-subsection` for
    parameters.
:param virial: Gradient subsections. See the following :ref:`gradient-subsection` for
    parameters.

A single string in a target section automatically expands, using the string as the
``read_from`` parameter.

.. _gradient-subsection:

Gradient Subsection
^^^^^^^^^^^^^^^^^^^

Each gradient subsection (like ``forces`` or ``stress``) has similar parameters:

:param read_from [optional]: The path to the file for gradient data. Defaults to
    ``targets.read_from`` if not provided.
:param reader [optional]: The reader library to use for parsing, behaves the same way as
    ``systems.reader``.
:param key [optional]: The key for reading from the file, defaulting to the subsection's
    name if not provided.

A single string in a gradient section automatically expands, using the string as the
``read_from`` parameter.

Sections set to ``true`` or ``on`` automatically expand with default parameters. A
warning is raised if requisite data for a gradient is missing, but training proceeds
without them. For instance,

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
validation and 20% for testing:

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

WandB Integration
=================

Optional section dealing with integration with `Weights and Biases (wandb)
<https://wandb.ai/site/>`_ logging. Leaving this section blank will simply disable wandb
integration. The parameters for this section is the same as that in `wandb.init
<https://docs.wandb.ai/ref/python/init/>`_. Here we provide a minimal example for the
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
    <https://docs.wandb.ai/ref/cli/wandb-login/>`_ for details on the setup.
