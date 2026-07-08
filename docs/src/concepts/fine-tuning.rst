.. _label_fine_tuning_concept:

Fine-tune a pre-trained model
=============================

Fine-tuning starts a new training run from an existing checkpoint. The model
architecture and weights are loaded from the checkpoint, while the optimizer and
learning-rate scheduler are initialized from the new options file. This is different
from restarting an interrupted training run, where the optimizer and scheduler states
are restored as well.

Not every architecture supports fine-tuning, and the exact options depend on the
architecture. To check if an architecture supports fine-tuning, see whether its training
options include ``architecture.training.finetune`` in the :ref:`architecture reference
<available-architectures>`. The examples use the PET syntax.

Once the options file is set up, start the run with ``mtt train options.yaml``. There is
a complete example in the tutorial section:
:ref:`sphx_glr_generated_examples_0-beginner_02-fine-tuning.py`.

Choosing a strategy
-------------------

The ``method`` option controls which parameters are trained after loading the
checkpoint.

.. image:: /../static/images/fine-tuning.svg
   :class: only-light
   :width: 700px
   :align: center

.. image:: /../static/images/fine-tuning_dark.svg
   :class: only-dark
   :width: 700px
   :align: center

.. list-table::
   :header-rows: 1

   * - Method
     - What is trained
     - When to use it
   * - ``full``
     - All model parameters
     - A good starting point when the new data differs noticeably from the original
       training data or when enough fine-tuning data is available.
   * - ``heads``
     - Only the selected prediction heads and last layers
     - Useful when the structures are similar to the original training data and the main
       change is the target definition, for example a related level of theory.
   * - ``lora``
     - Low-rank adapter parameters inserted into selected linear layers
     - Useful when only a small number of additional trainable parameters should be
       introduced.

With ``full`` and ``lora``, the backbone weights change, so the heads of targets that
are not part of the current training set would no longer match the features they
receive. Such targets are automatically removed from the model. Any target that should
be kept must therefore be listed in the training set, even if improving it is not the
goal. With ``heads``, the backbone is untouched and all existing targets are kept.

All strategies can use ``inherit_heads`` to initialize a new target head from an
existing one instead of starting from random weights (see :ref:`inheriting-heads`).

Regardless of the method, the composition weights (the per-atom-type baselines) of new
targets are always fitted directly on the fine-tuning dataset before training starts.
Targets already present in the checkpoint keep their composition weights.

The number of trainable parameters and their fraction of the total are logged at the
start of the run. This is a quick way to confirm that the intended parts of the model
are frozen.

Fine-tuning methods
-------------------

Full fine-tuning
^^^^^^^^^^^^^^^^

Full fine-tuning trains all weights of the loaded model:

.. code-block:: yaml

  architecture:
    training:
      learning_rate: 1e-5
      finetune:
        method: full
        read_from: path/to/checkpoint.ckpt

A lower learning rate than the one used for the original training is usually a good
starting point. For example, if the original training used the default ``1e-4``, start
with ``1e-5`` or lower and adjust based on validation error.

.. note::

  Full fine-tuning changes the shared model representation, so targets that are not part
  of the training set are removed from the model. To keep the checkpoint's original
  energy head, include the original target during fine-tuning as described in
  :ref:`multi-fidelity-fine-tuning`.

Heads only
^^^^^^^^^^

Head-only fine-tuning freezes the shared representation and trains only the selected
readout:

.. code-block:: yaml

  architecture:
    training:
      learning_rate: 1e-5
      finetune:
        method: heads
        read_from: path/to/checkpoint.ckpt
        config:
          head_modules: ["node_heads", "edge_heads"]
          last_layer_modules: ["node_last_layers", "edge_last_layers"]

The ``*_heads`` modules are usually multilayer perceptrons that transform the learned
features and the ``*_last_layers`` modules are the final linear layers that map the
result to the target values.

Because head-only fine-tuning leaves the shared representation frozen, existing targets
from the checkpoint remain usable even if they are not included in the new training set.

LoRA
^^^^

LoRA fine-tuning inserts low-rank adapter weights into selected linear layers and
freezes the rest of the model:

.. code-block:: yaml

  architecture:
    training:
      learning_rate: 1e-5
      finetune:
        method: lora
        read_from: path/to/checkpoint.ckpt
        config:
          rank: 4
          alpha: 8
          target_modules: ["input_linear", "output_linear"]

The ``target_modules`` entries are matched against module names. The values shown above
are the defaults. Increase ``rank`` to give the adapters more capacity and decrease it
when the fine-tuning set is small or overfitting appears quickly.

With ``method: lora``, everything except the adapter weights is frozen, including the
prediction heads. When fine-tuning on a new target variant, its newly created head would
therefore stay at its initialization during training. Use ``inherit_heads`` to
initialize it from an existing head instead of random weights. The inherited head is
still frozen afterwards and only the LoRA adapter weights are trained.

Working with target variants
----------------------------

A variant is an alternative version of a target, distinguished by a suffix in the target
name, such as ``energy/pbe``. A model can hold several variants of the same target and
the variant to use can be selected at evaluation and simulation time.

Creating a new variant
^^^^^^^^^^^^^^^^^^^^^^

When fine-tuning to a new energy definition, create a new energy variant if you want to
keep it separate from the checkpoint's original ``energy`` target:

.. code-block:: yaml

  training_set:
    systems:
      read_from: path/to/dataset.xyz
      length_unit: angstrom
    targets:
      energy/<variantname>:
        quantity: energy
        key: <energy-key>
        unit: <energy-unit>
        description: "description of your variant"

Variant names follow the pattern ``energy/<variantname>``. Good names are often based on
the level of theory, functional, or dataset, for example ``energy/pbe`` or
``energy/my-dataset``.

.. _inheriting-heads:

Inheriting weights from existing heads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the new target is close to a target already present in the checkpoint, the new head
can be initialized from the existing one:

.. code-block:: yaml

  architecture:
    training:
      finetune:
        method: full
        read_from: path/to/checkpoint.ckpt
        inherit_heads:
          energy/<variantname>: energy

The keys of ``inherit_heads`` are the new trainable targets from
``training_set.targets``. The values are the existing targets in the checkpoint. The
copied weights remain trainable during fine-tuning. The source target does not need to
be part of the new training set: the weights are copied before targets missing from the
training set are removed from the model.

Making a variant the default target
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulation engines look for a target literally named ``energy``. To make a fine-tuned
variant available under that name, set ``default_target``:

.. code-block:: yaml

  architecture:
    training:
      finetune:
        method: full
        read_from: path/to/checkpoint.ckpt
        default_target: energy/<variantname>

After training finishes, the full state of the variant (heads, composition weights and
scaler settings) is copied into the ``energy`` target, overwriting it if it already
exists. The variant itself is left in the model unchanged. This option is currently only
supported by PET.

.. _multi-fidelity-fine-tuning:

Keeping multiple energy heads useful
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A checkpoint can contain several variants of the same target. With ``full`` or ``lora``
fine-tuning, variants that are not part of the current training set are removed from the
model, since their heads would no longer match the updated representation. To keep
several variants, train on all of them during fine-tuning:

.. code-block:: yaml

  training_set:
    - systems:
        read_from: dataset_1.xyz
        length_unit: angstrom
      targets:
        energy/<variant1>:
          quantity: energy
          key: my_energy_label1
          unit: eV
          description: "my variant1 description"
    - systems:
        read_from: dataset_2.xyz
        length_unit: angstrom
      targets:
        energy/<variant2>:
          quantity: energy
          key: my_energy_label2
          unit: eV
          description: "my variant2 description"

The two targets can also come from the same structures file if both labels are stored
there. In that case, use the corresponding ``key`` for each target. See the
:ref:`Training YAML reference <train_yaml_config>` for details on training with multiple
datasets.

Using the fine-tuned model
--------------------------

Evaluating a variant
^^^^^^^^^^^^^^^^^^^^

The fine-tuned variant can be selected during evaluation by using the same target name
in the evaluation options:

.. code-block:: yaml

   systems:
     read_from: path/to/dataset.xyz
   targets:
     energy/<variantname>:
       key: <energy-key>
       unit: <energy-unit>
       forces:
         key: forces

Using variants in simulation engines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulation engines usually request the standard ``energy`` output. When the model
contains several energy variants, select the variant that should be used for energy and
force predictions. (If the model was fine-tuned with ``default_target``, the plain
``energy`` output already points to the chosen variant and no selection is needed.)

With ASE, pass the variant name to ``MetatomicCalculator``:

.. code-block:: python

  from metatomic_ase import MetatomicCalculator

  calc = MetatomicCalculator("model-ft.pt", variants={"energy": "finetune"})
  atoms.calc = calc

The dictionary maps the target quantity, here ``energy``, to the variant name,
here ``finetune``. This corresponds to the training target
``energy/finetune``.

With LAMMPS, use the ``variant`` keyword:

.. code-block::

  pair_style metatomic model-ft.pt [...other arguments...] variant finetune

Replace ``finetune`` with the part of the target name after ``energy/``.

For more details, see the :ref:`fine-tuning tutorial
<sphx_glr_generated_examples_0-beginner_02-fine-tuning.py>`, the `metatomic ASE
documentation`_, and the `metatomic LAMMPS documentation`_.

.. _metatomic ASE documentation: https://docs.metatensor.org/metatomic/latest/engines/ase.html#metatomic_ase.MetatomicCalculator
.. _metatomic LAMMPS documentation: https://docs.metatensor.org/metatomic/latest/engines/lammps.html
