.. _label_fine_tuning_concept:

Fine-tune a pre-trained model
=============================

Fine-tuning starts a new training run from an existing checkpoint. The model
architecture and weights are loaded from the checkpoint, while the optimizer and
learning-rate scheduler are initialized from the new options file. This is different
from restarting an interrupted training run, where the optimizer and scheduler states
are restored as well.

Finetuning may not be supported by every architecture and if supported the syntax to
start a finetuning may be different from how it is explained here. This section
describes the process of fine-tuning a pre-trained model to adapt it to new tasks or
datasets. Fine-tuning is a common technique used in machine learning, where a model is
trained on a large dataset and then fine-tuned on a smaller dataset to improve its
performance on specific tasks. So far the fine-tuning capabilities are only available
for the PET, FlashMD, FlashMDSymplectic and SPACE models.

Once the options file is set up, start the run with ``mtt train options.yaml``. There is
a complete example in the tutorial section: :ref:`finetuning-tutorial`.

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

Multi-fidelity training
-----------------------
With the ``full`` and ``lora`` fine-tuning methods, the backbone weights change, which
means that any target *not* part of the current fine-tuning run's dataset would end up
with a head that no longer matches the (now different) feature space produced by the
backbone. To avoid keeping such now-useless heads around, they are automatically
removed from the model: after a ``full`` or ``lora`` fine-tuning run, the model only
retains the targets that were part of that run's training set. Targets you want to
keep must therefore be included in the training set of every subsequent ``full``/
``lora`` fine-tuning run, even if you do not care about further improving them.

If you want to fine-tune and retain multiple functional heads, the recommended way is
to do full fine-tuning on a new target, but keep training the old energy head as well,
by including it in the same run's targets (for instance by using ``inherit_heads`` to
initialize the new head from the old one, see above, while still listing both targets
below). This will leave you with a model capable of using different variants for
energy and force prediction. Again, you are able to select the preferred head in
``LAMMPS`` or when creating a ``metatomic`` calculator object. Thus, you should specify
both variants in the ``targets`` section of your ``options.yaml``. In the code
snippet, we additionally assume that the energy labels come from different datasets.
Please note, if you have both references in one file, they can be selected by
selecting the corresponding keys from the same system, the same dataset.

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
                description: 'my variant1 description'
      - systems:
            read_from: dataset_2.xyz
            length_unit: angstrom
        targets:
            energy/<variant2>:
                quantity: energy
                key: my_energy_label2
                unit: eV
                description: 'my variant2 description'



You can find more about setting up training with multiple files in the
:ref:`Training YAML reference <train_yaml_config>`.


Training only the head weights can be an alternative, if one wants to keep the old energy
head, but the reference data it was trained are not available. In that case, the
internal model weights are frozen, and only the weights of the new target are trained.
Since the backbone does not change with this method, all existing targets/heads are
kept automatically, whether or not they are part of the current run's dataset (see
below).


Fine-tuning model Heads only
----------------------------

Adapting all the model weights to a new dataset is not always the best approach. If the
new dataset consist of the same or similar data computed with a slightly different level
of theory compared to the pre-trained models' dataset, you might want to keep the
learned representations of the crystal structures and only adapt the readout layers
(i.e. the model heads) to the new dataset. Since the backbone is frozen and therefore
unchanged, all targets/heads already present in the model are kept, regardless of
whether they are part of the current run's dataset -- unlike ``full``/``lora``
fine-tuning, which drops targets not included in the current run (see
"Multi-fidelity training" above).
In this case, the ``mtt train`` command needs to be accompanied by the specific training
options in the ``options.yaml`` file. The following options need to be set:

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

For more details, see the :ref:`fine-tuning tutorial <finetuning-tutorial>`, the
`metatomic ASE documentation`_, and the `metatomic LAMMPS documentation`_.

.. _metatomic ASE documentation: https://docs.metatensor.org/metatomic/latest/engines/ase.html#metatomic_ase.MetatomicCalculator
.. _metatomic LAMMPS documentation: https://docs.metatensor.org/metatomic/latest/engines/lammps.html
