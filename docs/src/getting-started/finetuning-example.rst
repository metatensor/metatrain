.. _fine-tuning-example:
Finetuning example
-----------------------------
.. warning::
  Finetuning is currently only available for the PET architecture.


This is a simple example for fine-tuning PET-MAD (or a general PET model), that
can be used as a template for general fine-tuning with metatrain. 
Fine-tuning a pretrained model allows you to obtain a model better suited for
your specific system. You need to provide a dataset of structures that have
been evaluated at a reference level of theory, usually DFT. Fine-tuning
a universal model such as PET-MAD allows for reasonable model performance even if little training
data is available.

First you need a valid checkpoint for the PET architecture, you can obtain a PET-MAD 
checkpoint from huggingface
.. code-block:: bash
  wget https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt

Next, we set up the ``options.yaml`` file. We can specify the fine-tuning method
in the ``finetune`` block in the ``training`` options of the ``architecture``. 
Here, the basic ``full`` option is chosen, which finetunes all weights of the model. 
All available fine-tuning methods are found in the advanced concepts 
:ref:`Fine-tuning <fine-tuning>`_. This section discusses implementation details,
options and recommended use cases. Other fine-tuning options can be simply substituted in this script, 
by changing the ``finetune`` block. 
   
Furthermore, you need to specify the checkpoint, that you want to fine-tune in
the ``read_from`` option.

A simple ``options.yaml`` file for this task could look like this:

.. code-block:: yaml

  architecture:
    name: pet
    training:
      num_epochs: 1000
      learning_rate: 1e-5
      finetune:
        method: full
        read_from: path/to/checkpoint.ckpt
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

  test_set: 0.1
  validation_set: 0.1

In this example, we specified generic but reasonable ``num_epochs`` and ``learning_rate``
parameters. The ``learning_rate`` is chosen to be relatively low to stabilise
training. 


We assumed that the pre-trained model is trained on the dataset ``dataset.xyz`` with 
energies, forces and stresses, which are provided as ``energy`` targets (and its derivatives) 
in the ``options.yaml`` file.
Further information on specifying
targets can be found in :ref:`Customize a Dataset Configuration
<dataset_conf>`_.


After setting up your ``options.yaml`` file, finetuning can then simply be run
via ``mtt train options.yaml``.


Further fine-tuning examples can be found in the 
:ref:`AtomisticCookbook <https://atomistic-cookbook.org/examples/pet-finetuning/pet-ft.html>`_
