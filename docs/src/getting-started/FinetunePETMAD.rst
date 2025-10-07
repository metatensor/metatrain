.. _fine-tuning-example:
Finetuning example
-----------------------------
This is a simple example for fine-tuning PET-MAD (or a general PET checkpoint).

Here, the ``heads`` options is chosen only the heads of the model will be trained.
All available finetuning methods are found in the advanced concepts 
:ref:`Fine-tuning <fine-tuning>`. This section discusses implementation details,
options and tips. Other finetuning options can be simply substituted in this script, 
by changing the ``finetune`` block. 
   
A simple ``options.yaml`` file for this task could look like this:

.. code-block:: yaml

  architecture:
    name: pet
    training:
      num_epochs: 1000
      learning_rate: 1e-5
      finetune:
        method: "heads"
        read_from: path/to/checkpoint.ckpt
        config:
          head_modules: ['node_heads', 'edge_heads']
          last_layer_modules: ['node_last_layers', 'edge_last_layers']
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

In this example, fine-tuning also uses forces and stress. 
Further we choose a relatively low ``learning_rate`` to stabilise training.
After setting up your ``options.yaml`` file, finetuning is simply run by:
   ``mtt train options.yaml``
