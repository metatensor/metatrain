.. _fine-tune-example:
Finetuning example
-----------------------------

All available finetuning methods are found in the advanced concepts 
:ref:`Fine-tuning <fine-tune>`. Let's say we want to
finetune only the heads of PET-MAD (or a general PET checkpoint). 
A possible ``options.yaml`` could look like this:

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

You simply include the ``finetune`` section. In this example, we finetune 
also on forces and stress. Further we choose a relatively low
``learning_rate``.
The finetuning is simply run by:
   .. code-block:: bash
   mtt train options-finetune.yaml
