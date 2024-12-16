.. _architecture-nanopet:

NanoPET
=======

.. warning::

  This is an **experimental model**.  You should not use it for anything important.

This is a more user-friendly re-implementation of the original PET (which lives in
https://github.com/spozdn/pet), with slightly improved training and evaluation speed.

Installation
------------
To install the package, you can run the following command in the root
directory of the repository:

.. code-block:: bash

    pip install .[nanopet]

This will install the package with the nanoPET dependencies.


Default Hyperparameters
-----------------------
The default hyperparameters for the nanoPET model are:

.. literalinclude:: ../../../src/metatrain/experimental/nanopet/default-hypers.yaml
   :language: yaml


Tuning Hyperparameters
----------------------
The default hyperparameters above will work well in most cases, but they
may not be optimal for your specific dataset. In general, the most important
hyperparameters to tune are (in decreasing order of importance):

- ``cutoff``: This should be set to a value after which most of the interactions between
  atoms is expected to be negligible. A lower cutoff will lead to faster models.
- ``learning_rate``: The learning rate for the neural network. This hyperparameter
  controls how much the weights of the network are updated at each step of the
  optimization. A larger learning rate will lead to faster training, but might cause
  instability and/or divergence.
- ``batch_size``: The number of samples to use in each batch of training. This
  hyperparameter controls the tradeoff between training speed and memory usage. In
  general, larger batch sizes will lead to faster training, but might require more
  memory.
- ``d_pet``: This hyperparameters controls width of the neural network. In general,
  increasing it might lead to better accuracy, especially on larger datasets, at the
  cost of increased training and evaluation time.
- ``num_gnn_layers``: The number of graph neural network layers. In general, decreasing
  this hyperparameter to 1 will lead to much faster models, at the expense of accuracy.
  Increasing it may or may not lead to better accuracy, depending on the dataset, at the
  cost of increased training and evaluation time.
- ``num_attention_layers``: The number of attention layers in each layer of the graph
  neural network. Depending on the dataset, increasing this hyperparameter might lead to
  better accuracy, at the cost of increased training and evaluation time.
- ``loss``: This section describes the loss function to be used, and it has three
  subsections. 1. ``weights``. This controls the weighting of different contributions
  to the loss (e.g., energy, forces, virial, etc.). The default values of 1.0 for all
  targets work well for most datasets, but they might need to be adjusted. For example,
  to set a weight of 1.0 for the energy and 0.1 for the forces, you can set the
  following in the ``options.yaml`` file under ``loss``:
  ``weights: {"energy": 1.0, "forces": 0.1}``. 2. ``type``. This controls the type of
  loss to be used. The default value is ``mse``, and other options are ``mae`` and
  ``huber``. ``huber`` is a subsection of its own, and it requires the user to specify
  the ``deltas`` parameters in a similar way to how the ``weights`` are specified (e.g.,
  ``deltas: {"energy": 0.1, "forces": 0.01}``). 3. ``reduction``. This controls how the
  loss is reduced over batches. The default value is ``mean``, and the other allowed
  option is ``sum``.


All Hyperparameters
-------------------
:param name: ``experimental.nanopet``

model
#####

The model-related hyperparameters are

:param cutoff: Spherical cutoff to use for atomic environments
:param cutoff_width: Width of the shifted cosine cutoff function
:param d_pet: Width of the neural network
:param num_heads: Number of attention heads
:param num_attention_layers: Number of attention layers in each GNN layer
:param num_gnn_layers: Number of GNN layers
:param heads: The type of head (linear or mlp) to use for each target
:param zbl: Whether to use the ZBL short-range repulsion as the baseline for the model

training
########
The hyperparameters for training are

:param distributed: Whether to use distributed training
:param distributed_port: Port to use for distributed training
:param batch_size: Batch size for training
:param num_epochs: Number of epochs to train for
:param learning_rate: Learning rate for the optimizer
:param scheduler_patience: Patience for the learning rate scheduler
:param scheduler_factor: Factor to reduce the learning rate by
:param log_interval: Interval at which to log training metrics
:param checkpoint_interval: Interval at which to save model checkpoints
:param fixed_composition_weights: Weights for fixed atomic contributions to scalar
  targets
:param per_structure_targets: Targets to calculate per-structure losses for
:param log_mae: Whether to log the MAE (mean absolute error) of the model in addition
  to the RMSE
:param loss: The loss function to use, with the subfields described in the previous
  section
