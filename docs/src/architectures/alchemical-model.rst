.. _architecture-alchemical-model:

Alchemical Model
================

This is an implementation of Alchemical Model: a Behler-Parrinello neural network
:footcite:p:`behler_generalized_2007` with Smooth overlab of atomic positions (SOAP)
features :footcite:p:`bartok_representing_2013` and Alchemical Compression of the
composition space :footcite:p:`willatt_feature_2018, lopanitsyna_modeling_2023,
mazitov_surface_2024`. This model is extremely useful for simulating systems with
large amount of chemical elements.


Installation
------------

To install the package, you can run the following command in the root
directory of the repository:

.. code-block:: bash

    pip install .[alchemical-model]

This will install the package with the Alchemical Model dependencies.


Architecture Hyperparameters
----------------------------
model
#####
soap
^^^^
:param num_pseudo_species: Number of pseudo species to use in the Alchemical Compression
    of the composition space.
:param cutoff_radius: Spherical cutoff (Å) to use for atomic environments.
:param basis_cutoff: The maximal eigenvalue of the Laplacian Eigenstates (LE) basis
    functions used as radial basis :footcite:p:`bigi_smooth_2022`.
:param radial_basis_type: A type of the LE basis functions used as radial basis. The
    supported radial basis functions are

    - ``LE``: Original Laplacian Eigenstates raidal basis. These radial basis functions
      can be set in the ``.yaml`` file as:

      .. code-block:: yaml

        radial_basis_type: "le"

    - ``Physical``: Physically-motivated basis functions. These radial basis functions
      can be set in

      .. code-block:: yaml

        radial_basis_type: "physical"

:param basis_scale: Scaling parameter of the radial basis functions, representing the
    characteristic width (in Å) of the basis functions.
:param trainable_basis: If ``True``, the raidal basis functions will be accompanied by
    the trainable multi-layer perceptron (MLP). If ``False``, the radial basis
    functions will be fixed.

bpnn
^^^^
:param num_hidden_layers: number of hidden layers
:param num_neurons_per_layer: number of neurons per hidden layer
:param activation_function: activation function to use in the hidden layers

training
########
The parameters for the training loop are

:param batch_size: batch size
:param num_epochs: number of training epochs
:param learning_rate: learning rate
:param log_interval: write a line to the log every 10 epochs
:param checkpoint_interval: save a checkpoint every 25 epochs



Default Hyperparameters
-----------------------
The default hyperparameters for the Alchemical Model model are:

.. literalinclude:: ../../../src/metatensor/models/cli/conf/architecture/experimental.alchemical_model.yaml
   :language: yaml


Tuning Hyperparameters
----------------------
The default hyperparameters above will work well in most cases, but they
may not be optimal for your specific dataset. In general, the most important
hyperparameters to tune are (in decreasing order of importance):

- ``cutoff_radius``: This should be set to a value after which most of the
  interactions between atoms is expected to be negligible.
- ``num_pseudo_species``: This number determines the number of pseudo species
  to use in the Alchemical Compression of the composition space. This value should
  be adjusted based on the prior knowledge of the size of original chemical space
  size.
- ``learning_rate``: The learning rate for the neural network. This hyperparameter
  controls how much the weights of the network are updated at each step of the
  optimization. A larger learning rate will lead to faster training, but might cause
  instability and/or divergence.
- ``batch_size``: The number of samples to use in each batch of training. This
  hyperparameter controls the tradeoff between training speed and memory usage. In
  general, larger batch sizes will lead to faster training, but might require more
  memory.
- ``num_hidden_layers``, ``num_neurons_per_layer``, ``max_radial``, ``max_angular``:
  These hyperparameters control the size and depth of the descriptors and the neural
  network. In general, increasing these hyperparameters might lead to better accuracy,
  especially on larger datasets, at the cost of increased training and evaluation time.

References
----------
.. footbibliography::


