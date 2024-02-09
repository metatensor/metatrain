.. _architecture-soap-bpnn:

SOAP-BPNN
=========

This is a Behler-Parrinello neural network with SOAP features.


Installation
------------

To install the package, you can run the following command in the root
directory of the repository:

.. code-block:: bash

    pip install .[soap-bpnn]

This will install the package with the SOAP-BPNN dependencies.


Hyperparameters
---------------

The hyperparameters (and relative default values) for the SOAP-BPNN model are:

.. literalinclude:: ../../../src/metatensor/models/cli/conf/architecture/experimental.soap_bpnn.yaml
   :language: yaml

Any of these hyperparameters can be overridden in the training parameter file.

Tuning hyperparameters
######################

The default hyperparameters above will work well in most cases, but they
may not be optimal for your specific dataset. In general, the most important
hyperparameters to tune are (in decreasing order of importance):

- ``cutoff``: The cutoff radius for the SOAP descriptor. This should be set to
  a value after which most of the interactions between atoms is expected to be
  negligible.
- ``learning_rate``: The learning rate for the neural network. This hyperparameter
  controls how much the weights of the network are updated at each step of the
  optimization. A larger learning rate will lead to faster training, but might
  cause instability and/or divergence.
- ``batch_size``: The number of samples to use in each batch of training. This
  hyperparameter controls the tradeoff between training speed and memory usage.
  In general, larger batch sizes will lead to faster training, but might require
  more memory.
- ``num_hidden_layers``, ``num_neurons_per_layer``, ``max_radial``, ``max_angular``:
  These hyperparameters control the size and depth of the descriptors and
  the neural network. In general, increasing these hyperparameters might lead
  to better accuracy, especially on larger datasets, at the cost of increased
  training and evaluation time.
- ``radial_scaling`` hyperparameters: These hyperparameters control the radial
  scaling of the SOAP descriptor. In general, the default values should work
  well, but they might need to be adjusted for specific datasets.
- ``layernorm``: Whether to use layer normalization before the neural network.
  Setting this hyperparameter to ``false`` will lead to slower convergence of training,
  but might lead to better generalization outside of the training set distribution.
