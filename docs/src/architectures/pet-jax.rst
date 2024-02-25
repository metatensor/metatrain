.. _architecture-pet-jax:

PET-JAX
=========

This is a JAX implementation of the PET architecture.

Installation
------------
To use PET-JAX within ``metatensor-models``, you should already have
JAX installed for your platform (see the official JAX installation instructions).
Then, you can run the following command in the root directory of the repository:

.. code-block:: bash

    pip install .[pet-jax]

Following this, it is also necessary to hot-fix a few lines of your torch installation
to allow PET-JAX models to be exported. This can be achieved by running the following
Python script:

.. literalinclude:: ../../../src/metatensor/models/experimental/pet_jax/hotfix_torch.py
    :language: python

Default Hyperparameters
-----------------------
The default hyperparameters for the PET-JAX model are:

.. literalinclude:: ../../../src/metatensor/models/cli/conf/architecture/experimental.pet_jax.yaml
   :language: yaml


Tuning Hyperparameters
----------------------
To be done.

References
----------
.. footbibliography::
