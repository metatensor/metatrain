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


Hyperparameters
---------------

The hyperparameters (and relative default values) for the Alchemical Model model are:

.. literalinclude:: ../../../src/metatensor/models/cli/conf/architecture/experimental.alchemical_model.yaml
   :language: yaml

Any of these hyperparameters can be overridden with the training parameter file.


