.. _architecture-alchemical-model:

Alchemical Model
=========

This is an implementation of Alchemical Model: a Behler-Parrinello neural network 
with SOAP features and Alchemical Compression of the composition space. This model 
is extremely useful for simulating systems with large amount of chemical elements. 

For further details, please refer to the original papers:
- Willatt, Michael J., Félix Musil, and Michele Ceriotti. "Feature optimization for atomistic machine learning yields a data-driven construction of the periodic table of the elements." Physical Chemistry Chemical Physics 20.47 (2018): 29661-29668.
- Lopanitsyna, Nataliya, et al. "Modeling high-entropy transition metal alloys with alchemical compression." Physical Review Materials 7.4 (2023): 045802.
- Mazitov, Arslan, et al. "Surface segregation in high-entropy alloys from alchemical machine learning." arXiv preprint arXiv:2310.07604 (2023).


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

.. literalinclude:: ../../../src/metatensor/models/cli/conf/architecture/alchemical_model.yaml
   :language: yaml

Any of these hyperparameters can be overridden with the training parameter file.


