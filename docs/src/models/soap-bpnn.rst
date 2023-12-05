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

.. literalinclude:: ../../../src/metatensor_models/default.yml
   :language: yaml

Any of these hyperparameters can be overridden in the training configuration file.


