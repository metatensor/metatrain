.. _architecture-sparse-gap:

Sparse GAP
==========

This is an implementation of the sparse `Gaussian Approximation Potential
<GAP_>`_ (GAP) using `Smooth Overlap of Atomic Positions <SOAP_>`_ (SOAP)
implemented in `rascaline <RASCALINE_>`_.


.. _SOAP: https://doi.org/10.1103/PhysRevB.87.184115
.. _GAP:  https://doi.org/10.1002/qua.24927
.. _RASCALINE: https://github.com/Luthaf/rascaline


Installation
------------

To install the package, you can run the following command in the root directory
of the repository:

.. code-block:: bash

    pip install .[gap]

This will install the package with the GAP dependencies.


Hyperparameters
---------------

The hyperparameters (and relative default values) for the SOAP-BPNN model are:

.. literalinclude:: ../../../src/metatensor/models/experimental/gap/default-hypers.yaml
   :language: yaml

Any of these hyperparameters can be overridden with the training parameter file.
