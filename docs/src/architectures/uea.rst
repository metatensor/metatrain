.. _architecture-uea:

UEA (experimental)
==================

.. warning::

  This is an **experimental model**. You should not use it for anything important.

This is the UEA architecture, which stands for Universal Equivariant Approximator.
It is an equivariant neural network which can approximate any equivariant function of
atomic structures using a finite number of layers.

Installation
------------

To install the package, you can run the following command in the root
directory of the repository:

.. code-block:: bash

    pip install metatrain[uea]

This will install the package with the UEA dependencies.


Default Hyperparameters
-----------------------

The default hyperparameters for the UEA model are:

.. literalinclude:: ../../../src/metatrain/experimental/uea/default-hypers.yaml
   :language: yaml


Tuning Hyperparameters
----------------------

Coming soon


All Hyperparameters
-------------------

Coming soon

References
----------

.. footbibliography::
