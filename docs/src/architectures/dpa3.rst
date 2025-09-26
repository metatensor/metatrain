.. _architecture-nanopet:

DPA3 (experimental)
======================

.. warning::

  This is an **experimental architecture**. You should not use it for anything important.

This is an interface to the DPA3 architecture described in https://arxiv.org/abs/2506.01686
and implemented in deepmd-kit (https://github.com/deepmodeling/deepmd-kit).

Installation
------------

To install the package, you can run the following command in the root
directory of the repository:

.. code-block:: bash

    pip install metatrain[dpa3]

This will install the package with the DPA3 dependencies.


Default Hyperparameters
-----------------------

The default hyperparameters for the DPA3 architecture are:

.. literalinclude:: ../../../src/metatrain/experimental/dpa3/default-hypers.yaml
   :language: yaml


Tuning Hyperparameters
----------------------

@littlepeachs this is where you can tell users how to tune the parameters of the model
to obtain different speed/accuracy tradeoffs

References
----------

.. footbibliography::
