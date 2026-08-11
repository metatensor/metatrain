.. _arch-{{architecture}}_installation:

Installation
------------

To install this architecture along with the ``metatrain`` package, run:

.. code-block:: bash

    pip install metatrain[{{architecture}}]

where the square brackets indicate that you want to install the optional
dependencies required for ``{{architecture}}``.

Alternatively, you can install via conda-forge:

.. code-block:: bash

    conda install -c conda-forge metatrain

For architectures with optional dependencies, you may need to install those
dependencies separately with pip, or use conda packages if they are available
on conda-forge. See the `installation` page for more details.
