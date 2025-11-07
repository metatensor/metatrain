.. _architecture-{architecture}:

LLPR
====

The LLPR architecture is a "wrapper" architecture that enables cheap uncertainty
quantification (UQ) via the last-layer prediction rigidity (LLPR) approach proposed
by Bigi et al. :footcite:p:`bigi_mlst_2024` It is compatible with the following
``metatrain`` models constructed from NN-based architectures: PET and SOAP-BPNN.
The implementation of the LLPR as a separate architecture within ``metatrain``
allows the users to compute the uncertainties without dealing with the fine details
of the LLPR implementation.

Installation
------------

To install this architecture along with the ``metatrain`` package, run:

.. code-block:: bash

    pip install metatrain[{architecture}]

where the square brackets indicate that you want to install the optional
dependencies required for ``{architecture}``.

.. _{architecture}_default_hypers:

Default Hyperparameters
-----------------------

The description of all the hyperparameters used in ``{architecture}`` is provided
further down this page. However, here we provide you with a yaml file containing all
the default hyperparameters, which might be convenient as a starting point to
create your own hyperparameter files:

.. literalinclude:: {default_hypers_path}
   :language: yaml

.. _{architecture}_model_hypers:

Model hyperparameters
------------------------

The parameters that go under the ``architecture.model`` section of the config file
are the following:

.. autoclass:: {model_hypers_path}
    :members:
    :undoc-members:

where the ensemble hyperparameters should adhere to the following structure:

.. autoclass:: metatrain.llpr.model.EnsemblesHypers
    :members:
    :undoc-members:

.. _{architecture}_trainer_hypers:

Trainer hyperparameters
-------------------------

The parameters that go under the ``architecture.trainer`` section of the config file
are the following:

.. autoclass:: {trainer_hypers_path}
    :members:
    :undoc-members:

References
----------

.. footbibliography::
