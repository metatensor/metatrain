.. _architecture-{architecture}:

SOAP-BPNN
=========

This is a Behler-Parrinello type neural network :footcite:p:`behler_generalized_2007`,
which, instead of their original atom-centered symmetry functions, we use the Smooth
overlap of atomic positions (SOAP) :footcite:p:`bartok_representing_2013` as the atomic
descriptors, computed with `torch-spex <https://github.com/lab-cosmo/torch-spex>`_.

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

with the following definitions needed to fully understand some of the parameters:

.. autoclass:: {architecture_path}.hypers.SOAPConfig
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.hypers.SOAPCutoffConfig
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.hypers.BPNNConfig
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
