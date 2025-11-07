.. _architecture-{architecture}:

GAP
===

This is an implementation of the sparse Gaussian Approximation Potential
(GAP) :footcite:p:`bartok_representing_2013` using Smooth Overlap of Atomic Positions
(SOAP) :footcite:p:`bartok_gaussian_2010` implemented in `featomic <FEATOMIC_>`_.

.. _FEATOMIC: https://github.com/Luthaf/featomic

The GAP model in metatrain can only train on CPU, but evaluation
is also supported on GPU.

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

.. autoclass:: {architecture_path}.hypers.KRRHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.hypers.SOAPHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.hypers.SOAPCutoffHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.hypers.SOAPCutoffSmoothingHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.hypers.SOAPDensityHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.hypers.SOAPDensityScalingHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.hypers.SOAPBasisHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.hypers.SOAPBasisRadialHypers
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
