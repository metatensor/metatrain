.. _architecture-{architecture}:

FlashMD
==============

FlashMD is a method for the direct prediction of positions and momenta in a molecular
dynamics simulation, presented in :footcite:p:`bigi_flashmd_2025`. When compared to
traditional molecular dynamics methods, it predicts the positions and momenta of atoms
after a long time interval, allowing the use of much larger time steps. Therefore, it
achieves a significant speedup (10-30x) compared to molecular dynamics using MLIPs.
The FlashMD architecture implemented in metatrain is based on the
:ref:`PET architecture <architecture-pet>`.

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

Tuning hyperparameters
----------------------

Most of the parameters of FlashMD are inherited from the PET architecure, although
they might have different default values.

.. container:: mtt-hypers-remove-classname

    - FlashMD-specific parameters for the model:

        .. autoattribute:: {model_hypers_path}.predict_momenta_as_difference
            :no-index:

    - FlashMD-specific parameters for the trainer:

        .. autoattribute:: {trainer_hypers_path}.timestep
            :no-index:

        .. autoattribute:: {trainer_hypers_path}.masses
            :no-index:

.. _{architecture}_model_hypers:

Model hyperparameters
------------------------

The parameters that go under the ``architecture.model`` section of the config file
are the following:

.. autoclass:: {model_hypers_path}
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

