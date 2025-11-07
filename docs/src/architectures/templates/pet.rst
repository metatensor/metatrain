.. _architecture-{architecture}:

PET
===

PET is a cleaner, more user-friendly reimplementation of the original
PET model :footcite:p:`pozdnyakov_smooth_2023`. It is designed for better
modularity and maintainability, while preseving compatibility with the original
PET implementation in ``metatrain``. It also adds new features like long-range
features, better fine-tuning implementation, a possibility to train on
arbitrarty targets, and a faster inference due to the ``fast attention``.

Installation
------------

PET model is included in the ``metatrain`` package and doesn't require any
additional installation steps. To install ``metatrain`` run:

.. code-block:: bash

    pip install metatrain

This will install the PET model along with the ``metatrain`` package.

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

The default hyperparameters above will work well in most cases, but they
may not be optimal for your specific dataset. There is good number of
parameters to tune, both for the :ref:`model <pet_model_hypers>` and the
:ref:`trainer <pet_trainer_hypers>`. Since seeing them for the first time
might be overwhelming, here we provide a **list of the parameters that are
in general the most important** (in decreasing order of importance):

.. container:: mtt-hypers-remove-classname

  .. autoattribute:: {model_hypers_path}.cutoff
      :no-index:

  .. autoattribute:: {trainer_hypers_path}.learning_rate
      :no-index:

  .. autoattribute:: {trainer_hypers_path}.batch_size
      :no-index:

  .. autoattribute:: {model_hypers_path}.d_pet
      :no-index:

  .. autoattribute:: {model_hypers_path}.d_node
      :no-index:

  .. autoattribute:: {model_hypers_path}.num_gnn_layers
      :no-index:

  .. autoattribute:: {model_hypers_path}.num_attention_layers
      :no-index:

  .. autoattribute:: {trainer_hypers_path}.loss
      :no-index:

  .. autoattribute:: {model_hypers_path}.long_range
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
