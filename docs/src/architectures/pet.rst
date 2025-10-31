.. _architecture-pet:

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

Default Hyperparameters
-----------------------

The description of all the hyperparameters used in PET is provided further
down this page. However, here we provide you with a yaml file containing all 
the default hyperparameters, which might be convenient as a starting point to
create your own hyperparameter files:

.. literalinclude:: ../../../src/metatrain/pet/default-hypers.yaml
   :language: yaml
   :lines: 2-

Tuning hyperparameters
----------------------

The default hyperparameters above will work well in most cases, but they
may not be optimal for your specific dataset. There is good number of
parameters to tune, both for the :ref:`model <pet_model_hypers>` and the
:ref:`trainer <pet_trainer_hypers>`. Since seeing them for the first time
might be overwhelming, here we provide a **list of the parameters that are
in general the most important** (in decreasing order of importance):

.. container:: mtt-hypers-remove-classname

  .. autoattribute:: metatrain.pet.hypers.PETHypers.cutoff
      :no-index:

  .. autoattribute:: metatrain.pet.hypers.PETTrainerHypers.learning_rate
      :no-index:

  .. autoattribute:: metatrain.pet.hypers.PETTrainerHypers.batch_size
      :no-index:

  .. autoattribute:: metatrain.pet.hypers.PETHypers.d_pet
      :no-index:

  .. autoattribute:: metatrain.pet.hypers.PETHypers.d_node
      :no-index:

  .. autoattribute:: metatrain.pet.hypers.PETHypers.num_gnn_layers
      :no-index:

  .. autoattribute:: metatrain.pet.hypers.PETHypers.num_attention_layers
      :no-index:

  .. autoattribute:: metatrain.pet.hypers.PETTrainerHypers.loss
      :no-index:

  .. autoattribute:: metatrain.pet.hypers.PETHypers.long_range
      :no-index:

.. _pet_model_hypers:

Model hyperparameters
------------------------

The parameters that go under the ``architecture.model`` section of the config file
are the following:

.. autoclass:: metatrain.pet.hypers.PETHypers
    :members:
    :undoc-members:

.. _pet_trainer_hypers:

Trainer hyperparameters
-------------------------

The parameters that go under the ``architecture.trainer`` section of the config file
are the following:

.. autoclass:: metatrain.pet.hypers.PETTrainerHypers
    :members:
    :undoc-members:

Configuration for fine-tuning
-----------------------------

This section contains the definitions needed to understand the ``training.finetune`` parameter.

.. autoclass:: metatrain.pet.modules.finetuning.FullFinetuneHypers
    :members:
    :undoc-members:

.. autoclass:: metatrain.pet.modules.finetuning.LoRaFinetuneHypers
    :members:
    :undoc-members:

.. autoclass:: metatrain.pet.modules.finetuning.HeadsFinetuneHypers
    :members:
    :undoc-members:

.. autoclass:: metatrain.pet.modules.finetuning.LoRaFinetuneConfig
    :members:
    :undoc-members:

.. autoclass:: metatrain.pet.modules.finetuning.HeadsFinetuneConfig
    :members:
    :undoc-members:

References
----------

.. footbibliography::
