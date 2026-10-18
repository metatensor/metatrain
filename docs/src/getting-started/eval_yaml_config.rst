.. _eval_yaml_config:

Evaluation options
==================

``mtt eval`` evaluates a trained model on a dataset described by an options file:

.. code-block:: bash

    mtt eval model.pt eval.yaml

The ``systems`` and ``targets`` sections of ``eval.yaml`` are the same as for a dataset
in the training options file (see :ref:`train_yaml_config`). When ``targets`` are
given, the errors of the model on them are reported. This page describes the options
that change which errors are reported.

.. autoclass:: metatrain.share.base_hypers.EvalDatasetDictHypers
    :members:
    :undoc-members:
    :no-index:


Errors for each block
---------------------

By default, the errors are aggregated over all the blocks of a target. To report them
separately for each block, e.g. for each ``o3_lambda`` and ``o3_sigma`` of a spherical
target, set ``log_separate_blocks``:

.. code-block:: yaml

    systems: dataset.xyz
    targets:
      mtt::dipole:
        type:
          spherical:
            irreps:
              - {o3_lambda: 1, o3_sigma: 1}
    log_separate_blocks: true

Each error is then labeled with the key of its block, and the aggregated errors are not
reported:

.. code-block:: text

    mtt::dipole (o3_lambda=1,o3_sigma=1) RMSE (per atom): 0.0123 | ...
