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


Equivariance error
------------------

Models that are not equivariant by construction, i.e. whose predictions do not rotate
exactly like the targets, e.g. PET, can learn equivariance only approximately. ``mtt eval`` can
measure how far a model is from being equivariant, by also evaluating every system on
many rotated and inverted copies, rotating the predictions back, and comparing them. Add
an ``equivariance`` section (``equivariance: {}`` uses the default settings), which
requires a ``targets`` section:

.. code-block:: yaml

    systems: dataset.xyz
    targets:
      energy:
        key: U0
    equivariance:
      max_angular_momentum_grid: 6
      output: symmetrized.zip

For each target, the following errors are then reported, in the units of the target:

- ``equivariance RMSE``: the spread of the back-rotated predictions over the
  orientations, which is zero for an equivariant model;
- ``O3-averaged RMSE`` and ``O3-averaged MAE``: the errors of the predictions averaged
  over all orientations, i.e. of the closest equivariant model;
- ``oriented RMSE``: ``sqrt(equivariance RMSE² + O3-averaged RMSE²)``, the RMSE the
  model would have on randomly oriented copies of the systems.

.. code-block:: text

    energy equivariance RMSE (per atom): 0.28 meV | energy O3-averaged RMSE (per atom): 0.63 meV | energy oriented RMSE (per atom): 0.69 meV | ...

With ``log_separate_blocks``, the equivariance error is also reported for each block.
For scalar and Cartesian targets, these blocks are labeled by the angular momentum
components (``o3_lambda``, ``o3_sigma``) of the target, rather than by its own keys.

The orientations are sampled on a quadrature grid, whose degree is set by
``max_angular_momentum_grid``. The default grid is exact only for models that are
equivariant by construction. ``mtt eval`` therefore re-measures the error of the first
batch on a finer grid, and warns if the two differ by more than 1%; in that case,
increase ``max_angular_momentum_grid`` until the error stops changing. The cost of the
evaluation grows steeply with the degree of the grid.

The equivariance error of gradients (e.g. forces) is not measured: a warning is printed
and only the target values are considered.

With ``output: symmetrized.zip``, a ``DiskDataset`` is written with one entry per
system, containing the system, the O(3)-averaged predictions under the name of each
target (e.g. ``energy``), and their variance over the orientations under
``o3::variance::<target>`` (e.g. ``o3::variance::energy``).

.. autoclass:: metatrain.share.base_hypers.EquivarianceHypers
    :members:
    :undoc-members:
    :no-index:
