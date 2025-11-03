.. _loss-functions:

Loss functions
==============

``metatrain`` supports a variety of loss functions, which can be configured
in the ``loss`` subsection of the ``training`` section for each ``architecture``
in the options file. The loss functions are designed to be flexible and can be
tailored to the specific needs of the dataset and the targets being predicted.

The ``loss`` subsection describes the loss functions to be used. The most basic
configuration is

.. code-block:: yaml

  loss: mse

which sets the loss function to mean squared error (MSE) for all targets and, if
present, their gradients. When training a potential energy surface on energy,
forces, and stress, for example, this configuration is internally expanded to

.. code-block:: yaml

  loss:
    energy:
      type: mse
      weight: 1.0
      reduction: mean
      gradients:
        positions:
          type: mse
          weight: 1.0
          reduction: mean
        strain:
          type: mse
          weight: 1.0
          reduction: mean

This example assumes the training set contains a target named ``energy``, which has
both forces and stress/virial gradients requested. In case the energy target has a
custom name (say, ``mtt::etot``), the configuration would instead be

.. code-block:: yaml

  loss:
    mtt::etot:
      type: mse
      weight: 1.0
      reduction: mean
      gradients:
        positions:
          type: mse
          weight: 1.0
          reduction: mean
        strain:
          type: mse
          weight: 1.0
          reduction: mean
  ...
  training_set:
    systems:
    ...
    targets:
      mtt::etot:
        quantity: energy
        forces: true # or some other allowed configuration
        stress: true # or some other allowed configuration
    ...

The internal, more detailed configuration can be used in the options file
to specify different loss functions for each target, or to override default
values for the parameters. The parameters accepted by each loss function term
are

1. ``type``. This controls the type of loss to be used. The default value is ``mse``,
   and other standard options are ``mae`` and ``huber``, which implement the equivalent
   PyTorch loss functions
   `MSELoss <https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>`_,
   `L1Loss <https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html>`_,
   and
   `HuberLoss <https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html>`_,
   respectively.
   There are also "masked" versions of these losses, which are useful when using
   padded targets with values that should be masked before computing the loss. The
   masked losses are named ``masked_mse``, ``masked_mae``, and ``masked_huber``.

2. ``weight``. This controls the weighting of different contributions to the loss
   (e.g., energy, forces, virial, etc.). The default value of 1.0 for all targets
   works well for most datasets, but can be adjusted if required.

3. ``reduction``. This controls how the overall loss is computed across batches.
   The default for this is to use the ``mean`` of the batch losses. The ``sum``
   function is also supported.

Some losses, like ``huber``, require additional parameters to be specified. Below is
a table summarizing losses that require or allow additional parameters:

.. list-table:: Loss Functions and Parameters
    :header-rows: 1
    :widths: 20 30 50

    * - Loss Type
      - Description
      - Additional Parameters
    * - ``mse``
      - Mean squared error
      - N/A
    * - ``mae``
      - Mean absolute error
      - N/A
    * - ``huber``
      - Huber loss
      - ``delta``: Threshold at which to switch from squared error to absolute error.
    * - ``masked_mse``
      - Masked mean squared error
      - N/A
    * - ``masked_mae``
      - Masked mean absolute error
      - N/A
    * - ``masked_huber``
      - Masked Huber loss
      - ``delta``: Threshold at which to switch from squared error to absolute error.


Simplified configurations
-------------------------

The internal specification of loss functions can be cumbersome for common use cases.
There are then shortcuts to simplify the configuration for standard scenarios.

The first example is that of machine-learning interatomic potentials (MLIPs).
Since often one wants to train on gradients, like forces and stress/virial, the loss
functions can be specified without explicitly defining the gradients subsection,
by using the shorthand names ``forces`` and ``stress`` or ``virial`` at the top level
of the loss configuration. For example, the following configuration is equivalent to the
one above for the ``energy`` target:

.. code-block:: yaml

  loss:
    energy:
      type: mse
      weight: 1.0
      reduction: mean
    forces:
      type: mse
      weight: 1.0
      reduction: mean
    stress:
      type: mse
      weight: 1.0
      reduction: mean

Another common scenario is when only the loss function type is to be specified.
In this case, it is possible to use the following shorthand notation:

.. code-block:: yaml

  loss:
    energy:
      type: mse
    forces:
      type: mae
    stress:
      type: huber

which is equivalent to the more detailed configuration:

.. code-block:: yaml

  loss:
    energy:
      type: mse
      weight: 1.0
      reduction: mean
    forces:
      type: mae
      weight: 1.0
      reduction: mean
    stress:
      type: huber
      weight: 1.0
      reduction: mean
      delta: 1.0




Masked loss functions
---------------------

Masked loss functions are particularly useful when dealing with datasets that contain
padded targets. In such cases, the loss function can be configured to ignore the padded
values during the loss computation. This is done by using the ``masked_`` prefix in
the loss type. For example, if the target contains padded values, you can use
``masked_mse`` or ``masked_mae`` to ensure that the loss is computed only on the
valid (non-padded) values. The values of the masks must be passed as ``extra_data``
in the training set, and the loss function will automatically apply the mask to
the target values. An example configuration for a masked loss is as follows:

 .. code-block:: yaml

  loss:
    energy:
      type: masked_mse
      weight: 1.0
      reduction: sum
    forces:
      type: masked_mae
      weight: 0.1
      reduction: sum
  ...

  training_set:
    systems:
    ...
    targets:
      mtt::my_target:
        ...
    ...
    extra_data:
      mtt::my_target_mask:
        read_from: my_target_mask.mts
