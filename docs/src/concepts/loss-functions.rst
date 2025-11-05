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


.. _dos-loss:

Masked DOS Loss Function
------------------------
The masked DOS loss function is a specialized loss function designed to support model training on the electronic density of states (DOS) projected on an energy grid where the structures in the dataset
have eigenvalues that span different energy ranges, while accounting for the lack of absolute energy reference in DOS calculations. This loss function allows for effective training
by focusing the loss computation on the relevant energy ranges for each structure, thereby providing a unified approach to handling DOS data with varying eigenvalue distributions.
The loss function accounts for the lack of absolute energy reference by allowing the user to specify a number of extra targets that the model predicts beyond the actual DOS values in the target.
Within the loss function, these extra targets are used to dynamically shift the energy grid for each structure during training, aligning the predicted DOS with the target DOS in a way that minimizes the loss.
After the alignment step, the loss function is comprised of three components:

- an integrated loss on the masked DOS values

.. code-block:: python

    masked_DOS_loss = torch.trapezoid((aligned_predictions - targets)**2 * mask, x_axis = energy_grid)

- an integrated loss on the gradient of the *unmasked* DOS values, to ensure that values outside the masked region are also learned smoothly

.. code-block:: python

    unmasked_gradient_loss = torch.trapezoid(aligned_predictions_gradient**2 * (~mask), x_axis = energy_grid)

- an integrated loss on the cumulative DOS values in the masked region

.. code-block:: python

    cumulative_aligned_predictions = torch.cumulative_trapezoid(aligned_predictions, x = energy_grid)
    cumulative_targets = torch.cumulative_trapezoid(targets, x = energy_grid)
    masked_cumulative_DOS_loss = torch.trapezoid((cumulative_aligned_predictions - cumulative_targets)**2 * mask, x_axis = energy_grid[1:])

Each component can be weighted independently to tailor the loss function to specific training needs.

.. code-block:: python

    loss = (masked_DOS_loss +
            grad_weight * unmasked_gradient_loss +
            int_weight * masked_cumulative_DOS_loss)

To use this loss function, you can refer to this code snippet for the ``loss`` section in your YAML configuration file:

.. code-block:: yaml

    loss:
      mtt::dos:
        type: "masked_dos"
        grad_weight: 1e-4
        int_weight: 2.0
        extra_targets: 200
        reduction: "mean"

:param name: key for the dos in the prediction/target dictionary. (mtt::dos in this case)
:param grad_weight: Multiplier for the gradient of the unmasked DOS component.
:param int_weight: Multiplier for the cumulative DOS component.
:param extra_targets: Number of extra targets predicted by the model.
:param reduction: reduction mode for torch loss. Options are "mean", "sum", or "none".

The values used in the above example are the ones used for PETMADDOS training and can be a reasonable starting point for other applications.
