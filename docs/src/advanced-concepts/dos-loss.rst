.. _dos-loss:
Masked DOS Loss Function
=========================
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
