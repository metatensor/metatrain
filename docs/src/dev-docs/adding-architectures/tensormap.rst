Using TensorMap
###############

The :py:class:`metatensor.torch.TensorMap` class is one of the core classes of the
`metatensor` library. It is used to store :py:class:`torch.Tensor` objects with
additional metadata. In order to access the underlying data, you can use the
following syntax:

.. code-block:: python

    tensor_map.block().values

which assumes that this ``TensorMap`` contains a single block (this will be the
case, for example, for an ``energy`` target). The gradients of the energies
with respect to positions (i.e., minus the forces) can be accessed using

.. code-block:: python

    tensor_map.block().gradient('positions').values
