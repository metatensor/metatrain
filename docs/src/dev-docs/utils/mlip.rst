MLIP Base Classes
=================

.. automodule:: metatrain.utils.mlip
    :members:
    :undoc-members:
    :show-inheritance:

MLIPModel
---------

The :py:class:`metatrain.utils.mlip.MLIPModel` class is a base class for
MLIP-only models that predict only energies and forces. It provides a common
implementation of the forward pass that processes neighbor lists and computes
energies using a simplified interface.

Derived classes only need to implement the
:py:meth:`~metatrain.utils.mlip.MLIPModel.compute_energy` method.

MLIPTrainer
-----------

The :py:class:`metatrain.utils.mlip.MLIPTrainer` class is a base trainer for
MLIP-only models. It implements the complete training loop and handles:

- Distributed training
- Data loading with optional rotational augmentation
- Loss computation
- Checkpointing

Derived classes only need to implement the
:py:meth:`~metatrain.utils.mlip.MLIPTrainer.use_rotational_augmentation` method
to specify whether rotational data augmentation should be used during training.

Example
^^^^^^^

Here's how to use the base classes to create a new MLIP architecture:

.. code-block:: python

    from metatrain.utils.mlip import MLIPModel, MLIPTrainer

    class MyMLIPModel(MLIPModel):
        def compute_energy(
            self,
            edge_vectors: torch.Tensor,
            species: torch.Tensor,
            centers: torch.Tensor,
            neighbors: torch.Tensor,
            system_indices: torch.Tensor,
        ) -> torch.Tensor:
            # Implement your energy computation here
            ...
            return energies  # shape: (N_systems,)

    class MyMLIPTrainer(MLIPTrainer):
        def use_rotational_augmentation(self) -> bool:
            # Return True to use rotational augmentation, False otherwise
            return False

Utilities
---------

.. autofunction:: metatrain.utils.mlip.get_mlip_scheduler
