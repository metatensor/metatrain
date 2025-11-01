.. _adding-new-mlip:

Adding a new MLIP-only architecture
====================================

For MLIP-only models (models that only predict energies and forces),
``metatrain`` provides base classes :py:class:`metatrain.utils.mlip.MLIPModel`
and :py:class:`metatrain.utils.mlip.MLIPTrainer` that implement most of the
boilerplate code. See :doc:`utils/mlip` for more details.

Example: Creating an MLIP-only architecture
--------------------------------------------

To demonstrate how easy it is to add a new MLIP-only architecture using the base
classes, let's look at the ``mlip_example`` architecture in ``metatrain``. This minimal
architecture always predicts zero energy, serving as a simple template for MLIP
development.

The model (``model.py``) only needs to implement the ``compute_energy`` method:

.. code-block:: python

    from metatrain.utils.mlip import MLIPModel

    class ZeroModel(MLIPModel):
        """A minimal example MLIP model that always predicts zero energy."""

        __checkpoint_version__ = 1

        def __init__(self, hypers, dataset_info):
            super().__init__(hypers, dataset_info)
            # Request a neighbor list with the cutoff from hyperparameters
            cutoff = hypers["cutoff"]
            self.request_neighbor_list(cutoff)

        def compute_energy(
            self,
            edge_vectors,
            species,
            centers,
            neighbors,
            system_indices,
        ):
            # Get the number of systems and return zeros
            n_systems = system_indices.max().item() + 1
            return torch.zeros(n_systems, device=edge_vectors.device)

The trainer (``trainer.py``) only needs to specify whether to use rotational
augmentation:

.. code-block:: python

    from metatrain.utils.mlip import MLIPTrainer

    class ZeroTrainer(MLIPTrainer):
        """Trainer for the ZeroModel."""

        __checkpoint_version__ = 1

        def use_rotational_augmentation(self):
            return False  # No rotational augmentation for this example

That's it! The base classes handle all the training loop, data loading,
composition weights, scaling, checkpointing, and export functionality. This
allows you to focus on implementing the core physics of your model in the
``compute_energy`` method.

The complete example architecture can be found in ``src/metatrain/mlip_example/``.
