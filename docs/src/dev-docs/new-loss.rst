.. _adding-new-loss:

Adding a new loss function
==========================

This page describes the required classes and files necessary for adding a new
loss function to ``metatrain``. Defining a new loss can be useful in case some extra 
data has to be used to compute the loss.

Loss functions in ``metatrain`` are implemented as subclasses of
:py:class:`metatrain.utils.loss.LossInterface`. This interface defines the
required method :py:meth:`compute`, which takes the model predictions and
the ground truth values as input and returns the computed loss value. The 
:py:meth:`compute` method accepts an additional argument ``extra_data`` on top of 
``predictions`` and ``targets``, that can be used to pass any extra information needed 
for the loss computation.

.. code-block:: python

    def NewLoss(LossInterface):
        def __init__(self, hypers):
            # Initialize the loss function with hyperparameters
            ...

        def compute(self, predictions, targets, extra_data):
            # Compute and return the loss value
            ...