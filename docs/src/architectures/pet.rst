.. _architecture-pet:

PET
===

.. warning::

  This is an **experimental model**.  You should not use it for anything important.


PET Hyperparameters
----------------------------

There are two significant groups of hyperparameters controlling PET fits. The 
first group consists of the hyperparameters related to the model architecture 
itself, such as the number of layers, type of activation function, etc. The 
second group consists of settings that control how the fitting is done, such 
as batch size, the total number of epochs, learning rate, parameters of the learning rate 
scheduler, and so on.


Within conventional wisdom originating from ``traditional`` models, such as linear and kernel 
regression, the second group of hyperparameters that controls the 
optimizer's behavior might seem unimportant. Indeed, when fitting linear
or kernel models, the exact value of the optimum is usually achieved by 
linear algebra methods, and thus, the particular choice of optimizer 
has little importance. 


However, with deep neural networks, the situation is drastically different. 
The exact minimum of the loss is typically never achieved; instead, the model 
asymptotically converges to it during fitting. It is essential to ensure that 
the total number of epochs is sufficient for the model to approach the optimum 
closely, thus achieving good accuracy.

**In the case of PET, there is only one hyperparameter that MUST be manually 
adjusted for each new dataset: the cutoff radius.** The selected cutoff 
significantly impacts the model's accuracy and fitting/inference times, making 
it very sensitive to this hyperparameter. All other hyperparameters can be safely set 
to their default values. The next reasonable step (after fitting with default settings), especially for large datasets 
is to try to increase the duration of fitting and see if it improves the accuracy of the obtained model. 

Selection of R_CUT
~~~~~~~~~~~~~~~~~~

A good starting point is to select a cutoff radius that ensures about 20 
neighbors on average. This can be done by analyzing the neighbor lists for 
different cutoffs before launching the training script. `This 
<https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html>`_ is an example of a 
neighbor list constructor in Python.

For finite configurations, such as small molecules in COLL/QM9/rmd17 datasets, 
it makes sense to select R_CUT large enough to encompass the whole molecule. 
For instance, it can be set to 100 Å, as there are no numerical instabilities 
for arbitrarily large cutoffs.

The hyperparameter for the cutoff radius is called ``R_CUT.``

Selection of fitting duration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second most important group of settings is the one that adjusts the fitting duration 
of the model. Unlike specifying a dataset-specific cutoff radius, this step is 
optional since reasonable results can be obtained with the default fitting 
duration. The time required to fit the model is a complex function of the 
model's size, the dataset's size, and the complexity of the studied 
interatomic interactions. The default value might be insufficient for 
large datasets. If the model is still underfit after the predefined number of 
epochs, the fitting procedure can be continued by relaunching the fitting 
script with the same calculation name.

However, the total number of epochs is only part of the equation. The other 
key aspect is the rate at which the learning rate decays. We use `StepLR 
<https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html>`_ 
as a learning rate scheduler. To increase the overall fitting time, one needs 
to specify a larger step size, which controls how fast StepLR decreases the 
learning rate. This can be done by specifying the hyperparameter 
``SCHEDULER_STEP_SIZE.`` 

For hyperparameters like ``SCHEDULER_STEP_SIZE,`` ``EPOCH_NUM,`` ``BATCH_SIZE,`` and 
``EPOCHS_WARMUP,`` either normal or atomic versions can be specified. Atomic 
versions are termed ``SCHEDULER_STEP_SIZE_ATOMIC,`` ``EPOCH_NUM_ATOMIC,`` 
``BATCH_SIZE_ATOMIC,`` and ``EPOCHS_WARMUP_ATOMIC.`` For instance, using the same 
batch size for datasets with structures of very different sizes makes no 
sense. If one dataset contains molecules with 10 atoms on average and another 
contains nanoparticles with 1000 atoms, it makes sense to use a 100 times 
larger batch size in the first case. If ``BATCH_SIZE_ATOMIC`` is specified, the 
normal batch size is computed as BATCH_SIZE = BATCH_SIZE_ATOMIC / 
(average_number_of_atoms_in_the_training_dataset). Similar logic applies to 
``SCHEDULER_STEP_SIZE,`` ``EPOCH_NUM,`` and ``EPOCHS_WARMUP.`` In these cases, 
normal versions are obtained by division by the total number of atoms of 
structures in the training dataset. All default values are given by atomic 
versions for better transferability across various datasets.

To increase the step size of the learning rate scheduler by, for example, 2 
times, take the default value for ``SCHEDULER_STEP_SIZE_ATOMIC`` from the 
default_hypers and specify a value that's twice as large.



Default Hyperparameters
-----------------------
The default hyperparameters for the PET model are:

.. literalinclude:: ../../../src/metatensor/models/experimental/pet/default-hypers.yaml
   :language: yaml
