Usage - command line
====================

`metatensor-models` is designed for an direct usage from the the command line (cli). The
general help of `metatensor-models` can be accessed using

.. code-block:: bash

    metatensor-models -h

We now demonstrate how to `train` and `evaluate` a model from the command line. For this
example we use the :ref:`architecture-soap-bpnn` architecture and a subset of the `QM9
dataset <https://paperswithcode.com/dataset/qm9>`_. You can obtain the reduced dataset
from our :download:`website <../../static/qm9_reduced_100.xyz>`.

Training
########

For performing a model training in `metatensor-models` uses the hydra framework. Hydra
allows to dynamical composition and override of config files and the command line and
has powerful tools to create multiple training runs with a single command. We will not
explain here how to use hydra in detail, as we only use a few functions ins this example
but rather refer to their good package documentation.

The sub-command to start a model training is

.. code-block:: bash

    metatensor-models train

To train a model you have to define your parameters. This includes the specific
architecture you want to use, the files for obtaining training structures and target
values as well as probable changes

The default model and training hyperparameter for each model are listed in their
corresponding documentation page. We will use the following config to run an example
training


.. literalinclude:: ../../static/parameters.yaml
   :language: yaml

For each training run a new output directory is based on the current date and time is
created. By default, this output directory is used to store Hydra output for the run
(Configuration, Logs etc). You can `override
<https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/>`_ this
behavior in the parameter file. To start the training using the ``parameters.yaml`` in
the current directory type.

.. literalinclude:: ../../../examples/train.sh
    :language: bash


Evaluation
##########

The sub-command to evaluate a already trained model is

.. code-block:: bash

    metatensor-models eval

.. literalinclude:: ../../../examples/eval.sh
    :language: bash
