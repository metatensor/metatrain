.. _label_basic_usage:

Basic Usage
===========

`metatensor-models` is designed for an direct usage from the the command line (cli). The
general help of `metatensor-models` can be accessed using

.. code-block:: bash

    metatensor-models --help

We now demonstrate how to `train` and `evaluate` a model from the command line. For this
example we use the :ref:`architecture-soap-bpnn` architecture and a subset of the `QM9
dataset <https://paperswithcode.com/dataset/qm9>`_. You can obtain the reduced dataset
from our :download:`website <../../static/qm9/qm9_reduced_100.xyz>`.

Training
########

To train models, `metatensor-models` uses the hydra framework. Hydra is a framework
developed by Facebook AI for elegantly configuring complex applications. It's primarily
used for managing command-line arguments in Python applications, allowing for a
structured and dynamic approach to configuration. It allows to dynamical composition and
override of config files and the command line and has powerful tools to create multiple
training runs with a single command. We will not explain here how to use hydra in
detail, as we only use a few functions ins this example but rather refer to their good
package documentation.

The sub-command to start a model training is

.. code-block:: bash

    metatensor-models train

To train a model you have to define your options. This includes the specific
architecture you want to use and the data including the training systems and target
values

The default model and training hyperparameter for each model are listed in their
corresponding documentation page. We will use these minimal options to run an example
training using the default hyperparameters of an SOAP BPNN model

.. literalinclude:: ../../static/qm9/options.yaml
   :language: yaml

For each training run a new output directory in the format
``output/YYYY-MM-DD/HH-MM-SS`` based on the current date and time is created. By
default, this output directory is used to store Hydra's output for the run
(configuration, Logs etc). You can `override
<https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/>`_ this
behavior in the options file. To start the training create an ``options.yaml`` file in
the current directory and type

.. literalinclude:: ../../../examples/basic_usage/usage.sh
    :language: bash
    :lines: 3-8


Evaluation
##########

The sub-command to evaluate an already trained model is

.. code-block:: bash

    metatensor-models eval

Besides the trained `model`, you will also have to provide a file containing the
system and possible target values for evaluation. The system of this ``eval.yaml``
is exactly the same as for a dataset in the ``options.yaml`` file.

.. literalinclude:: ../../static/qm9/eval.yaml
   :language: yaml

Note that the ``targets`` section is optional. If the ``targets`` section is present,
the function will calculate and report RMSE values of the predictions with respect to
the real values as loaded from the ``targets`` section. You can run an evaluation by
typing

.. literalinclude:: ../../../examples/basic_usage/usage.sh
    :language: bash
    :lines: 9-24


Molecular simulations
#####################

The trained model can also be used to run molecular simulations.
You can find how in the :ref:`tutorials` section.
