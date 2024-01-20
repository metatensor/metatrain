Override Architecture's Default Parameters
==========================================

In our initial tutorial, we used default parameters to train a model employing the
SOAP-BPNN architecture, as shown in the following config:

.. literalinclude:: ../../static/options.yaml
   :language: yaml

While default parameters often serve as a good starting point, depending on your
training target and dataset, it might be necessary to adjust the architecture's
parameters.

First, familiarize yourself with the specific parameters of the architecture you intend
to use. We provide a list of all architectures and their parameters in the
:ref:`available-architectures` section. For example, the parameters of the SOAP-BPNN
models are detailed at :ref:`architecture-soap-bpnn`.

Modifying Parameters (yaml)
---------------------------

As an example, let's increase the number of epochs (``num_epochs``) and the ``cutoff``
radius of the SOAP descriptor. To do this, create a new section in the ``options.yaml``
named ``architecture``. Within this section, you can override the architecture's
hyperparameters. The adjustments for ``num_epochs`` and ``cutoff`` look like this:

.. code-block:: yaml

   defaults:
      - architecture: soap_bpnn
      - _self_

   architecture:
      model:
         soap:
            cutoff: 7.0
      training:
         num_epochs: 200

   training_set:
   structures: "qm9_reduced_100.xyz"
   targets:
      energy:
         key: "U0"

   test_set: 0.1
   validation_set: 0.1

Modifying Parameters (Command Line Overrides)
---------------------------------------------

For quick adjustments, command-line overrides are also an option. The changes above can
be achieved by:

.. code-block:: bash

   metatensor-models train options.yaml \
      -y architecture.model.soap.cutoff=7.0 architecture.training.num_epochs=200

Here, the ``-y`` flag is used to parse the override flags. More details on override
syntax are available at https://hydra.cc/docs/advanced/override_grammar/basic/.

.. note::

   For your reference and reproducibility purposes `metatensor-models` always writes the
   fully expanded options to the ``.hydra`` subdirectory inside the ``output``
   directory of your current training run.


Understanding the Defaults Section
----------------------------------

You may have noticed the ``defaults`` section at the beginning of each file. This list
dictates which defaults should be loaded and how to compose the final config object and
is conventionally the first item in the config.

Append ``_self_`` to the end of the list to have your primary config override values
from the Defaults List. If you do not add a ``_self_`` entry still your primary config
Overrides values from the Defaults List, but Hydra will throw a warning. For more
background, visit https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/.
