.. _adding-new-architecture:

Adding a new architecture
=========================

This page describes the required classes and files necessary for adding a new
architecture to ``metatrain`` as experimental or stable architecture as
described on the :ref:`architecture-life-cycle` page.

What is a ``metatrain`` architecture?
-------------------------------------

To work with ``metatrain`` any architecture has to follow the same public API to
be called correctly within the :py:func:`metatrain.cli.train` function to
process the user's options. In brief, the core of the ``train`` function looks
similar to these lines

.. code-block:: python

    from architecture import __model__ as Model
    from architecture import __trainer__ as Trainer

    hypers = {...}
    dataset_info = DatasetInfo()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)

        trainer = Trainer.load_checkpoint(
            checkpoint, hypers=hypers["training"], context="restart")
        model = Model.load_checkpoint(checkpoint, context="restart")
        model = model.restart(dataset_info)
    else:
        trainer = Trainer(hypers["training"])

        if hasattr(hypers["training"], "finetune"):
            checkpoint = hypers["training"]["finetune"]["read_from"]
            model = Model.load_checkpoint(path=checkpoint, context="finetune")
        else:
            model = Model(hypers["model"], dataset_info)

    trainer.train(
        model=model,
        dtype=dtype,
        devices=[],
        train_datasets=[],
        val_datasets=[],
        checkpoint_dir="path",
    )

    model.save_checkpoint("model.ckpt")

    mts_atomistic_model = model.export()
    mts_atomistic_model.export("model.pt", collect_extensions="extensions/")

General code structure
----------------------

To follow this, a new architecture has to define two classes

- a ``Model`` class, defining the core of the architecture. This class must
  implement the interface documented in
  :py:class:`metatrain.utils.abc.ModelInterface`
- a ``Trainer`` class, used to train an architecture and produce a model that
  can be evaluated and exported. This class must implement the interface
  documented below in :py:class:`metatrain.utils.abc.TrainerInterface`.

.. note::

    ``metatrain`` does not know the types and numbers of targets/datasets an
    architecture can handle. As a result, it cannot generate useful error
    messages when a user attempts to train an architecture with unsupported
    target and dataset combinations. Therefore, it is the responsibility of the
    architecture developer to verify if the model and the trainer support the
    provided train_datasets and val_datasets passed to the Trainer, as well as
    the dataset_info passed to the model.

To comply with this design each architecture has to implement a couple of files
inside a new architecture directory, either inside the ``experimental``
subdirectory or in the ``root`` of the Python source if the new architecture
already complies with all requirements to be stable. The usual structure of
architecture looks as

.. code-block:: text

    myarchitecture
        ├── __init__.py
        ├── model.py
        └── trainer.py

.. note::
    Because achitectures can live in either ``src/metatrain/<architecture>``,
    ``src/metatrain/experimental/<architecture>``, or
    ``src/metatrain/deprecated/<architecture>``; the code inside should use
    absolute imports use the tools provided by metatrain.

    .. code-block:: python

        # do not do this
        from ..utils.dtype import dtype_to_str

        # Do this instead
        from metatrain.utils.dtype import dtype_to_str

Model class (``model.py``)
--------------------------

A model class has to follow the interface defined in
:py:class:`~metatrain.utils.abc.ModelInterface`. That is, all the
methods that are marked as abstract in the interface must be implemented
with the indicated API (same arguments and same return). At first sight,
the interface might feel overwhelming, therefore here is a summary of the
steps to take to implement a new model class:


- Implement the ``__init__`` method, which takes as input the model hyperparameters
  and the dataset information. This should initialize your model.
- Implement the ``forward`` method, which defines the forward pass of the model.
- Add some class attributes with ``__names_like_this__`` that will help metatrain
  understand how to treat your model. They are listed and described in the
  :py:class:`~metatrain.utils.abc.ModelInterface` documentation.
- Implement the rest of abstract methods, which in general deal with handling
  checkpoints, exporting the model, and restarting training from a checkpoint.

Here is an incomplete example of what a model implementation looks like:

.. literalinclude:: ./dummy_model.py
    :language: python

Trainer class (``trainer.py``)
------------------------------

A trainer class has to follow the interface defined in
:py:class:`~metatrain.utils.abc.TrainerInterface`. That is, all the
methods that are marked as abstract in the interface must be implemented
with the indicated API (same arguments and same return). We recommend
looking at existing implementations of trainers for inspiration. They
will look something like this:

.. literalinclude:: ./dummy_trainer.py
    :language: python

Init file (``__init__.py``)
---------------------------

You are free to name the ``Model`` and ``Trainer`` classes as you want. These
classes should then be made available in the ``__init__.py`` under the names
``__model__`` and ``__trainer__`` so metatrain knows where to find them.
``__init__.py`` must also contain definition for the original ``__authors__``
and current ``__maintainers__`` of the architecture.

.. code-block:: python

    from .model import ModelInterface
    from .trainer import TrainerInterface

    # class to use as the architecture's model
    __model__ = ModelInterface
    # class to use as the architecture's trainer
    __trainer__ = TrainerInterface

    # List of the original authors of the architecture, each with an email
    # address and GitHub handle.
    #
    # These authors are not necessarily currently in charge of maintaining the code
    __authors__ = [
        ("Jane Roe <jane.roe@myuniversity.org>", "@janeroe"),
        ("John Doe <john.doe@otheruniversity.edu>", "@johndoe"),
    ]

    # Current maintainers of the architecture code, using the same
    # style as ``__authors__``
    __maintainers__ = [("Joe Bloggs <joe.bloggs@sotacompany.com>", "@joebloggs")]

Hyperparameters documentation
-----------------------------

In the previous sections we mentioned that both the ``Model`` and ``Trainer``
need their hyperparameters to be indicated in their ``__hypers_cls__`` class
attribute. At the beginning, a simple class specifying default values for
each hyperparameter is sufficient, i.e.:

.. code-block:: python

    class ModelHypers:
        alpha = 1.2
        mode = "strict"

    class Model(ModelInterface):
        __hypers_cls__ = ModelHypers

However, for an architecture to be considered stable, the hyperparameters
must be:

- Documented using docstrings.
- Type hinted using Python's type hinting system.

This will allow ``metatrain`` to:

- Automatically generate documentation pages for the architecture
  (see :ref:`newarchitecture-documentation`).
- Validate the inputs provided by the user through the CLI. This is
  done through ``pydantic``.

For this, ``metatrain``'s convention is to use ``TypedDict`` classes to
define the hyperparameters. You can take inspiration from existing
architectures for complex examples of type hinting, but here is how
it would look like for the simple example above:

.. code-block:: python

    from typing import Literal
    from typing_extensions import TypedDict
    from metatrain.torch import DatasetInfo

    class ModelHypers(TypedDict):
        """Hyperparameters for the Model architecture."""

        alpha: float = 1.2
        """Scaling factor for the model predictions."""
        mode: Literal["strict", "lenient"] = "strict"
        """Mode of operation for the model."""

    class Model(ModelInterface):
        __hypers_cls__ = ModelHypers

        def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo):
            super().__init__(hypers, dataset_info)
            ...

Note that we also added the hypers class as type hint for the ``hypers``
argument of the ``__init__`` method of the model. This is not required
by ``metatrain`` to work, but it will help static type checkers to catch
bugs in your code, as well as improving the development experience in
IDE's like VSCode or PyCharm. So we strongly recommend it!

.. _newarchitecture-documentation:

Documentation
-------------

By following the guidelines for documenting hyperparameters, ``metatrain``
**will automatically generate a documentation page for the new architecture**.
This documentation page will contain information about how to install your
architecture, the default hyperparameters, and the descriptions of all
the hyperparameters for both the model and the trainer.

However, **you are always welcome to add more information to the documentation
page**, such as a description of the architecture, references to relevant
papers, or examples of usage. To do this, you just have to create a file
called ``<your_architecture_name>.rst`` in the
``docs/src/architectures/templates`` folder. This folder is called ``templates``
because the files here are preprocessed to generate the final documentation
``.rst`` files. This is done by simply reading the file as a python string
and calling ``template_string.format(...)`` with some variables passed that
you can use in your template. However, you are completely free to not use
any of the variables and just write a static ``.rst`` file if you prefer.
It will be treated as a template, but the effect will be simply that your
file will get copied to the final documentation folder without any changes.

.. _ckpt_version:

Checkpoint versioning
----------------------

Checkpoints are used to save the weights of a models and the state of the
trainer to disk, enabling to restart interupted training runs, to fine-tune
existing models on new dataset, and to export standalone models based on
TorchScript.

A checkpoint created for one version might need to be read again
by a later version of the architecture, where the internal structure might have
changed. To enable this, all ``Model`` classes are required to have a
``__checkpoint_version__`` class attribute containing the version of the
checkoint, as a strictly inreasing integer. Additionally, architectures should
provide an ``upgrade_checkpoint(checkpoint: Dict) -> Dict`` function, that will
be called when a user is trying to load some outdated checkpoint. This function
is responsible for updating the checkpoint data and returning a checkpoint
compatible with the current version.

Similarly, the ``Trainer`` state is also saved in checkpoint and used to restart
training. All trainer must thus have a ``__checkpoint_version__`` class
attribute as well as a ``upgrade_checkpoint(checkpoint: Dict) -> Dict`` function
to updgrade from previous checkpoints.
