.. _adding-new-architecture:

Adding a new architecture
=========================

This page describes the required classes and files necessary for adding a new
architecture to ``metatrain`` as experimental or stable architecture as described on the
:ref:`architecture-life-cycle` page. For **examples** refer to the already existing
architectures inside the source tree.

To work with ``metatrain`` any architecture has to follow the same public API to be
called correctly within the :py:func:`metatrain.cli.train` function to process the
user's options. In brief, the core of the ``train`` function looks similar to these
lines

.. code-block:: python

    from architecture import __model__ as Model
    from architecture import __trainer__ as Trainer

    hypers = {...}
    dataset_info = DatasetInfo()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model = Model.load_checkpoint(checkpoint)
        trainer = Trainer.load_checkpoint(checkpoint, hypers["training"])
        model = model.restart(dataset_info)
    else:
        model = Model(hypers["model"], dataset_info)
        trainer = Trainer(hypers["training"])

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


To follow this, a new architecture has to define two classes

- a ``Model`` class, defining the core of the architecture. This class must implement
  the interface documented below in :py:class:`ModelInterface`
- a ``Trainer`` class, used to train an architecture and produce a model that can be
  evaluated and exported. This class must implement the interface documented below in
  :py:class:`TrainerInterface`.

.. note::

    ``metatrain`` does not know the types and numbers of targets/datasets an
    architecture can handle. As a result, it cannot generate useful error messages when
    a user attempts to train an architecture with unsupported target and dataset
    combinations. Therefore, it is the responsibility of the architecture developer to
    verify if the model and the trainer support the provided train_datasets and
    val_datasets passed to the Trainer, as well as the dataset_info passed to the model.

To comply with this design each architecture has to implement a couple of files
inside a new architecture directory, either inside the ``experimental`` subdirectory or
in the ``root`` of the Python source if the new architecture already complies with all
requirements to be stable. The usual structure of architecture looks as

.. code-block:: text

    myarchitecture
        ├── model.py
        ├── trainer.py
        ├── __init__.py
        ├── default-hypers.yaml
        └── schema-hypers.json

.. note::
    A new architecture doesn't have to be registered somewhere in the file tree of
    ``metatrain``. Once a new architecture folder with the required files is created
    ``metatrain`` will include the architecture automatically.

Model class (``model.py``)
--------------------------
The ``ModelInterface``, is recommended to be located in a file called ``model.py``
inside the architecture folder is the main model class and must implement a
``save_checkpoint()``, ``load_checkpoint()`` as well as a ``restart()`` and ``export()``
method.

.. code-block:: python

    from metatensor.torch.atomistic import MetatensorAtomisticModel, ModelMetadata

    class ModelInterface:

        __supported_devices__ = ["cuda", "cpu"]
        __supported_dtypes__ = [torch.float64, torch.float32]
        __default_metadata__ = ModelMetadata(
            references = {"implementation": ["ref1"], "architecture": ["ref2"]}
        )

        def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo):
            self.hypers = model_hypers
            self.dataset_info = dataset_info

        @classmethod
        def load_checkpoint(
            cls,
            checkpoint: Dict[str, Any],
            context: Literal["restart", "finetune", "export"],
        ) -> "ModelInterface":
            """Create a model from a checkpoint's state dictionary.

            :param checkpoint: Checkpoint's state dictionary.
            :param context: Purpose of the model to load from the checkpoint file.
                Required values are "restart" and "finetune", "export" but can be
                extended to other values.
            """
            pass

        def restart(cls, dataset_info: DatasetInfo) -> "ModelInterface":
            """Restart training.

            This function is called whenever training restarts, with the same or a
            different dataset.

            It enables transfer learning (changing the targets), and fine-tuning (same
            targets, different datasets)
            """
            pass

            def export(
        self, metadata: Optional[ModelMetadata] = None
    ) -> MetatensorAtomisticModel:
            pass

Note that the ``ModelInterface`` does not necessarily inherit from
:py:class:`torch.nn.Module` since training can be performed in any way.
``__supported_devices__`` and ``__supported_dtypes__`` can be defined to set the
capabilities of the model. These two lists should be sorted in order of preference since
``metatrain`` will use these to determine, based on the user request and
machines' availability, the optimal ``dtype`` and ``device`` for training.

The ``__default_metadata__`` is a class attribute that can be used to provide references
that will be stored in the exported model. The references are stored in a dictionary
with keys ``implementation`` and ``architecture``. The ``implementation`` key should
contain references to the software used in the implementation of the architecture, while
the ``architecture`` key should contain references about the general architecture.

The ``export()`` method is required to transform a trained model into a standalone file
to be used in combination with molecular dynamic engines to run simulations. We provide
a helper function :py:func:`metatrain.utils.export.export` to export a torch
model to an :py:class:`MetatensorAtomisticModel
<metatensor.torch.atomistic.MetatensorAtomisticModel>`.

Trainer class (``trainer.py``)
------------------------------
The ``TrainerInterface`` class should have the following signature with required
methods for ``train()``, ``save_checkpoint()`` and ``load_checkpoint()``.

.. code-block:: python

    class TrainerInterface:
        def __init__(self, train_hypers):
            self.hypers = train_hypers

        def train(
            self,
            model: ModelInterface,
            dtype: torch.dtype,
            devices: List[torch.device],
            train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
            val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
            checkpoint_dir: str,
        ) -> None: ...

        def save_checkpoint(self, path: Union[str, Path]) -> None: ...

        @classmethod
        def load_checkpoint(
            cls,
            checkpoint: Dict[str, Any],
            train_hypers: Dict[str, Any],
            context: Literal["restart", "finetune"],
        ) -> "TrainerInterface":
            """Create a trainer from a checkpoint's state dictionary.

            :param checkpoint: Checkpoint's state dictionary.
            :param context: Purpose of the model to load from the checkpoint file.
                Required values are "restart" and "finetune" but can be
                extended to other values.
            :param train_hypers: Hyperparameters used to create the trainer.
            """
            pass

The format of checkpoints is not defined by ``metatrain`` and can be any format that
can be loaded by the trainer (to restart training) and by the model (to export the
checkpoint). The only requirements are that the checkpoint must be loadable with
``torch.load()``, it must be a dictionary, and it must contain the name of the
architecture under the ``architecture_name`` key.

Init file (``__init__.py``)
---------------------------
The names of the ``ModelInterface`` and the ``TrainerInterface`` are free to choose but
should be linked to constants in the ``__init__.py`` of each architecture. On top of
these two constants the ``__init__.py`` must contain constants for the original
``__authors__`` and current ``__maintainers__`` of the architecture.

.. code-block:: python

    from .model import ModelInterface
    from .trainer import TrainerInterface

    __model__ = ModelInterface
    __trainer__ = TrainerInterface

    __authors__ = [
        ("Jane Roe <jane.roe@myuniversity.org>", "@janeroe"),
        ("John Doe <john.doe@otheruniversity.edu>", "@johndoe"),
    ]

    __maintainers__ = [("Joe Bloggs <joe.bloggs@sotacompany.com>", "@joebloggs")]

:param __model__: Mapping of the custom ``ModelInterface`` to a general one to be loaded
    by ``metatrain``.
:param __trainer__: Same as ``__MODEL_CLASS__`` but the Trainer class.
:param __authors__: Tuple denoting the original authors with an email address and GitHub
    handle of an architecture. These do not necessarily be in charge of maintaining the
    architecture.
:param __maintainers__: Tuple denoting the current maintainers of the architecture. Uses
    the same style as the ``__authors__`` constant.


Default Hyperparamers (``default-hypers.yaml``)
-----------------------------------------------
The default hyperparameters for each architecture should be stored in a YAML file
``default-hypers.yaml`` inside the architecture directory. Reasonable default hypers are
required to improve usability. The default hypers must follow the structure

.. code-block:: yaml

    name: myarchitecture

    model:
        ...

    training:
        ...

``metatrain`` will parse this file and overwrite these default hypers with the
user-provided parameters and pass the merged ``model`` section as a Python dictionary to
the ``ModelInterface`` and the ``training`` section to the ``TrainerInterface``.

JSON schema (``schema-hypers.yaml``)
------------------------------------
To validate the user's input hyperparameters we are using `JSON schemas
<https://json-schema.org/>`_ stored in a schema file called ``schema-hypers.json``. For
an :ref:`experimental architecture <architecture-life-cycle>` it is not required to
provide such a schema along with its default hypers but it is highly recommended to
reduce possible errors of user input like typos in parameter names or wrong sections. If
no ``schema-hypers.json`` is provided no validation is performed and user hypers are
passed to the architecture model and trainer as is.

To create such a schema start by using `online tools <https://jsonformatter.org>`_ that
convert the ``default-hypers.yaml`` into a JSON schema. Besides online tools, we also
had success using ChatGPT/LLM for this for conversion.

Documentation
-------------

Each new architecture should be added to ``metatrain``'s documentation. A short page
describing the architecture and its default hyperparameters will be sufficient. You
can take inspiration from existing architectures. The various targets that the
architecture can fit should be added to the table in the "Fitting generic targets"
section.
