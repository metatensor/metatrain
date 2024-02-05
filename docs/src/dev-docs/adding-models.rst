.. _adding-new-models:

Adding new models
=================

To be done.
This section contains a quick introduction to adding a new model to
``metatensor-models``. In order to add a new model, two things are
required:

- a ``Model`` class
- a ``train_model`` function

The ``Model`` class should inherit from ``torch.nn.Module`` and implement
an ``__init__`` method and a ``forward`` method. The ``__init__`` method
should have the following signature:

```python
class Model(torch.nn.Module):
    def __init__(
        self, capabilities: ModelCapabilities, hypers: Dict
    ) -> None:
```

For more information on the ``ModelCapabilities`` class, see
:py:class:`metatensor.torch.atomistic.ModelCapabilities`.

The ``forward`` method should have the following signature:

```python
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
```

For more information on the ``System`` and ``ModelOutput`` classes, see
:py:class:`metatensor.torch.atomistic.System` and
:py:class:`metatensor.torch.atomistic.ModelOutput`.

The ``train_model`` function should have the following signature:

```python
def train(
    train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    validation_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    model_capabilities: ModelCapabilities,
    hypers: Dict = DEFAULT_HYPERS,
    output_dir: str = ".",
) -> torch.nn.Module:
```

For more information on the ``Dataset`` class, see
:py:class:`metatensor.operations.utils.data.Dataset`.

Finally, the new model should implement default hyperparameters in the
``src/metatensor/models/cli/conf/architecture`` folder.

A good example of a simple model is the ``SOAP-BPNN`` model.
