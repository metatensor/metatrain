Target data Readers
###################

Parsers for obtaining information from structures. All readers return a of
:py:class:`metatensor.torch.TensorMap`. The mapping which reader is used for which file
type is stored in

.. autodata:: metatensor_models.utils.data.readers.targets.TARGET_READERS

Implemented Readers
-------------------

.. autofunction:: metatensor_models.utils.data.readers.targets.read_ase
