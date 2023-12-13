Target data Reader
##################

Parsers for obtaining information from target files. All readers return a of
:py:class:`metatensor.torch.TensorMap`. The mapping which reader is used for which file
type is stored in

.. autodata:: metatensor.models.utils.data.readers.targets.TARGET_READERS

Implemented Readers
-------------------

.. autofunction:: metatensor.models.utils.data.readers.targets.read_ase
