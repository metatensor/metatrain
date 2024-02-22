Structure Readers
#################

Parsers for obtaining information from structures. All readers return a :py:class:`list`
of :py:class:`metatensor.torch.atomistic.System`. The mapping which reader is used for
which file type is stored in

.. autodata:: metatensor.models.utils.data.readers.structures.STRUCTURE_READERS

Implemented Readers
-------------------

.. autofunction:: metatensor.models.utils.data.readers.structures.read_structures_ase
