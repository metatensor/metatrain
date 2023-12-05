Structure Readers
#################

Parsers for obtaining information from structures. All readers return a :py:class:`list`
of :py:class:`rascaline.torch.system.System`. The mapping which reader is used for which
file type is stored in

.. autodata:: metatensor_models.utils.data.readers.structures.STRUCTURE_READERS

Implemented Readers
-------------------

.. autofunction:: metatensor_models.utils.data.readers.structures.read_ase
