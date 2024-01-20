Structure and Target data Readers
=================================

The main entry point for reading structure and target information are the two reader
functions

.. autofunction:: metatensor.models.utils.data.read_structures
.. autofunction:: metatensor.models.utils.data.read_targets

Target type specific readers
----------------------------

:func:`metatensor.models.utils.data.read_targets` uses sub-functions to parse supported
target properties like the `energy` or `forces`. Currently we support reading the
following target properties via

.. autofunction:: metatensor.models.utils.data.read_energy
.. autofunction:: metatensor.models.utils.data.read_forces
.. autofunction:: metatensor.models.utils.data.read_virial
.. autofunction:: metatensor.models.utils.data.read_stress

File type specific readers
--------------------------

Based on the provided `file_format` they chose which sub-reader they use. For details on
these refer to their documentation

.. toctree::
   :maxdepth: 1

   structure
   targets
