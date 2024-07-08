Readers
#######

Parsers for obtaining *system* and *target* information from disk. Currently,
``metatrain`` support the following libraries for reading data

.. list-table::
  :header-rows: 1

  * - Library
    - Supported targets
    - Default file format
  * - ``ase``
    - system, energy, forces, stress, virials
    - ``.xyz``, ``.extxyz``

One can override these defaults by setting the ``reader`` option to the desired library.
Below the synopsis of the reader functions in details.

system and Target data Readers
==============================

The main entry point for reading system and target information are the two reader
functions

.. autofunction:: metatrain.utils.data.read_systems
.. autofunction:: metatrain.utils.data.read_targets

Target type specific readers
----------------------------

:func:`metatrain.utils.data.read_targets` uses sub-functions to parse supported
target properties like the `energy` or `forces`. Currently we support reading the
following target properties via

.. autofunction:: metatrain.utils.data.read_energy
.. autofunction:: metatrain.utils.data.read_forces
.. autofunction:: metatrain.utils.data.read_virial
.. autofunction:: metatrain.utils.data.read_stress
