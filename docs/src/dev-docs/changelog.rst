.. _changelog:

Changelog
=========

All notable changes to ``metatrain`` are documented here, following the `keep a
changelog <https://keepachangelog.com/en/1.1.0/>`_ format. This project follows
`Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

.. Possible sections for each release:

.. Fixed
.. #####

.. Added
.. #####

.. Changed
.. #######

.. Removed
.. #######

Unreleased
----------

Version 2025.6 - 2025-04-27
---------------------------

Fixed
#####

- ``PET`` can now evaluate on single-atom structures without crashing
- The metatrain dataloader doesn't load all batches ahead of each epoch anymore

Added
#####

- ``NanoPET`` and ``PET`` can now train on non-conservative stresses
- Users can now choose the name of the extension directory in ``mtt train`` and
  ``mtt export`` via the ``--extensions`` (or ``-e``) option
- Update to ``metatensor-torch-0.7.6``, adding support for torch 2.7
- ``PET`` now supports gradient clipping as a new training hyperparameter

Changed
#######

- Training and exporting models without extensions will no longer lead to the creation
  of an empty directory for the extensions
- The SOAP-BPNN model now uses ``torch-spex`` instead of ``featomic`` as its SOAP
  backend
- ``PET`` from the previous version is now deprecated and accessible as
  ``deprecated.pet``, while the old ``NativePET`` (``experimental.nativepet``) is
  now called ``PET`` (``pet`` from training option files)

Version 2025.5 - 2025-04-13
---------------------------

Fixed
#####

- Fix more composition model issues

Added
#####

- Update to ``metatensor-torch-0.7.5`` to allow training on ``non_conservative_forces``
  and  ``non_conservative_stress`` targets
- Add ``NativePET`` as a readable, efficient, backward-compatible PET implementation
- Added Wandb logger
- Save loss history in a ``.csv`` file

Version 2025.4 - 2025-03-29
---------------------------

Changed
#######

- upgraded to ``metatensor.torch`` 0.7.4, which gives access to batched ASE evaluation

Version 2025.3 - 2025-03-25
---------------------------

Fixed
#####

- Fixed a bug in the composition model, affecting SOAP-BPNN and nanoPET

Changed
#######

- :func:`metatrain.util.io.load_model` does not copy a remote model to the current
  directory.

Version 2025.2 - 2025-03-11
---------------------------

Added
#####

- Implement a long-range featurizer as a utility for all models
- Speed up system preparation

Changed
#######

- Remove biases in SOAP-BPNN's linear layers

Fixed
#####

- Fix NanoPET multi-GPU error message
- Fix ``device`` for fixed composition weights

Version 2025.1 - 2025-02-20
---------------------------

Added
#####

- Support for Python 3.13 and ``ase`` >= 3.23

Fixed
#####

- Some irrelevant autograd warnings

Version 2025.0 - 2025-02-19
---------------------------

Added
#####

* First release outside of the lab
