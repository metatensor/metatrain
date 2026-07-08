.. _dataset-formats:

Dataset formats
===============

Training, validation, and test data are loaded from the ``read_from`` path in
the ``systems`` section of the options file. The path determines which dataset
reader is used:

- ASE-readable files, such as ``.xyz`` files, are read into memory
- ``.zip`` files are read from disk one structure at a time
- directories are interpreted as memory-mapped datasets, which is usually the
  best option for large datasets.

The ``targets`` and ``extra_data`` sections have the same meaning for all three
formats. Switching from an ``.xyz`` file to a zip archive or memory-mapped
directory usually only requires changing ``read_from``:

.. code-block:: yaml

    training_set:
      systems:
        read_from: dataset.xyz       # ASE-readable file, or
        # read_from: dataset.zip     # zip file, or
        # read_from: my_dataset/     # memory-mapped directory
        length_unit: angstrom
      targets:
        energy:
          key: energy
          unit: eV

ASE-readable files
------------------

Most examples in the documentation use files read by ASE. In this case,
per-structure quantities, such as energies or extra data like ``charge``, are
read from ``atoms.info[key]``. Per-atom quantities, such as forces, are read
from ``atoms.arrays[key]``.

Targets can also be read from metatensor ``.mts`` files by setting the target's
``read_from`` option to the corresponding file. See
:doc:`../getting-started/train_yaml_config` for the full data configuration
reference.

Zip files
---------

A zip dataset (:py:class:`metatrain.utils.data.DiskDataset`) contains one
folder for each structure, named by index:

.. code-block:: text

    dataset.zip
    ├── 0/
    │   ├── system.mta
    │   ├── energy.mts
    │   └── charge.mts
    ├── 1/
    │   └── ...
    └── ...

Each folder contains the structure in ``system.mta``, stored as a
``metatomic.torch.System``. Targets and extra data are stored next to it as
per-structure ``metatensor.torch.TensorMap`` objects in ``.mts`` files. The
file name, without the extension, must match the corresponding entry in the
options file: an ``energy:`` target is read from ``energy.mts``, a
``charge:`` extra-data entry from ``charge.mts``, and so on. The same rule
applies to ``spin_multiplicity.mts`` when using spin conditioning.

The easiest way to create such a file is the
:py:class:`metatrain.utils.data.writers.DiskDatasetWriter` class, which takes
care of the serialization details:

.. code-block:: python

    from metatrain.utils.data.writers import DiskDatasetWriter

    writer = DiskDatasetWriter("dataset.zip")
    for system, targets in zip(systems, target_dicts):
        # targets is a dictionary of per-structure TensorMaps, e.g.
        # {"energy": ..., "charge": ...}
        writer.write([system], targets)
    writer.finish()

Memory-mapped directories
-------------------------

A memory-mapped dataset (:py:class:`metatrain.utils.data.dataset.MemmapDataset`)
is a directory of NumPy and raw binary arrays. It uses the following files:

- ``ns.npy``: number of structures, shape ``(1,)``
- ``na.npy``: cumulative number of atoms per structure, shape ``(ns + 1,)``,
  ``int64``
- ``x.bin``: positions of all atoms, concatenated, shape ``(na[-1], 3)``,
  ``float32``
- ``a.bin``: atomic types of all atoms, concatenated, shape ``(na[-1],)``,
  ``int32``
- ``c.bin`` (optional): cell matrices, shape ``(ns, 3, 3)``, ``float32``
- ``<key>.bin``: one file per target and extra-data entry, named after the
  corresponding ``key`` option. Per-structure quantities have shape
  ``(ns, ..., num_subtargets)``, while per-atom quantities have shape
  ``(na[-1], ..., num_subtargets)``. These arrays are stored as ``float32``.

Forces and stresses of an energy target use the same convention, with file
names taken from their own ``key`` options. Extra data, for example
``charge.bin`` or ``spin_multiplicity.bin``, must contain per-structure scalar
values with shape ``(ns, 1)``.

Spherical targets and virials are not supported in this format. A complete
walkthrough, including forces and stresses, is available in
:ref:`sphx_glr_generated_examples_0-beginner_01-data_preparation.py`.
