Target data Writers
===================

The main entry point for writing target information is

.. autofunction:: metatrain.utils.data.writers.get_writer


Based on the provided filename the writer choses which child writer to use. The mapping
which writer is used for which file type is stored in

.. autodata:: metatrain.utils.data.writers.PREDICTIONS_WRITERS

Implemented Writers
-------------------

.. autofunction:: metatrain.utils.data.writers.Writer
.. autofunction:: metatrain.utils.data.writers.ASEWriter
.. autofunction:: metatrain.utils.data.writers.DiskDatasetWriter
.. autofunction:: metatrain.utils.data.writers.MetatensorWriter
