Auxiliary outputs
=================

These outputs, which are generally idenfified by the ``mtt::aux::`` prefix,
represent additional information that the model may provide. They are not
conventional trainable outputs, in the sense that they do not correspond to
training targets. For example, such quantities might be the internal
representation of the model, uncertainty estimates, or non-trainable quantities.

The following auxiliary outputs are currently supported by one or more
architectures in ``metatrain``:

- :ref:`mtt-aux-target-last-layer-features`: The representation of the model
  at the last layer, before the final linear transformation to produce
  target ``target``.
- :ref:`mtt-aux-target-uncertainty`: An uncertainty estimate for target
  ``target``, as computed by the LLPR module.
- :ref:`mtt-aux-target-ensemble`: An ensemble prediction for target
  ``target``, as computed by the LLPR module.


.. _mtt-aux-target-last-layer-features:

``mtt::aux::{target}_last_layer_features``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This output contains the representation of the model at the last layer, before
the final linear transformation to produce target ``target``. If the model
produces multiple targets, the corresponding representations might be different.

The last-layer features come in one of two forms:

- invariant or non-symmetrized features are a single block, used for all the blocks
  of the target;
- equivariant features have one block for each block of the target, with the same
  keys and components. The last layer maps each block of features to the block of
  the target with the same key, with the same weights for all the components.

.. list-table:: Metadata for last-layer features
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"`` if invariant or non-symmetrized, the keys of the target if
      equivariant
    - if invariant or non-symmetrized, a single dimension named ``"_"``, with a
      single entry set to ``0``, or, if equivariant, the keys of the target.

  * - samples
    - ``["system", "atom"]`` or ``["system"]``
    - if a ``sample_kind="atom"`` output is requested, the sample names will
      be ``["system", "atom"]``, if ``sample_kind="system"`` they will
      be ``["system"]``.

      ``"system"`` ranges from 0 to the number of systems given as input to
      the model. ``"atom"`` ranges between 0 and the number of
      atoms/particles in the corresponding system.

  * - components
    - No components if invariant or non-symmetrized, the components of the target
      if equivariant
    - Nothing if invariant or non-symmetrized, the components of the target if
      equivariant.

  * - properties
    - ``"feature"``
    - the last-layer features have a single property dimension named
      ``"feature"``, with entries ranging from 0 to the number of features
      in the last layer.

Last-layer features are supported by the following architectures:

+-----------------+-----------------------+------------------+-------------------+---------------------+
| :ref:`arch-pet` | :ref:`arch-soap_bpnn` | :ref:`arch-mace` | :ref:`arch-space` | :ref:`arch-flashmd` |
+-----------------+-----------------------+------------------+-------------------+---------------------+

SOAP-BPNN only provides them for scalar targets.

.. _mtt-aux-target-uncertainty:

``mtt::aux::{target}_uncertainty``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This output contains an uncertainty estimate for target ``target``, as computed
by the LLPR module. The metadata for this output is the same as for the
corresponding target: there is one uncertainty for each component and property of
the target.

When ``target`` is ``energy``, this output is instead named
``energy_uncertainty``, creating the `corresponding standard metatomic
output <mta-energy-uncertainty_>`_.

.. _mta-energy-uncertainty: https://docs.metatensor.org/metatomic/latest/outputs/energy.html#energy-uncertainty

This output is only supported by the :ref:`LLPR architecture <arch-llpr>`.

.. _mtt-aux-target-ensemble:

``mtt::aux::{target}_ensemble``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This output contains an ensemble prediction for target ``target``, as computed
by the LLPR module. The metadata for this output is the same as for the
corresponding target, except that the properties are named
``["ensemble_member", ...]``, followed by the property names of the target, with
one entry for each ensemble member and property of the target.

When ``target`` is ``energy``, this output is instead named
``energy_ensemble``, creating the `corresponding standard metatomic
output <mta-energy-ensemble_>`_, with a single ``"energy"`` property that holds the
index of each ensemble member.

.. _mta-energy-ensemble: https://docs.metatensor.org/metatomic/latest/outputs/energy.html#energy-ensemble

This output is only supported by the :ref:`LLPR architecture <arch-llpr>`.
