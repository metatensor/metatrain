Outputs
=======

Naming
------

The name and format of the outputs in ``metatensor-models`` are based on
those of the ``metatensor.torch.atomistic``
<https://lab-cosmo.github.io/metatensor/latest/atomistic/outputs.html>_
package. An immediate example is given by the ``energy`` output.

Any additional outputs present within the library are denoted by the
``mts-models::`` prefix. For example, some models can output their last-layer
features, which are named as ``mts-models::aux::last_layer_features``, where
``aux`` denotes an auxiliary output.

Outputs that are specific to a particular model should be named as
``mts-models::<model_name>::<output_name>``.


Arbitrary spherical tensors
---------------------------

In addition to physical, well-known quantities, some models can fit arbitrary
equivariant outputs expressed as spherical tensors. In that case, the ``quantity``
of the output should be set to ``mts-models::spherical_tensor_<lambda>_<sigma>``,
where ``<lambda>`` and ``<sigma>`` are the indices of the irreducible representation
of O(3) corresponding to the spherical tensor.

The metadata for such outputs is as follows:

.. list-table:: Metadata for spherical tensor outputs
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``["o3_lambda", "o3_sigma"]``
    - two dimensions named ``"o3_lambda"`` and ``"o3_sigma"`` with the 
      corresponding O3 equivariant indices.

  * - samples
    - ``["system", "atom"]`` or ``["system"]``
    - if doing ``per_atom`` output, the sample names must be ``["system",
      "atom"]``, otherwise the sample names must be ``["system"]``.

      ``"system"`` must range from 0 to the number of systems given as input to
      the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system.

  * - components
    - ``["o3_mu"]``
    - the O3 equivariant ``mu`` number, ranging from ``-o3_lambda`` to
      ``o3_lambda`` in integer steps.

  * - properties
    - ``"properties"``
    - spherical tensors must have a single property dimension named
      ``"property"``, with entries ranging from 0 to the number of properties.
