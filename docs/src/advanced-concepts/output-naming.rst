Output naming
=============

The name and format of the outputs in ``metatensor-models`` are based on
those of the ``metatensor.torch.atomistic``
<https://lab-cosmo.github.io/metatensor/latest/atomistic/outputs.html>_
package. An immediate example is given by the ``energy`` output.

Any additional outputs present within the library are denoted by the
``mtm::`` prefix. For example, some models can output their last-layer
features, which are named as ``mtm::aux::last_layer_features``, where
``aux`` denotes an auxiliary output.

Outputs that are specific to a particular model should be named as
``mtm::<model_name>::<output_name>``.
