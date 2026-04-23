Output naming
=============

The name and format of the outputs in ``metatrain`` are based on those of the
`metatomic <https://docs.metatensor.org/metatomic/latest/outputs/index.html>`_
package. An immediate example is given by the ``energy`` output.

Any additional outputs present within the library are denoted by the
``mtt::`` prefix. For example, some models can output their last-layer
features, which are named as ``mtt::aux::{target}_last_layer_features``,
where ``aux`` denotes an auxiliary output.

The reserved namespace ``mtt::features::{path}`` is used for opt-in diagnostic
feature captures from internal tensors. For example, PET can emit internal
tokens such as ``mtt::features::gnn_layers.0_node`` when requested explicitly
as outputs.

Outputs that are specific to a particular model should be named as
``mtt::<model_name>::<output_name>``.
