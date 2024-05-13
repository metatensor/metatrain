Auxiliary outputs
=================

These outputs, which are idenfified by the ``mts-models::aux::`` prefix,
represent additional information that the model may provide. They are not
conventional trainable outputs, and they often correspond to internal
information that the model is capable of providing, such as its internal
representation.

The following auxiliary outputs that are currently supported
by one or more architectures in the library:

- ``mts-models::aux::last_layer_features``: The internal representation
   of the model at the last layer, before the final linear transformation.

The following table shows the architectures that support each of the
auxiliary outputs:

+-----------------------------------------+-----------+------------------+-----+
| Auxiliary output                        | SOAP-BPNN | Alchemical model | PET |
+=========================================+===========+==================+=====+
| mts-models::aux::last_layer_features    | Yes       |       No         | No  |
+-----------------------------------------+-----------+------------------+-----+

