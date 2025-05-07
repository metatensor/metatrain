Example of doing multiproperty predictions with "SOAP-BPNN"
=========================================================
This is an example how to use the SOAP-BPNN model to predict chemical shieldings using
SOAP-BPNN and its individual components that contribute to the final shielding values.

The target DFT GIPAW shieldings of a dataset of mixed organic crystals 
containing at most the elements H and C, are stored in the files 'train.xyz', 'val.xyz' and 'test.xyz'.

The final expression for the shielding is given by:
.. math::
    \sigma_{\text{iso}} = \sigma_{\text{para}} + \sigma_{\text{dia}} + \sigma_{\text{bare}} + \sigma_{\text{shape}} + \sigma_{\text{para,oo}} + \sigma_{\text{para,lq}}

where \sigma_{\text{iso}} is the total isotropic shielding, 
\sigma_{\text{para}} is the paramagnetic contribution, 
\sigma_{\text{dia}} is the diamagnetic contribution, 
\sigma_{\text{bare}} is the bare shielding contribution,
 \sigma_{\text{shape}} is the shape contribution, 
 \sigma_{\text{para,oo}} is the para-oo contribution (correction term) 
 and \sigma_{\text{para,lq}} is the para-lq contribution (correction term).

In this example we are going to learn each contribution individually as well as having one read out layer for the total isotropic shielding.
This example highlights that different types of contributions can be learned simultaneously:
All contibutions in the sum are atom wise properties, except for the shape contribution which is a per-structure property.
