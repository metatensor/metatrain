.. _architecture-sparse-gap:

GAP
===

This is an implementation of the sparse `Gaussian Approximation Potential
<GAP_>`_ (GAP) using `Smooth Overlap of Atomic Positions <SOAP_>`_ (SOAP)
implemented in `rascaline <RASCALINE_>`_.


.. _SOAP: https://doi.org/10.1103/PhysRevB.87.184115
.. _GAP:  https://doi.org/10.1002/qua.24927
.. _RASCALINE: https://github.com/Luthaf/rascaline

The GAP model in metatensor-models can only train on CPU, but evaluation
is also supported on GPU.


Installation
------------

To install the package, you can run the following command in the root directory
of the repository:

.. code-block:: bash

    pip install .[gap]

This will install the package with the GAP dependencies.


Architecture Hyperparameters
----------------------------

:param name: ``experimental.gap``

model
#####
soap
^^^^
:param cutoff: Spherical cutoff (Å) to use for atomic environments. Default 5.0
:param max_radial: Number of radial basis function to use. Default 8
:param max_angular: Number of angular basis function to use also denoted by the  maximum
    degree of spherical harmonics. Default 6
:param atomic_gaussian_width: Width of the atom-centered gaussian creating the atomic
    density. Default 0.3
:param center_atom_weight: Weight of the central atom contribution to the features. If
    1.0 the center atom contribution is weighted the same as any other contribution. If
    0.0 the central atom does not contribute to the features at all. Default 1.0
:param cutoff_function: cutoff function used to smooth the behavior around the cutoff
    radius. The supported cutoff function are

    - ``Step``: Step function, 1 if ``r < cutoff`` and 0 if ``r >= cutoff``. This cutoff
      function takes no additional parameters and can set as in ``.yaml`` file:

      .. code-block:: yaml

        cutoff_function:
          Step:

    - ``ShiftedCosine`` (Default value): Shifted cosine switching function
      ``f(r) = 1/2 * (1 + cos(π (r- cutoff + width) / width ))``.
      This cutoff function takes the ``width``` as
      additional parameter and can set as in ``options.yaml`` file as:

      .. code-block:: yaml

        cutoff_function:
          ShiftedCosine:
            width: 1.0

:param radial_scaling: Radial scaling can be used to reduce the importance of neighbor
    atoms further away from the center, usually improving the performance of the model.
    The supported radial scaling functions are

    - ``None``: No radial scaling.

      .. code-block:: yaml

        radial_scaling:
          None:

    - ``Willatt2018`` (Default value): Use a long-range algebraic decay and
      smooth behavior at :math:`r
      \rightarrow 0`: as introduced by :footcite:t:`willatt_feature_2018` as ``f(r) =
      rate / (rate + (r / scale) ^ exponent)`` This radial scaling function can be set
      in the ``options.yaml`` file as.

      .. code-block:: yaml

        radial_scaling:
          Willatt2018:
            rate: 1.0
            scale: 2.0
            exponent: 7.0

.. note::

  Currently, we only support a Gaussian type orbitals (GTO) as radial basis functions
  and radial integrals.

krr
^^^^
:param degree: degree of the polynomial kernel. Default 2
:param num_sparse_points: number of pseudo points to select
    (by farthest point sampling). Default 500

training:
^^^^^^^^^
:param regularizer: value of the energy regularizer. Default 0.001
:param regularizer_forces: value of the forces regularizer. Default null


Default Hyperparameters
-----------------------
The default hyperparameters for the GAP model are:

.. literalinclude:: ../../../src/metatensor/models/experimental/gap/default-hypers.yaml
   :language: yaml

  
