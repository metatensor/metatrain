Fitting spherical targets on an atomic basis
============================================

.. note:: This section is a work in progress. It will be updated in the future to
   include more details and examples.


Targeting electronic structure quantities such as the electron density, Hamiltonian, and
density matrix is also possible with NanoPET in metatrain. These targets are ususally
expressed on an atom-centered quantum chemical basis set, and differ from normal
spherical targets in that they are per-atom (node) or per-atom pair (edge) quantities,
and crucially with a size that is dependent on the atomic type(s) of the atom center(s).



Preparing generic targets for reading by metatrain
--------------------------------------------------

Spherical targets on an atomic basis must be read stored in TensorMap format and written
to a DiskDataset


Input file
##########


.. code-block:: yaml

    targets:
      mtt::electron_density_coeffs:
        read_from: electron_density_coeffs.mts
        quantity: "node"
        unit: ""
        per_atom: True
        type:
         atomic_basis_spherical:
            n_center: 1
            basis:
            - key:
               o3_lambda: 0
               o3_sigma: 1
               center_type: 1
               num_subtargets: 2
            - key:
               o3_lambda: 0
               o3_sigma: 1
               center_type: 6
               num_subtargets: 10
            - key:
               o3_lambda: 0
               o3_sigma: 1
               center_type: 7
               num_subtargets: 10



Preparing your targets -- metatensor
####################################

TODO!
