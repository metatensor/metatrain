Fitting spherical targets on an atomic basis
============================================

.. note:: This tutorial is currently only applicable for model training with NanoPET.

.. note:: This section is a work in progress. It will be updated in the future to
   include more details and examples.

Targeting electronic structure quantities such as the electron density, Hamiltonian
matrix, and density matrix is also possible with NanoPET in metatrain. These targets are
ususally expressed on an atom-centered quantum chemical basis set, and differ from
normal spherical targets in that they are per-atom or per-pair quantities, and crucially
with a size that is dependent on the atomic type(s) of the atom center(s).


Defining atomic basis spherical targets
---------------------------------------

We will briefly outline a couple of examples of spherical targets on an atomic basis.
This section intends to describe the metadata structure of such targets, but the details
on how to generate these is outside the scope of this tutorial.

Electron density on a basis
###########################

First let's consider the electron density (though applicable also to any scalar field),
decomposed onto an atom-centered basis set. In this case, the model targets a set of
equivariant expansion coefficients.

These are stored in block-sparse TensorMap format according to the basis set definition.

For a methane (CH4) molecule, for instance, the keys may look like:

.. code-block::

   TensorMap with 8 blocks
   keys: o3_lambda  o3_sigma  center_type
            0         1           6
            1         1           6
            2         1           6
            3         1           6
            4         1           6
            0         1           1
            1         1           1
            2         1           1

where "o3_lambda" and "o3_sigma" are the angular order and inversion symmetry of the
irreducible components of the density decomposition (which is positive for a scalar
field, a real tensor), and "center_type" tracks the type of atom on which the
corresponding basis functions are centered.

For example for the block indexed by key:

.. code-block::

   LabelsEntry(o3_lambda=2, o3_sigma=-1, center_type=6)


.. code-block::

   TensorBlock
      samples (1): ['system', 'atom']
      components (5): ['o3_mu']
      properties (5): ['n']
      gradients: None

As this is a per-atom (single-center) quantity, the samples contain the "system" and
"atom" dimensions. As this is a Carbon block (type 6), there is one sample.

A single "o3_mu" dimension in the components tracks the components of the spherical
tensor of order "o3_lambda" (as indexed in the keys), and the radial basis functions are
enumerated in the properties with dimension "n".

Hamiltonian (or density matrix) on a coupled basis
##################################################

The elements of the Hamiltonian matrix (or analogously the density matrix) on the atomic
orbital basis can also be targeted. For now, this must be expressed in the coupled
angular momenta basis, and symmetrized with respect to permutations. Details on how to
represent Hamiltonian matrices in such a way are outside the scope of this tutorial but
will be covered elsewhere.

For methane, the keys of the Hamiltonian (coupled and symmetrized) are as follows:

.. code-block::

   TensorMap with 56 blocks
   keys: o3_lambda  o3_sigma  s2_pi  first_atom_type  second_atom_type
            0         1        0           6                6
            0         1        1           6                6
                                    ...
            2         1        1           1                1
            2         1       -1           1                1

where "o3_lambda" and "o3_sigma" are as before, and "s2_pi" tracks the permutational
symmetry of each block. A value of zero means unsymetrized - in the case of on-site
terms or off-site terms for atoms of different types. Values of +/- 1 refer to plus- or
minus-permutationally symmetrized blocks, and only exist for off-site atom pairs of the
same atomic type. "first_atom_type" and "second_atom_type" refer to the types of the
pair of atoms the matrix element corresponds to.

For example for the block indexed by key:

.. code-block::

   LabelsEntry(o3_lambda=1, o3_sigma=-1, s2_pi=0, first_atom_type=1, second_atom_type=6)


the block metadata is as follows:

.. code-block::

   TensorBlock
      samples (0): ['system', 'first_atom', 'second_atom', 'cell_shift_a', 'cell_shift_b', 'cell_shift_c']
      components (1): ['o3_mu']
      properties (39): ['l_1', 'l_2', 'n_1', 'n_2']
      gradients: None

As this is a per-pair (two-center) quantity, the samples dimensions are the standard
dimensions found in a neighbor list, tracking the atomic indices of the atoms in the
pair and the cell translation vector that separates them (which can be non-zero in
periodic systems).

The components axis is the same as for the density coefficients above, as the
Hamiltonian is expressed on the coupled angular momenta basis.

Finally the properties tracks the indices of the original orbitals in the uncoupled
basis.

Preparing atomic basis spherical targets for metatrain
------------------------------------------------------

Atomic basis spherical targets must be stored in TensorMap format and written to a
DiskDataset prior to calling metatrain. With targets stored with the metadata structure
as outlined above, one can create a DiskDataset by following the example in
"examples/programmatic/disk_dataset".

Then, the ``systems`` and ``targets`` section of the input file should be written as
follows:

Input file
##########


.. code-block:: yaml

   systems: disk_dataset.zip

   targets:

      mtt::electron_density_basis:
         read_from: disk_dataset.zip
         type: atomic_basis_spherical

      mtt::hamiltonian:
         read_from: disk_dataset.zip
         type: atomic_basis_spherical
         unit: Ha

Unlike normal spherical targets, the ``irreps`` do not need to be specified in the input
file and are instead inferred by reading the targets in the dataset. Whether the targets
are per-atom or per-pair is also inferred from the samples metadata of the targets, so
only the name (i.e. ``mtt::electron_density_basis``) and ``unit`` of the qunaitity needs to
be specified.
