import os
from typing import Dict, List, Tuple

import ase
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import torch
import xitorch as xt
from metatensor.torch import Labels, TensorBlock, TensorMap
from xitorch.linalg import symeig

from elearn.interface.metatensor.blocks_to_matrix import tensormap_to_dense
from elearn.interface.metatensor.couple import uncouple_tensor_blocks
from elearn.interface.metatensor.symmetrize import unsymmetrize_over_permutations

from .utils import pyscf_orbital_order

os.environ["PYSCFAD_BACKEND"] = "torch"
from pyscfad import gto  # noqa E402
from pyscfad.scf import hf  # noqa E402

__allowed_observables__ = ["eigenvalues", "gap", "dipole", "polarizability"]
__dummy_keys__ = Labels("_", torch.tensor([[0]], dtype=torch.int32))
__dummy_properties__ = Labels("value", torch.tensor([[0]], dtype=torch.int32))


def create_pyscfad_mf(atoms: ase.Atoms, basis: str):

    mol = gto.Mole()
    mol.atom = pyscf_ase.ase_atoms_to_pyscf(atoms)
    mol.basis = basis
    mol.build()

    mf = SCF(mol)

    return mf


class SCF(hf.SCF):

    def __init__(self, mol):
        super().__init__(mol)

        self.mo_coeff = None
        self.mo_energy = None
        self.fock = None

    def _eigh(self, h, s):
        if s is None:
            return symeig(xt.LinearOperator.m(h))
        e, c = symeig(xt.LinearOperator.m(h), M=xt.LinearOperator.m(s))
        return e, c

    def compute_mo_energies(self):
        assert self.fock is not None
        self.mo_energy, self.mo_coeff = self._eigh(self.fock, self.get_ovlp())

    def get_mo_energies(self):
        if self.mo_energy is None:
            self.compute_mo_energies()
        return self.mo_energy

    def get_mo_coeffs(self):
        if self.mo_energy is None:
            self.compute_mo_energies()
        return self.mo_coeff

    def get_occ(self, mo_energy=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        return torch.from_numpy(super().get_occ(mo_energy))

    def get_gap(self):
        if self.mo_energy is None:
            self.compute_mo_energies()
        self.mo_occ = self.get_occ(self.mo_energy)
        homo_energy = self.mo_energy[self.mo_occ > 0][-1]
        lumo_energy = self.mo_energy[~(self.mo_occ > 0)][0]
        self.gap = lumo_energy - homo_energy
        return self.gap

    def get_ovlp(self):
        return torch.from_numpy(super().get_ovlp())

    def set_fock(self, fock):
        self.fock = fock

    def make_rdm1(self, fock=None):
        if fock is None:
            assert self.fock is not None
            fock = self.fock
        if self.mo_energy is None:
            self.compute_mo_energies()
        self.mo_occ = self.get_occ(self.mo_energy)
        dm1 = super().make_rdm1(self.mo_coeff, self.mo_occ)
        return dm1

    def polarizability(self, unit="a.u.", verbose=0):
        assert self.fock is not None

        def _apply_field(field):
            # Apply perturbing electric field to Fock matrix
            perturbed_fock = (
                self.fock + torch.from_numpy(self.mol.intor("int1e_r")).T @ field
            )
            perturbed_dm1 = self.make_rdm1(perturbed_fock)
            perturbed_dipole = self.dip_moment(
                dm=perturbed_dm1, unit=unit, verbose=verbose
            )
            return perturbed_dipole

        # Compute polarizability tensor
        field = torch.zeros(3, dtype=self.fock.dtype, device=self.fock.device)
        polarizability = torch.autograd.functional.jacobian(_apply_field, field)
        return polarizability


def compute_observables(
    predictions: TensorMap,
    indirect_targets: List[str],
    pyscf_mfs: List[SCF],
    systems: List[ase.Atoms],
    system_idx: List[int],  # Check if feasible
    basis_set: Dict[int, List[int]],
) -> Dict[str, torch.Tensor]:

    for indirect_target in indirect_targets:
        assert indirect_target in __allowed_observables__

    assert (
        len(pyscf_mfs) == len(systems) and len(pyscf_mfs) > 0
    ), "Invalid number of systems"

    predicted_observables: Dict[str, torch.Tensor] = {
        target: [] for target in indirect_targets
    }

    # Get matrices
    blocks_to_transform = uncouple_tensor_blocks(
        unsymmetrize_over_permutations(predictions)
    )
    # Possibly pass also the indices
    matrices = tensormap_to_dense(
        blocks_to_transform, systems, basis_set, system_idx=system_idx
    )
    matrices = pyscf_orbital_order(matrices, systems, basis_set, to_spherical=False)

    for target in indirect_targets:
        for mf, H in zip(pyscf_mfs, matrices, strict=True):
            dm = None
            mf.set_fock(H)
            if target == "eigenvalues":
                prediction = mf.get_mo_energies()
            elif target == "gap":
                prediction = mf.get_gap()
            elif target == "dipole":
                if dm is None:
                    dm = mf.make_rdm1()
                prediction = mf.dip_moment(dm=dm, unit="a.u.", verbose=0)
            elif target == "polarizability":
                if dm is None:
                    dm = mf.make_rdm1()
                prediction = mf.polarizability()
            predicted_observables[target].append(prediction)
        predicted_observables[target] = to_tensormap(
            torch.stack(predicted_observables[target]),
            name=target,
            system_idx=system_idx,
            fixed_shape=None,
        )

    return predicted_observables


def pad_and_stack(
    tensors: List[torch.Tensor], target_shape: Tuple[int]
) -> torch.Tensor:
    """
    Zero-pads a list of tensors to match the target shape and stacks them into a single
    tensor. Assumes the target shape is greater than or equal to each tensor's shape in
    every dimension.

    Args:
        tensors (List[torch.Tensor]): List of input tensors with possibly different
        shapes.
        target_shape (Tuple[int, ...]): Desired shape, must be same length as input
        tensor shapes.

    Returns:
        torch.Tensor: Stacked tensor with shape (N, *target_shape), where N is the
        number of input tensors.
    """
    assert all(
        len(t.shape) == len(target_shape) for t in tensors
    ), "All tensors must have the same number of dimensions as target shape."

    padded_tensors = []
    for x in tensors:
        pad: List[int] = [0] * (2 * len(x.shape))  # Initialize fixed-size list
        for i in range(
            len(x.shape) - 1, -1, -1
        ):  # Iterate from last dimension to first
            pad[2 * i + 1] = max(0, target_shape[i] - x.shape[i])  # Right padding only
        padded_tensors.append(torch.nn.functional.pad(x, pad, "constant", 0.0))

    return torch.stack(padded_tensors)


def to_tensormap(
    quantity: torch.Tensor, name: str, system_idx: List[int], fixed_shape=None
) -> TensorMap:

    device = quantity[0].device
    if name == "eigenvalues":

        values = pad_and_stack(quantity, fixed_shape)

        tensor = TensorMap(
            Labels("_", torch.tensor([[0]], dtype=torch.int32, device=device)),
            [
                TensorBlock(
                    samples=Labels(
                        "system",
                        torch.tensor(
                            system_idx, dtype=torch.int32, device=device
                        ).unsqueeze(1),
                    ),
                    components=[],
                    properties=Labels(
                        "_",
                        torch.arange(fixed_shape[-1], device=device).unsqueeze(1),
                    ),
                    values=values,
                )
            ],
        )
    elif name == "gap":
        tensor = TensorMap(
            Labels("_", torch.tensor([[0]], dtype=torch.int32, device=device)),
            [
                TensorBlock(
                    samples=Labels(
                        "system",
                        torch.tensor(
                            system_idx, dtype=torch.int32, device=device
                        ).unsqueeze(1),
                    ),
                    components=[],
                    properties=Labels(
                        "_", torch.tensor([[0]], dtype=torch.int32, device=device)
                    ),
                    values=quantity.unsqueeze(1),
                )
            ],
        )
    elif name == "dipole":
        tensor = TensorMap(
            Labels("_", torch.tensor([[0]], dtype=torch.int32, device=device)),
            [
                TensorBlock(
                    samples=Labels(
                        "system",
                        torch.tensor(
                            system_idx, dtype=torch.int32, device=device
                        ).unsqueeze(1),
                    ),
                    components=[
                        Labels(
                            "xyz",
                            torch.arange(3, dtype=torch.int32, device=device).unsqueeze(
                                1
                            ),
                        )
                    ],
                    properties=Labels(
                        "_", torch.tensor([[0]], dtype=torch.int32, device=device)
                    ),
                    values=quantity.unsqueeze(2),
                )
            ],
        )
    elif name == "polarizability":
        tensor = TensorMap(
            Labels("_", torch.tensor([[0]], dtype=torch.int32, device=device)),
            [
                TensorBlock(
                    samples=Labels(
                        "system",
                        torch.tensor(
                            system_idx, dtype=torch.int32, device=device
                        ).unsqueeze(1),
                    ),
                    components=[
                        Labels(
                            "xyz_1",
                            torch.arange(3, dtype=torch.int32, device=device).unsqueeze(
                                1
                            ),
                        ),
                        Labels(
                            "xyz_2",
                            torch.arange(3, dtype=torch.int32, device=device).unsqueeze(
                                1
                            ),
                        ),
                    ],
                    properties=Labels(
                        "_", torch.tensor([[0]], dtype=torch.int32, device=device)
                    ),
                    values=quantity.unsqueeze(2),
                )
            ],
        )

    return tensor


def combine_matrices(
    node: Dict[int, Dict[Tuple[int], torch.Tensor]],
    edge: Dict[int, Dict[Tuple[int], torch.Tensor]],
    is_mol: bool = True,
) -> List[torch.Tensor]:

    if is_mol:
        out: Dict[int, torch.Tensor] = {}
        for A in edge:
            assert A in node, f"Node {A} not found in node"
            for k in edge[A]:
                assert k == (0, 0, 0)
                if k in node[A]:
                    out[A] = edge[A][k] + node[A][k]
                else:
                    out[A] = edge[A][k]
        return list(out.values())
