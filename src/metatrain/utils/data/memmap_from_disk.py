"""Dynamic, generic memory-mapped datasets built from a :class:`DiskDataset`.

Reading samples out of a :class:`DiskDataset` (a zip file) is convenient but slow,
because every access decompresses and deserializes a ``System`` and one or more
``TensorMap`` objects. For repeated access during training this IO cost dominates.

This module builds a *generic* memory-mapped copy of a :class:`DiskDataset` on the
fly, in a temporary directory, and exposes it through
:class:`MemmapFromDiskDataset`, a drop-in replacement that returns exactly the same
samples (same ``System`` and ``TensorMap`` metadata) but reads them from
``numpy.memmap`` arrays instead.

Unlike :class:`metatrain.utils.data.dataset.MemmapDataset` (which targets the
documented MLIP memmap format: scalar/Cartesian energy/forces/stress), this builder
serializes the *actual* ``TensorMap`` structure block by block. It therefore handles:

- arbitrary generic targets (beyond energy/forces/stresses),
- targets with multiple blocks,
- ``sample_kind: system`` and ``sample_kind: atom`` spherical targets,
- atomic-basis spherical targets, whose keys carry an ``atom_type`` dimension and
  whose blocks only index the atoms of that type.

The temporary directory is created in the platform temp location (honouring
``TMPDIR`` etc., so it works on a laptop or an HPC scratch space) and is removed
automatically when the training process exits.
"""

import atexit
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from metatensor.learn.data._namedtuple import namedtuple
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System
from torch.utils.data import Dataset as TorchDataset

from metatrain.utils.data.dataset import MemmapArray
from metatrain.utils.data.readers.metatensor import (
    _check_tensor_map_metadata,
    _empty_tensor_map_like,
)
from metatrain.utils.data.target_info import (
    TargetInfo,
    get_energy_target_info,
    get_generic_target_info,
)


logger = logging.getLogger(__name__)

# Filenames for the (target-independent) per-system arrays.
NUM_ATOMS_FILE = "num_atoms.npy"
TYPES_FILE = "atomic_types.bin"
POSITIONS_FILE = "atomic_positions.bin"
CELLS_FILE = "cells.bin"
MANIFEST_FILE = "manifest.json"

# Auxiliary field carrying the (global) structure index. The DiskDataset injects this
# into every sample as extra data; some architecture transforms (e.g. the atomic-basis
# helper) rely on it, so the memory-mapped dataset must provide it too.
SYSTEM_INDEX_KEY = "mtt::aux::system_index"


def _block_value_file(target_index: int, block_index: int) -> str:
    return f"target_{target_index}_block_{block_index}_values.bin"


def _block_counts_file(target_index: int, block_index: int) -> str:
    return f"target_{target_index}_block_{block_index}_counts.npy"


def _block_atom_file(target_index: int, block_index: int) -> str:
    return f"target_{target_index}_block_{block_index}_atom.bin"


def _labels_to_meta(labels: Labels) -> Dict[str, Any]:
    """
    Serialize a :class:`Labels` object to a JSON-friendly dictionary.

    :param labels: A :class:`Labels` object to serialize.
    :return: A dictionary containing the ``names`` and ``values`` of the
        :class:`Labels` object, with ``values`` converted to a list of int64
    """
    return {
        "names": list(labels.names),
        "values": labels.values.to(torch.int64).cpu().tolist(),
    }


def _meta_to_labels(meta: Dict[str, Any]) -> Labels:
    """
    Rebuild a :class:`Labels` object from its serialized form.

    :param meta: A dictionary containing the serialized ``names`` and ``values`` of a
        :class:`Labels` object.
    :return: A :class:`Labels` object reconstructed from the metadata dictionary.
    """
    return Labels(
        names=meta["names"],
        values=torch.tensor(meta["values"], dtype=torch.int32).reshape(
            len(meta["values"]), len(meta["names"])
        ),
    )


def build_memmap_from_disk_dataset(
    disk_dataset: Any, root: Union[str, Path], target_keys: List[str]
) -> None:
    """Build a generic memory-mapped dataset from a :class:`DiskDataset`.

    The dataset is scanned twice: once to discover the union of all blocks (across
    systems) and their metadata and sizes, and once to write the values to disk.

    :param disk_dataset: The source :class:`DiskDataset` (or any indexable dataset
        whose samples expose ``.system`` and ``sample[target_key]`` TensorMaps).
    :param root: Directory in which to write the memmap files. Created if needed.
    :param target_keys: The target names to serialize.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n_systems = len(disk_dataset)

    # ------------------------------------------------------------------ #
    # Pass 1: gather dimensions, block metadata, and per-block counts.   #
    # ------------------------------------------------------------------ #
    atom_counts = np.zeros(n_systems, dtype=np.int64)

    # For each target we keep an ordered registry of blocks (keyed by the integer
    # tuple of its key). ``blocks`` maps key-tuple -> metadata; ``key_order``
    # preserves first-seen order, which matches the canonical order used by the
    # DiskDataset samples.
    targets_meta: Dict[str, Dict[str, Any]] = {
        tkey: {"keys_names": None, "sample_kind": None, "key_order": [], "blocks": {}}
        for tkey in target_keys
    }
    # (target_key, key_tuple) -> per-system count array
    counts: Dict[Any, np.ndarray] = {}

    for i in range(n_systems):
        sample = disk_dataset[i]
        system = sample.system
        atom_counts[i] = len(system.types)
        for tkey in target_keys:
            tensor_map = sample[tkey]
            tmeta = targets_meta[tkey]
            if tmeta["keys_names"] is None:
                tmeta["keys_names"] = list(tensor_map.keys.names)
            for bi in range(len(tensor_map.keys)):
                key_tuple = tuple(int(x) for x in tensor_map.keys.values[bi].tolist())
                block = tensor_map.block(bi)
                if key_tuple not in tmeta["blocks"]:
                    if block.gradients_list():
                        raise NotImplementedError(
                            "Memory-mapping targets with gradients (e.g. "
                            "energy/forces/stress) from a DiskDataset is not "
                            "supported. Use the dedicated MemmapDataset format for "
                            "MLIP training instead."
                        )
                    sample_kind = "atom" if "atom" in block.samples.names else "system"
                    if tmeta["sample_kind"] is None:
                        tmeta["sample_kind"] = sample_kind
                    tmeta["key_order"].append(key_tuple)
                    tmeta["blocks"][key_tuple] = {
                        "key": list(key_tuple),
                        "components": [_labels_to_meta(c) for c in block.components],
                        "comp_sizes": [len(c) for c in block.components],
                        "properties": _labels_to_meta(block.properties),
                        "n_props": int(block.values.shape[-1]),
                    }
                    counts[(tkey, key_tuple)] = np.zeros(n_systems, dtype=np.int64)
                counts[(tkey, key_tuple)][i] = int(block.values.shape[0])

    # Cumulative atom counts (int64, required for very large datasets).
    num_atoms = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int64)
    total_atoms = int(num_atoms[-1])
    np.save(root / NUM_ATOMS_FILE, num_atoms)

    # Order the blocks of each target the same way a DiskDataset sample does, i.e.
    # sorted by the key columns with the rightmost column taking highest priority
    # (e.g. for ``(o3_lambda, o3_sigma, atom_type)`` keys, blocks are grouped by
    # ``atom_type`` first). This makes the reconstructed samples byte-identical to
    # the originals, regardless of the order in which keys were first encountered.
    for tmeta in targets_meta.values():
        tmeta["key_order"].sort(key=lambda kt: tuple(reversed(kt)))

    # ------------------------------------------------------------------ #
    # Allocate the memmap arrays.                                        #
    # ------------------------------------------------------------------ #
    types_mm = np.memmap(
        root / TYPES_FILE, dtype="int32", mode="w+", shape=(total_atoms,)
    )
    positions_mm = np.memmap(
        root / POSITIONS_FILE, dtype="float32", mode="w+", shape=(total_atoms, 3)
    )
    cells_mm = np.memmap(
        root / CELLS_FILE, dtype="float32", mode="w+", shape=(n_systems, 3, 3)
    )

    # Per-block value/atom memmaps and offsets, keyed by (target_key, key_tuple).
    block_value_mm: Dict[Any, np.memmap] = {}
    block_atom_mm: Dict[Any, np.memmap] = {}
    block_offsets: Dict[Any, np.ndarray] = {}
    manifest_targets = []
    for ti, tkey in enumerate(target_keys):
        tmeta = targets_meta[tkey]
        sample_kind = tmeta["sample_kind"] or "system"
        manifest_blocks = []
        for bi, key_tuple in enumerate(tmeta["key_order"]):
            bmeta = tmeta["blocks"][key_tuple]
            block_counts = counts[(tkey, key_tuple)]
            offsets = np.concatenate([[0], np.cumsum(block_counts)]).astype(np.int64)
            total = int(offsets[-1])
            block_offsets[(tkey, key_tuple)] = offsets
            np.save(root / _block_counts_file(ti, bi), block_counts)

            shape = (total, *bmeta["comp_sizes"], bmeta["n_props"])
            block_value_mm[(tkey, key_tuple)] = np.memmap(
                root / _block_value_file(ti, bi),
                dtype="float32",
                mode="w+",
                shape=shape,
            )
            if sample_kind == "atom":
                block_atom_mm[(tkey, key_tuple)] = np.memmap(
                    root / _block_atom_file(ti, bi),
                    dtype="int32",
                    mode="w+",
                    shape=(total,),
                )
            manifest_blocks.append(bmeta)
        manifest_targets.append(
            {
                "name": tkey,
                "index": ti,
                "keys_names": tmeta["keys_names"],
                "sample_kind": sample_kind,
                "blocks": manifest_blocks,
            }
        )

    # ------------------------------------------------------------------ #
    # Pass 2: write the values.                                          #
    # ------------------------------------------------------------------ #
    write_cursor: Dict[Any, int] = {key: 0 for key in block_value_mm}
    for i in range(n_systems):
        sample = disk_dataset[i]
        system = sample.system
        start, end = int(num_atoms[i]), int(num_atoms[i + 1])
        types_mm[start:end] = system.types.cpu().numpy().astype("int32")
        positions_mm[start:end] = system.positions.cpu().numpy().astype("float32")
        cells_mm[i] = system.cell.cpu().numpy().astype("float32")
        for tkey in target_keys:
            tensor_map = sample[tkey]
            sample_kind = targets_meta[tkey]["sample_kind"] or "system"
            for bi in range(len(tensor_map.keys)):
                key_tuple = tuple(int(x) for x in tensor_map.keys.values[bi].tolist())
                block = tensor_map.block(bi)
                cnt = int(block.values.shape[0])
                cursor = write_cursor[(tkey, key_tuple)]
                block_value_mm[(tkey, key_tuple)][cursor : cursor + cnt] = (
                    block.values.cpu().numpy().astype("float32")
                )
                if sample_kind == "atom":
                    atom_col = block.samples.names.index("atom")
                    atoms = block.samples.values[:, atom_col].cpu().numpy()
                    block_atom_mm[(tkey, key_tuple)][cursor : cursor + cnt] = (
                        atoms.astype("int32")
                    )
                write_cursor[(tkey, key_tuple)] = cursor + cnt

    # Flush everything to disk.
    for mm in (types_mm, positions_mm, cells_mm):
        mm.flush()
    for mm in block_value_mm.values():
        mm.flush()
    for mm in block_atom_mm.values():
        mm.flush()

    with open(root / MANIFEST_FILE, "w") as f:
        json.dump({"num_systems": n_systems, "targets": manifest_targets}, f)


class MemmapFromDiskDataset(TorchDataset):
    """A generic memory-mapped dataset built from a :class:`DiskDataset`.

    This reads the files written by :func:`build_memmap_from_disk_dataset` and
    reconstructs, for each structure, the same ``System`` and ``TensorMap`` objects
    that the original :class:`DiskDataset` would return. It is a drop-in replacement
    usable anywhere a metatrain dataset is expected.

    :param path: Directory containing the memmap files and manifest.
    :param target_options: Target configuration (expanded metatrain yaml format),
        used to build the :class:`TargetInfo` objects.
    """

    def __init__(self, path: Union[str, Path], target_options: Dict[str, Any]) -> None:
        path = Path(path)
        self.path = path
        self.target_config = target_options
        with open(path / MANIFEST_FILE, "r") as f:
            manifest = json.load(f)

        self.ns = int(manifest["num_systems"])
        self.na = np.load(path / NUM_ATOMS_FILE)
        if self.na.dtype != np.int64:
            raise ValueError(f"{NUM_ATOMS_FILE} must use int64 dtype.")
        total_atoms = int(self.na[-1])

        self.types = MemmapArray(path / TYPES_FILE, (total_atoms,), "int32", mode="r")
        self.positions = MemmapArray(
            path / POSITIONS_FILE, (total_atoms, 3), "float32", mode="r"
        )
        self.cells = MemmapArray(
            path / CELLS_FILE, (self.ns, 3, 3), "float32", mode="r"
        )

        # Parse the manifest into a per-target list of blocks, opening one memmap
        # per block lazily (MemmapArray reopens per worker).
        self.targets: List[Dict[str, Any]] = []
        for tmeta in manifest["targets"]:
            ti = tmeta["index"]
            sample_kind = tmeta["sample_kind"]
            blocks = []
            for bi, bmeta in enumerate(tmeta["blocks"]):
                block_counts = np.load(path / _block_counts_file(ti, bi))
                offsets = np.concatenate([[0], np.cumsum(block_counts)]).astype(
                    np.int64
                )
                total = int(offsets[-1])
                shape = (total, *bmeta["comp_sizes"], bmeta["n_props"])
                values = MemmapArray(
                    path / _block_value_file(ti, bi), shape, "float32", mode="r"
                )
                atom = None
                if sample_kind == "atom":
                    atom = MemmapArray(
                        path / _block_atom_file(ti, bi), (total,), "int32", mode="r"
                    )
                blocks.append(
                    {
                        "key": bmeta["key"],
                        "offsets": offsets,
                        "values": values,
                        "atom": atom,
                        "components": [_meta_to_labels(c) for c in bmeta["components"]],
                        "properties": _meta_to_labels(bmeta["properties"]),
                    }
                )
            self.targets.append(
                {
                    "name": tmeta["name"],
                    "keys_names": tmeta["keys_names"],
                    "sample_kind": sample_kind,
                    "blocks": blocks,
                }
            )

        self.sample_class = namedtuple(
            "Sample",
            ["system"] + list(self.target_config.keys()) + [SYSTEM_INDEX_KEY],
        )

    def __len__(self) -> int:
        return self.ns

    def get_num_atoms(self, i: int) -> int:
        """
        Number of atoms in structure ``i`` without loading the full sample.

        :param i: Structure index.
        :return: Number of atoms in structure ``i``.
        """
        return int(self.na[i + 1] - self.na[i])

    def get_all_atom_counts(self) -> np.ndarray:
        """
        Atom counts for all structures as a numpy int64 array.

        :return: A numpy array of shape (n_structures,) containing the number of atoms
            in each structure. The dtype is int64 to support very large datasets.
        """
        return np.diff(self.na).astype(np.int64)

    def __getitem__(self, i: int) -> Any:
        start, end = int(self.na[i]), int(self.na[i + 1])
        types = torch.tensor(self.types[start:end], dtype=torch.int32)
        positions = torch.tensor(self.positions[start:end], dtype=torch.float64)
        cell = torch.tensor(self.cells[i], dtype=torch.float64)
        system = System(
            positions=positions,
            types=types,
            cell=cell,
            pbc=torch.logical_not(torch.all(cell == 0.0, dim=1)),
        )

        joint: Dict[str, Any] = {"system": system}
        for tmeta in self.targets:
            sample_kind = tmeta["sample_kind"]
            present_keys: List[List[int]] = []
            blocks: List[TensorBlock] = []
            for bmeta in tmeta["blocks"]:
                offsets = bmeta["offsets"]
                o0, o1 = int(offsets[i]), int(offsets[i + 1])
                cnt = o1 - o0
                if cnt == 0:
                    # This block is not present for this structure (e.g. an
                    # atom_type that does not occur here). Skip it, exactly as the
                    # DiskDataset would.
                    continue
                values = torch.tensor(
                    np.asarray(bmeta["values"][o0:o1]), dtype=torch.float64
                )
                if sample_kind == "atom":
                    atoms = np.asarray(bmeta["atom"][o0:o1])
                    samples = Labels(
                        names=["system", "atom"],
                        values=torch.tensor(
                            [[i, int(a)] for a in atoms], dtype=torch.int32
                        ),
                    )
                else:
                    samples = Labels(
                        names=["system"],
                        values=torch.tensor([[i]], dtype=torch.int32),
                    )
                blocks.append(
                    TensorBlock(
                        values=values,
                        samples=samples,
                        components=bmeta["components"],
                        properties=bmeta["properties"],
                    )
                )
                present_keys.append(bmeta["key"])
            keys = Labels(
                names=tmeta["keys_names"],
                values=torch.tensor(present_keys, dtype=torch.int32).reshape(
                    len(present_keys), len(tmeta["keys_names"])
                ),
            )
            joint[tmeta["name"]] = TensorMap(keys=keys, blocks=blocks)

        # Auxiliary system index, identical to the one produced by DiskDataset.
        joint[SYSTEM_INDEX_KEY] = TensorMap(
            keys=Labels(["_"], torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[i]]).to(torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(["_"], torch.tensor([[0]])),
                )
            ],
        )

        return self.sample_class._make(
            [joint[name] for name in self.sample_class._fields]
        )

    def get_target_info(self) -> Dict[str, TargetInfo]:
        """Build :class:`TargetInfo` objects for all targets.

        This mirrors :meth:`DiskDataset.get_target_info` so that the resulting metadata
        is identical regardless of whether the data is read from the zip file or from
        the memory-mapped copy.

        :return: A dictionary mapping target names to :class:`TargetInfo` objects.
        """
        target_info_dict = {}
        for target_key, target in self.target_config.items():
            is_energy = (
                (target["quantity"] == "energy")
                and target["sample_kind"] == "system"
                and target["num_subtargets"] == 1
                and target["type"] == "scalar"
            )
            tensor_map = self[0][target_key]
            if is_energy:
                if len(tensor_map) != 1:
                    raise ValueError("Energy TensorMaps should have exactly one block.")
                add_position_gradients = tensor_map.block().has_gradient("positions")
                add_strain_gradients = tensor_map.block().has_gradient("strain")
                target_info = get_energy_target_info(
                    target_key, target, add_position_gradients, add_strain_gradients
                )
                _check_tensor_map_metadata(tensor_map, target_info.layout)
                target_info_dict[target_key] = target_info
            else:
                target_info = get_generic_target_info(target_key, target)
                _check_tensor_map_metadata(tensor_map, target_info.layout)
                if not target_info.is_atomic_basis:
                    target_info.layout = _empty_tensor_map_like(tensor_map)
                target_info_dict[target_key] = target_info
        return target_info_dict

    def get_extra_data_info(self) -> Dict[str, TargetInfo]:
        """Return the extra-data :class:`TargetInfo`, i.e. the system index.

        This mirrors the ``is_extra_data=True`` branch of
        :meth:`DiskDataset.get_target_info`.

        :return: A dictionary mapping the extra-data key to a :class:`TargetInfo`
            object.
        """
        return {
            SYSTEM_INDEX_KEY: get_generic_target_info(
                "system_index",
                {
                    "quantity": "",
                    "unit": "",
                    "type": "scalar",
                    "sample_kind": "system",
                    "num_subtargets": 1,
                },
            )
        }


def create_memmap_dataset_from_zip(
    zip_path: Union[str, Path],
    target_options: Dict[str, Any],
    extra_data_options: Optional[Dict[str, Any]] = None,
) -> MemmapFromDiskDataset:
    """Create a memory-mapped dataset from a ``DiskDataset`` zip file.

    The memmap files are written to a fresh temporary directory (in the platform
    temp location, honouring ``TMPDIR`` etc.), which is removed automatically when
    the (main) process exits.

    :param zip_path: Path to the ``.zip`` DiskDataset.
    :param target_options: Target configuration (expanded metatrain yaml format).
    :param extra_data_options: Optional extra-data configuration. Not supported for
        memory-mapped datasets; a non-empty value raises ``NotImplementedError``.
    :return: A :class:`MemmapFromDiskDataset` reading the temporary memmap files.
    """
    # Imported here to avoid a circular import at module load time.
    from metatrain.utils.data.dataset import DiskDataset

    if extra_data_options:
        raise NotImplementedError(
            "`extra_data` is not supported together with `to_memmap: true`. "
            "Please remove `to_memmap` or the `extra_data` section."
        )

    target_keys = list(target_options.keys())
    disk_dataset = DiskDataset(zip_path, fields=target_keys)

    tmp_root = Path(tempfile.mkdtemp(prefix="metatrain_memmap_"))
    logger.info(
        f"Building memory-mapped dataset from {zip_path} in {tmp_root} "
        f"({len(disk_dataset)} structures)"
    )
    build_memmap_from_disk_dataset(disk_dataset, tmp_root, target_keys)
    _register_cleanup(tmp_root)

    return MemmapFromDiskDataset(tmp_root, target_options)


def _register_cleanup(path: Union[str, Path]) -> None:
    """Register removal of ``path`` when the creating process exits.

    The guard on the creating PID prevents ``DataLoader`` worker processes (which
    inherit the ``atexit`` registry on fork) from deleting the directory while the
    main process is still using it.

    :param path: Directory to remove on exit.
    """
    path = str(path)
    creator_pid = os.getpid()

    def _cleanup() -> None:
        if os.getpid() != creator_pid:
            return
        shutil.rmtree(path, ignore_errors=True)
        logger.info(f"Removed temporary memory-mapped dataset at {path}")

    atexit.register(_cleanup)
