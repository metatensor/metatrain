import os
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import metatensor.torch as mts
import metatomic.torch as mta
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelCapabilities, System, load_system

from ..dataset import ATOM_COUNTS_MEMBER, _parse_disk_dataset_member_name
from .writers import Writer, _split_tensormaps


def _stamp_system_label(tensor_map: TensorMap, system_index: int) -> TensorMap:
    """Set the ``"system"`` sample column of every block to ``system_index``.

    Values, components, properties and gradients are preserved.

    :param tensor_map: A single-system TensorMap.
    :param system_index: The value to write into the ``"system"`` column.
    :return: The relabeled TensorMap.
    """
    blocks = []
    for block in tensor_map.blocks():
        samples = block.samples
        if "system" not in samples.names:
            raise ValueError(
                "DiskDatasetWriter targets must have a 'system' sample "
                f"dimension, got {samples.names}"
            )
        column = samples.names.index("system")
        values = samples.values.clone()
        if len(values) > 0 and len(torch.unique(values[:, column])) > 1:
            raise ValueError(
                "write() was called with a single system but a TensorMap "
                "containing multiple system ids "
                f"({torch.unique(values[:, column]).tolist()})"
            )
        values[:, column] = system_index
        new_block = TensorBlock(
            values=block.values,
            samples=Labels(samples.names, values),
            components=block.components,
            properties=block.properties,
        )
        for parameter in block.gradients_list():
            new_block.add_gradient(parameter, block.gradient(parameter).copy())
        blocks.append(new_block)
    return TensorMap(tensor_map.keys, blocks)


class DiskDatasetWriter(Writer):
    """
    Write systems and predictions to a zip file, each system in a separate folder inside
    the zip.

    Every finished zip carries a complete ``metadata/atom_counts.npy`` member
    (the atom count of each entry, in entry order), which is what makes the
    dataset usable with atom-count-based sampling. The ``metadata/`` folder is
    reserved for such dataset-level informative files.

    :param path: Path to the output zip file.
    :param capabilities: Model capabilities.
    :param append: If True, open the zip file in append mode.
    """

    def __init__(
        self,
        path: Union[str, Path],
        capabilities: Optional[
            ModelCapabilities
        ] = None,  # unused, but matches base signature
        append: Optional[bool] = False,  # if True, open zip in append mode
    ):
        super().__init__(filename=path, capabilities=capabilities, append=append)
        self._finished = False
        self._atom_counts_prefix = np.empty(0, dtype=np.int64)
        self.index = 0

        if append and os.path.exists(path) and os.path.getsize(path) > 0:
            # Continue indexing after the existing entries, instead of
            # overwriting them at "0/", "1/", ..., and reconstruct the
            # atom-count prefix so finish() always leaves a complete
            # `metadata/atom_counts.npy`. A corrupt existing zip raises
            # BadZipFile here: appending to it would silently orphan every
            # existing entry, because mode "a" appends after the corrupt
            # bytes and writes a fresh central directory that lists only the
            # new members.
            with zipfile.ZipFile(path, "r") as existing:
                self.index, self._atom_counts_prefix = self._scan_existing(
                    path, existing
                )

        mode: Literal["w", "a"] = "a" if append else "w"
        self.zip_file = zipfile.ZipFile(path, mode)
        self._atom_counts: List[int] = []

    @staticmethod
    def _scan_existing(
        path: Union[str, Path], existing: zipfile.ZipFile
    ) -> tuple[int, np.ndarray]:
        """Validate an existing dataset zip and gather its append state.

        :param path: The zip file path (for error messages).
        :param existing: The zip, opened for reading.
        :return: The next free entry index and the existing entries' atom
            counts.
        """
        entry_numbers = []
        counts: Optional[np.ndarray] = None
        for info in existing.infolist():
            name = info.filename
            if name == ATOM_COUNTS_MEMBER:
                with existing.open(name, "r") as f:
                    counts = np.load(f)
                continue
            # Members that are not part of the format are left alone here;
            # DiskDataset warns about them when the dataset is read.
            parsed = _parse_disk_dataset_member_name(name)
            if parsed is not None and parsed[1] == "system":
                entry_numbers.append(parsed[0])
        if not entry_numbers and counts is None:
            # A non-empty file with no recognizable dataset content: likely a
            # corrupt zip, or not a DiskDataset at all. Truncated zips can
            # still parse, exposing an embedded archive's members (system.mta
            # files are zips themselves). Appending would orphan the content.
            raise ValueError(
                f"Cannot append to '{path}': the existing file contains no "
                "DiskDataset entries; it is corrupted or not a DiskDataset."
            )

        num_existing = 0
        if entry_numbers:
            # Appended entries continue the dense range 0..N-1 that DiskDataset
            # requires; fail now, while the existing file is still intact,
            # rather than producing a zip the reader is guaranteed to reject.
            if not np.array_equal(
                np.sort(entry_numbers), np.arange(len(entry_numbers))
            ):
                raise ValueError(
                    f"Cannot append to '{path}': its entries are not the dense "
                    "range `0/`, `1/`, ... (gaps or duplicated entry numbers); "
                    "re-write the dataset."
                )
            num_existing = len(entry_numbers)

        if counts is not None and len(counts) >= num_existing:
            prefix = counts[:num_existing].astype(np.int64, copy=False)
        elif num_existing > 0:
            # The dataset has no usable atom-count file: compute the missing
            # prefix now (one read per existing system) so the finished zip
            # is complete.
            prefix = np.empty(num_existing, dtype=np.int64)
            for entry in range(num_existing):
                with existing.open(f"{entry}/system.mta", "r") as f:
                    prefix[entry] = len(load_system(f))
        else:
            prefix = np.empty(0, dtype=np.int64)
        return num_existing, prefix

    def write(self, systems: List[System], predictions: Dict[str, TensorMap]) -> None:
        """
        Write a single (system, predictions) into the zip under
        a new folder "<index>/".

        The stored TensorMaps' ``"system"`` sample labels always hold the
        entry number, whether the systems arrive batched or one at a time.
        Distinct labels are required to read the dataset back for training
        (per-sample maps are joined at collation).

        :param systems: List of systems to write.
        :param predictions: Dictionary of TensorMaps with predictions for the systems.
        """

        if len(systems) == 1:
            # A batch of one carries a single system id already; stamping the
            # entry number over it (instead of splitting) also supports maps
            # whose id is not 0, e.g. carrying the sample id of a source
            # dataset.
            split_predictions = [
                {
                    name: _stamp_system_label(tensor_map, self.index)
                    for name, tensor_map in predictions.items()
                }
            ]
        else:
            # _split_tensormaps selects samples by batch-local system ids;
            # any other labeling would silently produce empty members.
            for name, tensor_map in predictions.items():
                ids = torch.unique(
                    torch.cat(
                        [
                            block.samples.column("system")
                            for block in tensor_map.blocks()
                        ]
                    )
                )
                if ids.tolist() != list(range(len(systems))):
                    raise ValueError(
                        f"batched write() requires the '{name}' TensorMap to "
                        f"label its {len(systems)} systems 0..{len(systems) - 1}, "
                        f"got system ids {ids.tolist()}"
                    )
            split_predictions = _split_tensormaps(
                systems, predictions, istart_system=self.index
            )

        for system, preds in zip(systems, split_predictions, strict=True):
            # system
            with self.zip_file.open(f"{self.index}/system.mta", "w") as f:
                mta.save(f, system.to("cpu").to(torch.float64))
            self._atom_counts.append(len(system))

            # each target
            for target_name, tensor_map in preds.items():
                with self.zip_file.open(f"{self.index}/{target_name}.mts", "w") as f:
                    tensor_map = mts.make_contiguous(tensor_map)
                    buf = tensor_map.to("cpu").to(torch.float64)
                    # metatensor.torch.save_buffer returns a torch.Tensor buffer
                    buffer = buf.save_buffer()
                    np.save(f, buffer.numpy())

            self.index += 1

    def finish(self) -> None:
        """
        Write the complete per-structure atom-count file
        (``metadata/atom_counts.npy``) and close the zip file. Calling it
        again is a no-op.
        """
        if self._finished:
            return
        full_counts = np.concatenate(
            [self._atom_counts_prefix, np.asarray(self._atom_counts, dtype=np.int64)]
        )
        # Zip files don't support in-place replacement of an entry: on
        # append, this adds a second "metadata/atom_counts.npy" rather than
        # overwriting the first. That is harmless, since reading a duplicated
        # name returns the last copy written.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Duplicate name")
            with self.zip_file.open(ATOM_COUNTS_MEMBER, "w") as f:
                np.save(f, full_counts)
        self.zip_file.close()
        self._finished = True
