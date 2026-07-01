import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.torch import TensorMap
from metatomic.torch import ModelCapabilities, System

from metatrain.utils.external_naming import to_external_name

from .metatensor import _concatenate_tensormaps_flat
from .writers import Writer


class MemMapWriter(Writer):
    """
    Write predictions to memory-mapped numpy files (``.npy``), one per target
    and per gradient.

    Mirrors :class:`MetatensorWriter`: ``write()`` stashes each batch in
    temporary ``.mts`` files to release GPU/CPU tensors immediately, avoiding
    unbounded memory growth over a long evaluation loop.  ``finish()`` then
    reuses :func:`_concatenate_tensormaps_flat` (same helper used by
    :class:`MetatensorWriter`) to join the per-batch TensorMaps with correct
    "system" label offsets, extracts the raw value arrays, and writes them as
    memory-mapped ``.npy`` files via :func:`numpy.lib.format.open_memmap` so
    they can be lazily reloaded with ``numpy.load(path, mmap_mode="r")``.

    Values are cast to ``float32`` before writing. This does not follow the
    on-disk layout of :class:`metatrain.utils.data.dataset.MemmapDataset`
    (which stores raw ``.bin`` shards alongside external shape metadata);
    the two only share the "float32, memory-mappable" convention.

    Gradient arrays are named and converted using the same physical
    convention as :class:`ASEWriter`: position gradients of an energy become
    negated ``forces``, and strain gradients become both a negated
    ``virial`` and a volume-normalized ``stress`` (NaN for systems without a
    valid cell).

    Only single-block :class:`metatensor.torch.TensorMap` outputs are
    supported; use the ``.mts`` format (:class:`MetatensorWriter`) for
    multi-block targets such as spherical tensors.

    :param filename: Base path for the output files, e.g.
        ``train_predictions.npy``.  Each target/gradient is saved next to it
        as ``{stem}_{name}.npy``, where the parent directory is fully
        preserved.
    :param capabilities: Model capabilities, used to give gradient arrays
        user-facing names (e.g. ``forces`` instead of
        ``energy_positions_gradients``).
    :param append: Unused; kept to match the base-class signature.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        capabilities: Optional[ModelCapabilities] = None,
        append: Optional[bool] = False,  # unused, but matches base signature
    ) -> None:
        super().__init__(filename, capabilities, append)
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._batch_idx = 0
        self._target_names: List[str] = []

    def write(self, systems: List[System], predictions: Dict[str, TensorMap]) -> None:
        """
        Save batch predictions to temporary files, releasing GPU/CPU memory
        immediately.

        :param systems: Systems for the current batch. Only their cells are
            used, to normalize strain-gradient outputs (e.g. stress) by the
            cell volume.
        :param predictions: Dictionary of TensorMaps with predictions for the
            systems.
        """
        if self._batch_idx == 0:
            self._target_names = list(predictions.keys())

        tmp_path = Path(self._tmp_dir.name)
        for target_name, tmap in predictions.items():
            if len(tmap.keys) != 1:
                raise ValueError(
                    "Only single-block `TensorMap`s can be written to memmap "
                    f"files, but target {target_name!r} has {len(tmap.keys)} "
                    "blocks. Use the '.mts' format for multi-block targets."
                )
            fname = tmp_path / f"{self._batch_idx}_{target_name}.mts"
            # Detach from GPU and convert to float64 before saving, mirroring
            # MetatensorWriter.write() so intermediate temp files are consistent.
            mts.save(str(fname), tmap.to("cpu").to(torch.float64))

        volumes = np.array(
            [
                torch.det(system.cell).item() if torch.any(system.cell != 0) else 0.0
                for system in systems
            ],
            dtype=np.float64,
        )
        volumes[volumes == 0.0] = np.nan
        np.save(tmp_path / f"{self._batch_idx}_volumes.npy", volumes)

        self._batch_idx += 1

    def __del__(self) -> None:
        # Guard against ResourceWarning when finish() is never called (e.g.
        # write() raised before the evaluation loop completed).
        try:
            self._tmp_dir.cleanup()
        except Exception:
            pass

    def finish(self) -> None:
        """
        Join the buffered batches and write one memory-mapped ``.npy`` file
        per target (and per gradient).
        """
        if self._batch_idx == 0:
            return

        tmp_path = Path(self._tmp_dir.name)
        out_filename = Path(self.filename)
        stem = out_filename.stem
        out_dir = out_filename.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # Cell volumes, in the same per-system order as the merged blocks below
        # (batches are concatenated in the order `write()` was called, and
        # `_concatenate_tensormaps_flat` preserves that row order).
        volumes = np.concatenate(
            [np.load(tmp_path / f"{i}_volumes.npy") for i in range(self._batch_idx)]
        )

        for target_name in self._target_names:
            batch_tmaps = [
                mts.load(str(tmp_path / f"{i}_{target_name}.mts"))
                for i in range(self._batch_idx)
            ]
            # Reuse the same helper as MetatensorWriter to merge batches with
            # correctly shifted "system" sample labels.
            merged = _concatenate_tensormaps_flat(batch_tmaps)

            if len(merged.keys) != 1:
                raise ValueError(
                    "Only single-block `TensorMap`s can be written to memmap "
                    f"files, but target {target_name!r} has {len(merged.keys)} "
                    "blocks. Use the '.mts' format for multi-block targets."
                )
            block = merged.block()

            self._write_array(out_dir / f"{stem}_{target_name}.npy", block.values)

            for gradient_name, gradient_block in block.gradients():
                internal_name = f"{target_name}_{gradient_name}_gradients"
                assert self.capabilities is not None
                external_name = to_external_name(
                    internal_name, self.capabilities.outputs
                )

                if "forces" in external_name:
                    self._write_array(
                        out_dir / f"{stem}_{external_name}.npy",
                        -gradient_block.values,
                    )
                elif "virial" in external_name:
                    strain_derivatives = (
                        gradient_block.values.detach().cpu().numpy().astype(np.float32)
                    )
                    volumes_f32 = volumes.astype(np.float32)
                    invalid = ~np.isfinite(volumes_f32)
                    strain_derivatives[invalid] = np.nan

                    self._write_array_np(
                        out_dir / f"{stem}_{external_name}.npy",
                        -strain_derivatives,
                    )
                    stress_name = external_name.replace("virial", "stress")
                    self._write_array_np(
                        out_dir / f"{stem}_{stress_name}.npy",
                        strain_derivatives / volumes_f32[:, None, None, None],
                    )
                else:
                    self._write_array(
                        out_dir / f"{stem}_{external_name}.npy",
                        gradient_block.values,
                    )

        self._tmp_dir.cleanup()

    @staticmethod
    def _write_array(path: Path, values: torch.Tensor) -> None:
        # Cast to float32 to match the MemmapDataset / data-preparation
        # convention (see examples/0-beginner/01-data_preparation.py).
        array = values.detach().cpu().numpy().astype(np.float32)
        MemMapWriter._write_array_np(path, array)

    @staticmethod
    def _write_array_np(path: Path, array: np.ndarray) -> None:
        array = array.astype(np.float32)
        fp = np.lib.format.open_memmap(
            path, mode="w+", dtype=array.dtype, shape=array.shape
        )
        fp[:] = array
        fp.flush()
