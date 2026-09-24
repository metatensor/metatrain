"""Torch-side analogue of jax-pme mixed / tiled batch bookkeeping.

jax-pme ``batched_tiled`` quantizes PBC atom counts to a tile size (``BM``)
and the k-grid to another (``BK``). torch-pme evaluates one ``System`` at a
time (a Python list batch). These helpers expose the same padding
arithmetic and PBC split so a later packed kernel could reuse this
bookkeeping without changing today's per-system potentials.
"""

from typing import List, Sequence, Tuple

from metatomic.torch import System


def pad_to_tile(count: int, tile: int) -> int:
    """Round ``count`` up to a multiple of ``tile`` (``BM`` / ``BK`` tiling)."""
    if tile <= 0:
        raise ValueError(f"tile must be > 0, got {tile}")
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    remainder = count % tile
    if remainder == 0:
        return count
    return count + (tile - remainder)


def split_pbc_indices(systems: Sequence[System]) -> Tuple[List[int], List[int]]:
    """Split batch indices into periodic vs non-periodic (jax-pme mixed)."""
    periodic: List[int] = []
    nonperiodic: List[int] = []
    for index, system in enumerate(systems):
        if bool(system.pbc.any()):
            periodic.append(index)
        else:
            nonperiodic.append(index)
    return periodic, nonperiodic
