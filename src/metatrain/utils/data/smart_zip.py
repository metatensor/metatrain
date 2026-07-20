"""A read-only zip reader that scales to zips with millions of members."""

import os
import zipfile
from pathlib import Path
from typing import IO, Any, Dict, Iterator, Optional, Union

import numpy as np


class _LazyZipFile(zipfile.ZipFile):
    """Read-mode ``ZipFile`` that skips central-directory parsing.

    Only used internally by :py:class:`SmartZip`, which indexes the member
    locations itself and opens members by passing reconstructed ``ZipInfo``
    objects to ``open()`` (public API). The inherited ``open()`` still
    validates the local header signature, cross-checks the member name, and
    verifies CRC-32 while reading.
    """

    def _RealGetContents(self) -> None:
        self.start_dir = 0


class SmartZip:
    """Read-only zip reader that scales to zips with millions of members.

    Implements the subset of the ``zipfile.ZipFile`` reading API that
    metatrain uses (``open``, ``namelist``, ``infoiter``, ``len``, context
    manager), with one structural difference: the central directory is
    parsed exactly once, at construction, into flat numpy arrays instead of
    one ``ZipInfo`` Python object per member. Compared to keeping a parsed
    ``ZipFile`` around, the index is

    - small: tens of bytes per member instead of ~650 (a 13M-member
      ``ZipFile`` costs ~10 GB of RAM and ~80 s to construct);
    - picklable: instances can be sent to ``spawn``/``forkserver``
      dataloader workers (the per-process file handle is dropped on pickling
      and reopened lazily);
    - fork-friendly: numpy data pages contain no Python object headers, so
      reads never write to them and they stay shared under copy-on-write,
      while ``ZipInfo`` objects are unshared page by page by refcount and
      garbage-collector writes.

    Reads go through ``ZipFile.open()`` with a ``ZipInfo`` rebuilt from the
    index, so local-header validation, member-name cross-checking, CRC-32
    verification and decompression are all standard-library behavior; in
    particular, compressed members are supported. ``open()`` transparently
    opens one file handle per process, so a single instance can be shared
    with dataloader workers.

    Known API divergences from ``ZipFile``: ``namelist()``/``infoiter()``
    are generators (a list of millions of names would defeat the purpose)
    and yield in sorted-name order rather than archive order; directory
    entries are skipped; encrypted members are rejected at construction.
    Like ``ZipFile``, ``open()`` on a duplicated member name reads the last
    copy in archive order.

    :param path: Path to the zip file.
    """

    # index columns, per member
    _OFFSET, _SIZE, _CRC, _COMPRESS_TYPE, _COMPRESS_SIZE = range(5)

    def __init__(self, path: Union[str, Path]):
        self.path = path

        names = []
        locs = []
        with zipfile.ZipFile(path, "r") as zip_file:
            for info in zip_file.infolist():
                if info.is_dir():
                    continue
                if info.flag_bits & 0x1:
                    raise ValueError(
                        f"cannot read '{path}': member '{info.filename}' is encrypted"
                    )
                names.append(info.filename.encode("utf-8"))
                locs.append(
                    (
                        info.header_offset,
                        info.file_size,
                        info.CRC,
                        info.compress_type,
                        info.compress_size,
                    )
                )

        self._names = np.array(names) if names else np.empty(0, dtype="S1")
        self._locs = (
            np.array(locs, dtype=np.int64) if locs else np.empty((0, 5), dtype=np.int64)
        )
        # sort by name for O(log n) lookups; stable, so duplicated names
        # keep their archive order
        order = np.argsort(self._names, kind="stable")
        self._names = self._names[order]
        self._locs = self._locs[order]

        self._fp: Optional[_LazyZipFile] = None
        self._fp_pid: Optional[int] = None

    def __len__(self) -> int:
        return len(self._names)

    def namelist(self) -> Iterator[str]:
        """Yield the name of every member (directory entries excluded).

        :yield: Member names, in sorted order.
        :ytype: str
        """
        for name in self._names:
            yield name.decode("utf-8")

    def infoiter(self) -> Iterator[zipfile.ZipInfo]:
        """Yield a reconstructed ``ZipInfo`` for every member.

        The objects are rebuilt from the index one at a time (holding a list
        of all of them is what this class exists to avoid) and carry the
        fields the index stores: ``filename``, ``header_offset``,
        ``file_size``, ``CRC``, ``compress_type`` and ``compress_size``.

        :yield: Member infos, in sorted-name order.
        :ytype: zipfile.ZipInfo
        """
        for row in range(len(self._names)):
            yield self._make_zipinfo(row)

    def _make_zipinfo(self, row: int) -> zipfile.ZipInfo:
        zinfo = zipfile.ZipInfo(self._names[row].decode("utf-8"))
        (
            zinfo.header_offset,
            zinfo.file_size,
            zinfo.CRC,
            zinfo.compress_type,
            zinfo.compress_size,
        ) = (int(value) for value in self._locs[row])
        return zinfo

    def _ensure_fp(self) -> _LazyZipFile:
        """The zip handle for the current process, opened on first use.

        A handle cannot be shared across processes (the underlying file
        offset would be shared too, and concurrent seeks would interleave),
        so each process — e.g. each dataloader worker — lazily opens its
        own. ``_LazyZipFile`` skips the central-directory parse, so this is
        O(1) in time and memory.

        :return: The per-process zip handle.
        """
        pid = os.getpid()
        fp = self._fp
        if fp is None or self._fp_pid != pid:
            if fp is not None:
                fp.close()
            fp = _LazyZipFile(self.path, "r")
            self._fp = fp
            self._fp_pid = pid
        return fp

    def open(self, name: str, mode: str = "r") -> IO[bytes]:
        """Open the member ``name`` for reading, like ``ZipFile.open``.

        :param name: The member name.
        :param mode: Only ``"r"`` is supported (``ZipFile`` also writes).
        :return: A file-like object with the member's (decompressed)
            contents.
        """
        if mode != "r":
            raise ValueError("SmartZip only supports opening members with mode='r'")
        key = name.encode("utf-8")
        # rightmost match: with the stable name sort, duplicated names keep
        # their archive order, so this reads the last-written copy exactly
        # like ZipFile does (e.g. a re-written `metadata/atom_counts.npy`
        # appended by DiskDatasetWriter shadows the stale one)
        row = int(np.searchsorted(self._names, key, side="right")) - 1
        if row < 0 or self._names[row] != key:
            raise KeyError(f"There is no item named {name!r} in the archive")
        return self._ensure_fp().open(self._make_zipinfo(row))

    def close(self) -> None:
        """Close the per-process file handle, if this process opened one."""
        if self._fp is not None:
            self._fp.close()
            self._fp = None
            self._fp_pid = None

    def __enter__(self) -> "SmartZip":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:
        fp = getattr(self, "_fp", None)
        if fp is not None:
            fp.close()

    def __getstate__(self) -> Dict[str, Any]:
        # the open handle is process-local and unpicklable; drop it and let
        # the unpickling process (e.g. a "spawn" dataloader worker) reopen
        # lazily
        state = self.__dict__.copy()
        state["_fp"] = None
        state["_fp_pid"] = None
        return state
