import pickle
import struct
import zipfile

import pytest

from metatrain.utils.data.smart_zip import SmartZip


PAYLOADS = {
    "0/system.mta": b"alpha" * 100,
    "1/system.mta": b"bravo" * 200,
    "readme.txt": b"hello",
}


def _make_zip(path, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression) as zf:
        for name, data in PAYLOADS.items():
            zf.writestr(name, data)
        zf.mkdir("empty_dir")
    return path


def test_open_reads_members_back(tmp_path):
    """SmartZip must return exactly the bytes that were written, for any
    member, in any access order — it replaces ZipFile as the read path of
    DiskDataset."""
    smart_zip = SmartZip(_make_zip(tmp_path / "data.zip"))
    for name in reversed(list(PAYLOADS)):
        with smart_zip.open(name) as f:
            assert f.read() == PAYLOADS[name]


def test_supports_compressed_members(tmp_path):
    """Decompression is delegated to ZipExtFile, so compressed zips must read
    back identically: this is what lets DiskDataset (or future users) support
    compressed datasets without a different code path."""
    smart_zip = SmartZip(_make_zip(tmp_path / "data.zip", zipfile.ZIP_DEFLATED))
    for name in PAYLOADS:
        with smart_zip.open(name) as f:
            assert f.read() == PAYLOADS[name]


def test_namelist_and_len_skip_directories(tmp_path):
    """Directory entries are not readable members; listing them would make
    every consumer re-filter (and DiskDataset would warn about them as
    unknown members)."""
    smart_zip = SmartZip(_make_zip(tmp_path / "data.zip"))
    assert sorted(PAYLOADS) == list(smart_zip.namelist())
    assert len(smart_zip) == len(PAYLOADS)
    assert [i.filename for i in smart_zip.infoiter()] == sorted(PAYLOADS)


def test_missing_member_is_keyerror(tmp_path):
    """Same exception type and message as ZipFile.open, so SmartZip can be
    swapped for ZipFile without changing callers' error handling."""
    smart_zip = SmartZip(_make_zip(tmp_path / "data.zip"))
    with pytest.raises(KeyError, match="There is no item named '2/system.mta'"):
        smart_zip.open("2/system.mta")


def test_write_mode_is_rejected(tmp_path):
    """SmartZip is read-only: a caller expecting ZipFile's mode='w' must fail
    loudly instead of silently corrupting the archive."""
    smart_zip = SmartZip(_make_zip(tmp_path / "data.zip"))
    with pytest.raises(ValueError, match="mode='r'"):
        smart_zip.open("0/system.mta", mode="w")


def test_corrupted_member_is_detected(tmp_path):
    """Reads verify the member's CRC-32 from the central directory: bit rot
    must raise instead of returning corrupted bytes."""
    path = _make_zip(tmp_path / "data.zip")
    smart_zip = SmartZip(path)
    with zipfile.ZipFile(path) as zf:
        offset = zf.getinfo("1/system.mta").header_offset
    with open(path, "r+b") as f:
        f.seek(offset + 60)  # inside the member's data
        byte = f.read(1)
        f.seek(offset + 60)
        f.write(bytes([byte[0] ^ 0xFF]))

    with pytest.raises(zipfile.BadZipFile, match="Bad CRC-32"):
        with smart_zip.open("1/system.mta") as f:
            f.read()


def test_encrypted_members_are_rejected(tmp_path):
    """SmartZip cannot decrypt members (it rebuilds ZipInfo objects with no
    flag bits), so encrypted archives must fail at construction, not with a
    CRC error at some later read."""
    path = _make_zip(tmp_path / "data.zip")
    # the stdlib cannot write encrypted zips: flip the encryption flag bit of
    # the first central-directory entry (2 bytes at offset 8 from the
    # signature) by hand
    data = bytearray(path.read_bytes())
    entry = data.index(b"PK\x01\x02")
    flags = struct.unpack_from("<H", data, entry + 8)[0]
    struct.pack_into("<H", data, entry + 8, flags | 0x1)
    path.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="is encrypted"):
        SmartZip(path)


def test_reads_do_not_construct_zipfile(tmp_path, monkeypatch):
    """After construction, reads must never re-open zipfile.ZipFile: parsing
    the central directory materializes one ZipInfo per member, which is what
    made dataloader workers OOM on large datasets (~10 GB RSS per open for a
    13M-member zip)."""
    smart_zip = SmartZip(_make_zip(tmp_path / "data.zip"))

    def _no_zipfile(*args, **kwargs):
        raise AssertionError("reads must not construct zipfile.ZipFile")

    monkeypatch.setattr(zipfile, "ZipFile", _no_zipfile)
    with smart_zip.open("0/system.mta") as f:
        assert f.read() == PAYLOADS["0/system.mta"]


def test_duplicated_names_read_last_copy(tmp_path):
    """ZipFile reads the last copy of a duplicated member name; SmartZip must
    do the same — DiskDatasetWriter's append mode relies on it, since it
    appends a fresh `metadata/atom_counts.npy` that shadows the stale one."""
    path = tmp_path / "data.zip"
    with warnings_as_duplicate_is_expected():
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("metadata/atom_counts.npy", b"stale")
            zf.writestr("metadata/atom_counts.npy", b"fresh")

    smart_zip = SmartZip(path)
    with smart_zip.open("metadata/atom_counts.npy") as f:
        assert f.read() == b"fresh"


def warnings_as_duplicate_is_expected():
    return pytest.warns(UserWarning, match="Duplicate name")


def test_picklable_and_readable_after_unpickling(tmp_path):
    """Datasets are pickled to dataloader workers under the spawn/forkserver
    start methods; the process-local file handle must be dropped and lazily
    reopened, not break pickling (as an open ZipFile would)."""
    smart_zip = SmartZip(_make_zip(tmp_path / "data.zip"))
    with smart_zip.open("0/system.mta") as f:  # open a handle to be dropped
        f.read()

    clone = pickle.loads(pickle.dumps(smart_zip))
    with clone.open("1/system.mta") as f:
        assert f.read() == PAYLOADS["1/system.mta"]
