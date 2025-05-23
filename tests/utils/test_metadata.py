import json

from metatomic.torch import ModelMetadata

from metatrain.utils.metadata import append_metadata_references


def test_append_metadata_new_keys():
    self_meta = ModelMetadata(references={"implementation": ["ref1"]})
    other_meta = ModelMetadata(references={"architecture": ["ref2"]})

    append_metadata_references(self_meta, other_meta)

    result = json.loads(self_meta._get_method("__getstate__")())
    assert result["references"]["implementation"] == ["ref1"]
    assert result["references"]["architecture"] == ["ref2"]


def test_append_metadata_existing_keys():
    self_meta = ModelMetadata(references={"implementation": ["ref1"]})
    other_meta = ModelMetadata(references={"implementation": ["ref2"]})

    append_metadata_references(self_meta, other_meta)

    result = json.loads(self_meta._get_method("__getstate__")())
    assert result["references"]["implementation"] == ["ref1", "ref2"]


def test_append_metadata_mixed_keys():
    self_meta = ModelMetadata(references={"implementation": ["ref1"]})
    other_meta = ModelMetadata(
        references={"implementation": ["ref2"], "architecture": ["ref3"]}
    )

    append_metadata_references(self_meta, other_meta)

    result = json.loads(self_meta._get_method("__getstate__")())
    assert result["references"]["implementation"] == ["ref1", "ref2"]
    assert result["references"]["architecture"] == ["ref3"]
