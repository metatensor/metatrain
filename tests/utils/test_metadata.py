from metatomic.torch import ModelMetadata

from metatrain.utils.metadata import merge_metadata


def test_append_metadata_new_keys():
    self_meta = ModelMetadata(references={"implementation": ["ref1"]})
    other_meta = ModelMetadata(references={"architecture": ["ref2"]})

    result = merge_metadata(self_meta, other_meta)

    assert result.references["implementation"] == ["ref1"]
    assert result.references["architecture"] == ["ref2"]


def test_append_metadata_existing_keys():
    self_meta = ModelMetadata(references={"implementation": ["ref1"]})
    other_meta = ModelMetadata(references={"implementation": ["ref2"]})

    result = merge_metadata(self_meta, other_meta)

    assert result.references["implementation"] == ["ref1", "ref2"]


def test_append_metadata_mixed_keys():
    self_meta = ModelMetadata(references={"implementation": ["ref1"]})
    other_meta = ModelMetadata(
        references={"implementation": ["ref2"], "architecture": ["ref3"]}
    )

    result = merge_metadata(self_meta, other_meta)

    assert result.references["implementation"] == ["ref1", "ref2"]
    assert result.references["architecture"] == ["ref3"]


def test_merge_metadata():
    self_meta = ModelMetadata(
        name="self_meta",
        description="self_meta",
        authors=[
            "John Doe",
            "Jane Smith",
        ],
        references={
            "architecture": ["ref1"],
            "model": ["ref2"],
        },
    )
    other_meta = ModelMetadata(
        name="other_meta",
        description="other_meta",
        authors=[
            "John Doe",
            "Alice Johnson",
        ],
        references={
            "model": ["ref3"],
            "implementation": ["ref4"],
        },
    )

    result = merge_metadata(self_meta, other_meta)

    assert result.name == "other_meta"
    assert result.description == "other_meta"
    assert result.authors == [
        "John Doe",
        "Jane Smith",
        "Alice Johnson",
    ]
    assert result.references["architecture"] == ["ref1"]
    assert result.references["model"] == ["ref2", "ref3"]
    assert result.references["implementation"] == ["ref4"]
