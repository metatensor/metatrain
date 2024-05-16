from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.utils.merge_capabilities import merge_capabilities


def test_merge_capabilities():
    old_capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 6],
        outputs={
            "energy": ModelOutput(quantity="energy", unit="eV"),
            "mtm::forces": ModelOutput(quantity="mtm::forces", unit="eV/Angstrom"),
        },
        interaction_range=1.0,
        dtype="float32",
    )

    new_capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1],
        outputs={
            "energy": ModelOutput(quantity="energy", unit="eV"),
            "mtm::forces": ModelOutput(quantity="mtm::forces", unit="eV/Angstrom"),
            "mtm::stress": ModelOutput(quantity="mtm::stress", unit="GPa"),
        },
        interaction_range=1.0,
        dtype="float32",
    )

    merged, novel = merge_capabilities(old_capabilities, new_capabilities)

    assert merged.length_unit == "angstrom"
    assert merged.atomic_types == [1, 6]
    assert merged.outputs["energy"].quantity == "energy"
    assert merged.outputs["energy"].unit == "eV"
    assert merged.outputs["mtm::forces"].quantity == "mtm::forces"
    assert merged.outputs["mtm::forces"].unit == "eV/Angstrom"
    assert merged.outputs["mtm::stress"].quantity == "mtm::stress"
    assert merged.outputs["mtm::stress"].unit == "GPa"
    assert merged.interaction_range == 1.0
    assert merged.dtype == "float32"

    assert novel.length_unit == "angstrom"
    assert novel.atomic_types == [1, 6]
    assert novel.outputs["mtm::stress"].quantity == "mtm::stress"
    assert novel.outputs["mtm::stress"].unit == "GPa"
    assert novel.interaction_range == 1.0
    assert novel.dtype == "float32"
