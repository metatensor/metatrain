from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.utils.merge_capabilities import merge_capabilities


def test_merge_capabilities():
    old_capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 6],
        outputs={
            "energy": ModelOutput(quantity="energy", unit="eV"),
            "forces": ModelOutput(quantity="forces", unit="eV/Angstrom"),
        },
        interaction_range=1.0,
    )

    new_capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1],
        outputs={
            "energy": ModelOutput(quantity="energy", unit="eV"),
            "forces": ModelOutput(quantity="forces", unit="eV/Angstrom"),
            "stress": ModelOutput(quantity="stress", unit="GPa"),
        },
        interaction_range=1.0,
    )

    merged, novel = merge_capabilities(old_capabilities, new_capabilities)

    assert merged.length_unit == "angstrom"
    assert merged.atomic_types == [1, 6]
    assert merged.outputs["energy"].quantity == "energy"
    assert merged.outputs["energy"].unit == "eV"
    assert merged.outputs["forces"].quantity == "forces"
    assert merged.outputs["forces"].unit == "eV/Angstrom"
    assert merged.outputs["stress"].quantity == "stress"
    assert merged.outputs["stress"].unit == "GPa"

    assert novel.length_unit == "angstrom"
    assert novel.atomic_types == [1, 6]
    assert novel.outputs["stress"].quantity == "stress"
    assert novel.outputs["stress"].unit == "GPa"
