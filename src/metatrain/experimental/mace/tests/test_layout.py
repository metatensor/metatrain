from e3nn import o3

from metatrain.experimental.mace.utils.mts import get_e3nn_mts_layout


def test_layout_property_name_drops_variant_and_prefix():
    """The default property label follows ``property_label_name``."""
    layout = get_e3nn_mts_layout(
        o3.Irreps("1x0e"),
        "mtt::dipole/dft",
        {"sample_kind": "system", "type": "spherical"},
    )
    assert layout.block(0).properties.names == ["dipole"]


def test_layout_explicit_properties_name_is_kept():
    layout = get_e3nn_mts_layout(
        o3.Irreps("1x0e"),
        "mtt::dipole/dft",
        {
            "sample_kind": "system",
            "type": "spherical",
            "properties_name": "feature",
        },
    )
    assert layout.block(0).properties.names == ["feature"]
