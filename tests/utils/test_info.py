from metatensor.models.utils.info import (
    finalize_aggregated_info,
    update_aggregated_info,
)


def test_update_aggregated_info():
    """Tests the `update_aggregated_info` function."""

    aggregated_info = {"a": (1.0, 1), "b": (2.0, 2)}
    new_info = {"a": (1.0, 1), "b": (2.0, 2), "c": (3.0, 3)}
    expected = {"a": (2.0, 2), "b": (4.0, 4), "c": (3.0, 3)}
    actual = update_aggregated_info(aggregated_info, new_info)
    assert actual == expected


def test_finalize_aggregated_info():
    """Tests the `finalize_aggregated_info` function."""

    aggregated_info = {"a": (1.0, 1), "b": (2.0, 2), "c": (3.0, 4)}
    expected = {"a": 1.0, "b": 1.0, "c": (3.0 / 4.0) ** 0.5}
    actual = finalize_aggregated_info(aggregated_info)
    assert actual == expected
