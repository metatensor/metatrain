import pytest

from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.reported_metrics import DEFAULT_SUBSETS, metrics_for, parse_metrics


def _targets(*names):
    return {
        name: get_generic_target_info(
            name,
            {
                "quantity": "",
                "unit": "",
                "type": "scalar",
                "num_subtargets": 1,
                "sample_kind": "structure",
            },
        )
        for name in names
    }


def test_absent_block_configures_nothing():
    assert parse_metrics(None, _targets("energy")) == {}
    assert parse_metrics({}, _targets("energy")) == {}


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError, match="not among the targets"):
        parse_metrics({"nope": {"type": "mse"}}, _targets("energy"))


def test_malformed_entries_are_rejected():
    """A clear error here beats a confusing one when the loss is built."""
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_metrics({"energy": ["rmse"]}, _targets("energy"))
    with pytest.raises(ValueError, match="missing its 'type'"):
        parse_metrics({"energy": {"aux_basis": "x"}}, _targets("energy"))


def test_subsets_default_to_every_dataset():
    specs = parse_metrics({"energy": {"type": "mse"}}, _targets("energy"))
    for subset in DEFAULT_SUBSETS:
        assert "energy" in metrics_for(specs, subset)


def test_subsets_are_honoured():
    specs = parse_metrics(
        {"energy": {"type": "mse", "subsets": ["validation"]}}, _targets("energy")
    )
    assert "energy" in metrics_for(specs, "validation")
    assert metrics_for(specs, "training") == {}
    # no subset at all applies everything, which is what a bare `mtt eval` wants
    assert "energy" in metrics_for(specs, None)


def test_log_interval_is_honoured_and_only_during_training():
    specs = parse_metrics(
        {"energy": {"type": "mse", "log_interval": 3}}, _targets("energy")
    )
    assert "energy" in metrics_for(specs, "validation", 0)
    assert metrics_for(specs, "validation", 1) == {}
    assert "energy" in metrics_for(specs, "validation", 3)
    # evaluation happens once, so the interval does not apply there
    assert "energy" in metrics_for(specs, "validation", None)


def test_selection_keys_are_stripped():
    """`subsets` and `log_interval` select the metric; they are not loss parameters."""
    specs = parse_metrics(
        {
            "energy": {
                "type": "mse",
                "subsets": ["validation"],
                "log_interval": 2,
                "delta": 0.5,
            }
        },
        _targets("energy"),
    )
    assert metrics_for(specs, "validation", 0)["energy"] == {
        "type": "mse",
        "delta": 0.5,
    }


@pytest.mark.parametrize("metric_type", ["rmse", "mae", "MAE"])
def test_built_in_metrics_are_rejected(metric_type):
    """
    RMSE and MAE come from the trainers' own accumulators, not from here.

    Allowing them would report a second value under a near-identical name, computed
    differently: the accumulators normalise per atom, a loss built here averages per
    system. The error points at the hyperparameter that does control them.
    """
    with pytest.raises(ValueError, match="cannot be requested in the 'metrics' block"):
        parse_metrics({"energy": {"type": metric_type}}, _targets("energy"))


def test_the_error_names_the_right_hyperparameter():
    with pytest.raises(ValueError, match="log_mae"):
        parse_metrics({"energy": {"type": "mae"}}, _targets("energy"))
    with pytest.raises(ValueError, match="always reported"):
        parse_metrics({"energy": {"type": "rmse"}}, _targets("energy"))
