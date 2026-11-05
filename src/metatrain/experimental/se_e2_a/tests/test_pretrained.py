"""Tests for loading a pretrained se_e2_a model via ``se_e2_a_model``."""

import collections
import copy
import json

import pytest
import torch
from metatomic.torch import ModelOutput

from metatrain.experimental.se_e2_a import SeE2A
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.readers import read_systems
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, MODEL_HYPERS
from .test_basic import _minimal_hypers


_BETA_SHIFT = 1.0


def _make_dataset_info():
    targets = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }
    return DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )


def _build_base_model():
    return SeE2A(_minimal_hypers("experimental.se_e2_a"), _make_dataset_info())


def _small_pretrained_checkpoint(path, branches=None):
    """Write a deepmd-kit checkpoint, optionally with several tasks."""
    base = _build_base_model()
    config = json.loads(base.model.get_model_def_script())
    state = collections.OrderedDict()
    for prefix in [f"model.{task}." for task in branches or ["Default"]]:
        for key, value in base.model.state_dict().items():
            if prefix == "model.Beta.":
                state[prefix + key] = value + _BETA_SHIFT
            else:
                state[prefix + key] = value
    state["_extra_state"] = {
        "model_params": (
            {"model_dict": {branch: config for branch in branches}}
            if branches
            else config
        ),
        "train_infos": {"lr": 0, "step": 0},
    }
    torch.save({"model": state}, path)
    return base


def _hypers_for(checkpoint_path, branch=None):
    """Full-size hypers pointing at a small checkpoint.

    Anything read off the loaded module (cutoff, network widths) then differs
    from these hypers, so a bug that ignores the checkpoint is visible.
    """
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["se_e2_a_model"] = str(checkpoint_path)
    hypers["se_e2_a_model_branch"] = branch
    return hypers


@pytest.mark.parametrize("branch", ["Alpha", "Beta"])
def test_multi_task_branch_loading(tmp_path, branch):
    path = tmp_path / "multi_task.pt"
    base = _small_pretrained_checkpoint(path, branches=["Alpha", "Beta"])
    pretrained = SeE2A(_hypers_for(path, branch), _make_dataset_info())
    shift = _BETA_SHIFT if branch == "Beta" else 0.0
    base_state = base.model.state_dict()
    loaded_state = pretrained.model.state_dict()
    assert set(loaded_state) == set(base_state)
    for key, value in base_state.items():
        if key in ("atomic_model.out_bias", "atomic_model.out_std"):
            continue
        torch.testing.assert_close(loaded_state[key], value + shift)


def test_single_task_loading(tmp_path):
    path = tmp_path / "single_task.pt"
    base = _small_pretrained_checkpoint(path)
    pretrained = SeE2A(_hypers_for(path), _make_dataset_info())
    assert pretrained.loaded_se_e2_a is True
    torch.testing.assert_close(
        pretrained.model.state_dict(), base.model.state_dict(), rtol=0, atol=0
    )


def test_multi_task_requires_a_branch(tmp_path):
    path = tmp_path / "multi_task.pt"
    _small_pretrained_checkpoint(path, branches=["Alpha", "Beta"])
    message = (
        "The pretrained se_e2_a model has multiple branches. Please specify the "
        "branch to load via the 'se_e2_a_model_branch' hyperparameter. "
        "Available branches: Alpha, Beta"
    )
    with pytest.raises(ValueError, match=message):
        SeE2A(_hypers_for(path), _make_dataset_info())


def test_multi_task_unknown_branch(tmp_path):
    path = tmp_path / "multi_task.pt"
    _small_pretrained_checkpoint(path, branches=["Alpha", "Beta"])
    message = (
        "The specified branch 'Gamma' is not available in the pretrained se_e2_a "
        "model. Available branches: Alpha, Beta"
    )
    with pytest.raises(ValueError, match=message):
        SeE2A(_hypers_for(path, "Gamma"), _make_dataset_info())


def test_pretrained_module_loading():
    base = _build_base_model()
    hypers = _minimal_hypers("experimental.se_e2_a")
    hypers["se_e2_a_model"] = base.model
    pretrained = SeE2A(hypers, _make_dataset_info())
    assert pretrained.loaded_se_e2_a is True
    assert pretrained.model is base.model


def test_pretrained_extracts_bias_and_std():
    base = _build_base_model()
    atomic_model = base.model.atomic_model
    has_bias = hasattr(atomic_model, "out_bias")
    has_std = hasattr(atomic_model, "out_std")
    if has_bias:
        orig_bias = atomic_model.out_bias.clone()
    if has_std:
        orig_std = atomic_model.out_std.clone()

    hypers = _minimal_hypers("experimental.se_e2_a")
    hypers["se_e2_a_model"] = base.model
    pretrained = SeE2A(hypers, _make_dataset_info())
    if has_bias:
        torch.testing.assert_close(pretrained._loaded_out_bias, orig_bias)
        assert (atomic_model.out_bias == 0).all()
    if has_std:
        torch.testing.assert_close(pretrained._loaded_out_std, orig_std)
        assert (atomic_model.out_std == 1.0).all()
    assert getattr(base.model, "_metatrain_extracted_scaleshift", False) is True


def test_pretrained_fixed_weights():
    base = _build_base_model()
    hypers = _minimal_hypers("experimental.se_e2_a")
    hypers["se_e2_a_model"] = base.model
    pretrained = SeE2A(hypers, _make_dataset_info())
    comp_weights = pretrained.get_fixed_composition_weights()
    scale_weights = pretrained.get_fixed_scaling_weights()
    if hasattr(base.model.atomic_model, "out_bias"):
        assert set(comp_weights["mtt::U0"]) == {1, 6, 7, 8}
    else:
        assert comp_weights == {}
    if hasattr(base.model.atomic_model, "out_std"):
        assert scale_weights["mtt::U0"] == 1.0
    else:
        assert scale_weights == {}


def test_pretrained_no_fixed_weights_without_loading():
    model = _build_base_model()
    assert model.get_fixed_composition_weights() == {}
    assert model.get_fixed_scaling_weights() == {}


def test_pretrained_forward_matches():
    torch.manual_seed(0)
    base = _build_base_model()
    base_state = copy.deepcopy(base.state_dict())
    hypers = _minimal_hypers("experimental.se_e2_a")
    hypers["se_e2_a_model"] = copy.deepcopy(base.model)
    pretrained = SeE2A(hypers, _make_dataset_info())
    pretrained.load_state_dict(base_state)

    systems = [system.to(torch.float64) for system in read_systems(DATASET_PATH)[:2]]
    for system in systems:
        get_system_with_neighbor_lists(system, base.requested_neighbor_lists())
    request = {"mtt::U0": ModelOutput(unit="", sample_kind="system")}
    torch.testing.assert_close(
        base(systems, request)["mtt::U0"].block().values,
        pretrained(systems, request)["mtt::U0"].block().values,
    )


def test_pretrained_checkpoint_roundtrip():
    base = _build_base_model()
    hypers = _minimal_hypers("experimental.se_e2_a")
    hypers["se_e2_a_model"] = base.model
    pretrained = SeE2A(hypers, _make_dataset_info())
    checkpoint = pretrained.get_checkpoint()
    assert isinstance(checkpoint["model_data"]["model_hypers"]["se_e2_a_model"], dict)

    reloaded = SeE2A.load_checkpoint(checkpoint, context="restart")
    assert reloaded.loaded_se_e2_a is True
    assert reloaded._loaded_out_bias is None
    systems = [system.to(torch.float64) for system in read_systems(DATASET_PATH)[:2]]
    for system in systems:
        get_system_with_neighbor_lists(system, pretrained.requested_neighbor_lists())
    request = {"mtt::U0": ModelOutput(unit="", sample_kind="system")}
    torch.testing.assert_close(
        pretrained(systems, request)["mtt::U0"].block().values,
        reloaded(systems, request)["mtt::U0"].block().values,
    )


def test_pretrained_no_double_extraction():
    base = _build_base_model()
    hypers = _minimal_hypers("experimental.se_e2_a")
    hypers["se_e2_a_model"] = base.model
    dataset_info = _make_dataset_info()
    SeE2A(hypers, dataset_info)
    second = SeE2A(hypers, dataset_info)
    assert second._loaded_out_bias is None
    assert second._loaded_out_std is None
