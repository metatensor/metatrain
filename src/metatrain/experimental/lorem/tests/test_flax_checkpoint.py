"""LOREM loaded from lorem-jax parameters reproduces lorem-jax's predictions.

The references in ``assets/lorem_jax`` come from ``assets/lorem_jax/generate.py``
(run once with lorem-jax installed), so these tests do not need JAX.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from metatomic.torch import ModelOutput, System

from metatrain.experimental.lorem import LOREM
from metatrain.experimental.lorem.modules.flax_checkpoint import (
    load_checkpoint,
    model_hypers_from_yaml,
    read_export,
)
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from .test_bec import _bec_target_info


pytest.importorskip("torchpme")

ASSETS = Path(__file__).parent / "assets" / "lorem_jax"
CASES = ["lorem", "lorem_scalar_messages", "lorem_bec", "lorem_l4"]
BASELINE = {1: -0.5, 6: -3.0, 8: -4.5}


def _export(case: str, folder: Path) -> Path:
    """Write a reference as a lorem-jax export folder (lab-cosmo/lorem-jax#42)."""
    data = np.load(ASSETS / f"{case}.npz")
    params = {
        k[len("params/") :]: data[k] for k in data.files if k.startswith("params/")
    }
    np.savez(folder / "params.npz", **params)
    (folder / "model.yaml").write_text(str(data["model_yaml"]))
    (folder / "baseline.yaml").write_text(yaml.safe_dump({"elemental": BASELINE}))
    (folder / "export.yaml").write_text("format: 1\n")
    return folder


def _model(name: str, hypers: Any) -> LOREM:
    targets = {
        "energy": get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"})
    }
    if name == "lorem.models.bec.LoremBEC":
        targets["mtt::bec"] = _bec_target_info()
    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6, 8], targets=targets)
    return LOREM(hypers, info).double()


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("label", ["molecule", "crystal"])
def test_matches_lorem_jax(case, label, tmp_path):
    name, hypers, params, baseline = read_export(_export(case, tmp_path))
    assert baseline == BASELINE
    model = _model(name, hypers)
    load_checkpoint(model, params)

    reference = np.load(ASSETS / f"{case}.npz")
    positions = torch.tensor(reference[f"{label}/positions"], requires_grad=True)
    system = System(
        types=torch.tensor(reference[f"{label}/numbers"], dtype=torch.int32),
        positions=positions,
        cell=torch.tensor(reference[f"{label}/cell"]),
        pbc=torch.tensor(reference[f"{label}/pbc"]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {name: ModelOutput(sample_kind="atom") for name in model.outputs}
    # training mode: the raw prediction, without the scaler and composition model
    predictions = model.train()([system], outputs)

    energy = predictions["energy"].block().values.sum()
    (forces,) = torch.autograd.grad(-energy, positions)
    expected = torch.tensor(float(reference[f"{label}/energy"]), dtype=torch.float64)
    # both sides run in float64 and agree to <= 1e-6 in energy; swapping the
    # message-pass tensor-product factors, or torch's default LayerNorm
    # epsilon, moves the energy by >= 2e-4 on these fixtures
    torch.testing.assert_close(energy, expected, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(
        forces,
        torch.tensor(reference[f"{label}/forces"], dtype=torch.float64),
        rtol=0.0,
        atol=1e-4,
    )
    if "mtt::bec" in predictions:
        torch.testing.assert_close(
            predictions["mtt::bec"].block().values[..., 0],
            torch.tensor(reference[f"{label}/apt"], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-7,
        )


def test_params_prefix_is_optional(tmp_path):
    """A plain ``flatten_dict(params["params"])`` dump loads like an export."""
    name, hypers, params, _ = read_export(_export("lorem", tmp_path))
    plain, prefixed = _model(name, hypers), _model(name, hypers)
    load_checkpoint(plain, params)
    load_checkpoint(prefixed, {f"params/{k}": v for k, v in params.items()})
    for module in ["sr", "lr"]:
        expected = getattr(plain, module).state_dict()
        for key, value in getattr(prefixed, module).state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("case", CASES)
def test_loaded_model_is_scriptable(case, tmp_path):
    """Covers scalar-only message passing and ``initialize_node_features=False``."""
    name, hypers, params, _ = read_export(_export(case, tmp_path))
    model = _model(name, hypers)
    load_checkpoint(model, params)
    model.eval()
    reference = np.load(ASSETS / f"{case}.npz")
    system = System(
        types=torch.tensor(reference["crystal/numbers"], dtype=torch.int32),
        positions=torch.tensor(reference["crystal/positions"]),
        cell=torch.tensor(reference["crystal/cell"]),
        pbc=torch.tensor(reference["crystal/pbc"]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {name: ModelOutput(sample_kind="atom") for name in model.outputs}
    eager = model([system], outputs)
    scripted = torch.jit.script(model)([system], outputs)
    for name in outputs:
        torch.testing.assert_close(
            scripted[name].block().values, eager[name].block().values
        )


def test_mismatched_models_are_rejected(tmp_path):
    name, hypers, params, _ = read_export(_export("lorem_bec", tmp_path))
    # no Born-charge target: the PerParticleTensorPredictor leaves are left over
    with pytest.raises(ValueError, match="not used"):
        load_checkpoint(_model("lorem.models.mlip.Lorem", hypers), params)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_checkpoint(_model(name, {**hypers, "num_features": 8}), params)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_checkpoint(_model(name, {**hypers, "num_message_passing": 0}), params)


def _model_yaml(case: str) -> dict:
    return yaml.safe_load(str(np.load(ASSETS / f"{case}.npz")["model_yaml"]))


def test_model_yaml_is_the_checkpoint_spec():
    name, hypers = model_hypers_from_yaml(_model_yaml("lorem_bec"))
    assert name == "lorem.models.bec.LoremBEC"
    assert not hypers["initialize_node_features"]
    assert not hypers["equivariant_message_passing"]
    # lorem-jax checkpoints after lab-cosmo/lorem-jax#37 also write the inputs
    ((name, fields),) = _model_yaml("lorem").items()
    _, hypers = model_hypers_from_yaml({name: {**fields, "inputs": []}})
    assert hypers["max_degree"] == 2


@pytest.mark.parametrize(
    "fields",
    [
        {"radial_basis": "gaussian"},
        {"lr": False},
        {"inputs": ["charge"]},
        {"not_a_field": 1},
        {"num_species": None},
    ],
)
def test_unsupported_model_yaml(fields):
    ((name, spec),) = _model_yaml("lorem").items()
    spec = {k: v for k, v in {**spec, **fields}.items() if v is not None}
    with pytest.raises(ValueError):
        model_hypers_from_yaml({name: spec})
    with pytest.raises(ValueError, match="unsupported"):
        model_hypers_from_yaml({"lorem.Lorem": spec})


def test_export_format_is_checked(tmp_path):
    (_export("lorem", tmp_path) / "export.yaml").write_text("format: 2\n")
    with pytest.raises(ValueError, match="export format 2"):
        read_export(tmp_path)
