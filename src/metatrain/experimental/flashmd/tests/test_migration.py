import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.experimental.flashmd.checkpoints import model_update_v4_v5
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import TargetInfo


def _cartesian_rank1_per_atom_target_info(name: str, unit: str) -> TargetInfo:
    block = TensorBlock(
        values=torch.empty((0, 3, 1), dtype=torch.float64),
        samples=Labels(["system", "atom"], torch.empty((0, 2), dtype=torch.int32)),
        components=[Labels(["xyz"], torch.arange(3, dtype=torch.int32).reshape(-1, 1))],
        properties=Labels([name], torch.zeros((1, 1), dtype=torch.int32)),
    )
    return TargetInfo(
        layout=TensorMap(keys=Labels.single(), blocks=[block]),
        quantity="momentum" if "moment" in name else "length",
        unit=unit,
    )


def _scaler_buffer(name: str) -> torch.Tensor:
    tensor_map = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.ones((1, 1), dtype=torch.float64),
                samples=Labels(["_"], torch.zeros((1, 1), dtype=torch.int32)),
                components=[],
                properties=Labels([name], torch.zeros((1, 1), dtype=torch.int32)),
            )
        ],
    )
    return mts.save_buffer(mts.make_contiguous(tensor_map))


def _make_checkpoint() -> dict:
    weight = torch.zeros(1)
    state_dict = {
        # heads
        "node_heads.positions.0.0.weight": weight,
        "edge_heads.momenta.0.0.weight": weight,
        # last-layers
        "node_last_layers.positions.0.positions___0.weight": weight,
        "edge_last_layers.momenta.0.momenta___0.bias": weight,
        # scaler
        "scaler.positions_scaler_buffer": _scaler_buffer("positions"),
        "scaler.momenta_per_target_scaler_buffer": _scaler_buffer("momenta"),
        "scaler.momenta_per_property_scaler_buffer": _scaler_buffer("momenta"),
        # unrelated python keys
        "node_embedders.0.momenta_encoder.weight": weight,
        "masses": weight,
    }
    return {
        "model_data": {
            "dataset_info": DatasetInfo(
                length_unit="A",
                atomic_types=[1, 6],
                targets={
                    "positions": _cartesian_rank1_per_atom_target_info(
                        "positions", "A"
                    ),
                    "momenta": _cartesian_rank1_per_atom_target_info(
                        "momenta", "(eV*u)^(1/2)"
                    ),
                },
            ),
        },
        "train_hypers": {
            "loss": {"positions": {"weight": 1.0}, "momenta": {"weight": 1.0}},
        },
        "model_state_dict": dict(state_dict),
        "best_model_state_dict": dict(state_dict),
    }


def test_model_update_v4_v5_renames_targets_loss_and_layout():
    checkpoint = _make_checkpoint()
    model_update_v4_v5(checkpoint)

    targets = checkpoint["model_data"]["dataset_info"].targets
    assert "positions" not in targets and "position" in targets
    assert "momenta" not in targets and "momentum" in targets

    assert targets["position"].layout.block(0).properties.names == ["position"]
    assert targets["momentum"].layout.block(0).properties.names == ["momentum"]

    loss = checkpoint["train_hypers"]["loss"]
    assert "positions" not in loss and "position" in loss
    assert "momenta" not in loss and "momentum" in loss


def test_model_update_v4_v5_renames_state_dict_keys_and_scaler_buffers():
    checkpoint = _make_checkpoint()
    model_update_v4_v5(checkpoint)

    for sd_key in ("model_state_dict", "best_model_state_dict"):
        state_dict = checkpoint[sd_key]
        # renames
        assert "node_heads.position.0.0.weight" in state_dict
        assert "edge_heads.momentum.0.0.weight" in state_dict
        assert "node_last_layers.position.0.position___0.weight" in state_dict
        assert "edge_last_layers.momentum.0.momentum___0.bias" in state_dict
        assert "scaler.position_scaler_buffer" in state_dict
        assert "scaler.momentum_per_target_scaler_buffer" in state_dict
        assert "scaler.momentum_per_property_scaler_buffer" in state_dict
        # old paths gone
        assert "node_heads.positions.0.0.weight" not in state_dict
        assert "edge_heads.momenta.0.0.weight" not in state_dict
        # unrelated python are attributes untouched !
        assert "node_embedders.0.momenta_encoder.weight" in state_dict
        assert "masses" in state_dict

        tm = mts.load_buffer(state_dict["scaler.position_scaler_buffer"])
        assert tm.block(0).properties.names == ["position"]
        tm = mts.load_buffer(state_dict["scaler.momentum_per_target_scaler_buffer"])
        assert tm.block(0).properties.names == ["momentum"]
