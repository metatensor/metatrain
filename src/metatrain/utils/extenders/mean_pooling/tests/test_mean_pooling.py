import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_features_info
from metatrain.utils.extenders import get_extender_hypers_sanitizer
from metatrain.utils.extenders.helpers import UnavailableOutputError, load_extender


TARGET = "mtt::refractive_index"
INPUT = "mtt::extender_aux::mean_pooling::refractive_index"


def test_mean_by_system(tmp_path):
    info = get_features_info(1, sample_kind="system")
    cls = load_extender("mean_pooling")
    hypers = get_extender_hypers_sanitizer("mean_pooling")(TARGET)
    pool = cls(hypers, DatasetInfo("angstrom", [1], {TARGET: info}))
    assert pool.supported_outputs() == {TARGET: info}
    assert pool.requested_extender_inputs({}) == {}
    atomic = pool.requested_extender_inputs()[INPUT]
    assert atomic.sample_kind == "atom"
    assert atomic.layout.block().properties == info.layout.block().properties
    path = tmp_path / "extender.pt"
    torch.jit.script(cls.load_checkpoint(pool.get_checkpoint(), "export")).save(
        str(path)
    )
    module = torch.jit.load(str(path))
    assert module([], {}, {}) == {}

    # Only selected centers are supplied, interleaved across unequal-size groups.
    samples = Labels(["system", "atom"], torch.tensor([[1, 4], [0, 2], [1, 0], [1, 2]]))
    values = torch.tensor(
        [[1.0], [8.0], [2.0], [9.0]], dtype=torch.float64, requires_grad=True
    )
    block = TensorBlock(values, samples, [], atomic.layout.block().properties)
    inputs = {INPUT: TensorMap(atomic.layout.keys, [block])}
    result = module([], {TARGET: ModelOutput(sample_kind="system")}, inputs, samples)[
        TARGET
    ]
    pooled = result.block()
    assert pooled.samples == Labels.range("system", 2)
    assert pooled.properties == info.layout.block().properties
    assert result.keys == info.layout.keys
    torch.testing.assert_close(pooled.values, values.new_tensor([[8.0], [4.0]]))
    pooled.values.sum().backward()
    torch.testing.assert_close(
        values.grad, values.new_tensor([[1 / 3], [1.0], [1 / 3], [1 / 3]])
    )
    with pytest.raises(ValueError, match="system-level"):
        pool([], {TARGET: ModelOutput(sample_kind="atom")}, inputs)


def test_target_layout():
    cls = load_extender("mean_pooling")
    with pytest.raises(UnavailableOutputError, match="Missing"):
        cls({"outputs": TARGET}, DatasetInfo("angstrom", [1], {}))
    for info in (get_features_info(1), get_features_info(2, sample_kind="system")):
        with pytest.raises(ValueError, match="single scalar"):
            cls({"outputs": TARGET}, DatasetInfo("angstrom", [1], {TARGET: info}))
