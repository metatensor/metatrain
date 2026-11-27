import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_features_info
from metatrain.utils.extenders import get_extender_hypers_sanitizer
from metatrain.utils.extenders.helpers import UnavailableOutputError, setup_extenders
from metatrain.utils.extenders.testing.extender import ExtenderTests
from metatrain.utils.extenders.testing.inputs import InputTests
from metatrain.utils.extenders.testing.output import OutputTests
from metatrain.utils.extenders.testing.torchscript import TorchscriptTests


TARGET = "mtt::energy"
INPUT = "mtt::extender_aux::mean_pooling::energy"


class MeanPoolingTests(ExtenderTests):
    extender = "mean_pooling"

    @pytest.fixture
    def hypers(self):
        return get_extender_hypers_sanitizer(self.extender)(TARGET)

    def build_hypers(self, input, output):
        # The single configurable name is the system output.
        assert input is None
        name = output["energy"]
        return {"outputs": name}, [INPUT], [name]


class TestTorchscript(TorchscriptTests, MeanPoolingTests):
    def test_checkpoint_round_trip(self, hypers, dataset_info, tmp_path):
        pool = self.extender_cls(hypers, dataset_info)
        restored = self.extender_cls.load_checkpoint(pool.get_checkpoint(), "export")
        path = tmp_path / "extender.pt"
        torch.jit.script(restored).save(str(path))
        module = torch.jit.load(str(path))
        atomic = pool.requested_extender_inputs()[INPUT]
        block = TensorBlock(
            torch.tensor([[2.0], [4.0]]),
            Labels(["system", "atom"], torch.tensor([[0, 0], [0, 1]])),
            [],
            atomic.layout.block().properties,
        )
        result = module(
            [],
            {TARGET: ModelOutput(sample_kind="system")},
            {INPUT: TensorMap(atomic.layout.keys, [block])},
        )
        torch.testing.assert_close(result[TARGET].block().values, torch.tensor([[3.0]]))


class TestInputs(InputTests, MeanPoolingTests):
    def test_requested_atomic_scalar(self, hypers, dataset_info):
        pool = self.extender_cls(hypers, dataset_info)
        assert pool.requested_extender_inputs({}) == {}
        atomic = pool.requested_extender_inputs({TARGET: ModelOutput()})[INPUT]
        target = dataset_info.targets[TARGET]
        assert atomic.sample_kind == "atom"
        assert atomic.quantity == target.quantity
        assert atomic.unit == target.unit
        assert atomic.layout.block().properties == target.layout.block().properties

    def test_output_and_specific_input(self, hypers):
        atomic = get_features_info(1, description="Learned atomic contribution")
        targets = {TARGET: get_features_info(1, sample_kind="system"), INPUT: atomic}
        pool = self.extender_cls(hypers, DatasetInfo("angstrom", [1], targets))
        assert pool.requested_extender_inputs()[INPUT] is atomic

        targets[INPUT] = get_features_info(2)
        with pytest.raises(ValueError, match="Incompatible layout"):
            self.extender_cls(hypers, DatasetInfo("angstrom", [1], targets))

    def test_output_and_extradata_input(self, hypers):
        targets = {TARGET: get_features_info(1, sample_kind="system")}
        dataset = DatasetInfo("angstrom", [1], targets, {INPUT: get_features_info(1)})
        with pytest.raises(ValueError, match="present only in extra_data"):
            setup_extenders(dataset, {self.extender: hypers})

    def test_target_layout(self, hypers):
        with pytest.raises(UnavailableOutputError, match="Missing"):
            self.extender_cls(hypers, DatasetInfo("angstrom", [1], {}))
        for info in (get_features_info(1), get_features_info(2, sample_kind="system")):
            with pytest.raises(ValueError, match="single scalar"):
                self.extender_cls(hypers, DatasetInfo("angstrom", [1], {TARGET: info}))


class TestOutputs(OutputTests, MeanPoolingTests):
    def test_output(self, hypers, dataset_info, dtype, device):
        pool = self.extender_cls(hypers, dataset_info).to(dtype=dtype, device=device)
        target = dataset_info.targets[TARGET].layout.to(dtype=dtype, device=device)
        # Selected atoms are interleaved across two unequal-size groups.
        samples = Labels(
            ["system", "atom"],
            torch.tensor([[1, 4], [0, 2], [1, 0], [1, 2]], device=device),
        )
        values = torch.tensor(
            [[1.0], [8.0], [2.0], [9.0]], dtype=dtype, device=device, requires_grad=True
        )
        block = TensorBlock(values, samples, [], target.block().properties)
        result = pool(
            [],
            {TARGET: ModelOutput(sample_kind="system")},
            {INPUT: TensorMap(target.keys, [block])},
            samples,
        )
        pooled = result[TARGET].block()
        assert result[TARGET].keys == target.keys
        assert pooled.samples == Labels.range("system", 2).to(device)
        assert pooled.properties == target.block().properties
        torch.testing.assert_close(pooled.values, values.new_tensor([[8.0], [4.0]]))
        pooled.values.sum().backward()
        torch.testing.assert_close(
            values.grad, values.new_tensor([[1 / 3], [1.0], [1 / 3], [1 / 3]])
        )

    def test_unsupported_requests(self, hypers, dataset_info):
        pool = self.extender_cls(hypers, dataset_info)
        with pytest.raises(ValueError, match="system-level"):
            pool([], {TARGET: ModelOutput(sample_kind="atom")}, {})
        with pytest.raises(ValueError, match="explicit gradients"):
            request = ModelOutput(
                sample_kind="system", explicit_gradients=["positions"]
            )
            pool([], {TARGET: request}, {})
