"""Reproduce nine annotation mismatches; success means they remain present.

    uv run --no-project --python 3.12 --with-editable . \
        python type_annotations_repro.py
"""

import inspect
import platform
import warnings
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args, get_origin, get_type_hints

import ase
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, systems_to_torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from metatrain.composition.model import CompositionModel
from metatrain.composition.trainer import Trainer
from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import CombinedDataLoader, DatasetInfo, get_dataset
from metatrain.utils.data.dataset import (
    Dataset,
    DiskDataset,
    MemmapDataset,
    _train_test_random_split,
)
from metatrain.utils.data.target_info import (
    TargetInfo,
    _get_spherical_irreps_iter,
    get_energy_target_info,
)
from metatrain.utils.data.writers import MemmapWriter
from metatrain.utils.external_naming import to_external_name
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists_transform
from metatrain.utils.output_gradient import compute_gradient
from metatrain.utils.wrapper import MetatrainModel


def format_annotation(annotation: object) -> str:
    """Format types, including TorchScript classes."""
    for name, cls in (
        ("Labels", Labels),
        ("TensorBlock", TensorBlock),
        ("TensorMap", TensorMap),
        ("ModelOutput", ModelOutput),
    ):
        if annotation is cls:
            return name
    origin = get_origin(annotation)
    if origin is not None:
        name = origin.__name__
        args = get_args(annotation)
        if args:
            return f"{name}[{', '.join(format_annotation(arg) for arg in args)}]"
        return name
    return getattr(annotation, "__name__", str(annotation))


def report(name: str, annotation: object, observed: str) -> None:
    """Print declared and observed types."""
    print(f"{name}: declared={format_annotation(annotation)}; actual={observed}")


def probe_gradient() -> None:
    """Probe gradient containers."""
    annotation = get_type_hints(compute_gradient)["return"]
    assert get_origin(annotation) is list
    assert get_args(annotation) == (torch.Tensor,)

    x = torch.tensor(2.0, requires_grad=True)
    normal = compute_gradient(x * x, [x], is_training=False, destroy_graph=True)
    assert isinstance(normal, tuple)
    torch.testing.assert_close(normal[0], torch.tensor(4.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        constant = compute_gradient(
            torch.tensor(1.0), [x], is_training=False, destroy_graph=True
        )
    assert any("GRADIENT WARNING" in str(w.message) for w in caught)
    assert isinstance(constant, list)
    torch.testing.assert_close(constant[0], torch.tensor(0.0))
    report(
        "1. compute_gradient",
        annotation,
        f"{type(normal).__name__} (normal), {type(constant).__name__} (constant)",
    )


def probe_llpr_checkpoint() -> None:
    """Probe fine-tuning's return type."""
    annotation = get_type_hints(LLPRUncertaintyModel.load_checkpoint)["return"]
    assert annotation is LLPRUncertaintyModel

    wrapped = Trainer({"densify_atomic_basis": False}).setup(
        {}, DatasetInfo("", [1], {})
    )
    result = LLPRUncertaintyModel.load_checkpoint(
        {"wrapped_model_checkpoint": wrapped.get_checkpoint()}, context="finetune"
    )
    assert isinstance(result, MetatrainModel)
    assert not isinstance(result, annotation)
    report(
        "2. LLPR.load_checkpoint",
        annotation,
        f"{type(result).__name__} (finetune)",
    )


def probe_dataset_factory() -> None:
    """Probe ZIP and memmap dataset types."""
    annotation = get_type_hints(get_dataset)["return"]
    declared_dataset = get_args(annotation)[0]
    assert declared_dataset is Dataset

    fixture = (
        Path(__file__).resolve().parent / "tests/resources/spherical_disk_dataset.zip"
    )
    dataset, _, _ = get_dataset(
        OmegaConf.create({"systems": {"read_from": str(fixture)}, "targets": {}})
    )
    assert isinstance(dataset, DiskDataset)
    assert not isinstance(dataset, declared_dataset)
    report(
        "3. get_dataset()[0] (ZIP)",
        declared_dataset,
        type(dataset).__name__,
    )

    with TemporaryDirectory() as directory:
        writer = MemmapWriter(directory)
        energy = TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.zeros((1, 1), dtype=torch.float64),
                    samples=Labels.range("system", 1),
                    components=[],
                    properties=Labels.range("energy", 1),
                )
            ],
        )
        writer.write(
            [systems_to_torch(ase.Atoms("H"), dtype=torch.float64)], {"energy": energy}
        )
        writer.finish()
        memmap, _, _ = get_dataset(
            OmegaConf.create({"systems": {"read_from": directory}, "targets": {}})
        )
        assert isinstance(memmap, MemmapDataset)
        assert not isinstance(memmap, declared_dataset)
        report(
            "3. get_dataset()[0] (memmap)",
            declared_dataset,
            type(memmap).__name__,
        )


def probe_combined_loader() -> None:
    """Probe the yielded batch type."""
    annotation = get_type_hints(CombinedDataLoader.__next__)["return"]
    assert annotation is DataLoader

    loader = CombinedDataLoader([DataLoader([1, 2], batch_size=2)], shuffle=False)
    batch = next(loader)
    assert isinstance(batch, torch.Tensor)
    assert not isinstance(batch, annotation)
    assert batch.tolist() == [1, 2]
    report(
        "4. CombinedDataLoader.__next__",
        annotation,
        type(batch).__name__,
    )


def probe_random_split() -> None:
    """Probe split dataset types."""
    annotation = get_type_hints(_train_test_random_split)["return"]
    assert get_origin(annotation) is list
    declared_dataset = get_args(annotation)[0]
    assert declared_dataset is Dataset

    original = Dataset.from_dict({"value": [0, 1, 2, 3]})
    # Keep the probe deterministic without changing global RNG state.
    state = np.random.get_state()
    try:
        np.random.seed(0)
        parts = _train_test_random_split(original, 0.5, 0.5)
    finally:
        np.random.set_state(state)
    assert all(isinstance(part, Subset) for part in parts)
    assert all(not isinstance(part, declared_dataset) for part in parts)
    assert sorted(parts[0].indices + parts[1].indices) == [0, 1, 2, 3]
    report(
        "5. _train_test_random_split",
        annotation,
        f"list[{type(parts[0]).__name__}]",
    )


def probe_spherical_properties() -> None:
    """Probe optional property labels."""
    annotation = get_type_hints(_get_spherical_irreps_iter)["return"]
    properties_annotation = get_args(annotation)[1]
    assert get_origin(properties_annotation) is list
    assert get_args(properties_annotation)[0] is Labels

    for product in (None, "cartesian"):
        _, properties = _get_spherical_irreps_iter(
            [{"o3_lambda": 0, "o3_sigma": 1}],
            {"num_subtargets": 1},
            product=product,
        )
        assert properties == [None]
    report(
        "6. _get_spherical_irreps_iter()[1]",
        properties_annotation,
        "[None] (default/cartesian)",
    )


def probe_supported_devices() -> None:
    """Probe device names."""
    annotation = inspect.get_annotations(ModelInterface)["__supported_devices__"]
    assert get_origin(annotation) is list
    assert get_args(annotation) == (torch.device,)

    devices = CompositionModel.__supported_devices__
    assert devices == ["cuda", "cpu"]
    assert all(isinstance(device, str) for device in devices)
    assert all(not isinstance(device, torch.device) for device in devices)
    report("7. ModelInterface.__supported_devices__", annotation, repr(devices))


def probe_neighbor_transform() -> None:
    """Probe callback dictionary types."""
    transform = get_system_with_neighbor_lists_transform([])
    annotations = get_type_hints(transform)
    assert get_args(annotations["targets"])[1] is TensorBlock
    assert get_args(annotations["extra"])[1] is TensorBlock
    assert get_args(get_args(annotations["return"])[1])[1] is TensorBlock
    assert get_args(get_args(annotations["return"])[2])[1] is TensorBlock

    info = get_energy_target_info("energy", {"unit": "eV"})
    system = systems_to_torch(ase.Atoms("H"), dtype=torch.float64)
    _, targets, extra = transform(
        [system], {"energy": info.layout}, {"aux": info.layout}
    )
    assert targets["energy"]._type().name() == "TensorMap"
    assert extra["aux"]._type().name() == "TensorMap"
    report(
        "8. neighbor transform targets/extra",
        annotations["targets"],
        f"dict[str, {targets['energy']._type().name()}]",
    )


def probe_external_naming() -> None:
    """Probe TargetInfo inputs."""
    annotation = get_type_hints(to_external_name)["quantities"]
    assert get_origin(annotation) is dict
    assert get_args(annotation)[1] is ModelOutput

    info = get_energy_target_info("energy", {"unit": "eV"})
    assert isinstance(info, TargetInfo)
    result = to_external_name("energy_positions_gradients", {"energy": info})
    assert result == "forces"
    report(
        "9. to_external_name quantities",
        annotation,
        f"dict[str, {type(info).__name__}]",
    )


def main() -> None:
    """Run all probes."""
    packages = ("metatrain", "torch", "metatensor-learn", "metatensor-torch")
    versions = ", ".join(f"{package}={version(package)}" for package in packages)
    print(f"Python={platform.python_version()}, {versions}")
    for probe in (
        probe_gradient,
        probe_llpr_checkpoint,
        probe_dataset_factory,
        probe_combined_loader,
        probe_random_split,
        probe_spherical_properties,
        probe_supported_devices,
        probe_neighbor_transform,
        probe_external_naming,
    ):
        probe()


if __name__ == "__main__":
    main()
