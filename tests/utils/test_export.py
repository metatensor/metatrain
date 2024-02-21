from pathlib import Path

from metatensor.torch.atomistic import ModelCapabilities

from metatensor.models.experimental.soap_bpnn import Model
from metatensor.models.utils.export import export, is_exported
from metatensor.models.utils.model_io import load_checkpoint, load_exported_model


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


def test_export(monkeypatch, tmp_path):
    """Tests the export function"""

    model = Model(capabilities=ModelCapabilities(species=[1]))

    monkeypatch.chdir(tmp_path)

    export(model, "exported.pt")

    assert Path("exported.pt").is_file()


def test_is_exported():
    """Tests the is_exported function"""

    checkpoint = load_checkpoint(RESOURCES_PATH / "bpnn-model.ckpt")
    exported_model = load_exported_model(RESOURCES_PATH / "bpnn-model.pt")

    assert is_exported(exported_model)
    assert not is_exported(checkpoint)
