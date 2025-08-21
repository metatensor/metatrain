from urllib.parse import urlparse
from urllib.request import urlretrieve

import pytest
import torch
from metatomic.torch import ModelOutput

from metatrain.pet.modules.compatibility import convert_checkpoint_from_legacy_pet
from metatrain.utils.data import read_systems
from metatrain.utils.io import load_model, model_from_checkpoint
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_WITH_FORCES_PATH


LEGACY_VERSIONS = ["0.3.2", "0.4.1", "1.0.0"]
STABLE_VERSIONS = ["1.0.1", "1.1.0"]


@pytest.mark.parametrize("version", STABLE_VERSIONS)
def test_pet_mad_consistency(version, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = f"https://huggingface.co/lab-cosmo/pet-mad/resolve/v{version}/models/pet-mad-v{version}.ckpt"

    if version in LEGACY_VERSIONS:
        if urlparse(path).scheme:
            path, _ = urlretrieve(path)

        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        checkpoint = convert_checkpoint_from_legacy_pet(checkpoint)
        pet_mad_model = model_from_checkpoint(checkpoint, context="export").eval()

    msg1 = (
        "trying to upgrade an old model checkpoint with unknown version, this "
        "might fail and require manual modifications"
    )
    msg2 = (
        "PET assumes that Cartesian tensors of rank 2 are stress-like, meaning that "
        "they are symmetric and intensive. If this is not the case, please use a "
        "different model."
    )
    with pytest.warns(UserWarning, match=f"({msg1}|{msg2})"):
        pet_mad_model = load_model(path).eval()

    systems = read_systems(DATASET_WITH_FORCES_PATH)[:5]
    for system in systems:
        system.positions.requires_grad_(True)
        get_system_with_neighbor_lists(system, pet_mad_model.requested_neighbor_lists())

    systems = [system.to(torch.float32) for system in systems]

    outputs = {"energy": ModelOutput(per_atom=False)}

    predictions = pet_mad_model(systems, outputs)

    expected_output = torch.tensor(
        [
            [-36.721317291260],
            [-37.199848175049],
            [-37.049076080322],
            [-36.586517333984],
            [-37.476135253906],
        ]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print("VERSION", version)
    # print(predictions["energy"].block().values[:5])

    torch.testing.assert_close(
        predictions["energy"].block().values[:5], expected_output, atol=5e-2, rtol=5e-2
    )
