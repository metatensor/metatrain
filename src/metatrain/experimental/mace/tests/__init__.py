import urllib.request
from pathlib import Path


mace_model_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model"
model_path = Path(__file__).parent / "mace_small.model"

if not model_path.exists():
    urllib.request.urlretrieve(mace_model_url, model_path)
