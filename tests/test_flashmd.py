from metatensor.torch import TensorMap, Labels, TensorBlock
import torch
from metatrain.flashmd.model import FlashMD
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import TargetInfo
from omegaconf import OmegaConf
from metatomic.torch import System, ModelOutput
from metatrain.utils.neighbor_lists import get_requested_neighbor_lists, get_system_with_neighbor_lists


def test_it_works():
  pet_default_hypers = OmegaConf.load("src/metatrain/pet/default-hypers.yaml")
  
  model_hypers = {
    "d_pet": 16,
    "hamiltonian": "direct",
    "integrator": "euler",
    "heads": {
      "mtt::delta_q": "linear",
      "mtt::delta_p": "linear",
    },
  }
  model_hypers = {**dict(pet_default_hypers)["architecture"]["model"], **model_hypers}

  dataset_info = DatasetInfo(
    length_unit="angstrom",
    atomic_types=[1, 6],
    targets={
      name: TargetInfo(
        layout=TensorMap(
          keys=Labels.single(),
          blocks=[
            TensorBlock(
              values=torch.empty((0, 1)),
              samples=Labels(
                names=["system"],
                values=torch.empty((0,1), dtype=int),
              ),
              components=[],
              properties=Labels.range("length", 1),
            )
          ]
        ),
        quantity="length",
        unit="angstrom",
      )
      for name in ["mtt::delta_q", "mtt::delta_p"]
    },
  )

  # create a FlashMD model and attach a (random) raw PET model
  model = FlashMD(model_hypers, dataset_info)

  # define example systems
  dtype = torch.float32
  systems = []
  systems.append(
    # system 0: 3 atoms
    System(
      types=torch.tensor([1, 6, 1]),
      positions=torch.randn(3, 3, dtype=dtype),
      cell=torch.eye(3, dtype=dtype),
      pbc=torch.tensor([True] * 3),
    )
  )
  systems.append(
    # system 1: 2 atoms
    System(
      types=torch.tensor([6, 1]),
      positions=torch.randn(2, 3, dtype=dtype),
      cell=torch.eye(3, dtype=dtype),
      pbc=torch.tensor([True] * 3),
    )
  )

  # attach neighbor lists to the systems
  nbr_lists = get_requested_neighbor_lists(model)
  systems = [
    get_system_with_neighbor_lists(system, nbr_lists)
    for system in systems
  ]

  # add random momenta to the systems

  outputs = {
    "mtt::delta_q": ModelOutput(quantity="length", unit="angstrom", per_atom=True),
    "mtt::delta_p": ModelOutput(quantity="length", unit="angstrom", per_atom=True),
  }
  result_dict = model(systems, outputs)
