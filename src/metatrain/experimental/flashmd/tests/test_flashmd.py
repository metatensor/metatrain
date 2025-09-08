import pytest
from metatensor.torch import TensorMap, Labels, TensorBlock
import torch
from metatrain.experimental.flashmd.model import FlashMD
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import TargetInfo
from omegaconf import OmegaConf
from metatomic.torch import System, ModelOutput
from metatrain.utils.neighbor_lists import get_requested_neighbor_lists, get_system_with_neighbor_lists


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_it_works():
  "Run a forward pass of FlashMD on two small systems and verify the output shapes."

  # load default hyper parameters for FlashMD
  full_hypers = OmegaConf.load("../default-hypers.yaml")
  model_hypers = dict(full_hypers)["architecture"]["model"]

  # define dataset (especially the targets)
  dataset_info = DatasetInfo(
    length_unit="angstrom",
    atomic_types=[1, 6],
    targets={
      name: TargetInfo(
        layout=TensorMap(
          keys=Labels.single(),
          blocks=[
            TensorBlock(
              values=torch.empty((0, 3, 1)),
              samples=Labels(
                names=["system", "atom"],
                values=torch.empty((0, 2), dtype=int),
              ),
              components=[
                Labels.range("xyz", 3),
              ],
              properties=Labels.range("length", 1),
            )
          ]
        ),
        quantity="length",
        unit="angstrom",
      )
      for name in ["positions", "momenta"]
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
  for system in systems:
    num_atoms = len(system)
    tmap = TensorMap(
      keys=Labels.single(),
      blocks=[
        TensorBlock(
          # TODO: get the momenta from the system if available!
          values=torch.randn(num_atoms, 3, dtype=dtype),
          samples=Labels(
            names=["system"],
            values=torch.arange(num_atoms, dtype=int).unsqueeze(-1),
          ),
          components=[],
          properties=Labels.range("length", 3),
        ),
      ]
    )
    system.add_data("momenta", tmap)

  outputs = {
    "positions": ModelOutput(quantity="length", unit="angstrom", per_atom=True),
    "momenta": ModelOutput(quantity="length", unit="angstrom", per_atom=True),
  }
  result_dict = model(systems, outputs)

  assert set(result_dict.keys()) == set(outputs.keys())

  # 2+3=5 atoms in total, both outputs are 3D vectors per atom
  assert result_dict["positions"][0].values.shape == (5, 3, 1)
  assert result_dict["momenta"][0].values.shape == (5, 3, 1)
