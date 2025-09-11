import subprocess

import pytest
import torch
from metatomic.torch.ase_calculator import MetatomicCalculator
from omegaconf import OmegaConf

from metatrain.utils.architectures import get_default_hypers, import_architecture

from .....tests.utils import RESOURCES_PATH


torch.manual_seed(42)


def test_llpr(tmpdir):
    """
    Tests the functionalities of the LLPRUncertaintyModel, mainly from the CLI.
    """
    # 1. Train a PET model on a subset of the QM7 dataset:
