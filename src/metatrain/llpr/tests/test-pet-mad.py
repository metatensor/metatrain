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
    Tests the LLPR wrapper with PET-MAD v1.0.2, which was published using a LLPR
    checkpoint version 1 (before refactoring the LLPR into an architecture).
    """
    # 1. Get the PET-MAD model checkpoint:
