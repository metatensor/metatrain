import warnings
from collections import defaultdict
from typing import Callable, Optional

import graph2mat
import sisl
import torch
from graph2mat import (
    BasisConfiguration,
    BasisMatrix,
    MatrixDataProcessor,
)
from graph2mat.bindings.torch import TorchBasisMatrixDataset
from graph2mat.core.data.basis import NoBasisAtom
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.data import TargetInfo, system_to_ase

from .conversions import get_target_converters
from .mtt import g2m_labels_to_tensormap

def _log_tmap(tmap: TensorMap) -> TensorMap:
    new_blocks = []
    for block in tmap.blocks():
        new_block = TensorBlock(
            values=torch.sign(block.values) * torch.log(block.values.abs() + 1),
            samples=block.samples,
            components=block.components,
            properties=block.properties
        )
        new_blocks.append(new_block)
    return TensorMap(keys=tmap.keys, blocks=new_blocks)

def _unlog_tmap(tmap: TensorMap) -> TensorMap:
    new_blocks = []
    for block in tmap.blocks():
        new_block = TensorBlock(
            values=torch.sign(block.values) * (torch.exp(block.values.abs()) - 1),
            samples=block.samples,
            components=block.components,
            properties=block.properties
        )
        new_blocks.append(new_block)
    return TensorMap(keys=tmap.keys, blocks=new_blocks)

def get_log_transform(
    matrices: dict[str, dict],
) -> Callable:
    
    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ):
        
        for matrix_spec in matrices.values():
            if not matrix_spec.get("learn_log", False):
                continue
            
            nodes_target = matrix_spec["nodes"]
            edges_target = matrix_spec["edges"]

            targets[nodes_target] = _log_tmap(targets[nodes_target])
            targets[edges_target] = _log_tmap(targets[edges_target])

        return systems, targets, extra

    return transform

def get_unlog_transform(
    matrices: dict[str, dict],
) -> Callable:
    
    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ):
        
        for matrix_spec in matrices.values():
            if not matrix_spec.get("learn_log", False):
                continue
            
            nodes_target = matrix_spec["nodes"]
            edges_target = matrix_spec["edges"]

            targets[nodes_target] = _unlog_tmap(targets[nodes_target])
            targets[edges_target] = _unlog_tmap(targets[edges_target])

        return systems, targets, extra

    return transform
