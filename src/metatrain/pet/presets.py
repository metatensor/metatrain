"""Preset configurations for PET models.

This module defines preset hyperparameter configurations for PET models,
providing users with pre-configured options optimized for different use cases.
"""

from typing import Dict, Literal, Optional

# Preset types that are available
PETPresets = Literal["default", "fast", "medium", "large"]


def get_preset_hypers(preset: PETPresets) -> Dict:
    """Get hyperparameter preset for PET models.

    :param preset: The preset name ("default", "fast", "medium", or "large")
    :return: Dictionary with preset hyperparameters
    :raises ValueError: If preset is not recognized
    """
    presets = {
        "default": {},
        "fast": {
            "model": {
                "cutoff": 3.5,
                "d_pet": 64,
                "d_node": 128,
                "d_feedforward": 128,
                "num_heads": 4,
                "num_attention_layers": 1,
                "num_gnn_layers": 1,
                "long_range": {"enabled": False},
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 2e-4,
            },
        },
        "medium": {
            "model": {
                "cutoff": 4.5,
                "d_pet": 128,
                "d_node": 256,
                "d_feedforward": 256,
                "num_heads": 8,
                "num_attention_layers": 2,
                "num_gnn_layers": 2,
                "long_range": {"enabled": False},
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 1e-4,
            },
        },
        "large": {
            "model": {
                "cutoff": 5.5,
                "d_pet": 256,
                "d_node": 512,
                "d_feedforward": 512,
                "num_heads": 16,
                "num_attention_layers": 3,
                "num_gnn_layers": 3,
                "long_range": {"enabled": False},
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 5e-5,
            },
        },
    }

    if preset not in presets:
        raise ValueError(
            f"Unknown preset '{preset}'. Available presets are: {list(presets.keys())}"
        )

    return presets[preset]


PRESET_DESCRIPTIONS = {
    "default": "The default hyperparameters as defined in the PET documentation.",
    "fast": "Fast preset optimized for quick training and evaluation. "
    "Smaller model with reduced dimensionality and layers.",
    "medium": "Medium preset providing a balance between speed and accuracy. "
    "Moderate model size suitable for most use cases.",
    "large": "Large preset for maximum accuracy. "
    "Full-featured model with high dimensionality and more layers.",
}
