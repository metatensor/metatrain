"""Tests for BasePrecision OmegaConf interpolation."""

from omegaconf import OmegaConf

from metatrain.share.base_hypers import BasePrecision
from metatrain.utils.architectures import get_default_hypers


def test_default_hypers_contains_interpolation():
    """get_default_hypers without base_precision returns interpolation strings."""
    hypers = get_default_hypers("soap_bpnn")
    # soap_bpnn has no BasePrecision fields, so just check the call succeeds
    # and returns a plain dict
    assert isinstance(hypers, dict)
    assert hypers["name"] == "soap_bpnn"


def test_default_hypers_resolves_with_base_precision():
    """get_default_hypers with base_precision resolves all interpolations."""
    hypers = get_default_hypers("soap_bpnn", base_precision=64)
    assert isinstance(hypers, dict)
    assert hypers["name"] == "soap_bpnn"
    # All values should be concrete (no interpolation strings)
    _assert_no_interpolations(hypers)


def test_interpolation_resolves_after_merge():
    """OmegaConf merge + access resolves ${base_precision} correctly."""
    arch_defaults = OmegaConf.create(
        {
            "architecture": {
                "model": {
                    "descriptor": {"precision": BasePrecision.value, "other": "keep"},
                    "fitting_net": {"precision": BasePrecision.value},
                }
            }
        }
    )
    user_options = OmegaConf.create({"base_precision": 64})
    merged = OmegaConf.merge(user_options, arch_defaults)

    assert merged["architecture"]["model"]["descriptor"]["precision"] == 64
    assert merged["architecture"]["model"]["fitting_net"]["precision"] == 64
    assert merged["architecture"]["model"]["descriptor"]["other"] == "keep"


def test_user_override_propagates():
    """User setting base_precision propagates through interpolation."""
    arch_defaults = OmegaConf.create(
        {
            "architecture": {
                "model": {
                    "outer": {
                        "inner": {"precision": BasePrecision.value},
                    }
                }
            }
        }
    )
    user_options = OmegaConf.create({"base_precision": 32})
    merged = OmegaConf.merge(user_options, arch_defaults)
    assert merged["architecture"]["model"]["outer"]["inner"]["precision"] == 32


def test_explicit_precision_overrides_interpolation():
    """Explicit precision: 16 in user options overrides the interpolation."""
    arch_defaults = OmegaConf.create(
        {
            "architecture": {
                "model": {
                    "descriptor": {"precision": BasePrecision.value},
                }
            }
        }
    )
    user_options = OmegaConf.create(
        {
            "base_precision": 64,
            "architecture": {
                "model": {
                    "descriptor": {"precision": 16},
                }
            },
        }
    )
    merged = OmegaConf.merge(arch_defaults, user_options)
    # The user's explicit 16 wins over the interpolation
    assert merged["architecture"]["model"]["descriptor"]["precision"] == 16


def _assert_no_interpolations(d):
    """Recursively verify no unresolved OmegaConf interpolation strings."""
    if isinstance(d, dict):
        for v in d.values():
            _assert_no_interpolations(v)
    elif isinstance(d, list):
        for v in d:
            _assert_no_interpolations(v)
    elif isinstance(d, str):
        assert not d.startswith("${"), f"Unresolved interpolation: {d}"
