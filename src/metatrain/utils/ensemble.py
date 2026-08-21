# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
"""Shallow-ensemble building blocks, shared across architectures.

A "shallow ensemble" replicates only a model's head and/or last-layer (readout)
modules ``members`` times, keeping the (expensive) backbone shared and trained
jointly. Deliberately minimal: an ensemble member is nothing more than one more
call to a module factory, so this module provides exactly:

* :class:`ShallowEnsembleHypers` -- the ``{scope, members}`` hyperparameter block,
  and :func:`validate_shallow_ensemble_hypers`, its options-validation hook.
* :func:`make_ensemble_members` -- wraps a module factory into ``members``
  independently-initialized copies, stored under ``member_{e}`` keys so that they
  show up as such in the state dict.
* :func:`uncertainty_output_name` -- the ``{target}_uncertainty`` auxiliary
  output naming convention (matching the one already used by
  :mod:`metatrain.llpr.model` for its own, unrelated, post-hoc uncertainties),
  used for the ensemble variance.

Everything else (how the mean and variance across members are computed, how the
variance is exposed and scaled, how it enters the loss) is architecture code --
see ``metatrain.pet.modules.backend``, ``metatrain.pet.model`` and
``metatrain.utils.loss.TensorMapEnsembleNLLLoss``.
"""

from typing import Any, Callable, Literal

import torch
from pydantic import BeforeValidator
from typing_extensions import Annotated, TypedDict

from .hypers import init_with_defaults


class ShallowEnsembleHypers(TypedDict):
    """Hyperparameters for an optional shallow ensemble of a model's output heads."""

    scope: Literal["head", "readout"] = "head"
    """Which modules get replicated per ensemble member.

    ``"readout"``: only the (strictly linear) last-layer readout is replicated;
    the (nonlinear) head stays shared across members. This is the cheapest option,
    and matches the "shallow ensemble" construction of replicating only the last
    linear layer.

    ``"head"``: both the head and the readout are replicated per member, so each
    member has its own independent nonlinear head as well as its own readout.
    More expressive (and more expensive) than ``"readout"``, but still cheap
    relative to the shared backbone.
    """
    members: int = 1
    """Number of ensemble members. Must be ``> 1`` for ensembling to actually be
    enabled -- a ``shallow_ensemble`` block with ``members: 1`` is rejected at
    options-validation time (use ``shallow_ensemble: null``, the default, to
    disable ensembling instead)."""


def validate_shallow_ensemble_hypers(raw: Any) -> Any:
    """Options-validation hook for :class:`ShallowEnsembleHypers`.

    Two things happen here, both because pydantic does not fill in a
    ``TypedDict`` field's own ``= default`` when validating it as a *nested*
    model (only ``init_with_defaults``/OmegaConf-merged top-level hypers get
    that treatment elsewhere in this codebase) -- so a user-supplied
    ``shallow_ensemble: {members: 4}`` would otherwise fail with a confusing
    "scope: Field required", even though ``scope`` has a perfectly good default:

    1. Missing keys are filled from :func:`init_with_defaults
       <metatrain.utils.hypers.init_with_defaults>` before pydantic's own
       ``TypedDict`` field validation runs (this must be a ``BeforeValidator``,
       not an ``AfterValidator``, for that reason).
    2. ``members <= 1`` is rejected: a present-but-inert ``shallow_ensemble``
       block looks like a misconfiguration (the user presumably meant to enable
       ensembling), so this is a hard validation error rather than a silent
       no-op. Use ``shallow_ensemble: null`` (the field's default) to disable
       ensembling instead.

    :param raw: the raw, possibly-partial ``shallow_ensemble`` value as parsed
        from ``options.yaml`` (or ``None``, passed through untouched).
    :return: ``raw`` with missing keys filled in, ready for pydantic's own
        ``TypedDict`` field validation.
    :raises ValueError: if ``members <= 1``.
    """
    if not isinstance(raw, dict):
        # not a dict (including ``None``): let pydantic's own type checking
        # produce the error, this validator has nothing useful to add
        return raw
    filled = {**init_with_defaults(ShallowEnsembleHypers), **raw}
    if filled["members"] <= 1:
        raise ValueError(
            "'shallow_ensemble.members' must be > 1 for ensembling to have any "
            f"effect (got {filled['members']}). Use 'shallow_ensemble: null' "
            "(the default) to disable ensembling entirely."
        )
    return filled


ShallowEnsembleHypersField = Annotated[
    ShallowEnsembleHypers, BeforeValidator(validate_shallow_ensemble_hypers)
]
"""``Optional[ShallowEnsembleHypersField] = None`` is the pattern to use on a
model's ``ModelHypers`` TypedDict to expose this hyperparameter block."""


def make_ensemble_members(
    factory: Callable[[], torch.nn.Module], members: int
) -> torch.nn.ModuleDict:
    """Build ``members`` independently-initialized copies of a module.

    Each call to ``factory()`` should construct a fresh module (e.g. via a
    ``lambda`` closing over the constructor arguments), so that the ``members``
    copies get independent (PyTorch-default-random) initializations. They are
    stored under ``member_0``, ``member_1``, ... keys, so that ``member_{e}``
    appears as a path segment in the resulting state dict, and so that plain
    ``.values()`` iteration (TorchScript-compatible, as already used elsewhere in
    this codebase for ``nn.ModuleDict``/``nn.ModuleList`` collections of
    independent submodules, e.g. :class:`metatrain.utils.readout.MoEReadout`)
    visits the members in a fixed, deterministic order.

    :param factory: zero-argument callable constructing one ensemble member.
    :param members: number of members to build. Must be ``> 1``; this is not
        re-validated here (see :func:`validate_shallow_ensemble_hypers`).
    :return: a ``ModuleDict`` with keys ``member_0``, ..., ``member_{members-1}``.
    """
    return torch.nn.ModuleDict({f"member_{e}": factory() for e in range(members)})


def uncertainty_output_name(target_name: str) -> str:
    """Name of the ``_uncertainty`` auxiliary output for a target.

    Matches the convention already used by :mod:`metatrain.llpr.model` for its
    own (unrelated, post-hoc) uncertainties:
    ``mtt::aux::{target}_uncertainty``, except for ``"energy"`` which
    special-cases to the bare ``"energy_uncertainty"``.

    :param target_name: name of the base target (e.g. ``"energy"``).
    :return: the name of the corresponding uncertainty auxiliary output.
    """
    if target_name == "energy":
        return "energy_uncertainty"
    return "mtt::aux::" + target_name.replace("mtt::", "") + "_uncertainty"
