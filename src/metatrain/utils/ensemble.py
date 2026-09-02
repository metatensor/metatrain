# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
"""Shallow-ensemble building blocks, shared across architectures.

A "shallow ensemble" replicates only a model's head and/or last-layer (readout)
modules ``members`` times, keeping the (expensive) backbone shared and trained
jointly. Deliberately minimal: an ensemble member is nothing more than one more
call to a module factory, so this module provides exactly:

* :class:`ShallowEnsembleHypers` -- the ``{scope, members, dropout, bagging}``
  hyperparameter block, and :func:`validate_shallow_ensemble_hypers`, its
  options-validation hook. ``dropout``/``bagging`` are two independent, optional
  ways to inject more diversity between members than independent initialization
  alone provides on a shared backbone -- see their own docstrings.
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
    dropout: float = 0.0
    """Dropout probability applied inside each ensemble member's own head MLP.

    Only meaningful for ``scope="head"``: with ``scope="readout"`` the head is
    shared and evaluated exactly once per forward pass, so every member would
    see the identical dropout mask, injecting no diversity at all -- rejected
    at options-validation time rather than silently doing nothing. With
    ``scope="head"`` each member's head is its own module instance, evaluated
    separately, so each draws its own independent mask: this gives members a
    genuine source of diversity in their own training trajectory beyond
    independent initialization, which on a shared backbone is otherwise the
    only thing that keeps them from converging towards the same function. ``0``
    (the default) disables it, reproducing the exact model structure and
    behavior from before this hyperparameter was added.
    """
    bagging: float = 1.0
    """Per-(member, atom) keep probability for an online-bagging-style training
    regularizer (streaming approximation to bootstrap aggregation, e.g. Oza &
    Russell, 2001).

    During training only, each member's contribution to the mean prediction
    that reaches the loss is included with this probability, drawn
    independently per member, per atom, and per forward pass; an atom for
    which every member happens to be excluded that draw falls back to the
    plain, uniform mean, so the mean is never left undefined. This gives every
    member a different, resampled view of the training signal on every step,
    without needing separate dataloaders per member.

    Does not affect the reported variance/uncertainty, which is always the
    plain (unbiased) variance of the raw, unweighted member predictions, nor
    evaluation: ``model.eval()`` always uses the plain, uniform mean,
    regardless of this setting. ``1.0`` (the default, meaning "always keep
    every member") disables it, reproducing the exact computation from before
    this hyperparameter was added.
    """


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
    if not (0.0 <= filled["dropout"] < 1.0):
        raise ValueError(
            f"'shallow_ensemble.dropout' must be in [0, 1) (got {filled['dropout']})."
        )
    if not (0.0 < filled["bagging"] <= 1.0):
        raise ValueError(
            f"'shallow_ensemble.bagging' must be in (0, 1] (got {filled['bagging']})."
        )
    if filled["dropout"] > 0.0 and filled["scope"] == "readout":
        raise ValueError(
            "'shallow_ensemble.dropout' has no effect with scope='readout': the "
            "head is shared and evaluated once per forward pass, so every member "
            "would see the identical dropout mask. Use scope='head' instead, or "
            "leave 'dropout' at its default (0)."
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
