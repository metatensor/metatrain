Internal E-PET alignment note
=============================

This note records the local integration state of ``experimental.e_pet``. It is
branch-maintenance context, not user documentation; keep it out of a PR commit stack
unless reviewers explicitly ask for provenance.

Baseline
--------

- Original replay baseline: ``origin/main`` commit
  ``45c3f6bed7fe91764eabaecbfb1b1418c8483f47``.
- Current replay baseline: ``origin/main`` commit
  ``1e12a4ad662e239ea2b106903c128cde99d80659``.
- Archival source branch: ``deep_token`` at commit ``6ec71034``. It remains
  read-only source material, not an integration target.

PR-facing retained surface
--------------------------

- ``metatrain.experimental.e_pet``:
  PET-backed E-PET model, trainer, docs, and tests.
- Tensor-basis readouts:
  E-PET spherical blocks infer angular order from target ``o3_lambda``. The public
  E-PET tensor-basis SOAP options are deliberately narrow: ``max_radial`` and
  ``cutoff``.
- Optimizer surface:
  ``learning_rate`` applies to PET trunk, PET heads, scalar paths, and coefficient
  readouts. ``tensor_basis_learning_rate`` is the only optional separate LR and
  applies only to ``basis_calculators.*``.
- Cartesian rank-2 targets:
  public Cartesian stress-like targets are converted internally into hidden
  spherical ``l=0`` / ``l=2`` readouts, reconstructed to Cartesian form, and then
  passed through loss, metrics, scaling, and optional volume normalization under the
  public target name.
- Atomic-basis spherical targets:
  E-PET follows PET's densified training / sparse evaluation contract, with one
  target-level head by default and one tensor-basis calculator per densified irrep.
- PET compatibility features needed by comparison workflows:
  token-rotation-compatible feature outputs, ``volume_normalized_targets``,
  ``shared_head_groups``, and invariant losses.

Default-off retained controls
-----------------------------

- ``atomic_basis_irrep_balanced_loss``:
  optional PET/E-PET fair-comparison objective for per-atom spherical atomic-basis
  targets. It is retained because it produced the stable atomic-basis comparison
  surface; it remains default-off and is not a replacement for generic losses.
- ``basis_gram_weight`` and ``coefficient_l2_weight``:
  optional E-PET regularizers. ``coefficient_l2_exclude_spherical_l0`` is retained
  so hidden or public scalar spherical blocks can stay PET-like while nontrivial
  tensor-basis coefficients are regularized.
- ``legacy`` tensor-basis layout:
  compatibility switch for existing SOAP-BPNN tensor-basis species layouts. It is
  retained because it selects a real tensor-basis implementation path, not because
  of branch history.

Dropped branch artifacts
------------------------

- Numbered ``pet_*`` snapshot architectures and prototype runtime wrappers.
- Benchmark-only metrics/reporting and branch-local experiment orchestration.
- Deprecated E-PET tensor-basis options:
  ``max_angular``, ``max_lambda``, ``add_l1_species_dependent_vector``, and
  ``l1_species_dependent_vector_soap``.
- Exploratory readout-specific LR options:
  ``pet_trunk_learning_rate``, ``readout_learning_rate``, and
  ``spherical_l0_readout_learning_rate``. Exact historical reproductions are kept
  through the local safety branch, not through PR-facing deprecated options.
- Exploratory tensor-basis normalization and scaler-floor diagnostics. The retained
  default workflow uses the raw tensor basis together with the fitted target scaler
  and, when needed, the default-off irrep-balanced objective.
