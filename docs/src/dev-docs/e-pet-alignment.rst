E-PET alignment note
====================

This branch was first realigned from ``deep_token`` onto ``origin/main`` commit
``45c3f6bed7fe91764eabaecbfb1b1418c8483f47`` and then replayed onto latest
``origin/main`` commit ``1e12a4ad662e239ea2b106903c128cde99d80659``. The
``deep_token`` branch remains archival source material, not an integration target.

Replay source
-------------

- Archival source branch: ``deep_token`` at commit ``6ec71034``.
- Selective carry-over source: the corresponding local working tree on top of that
  branch, which contained the ``experimental/e_pet`` architecture and local PET
  diagnostic-hook refinements.

Retained suite surface
----------------------

- ``metatrain.experimental.e_pet``:
  the E-PET model, trainer, defaults, and tests. The promoted PET-OMat
  reference default uses split learning rates with PET trunk ``2e-4``,
  tensor-basis ``1e-3``, and readout ``1e-3``.
- E-PET Cartesian rank-2 targets:
  public Cartesian stress-like targets can be decomposed internally into hidden
  spherical ``l=0`` / ``l=2`` readouts, reconstructed, and normalized by volume
  once under the public target name.
- E-PET atomic-basis support:
  per-atom spherical atomic-basis targets use PET's densified training path,
  one target-level head by default, and one tensor basis per densified irrep.
- ``metatrain.soap_bpnn.modules.tensor_basis``:
  the tensor-basis extensions required by E-PET. E-PET tensor-basis angular
  order is target-derived from each block's ``o3_lambda``; there is no
  E-PET-specific ``max_angular`` or ``max_lambda`` option.
- ``metatrain.pet`` diagnostic compatibility:
  opt-in ``mtt::features::{path}`` captures, plus the existing
  ``features`` and ``mtt::aux::{target}_last_layer_features`` outputs.
- Generic PET-OMat enablement in the suite:
  ``invariant_mse`` / ``invariant_huber`` in ``metatrain.utils.loss``,
  plus default-preserving PET-side ``volume_normalized_targets`` and
  ``shared_head_groups``.

Default-off retained diagnostics
--------------------------------

- E-PET ``basis_normalization`` and ``scale_property_floor_ratio`` remain
  documented diagnostic controls and default to disabled.
- PET/E-PET ``atomic_basis_irrep_balanced_loss`` remains a fair-comparison
  objective for atomic-basis studies and defaults to disabled.
- PET ``edge_harmonics`` remains a default-off trunk-input experiment for fair
  PET/E-PET comparisons.

Intentionally dropped from the replay
-------------------------------------

- Numbered ``pet_*`` snapshot architectures and other branch-history replicas.
- Comparison-specific runtime wrappers from ``pet_tensorbasis_prototype``.
- Benchmark-only metrics, reporting, and other branch-local experimentation
  unrelated to E-PET runtime or token-rotation compatibility.
- Deprecated E-PET-only single species-dependent ``l=1`` vector-basis options;
  E-PET keeps the explicit ``extra_l1_vector_basis_branches`` surface instead.
