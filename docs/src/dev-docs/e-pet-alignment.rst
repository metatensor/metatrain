E-PET alignment note
====================

This branch is intentionally replayed on top of ``origin/main`` commit
``45c3f6bed7fe91764eabaecbfb1b1418c8483f47`` instead of being cleaned up in place
from ``deep_token``.

Replay source
-------------

- Archival source branch: ``deep_token`` at commit ``6ec71034``.
- Selective carry-over source: the corresponding local working tree on top of that
  branch, which contained the ``experimental/e_pet`` architecture and local PET
  diagnostic-hook refinements.

Retained suite surface
----------------------

- ``metatrain.experimental.e_pet``:
  the E-PET model, trainer, defaults, and tests.
- ``metatrain.soap_bpnn.modules.tensor_basis``:
  the tensor-basis extensions required by E-PET.
- ``metatrain.pet`` diagnostic compatibility:
  opt-in ``mtt::features::{path}`` captures, plus the existing
  ``features`` and ``mtt::aux::{target}_last_layer_features`` outputs.
- Generic PET-OMat enablement in the suite:
  ``invariant_mse`` / ``invariant_huber`` in ``metatrain.utils.loss``,
  plus PET-side ``volume_normalized_targets`` and ``shared_head_groups``.

Intentionally dropped from the replay
-------------------------------------

- Numbered ``pet_*`` snapshot architectures and other branch-history replicas.
- Comparison-specific runtime wrappers from ``pet_tensorbasis_prototype``.
- Benchmark-only metrics, reporting, and other branch-local experimentation
  unrelated to E-PET runtime or token-rotation compatibility.
