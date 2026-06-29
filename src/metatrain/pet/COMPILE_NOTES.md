# PET `torch.compile` notes

This documents the data-dependent-shape problems that block `torch.compile` of the
PET core (`core.py` / `modules/structures.py`) and the solutions, so we don't
re-derive them.

## The problem

`PETCore` is split into `preprocess` → `compute_features` → `predict`.
`compute_features` and `predict` are the compute-heavy parts and compile cleanly with
`fullgraph=True` out of the box. `preprocess` (structure → NEF batch tensors) did not,
because of **data-dependent shapes**:

1. **`max_edges_per_node = int(torch.max(num_neighbors))`** — the largest neighbour
   count of any atom becomes the *second dimension of the NEF grid*
   `[n_nodes, max_edges_per_node]`. Extracting it is a tensor→host scalar (an
   `.item()`-class op), so by default Dynamo **graph-breaks** here.

2. **`num_neighbors = torch.bincount(centers, minlength=num_nodes)`** — `bincount` has
   a **data-dependent output shape** (see below), so under
   `capture_dynamic_output_shape_ops` its result is an *unbacked-shaped* tensor. That
   makes `n_nodes = num_neighbors.shape[0]` unbacked and silently corrupts the entire
   NEF layout (wrong `padding_mask`, shuffled `edge_vectors`, negative
   `nef_to_edges_neighbor`, …).

3. **`reverse_neighbor_index[~nef_mask] = torch.arange(num_padded)`** — a boolean-mask
   assignment is a data-dependent *output-shape* op; it can't be captured into a full
   graph at all (it errors under `fullgraph=True`).

> Why `bincount` is data-dependent even with `minlength=num_nodes`:
> `bincount` output size is `max(minlength, input.max() + 1)`. `minlength` is only a
> *lower* bound — the compiler cannot prove `centers.max() + 1 <= num_nodes` (it depends
> on the runtime *values* in `centers`, not on shapes), so it conservatively treats the
> output shape as unbacked/data-dependent. Allocating the buffer ourselves
> (`torch.zeros(num_nodes)`) makes the shape *backed* (it is `num_nodes` by
> construction), and `scatter_add_` just writes into that fixed-size buffer. Same
> result, but the shape provenance is now visible to the compiler.

## The naive "fix" that silently miscompiles

Setting `torch._dynamo.config.capture_scalar_outputs = True` (what the graph-break
warning suggests) removes break (1), but on its own produces **wrong numbers**: the
unbacked `max_edges_per_node` and the unbacked `bincount` shape are mishandled. This is
a *silent* miscompile (no error), which is the dangerous part.

## What we did (the implemented fix — "Path A")

In `modules/structures.py :: compute_batch_tensors`:

- **(2)** replace `bincount` with a static-shape `scatter_add_` into a
  `num_nodes`-sized buffer.
- **(1)** keep `int(torch.max(num_neighbors))` but add
  `torch._check(max_edges_per_node >= 0)` (guarded by
  `if not torch.jit.is_scripting()`, since it is a compile-only hint and not
  TorchScript-able). This tells the compiler the symbolic NEF dimension is a valid
  (non-negative) size. (`torch._check(x >= 0)` is the forward-compatible replacement
  for the now-deprecated `torch._check_is_size`.)
- **(3)** the boolean-mask assignment was already rewritten to a static-shape
  `torch.where(mask, value, cumsum(~mask) - 1)` (bit-identical; verified over 2000
  random trials).

### Adaptive-cutoff path (the one production models use)

When `num_neighbors_adaptive` is set, `compute_batch_tensors` filters edges with
`keep = torch.nonzero(edge_distances <= pair_cutoffs)` + `index_select`. `nonzero` is a
data-dependent *output-shape* op, so the post-filter edge count is an **unbacked**
symbolic size, and it flows into `get_corresponding_edges` (`modules/nef.py`). That
function has an empty-edge early return whose `centers.numel() == 0` cannot be guarded
on an unbacked size, and its `amin/amax(dim=0)` reductions over the edge dimension
internally guard `Ne(edge_count, 0)`. Fixes in `get_corresponding_edges`:

- resolve the empty check with `guard_or_false(centers.numel() == 0)` (imported at
  module level with an identity fallback; only *called* inside
  `if not torch.jit.is_scripting():` so TorchScript dead-code-eliminates it). An
  unbacked size yields `False`, so the compiled path falls through to the non-empty
  branch. (`guard_or_false` is the forward-compatible replacement for the now-deprecated
  `guard_size_oblivious`.)
- right after, `torch._check(centers.shape[0] > 0)` so the `amin/amax` reductions know
  the edge dimension is non-empty.

**Limitation:** the *compiled* adaptive path assumes ≥ 1 edge. A 0-edge system (isolated
atom) under the compiled adaptive path would hit the reductions; eager / TorchScript keep
full empty-system support (the early return still fires there).

### Compiling

To compile `preprocess` (either cutoff mode), the caller must enable:

```python
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
core_pre = torch.compile(core.preprocess, fullgraph=True)
```

Verified: `fullgraph=True` parity with eager on multiple system sizes for **both** the
default and the adaptive-cutoff paths
(`tests/test_core.py::test_core_preprocess_fullgraph_compile[*]`), TorchScript export
still works (incl. an adaptive model), and the full PET suite passes (eager numerics
unchanged: `scatter_add` ≡ `bincount`).

### Caveats / fragility of Path A

- Relies on `capture_scalar_outputs` / `capture_dynamic_output_shape_ops` (global,
  non-default Dynamo flags) and on `torch._check` / `guard_or_false`. These are
  torch-version sensitive — re-run the compile test when bumping torch.
- The compiled adaptive path assumes ≥ 1 edge (see above).

## Alternatives considered (not implemented)

- **Static `max_edges` cap (Option B).** Pad the NEF grid to a fixed compile-time
  constant (e.g. derived from `num_neighbors_adaptive`, or an explicit `max_neighbors`
  hyper) instead of `int(max(...))`. Proven to compile `fullgraph=True` with **no**
  Dynamo flags — the most robust option — but it wastes memory on padding and needs an
  overflow policy (truncate-by-distance or error) for atoms exceeding the cap.
- **Flat scatter-based GNN (Option C).** Drop the padded `[n_nodes, max_edges, d]` NEF
  layout entirely; keep edges flat `[n_edges, d]` and use `segment`/`scatter` reductions
  (incl. a segment-softmax in `CartesianTransformer`). `n_edges` is a *backed* symint,
  so no host sync and no padding. The "correct" long-term fix, but a large rewrite of
  the transformer and a checkpoint break.

If Path A becomes flaky across torch versions, switch to Option B; reach for Option C
only if padding memory or the global Dynamo flags become a real constraint.
