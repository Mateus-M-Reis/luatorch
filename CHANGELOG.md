# Changelog

All notable changes to luatorch are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

_No unreleased changes yet._

## [0.17.0] — 2026-11-21

### Added

- `torch.einsum(equation, ...)` backed by `at::einsum`, with a Lua-side
  validation pass for operand counts, subscripts, ellipses, and output labels;
- `torch.isnan(t)` / `torch.isinf(t)` and `Tensor:isnan()` / `Tensor:isinf()`,
  returning bool tensors (float16/32/64 are checked; integer and bool inputs
  are always false);
- direct `float16` construction from Lua tables and scalars via
  `torch.tensor(..., {dtype = torch.float16})` and `torch.HalfTensor{...}`;
- tests for einsum equations, errors, and autograd; float16 creation,
  rounding, and printing; and the NaN/Inf predicates.

### Changed

- `float16` tensors are now staged as float32 and converted by ATen, which
  applies IEEE round-to-nearest-even and overflow to `inf`; the previous
  "cannot be created from Lua tables" error is gone.

## [0.16.0] — 2026-11-21

### Added

- `torch.linalg.cond` for 2, 1, inf, and Frobenius condition numbers;
- `torch.linalg.matrix_rank` with an optional singular-value tolerance;
- tests for identity, singular matrices, and rank thresholds.

## [0.15.0] — 2026-11-21

### Added

- `matrix_norm` orders `2` (largest singular value) and `-2` (smallest)
  through the existing SVD path;
- tests for spectral norms and unsupported orders.

## [0.14.0] — 2026-11-21

### Added

- `Tensor:take_along_dim(indices, dim)` with 1-based dims/indices;
- `torch.linalg.vector_norm(a[, ord[, dim[, keepdim]]])`;
- tests for gather-aligned selection and vector norm orders.

## [0.13.0] — 2026-11-21

### Added

- `torch.ger` outer-product alias with 1-D validation;
- `Tensor:quantile(q[, dim[, keepdim]])` with linear interpolation;
- `torch.linalg.matrix_norm` for Frobenius, 1, -1, +inf, and -inf norms;
- tests for quantile, outer products, and matrix norm orders.

## [0.12.0] — 2026-11-21

### Added

- `Tensor:median([dim])` for flattened and dimension medians;
- `torch.bincount(input[, weights[, minlength]])`;
- `torch.histc(input[, bins[, min[, max]]])`;
- tests for weighted counts, histogram edges, median shapes, and empty tensors.

### Fixed

- `Tensor:totable()` on an empty tensor now returns an empty table instead of
  raising on a null data pointer.

## [0.11.0] — 2026-11-21

### Added

- `DataLoader` cooperative Lua coroutine workers (`num_workers`), bounded
  `prefetch`, deterministic loader-local `seed`, explicit
  `iterator:close()`/`loader:close()`, and ordered worker-error propagation;
- runnable `examples/dataloader.lua` covering batching, seeded shuffling,
  prefetch, early close, `drop_last`, and empty datasets;
- `docs/data.md` reference for the dataset protocol, loader options, iterator
  lifecycle, worker model, and limits.

### Changed

- `DataLoader:iter()` now returns a callable iterator object; existing
  `for ... in loader:iter()` and `loader:iter()()` usage is unchanged.

### Fixed

- `shuffle = true` on an empty dataset no longer crashes while building an
  empty permutation;
- with `num_workers = 0` and `prefetch > 0`, a look-ahead error now waits for
  earlier prepared batches and is raised at its own batch position, matching
  worker behavior.

### Documentation

- Document that workers are cooperative coroutines in the current Lua state,
  not OS threads or processes, and that they do not overlap with native ATen
  work.

### Tests

- 311 dependency-free tests, including `num_workers = 0` preparation, empty
  datasets across option combinations, exhausted and closed iterators, repeated
  close, and worker/collate error positions.

## [0.10.0] — 2026-11-19

### Added

- SIMD-friendly polynomial `exp`/`tanh`/`sigmoid` helpers in the fused C
  backend; float32 users trade a small accuracy margin for vectorizable math;
- `#pragma omp parallel for simd` on the elementwise kernel.

### Tests

- 303 tests.

## [0.9.0] — 2026-11-19

### Added

- multi-layer RNN/LSTM stacks with PyTorch-compatible `_lN` parameter names;
- bidirectional RNN/LSTM traversal and concatenated outputs;
- inter-layer dropout, disabled in eval mode;
- explicit hidden/cell state tensors per layer and direction;
- 303 total tests.

## [0.8.0] — 2026-11-17

### Added

- PyTorch-style optimizer parameter groups, per-group hyperparameters, and
  `Optimizer:add_param_group`;
- learning-rate schedulers that update every parameter group;
- class weights, `ignore_index`, and label smoothing for
  `CrossEntropyLoss`/`NLLLoss`;
- single-layer `nn.RNN` and `nn.LSTM` with sequence-first and batch-first
  layouts, optional hidden/cell states, and PyTorch parameter names;
- safetensors state-dict interchange with `Module:save_safetensors` and
  `Module:load_safetensors`, including prefix stripping;
- batched `Tensor:index_select_batch` backed by one native C call;
- fused softmax/log-softmax over arbitrary dimensions via a contiguous layout
  move;
- `luatorch-0.8.0-1.rockspec`, `make install`, and `scripts/install.sh`.

### Changed

- `optimizer.lr` mirrors the first parameter group after each step;
- `nn.Softmax`/`nn.LogSoftmax` documentation now covers arbitrary dimensions.

### Tests

- 295 dependency-free tests.

## [0.7.0] — 2026-11-16

- per-tensor-method profiler and release polish;
- direct `from_blob` and fixed Clang build.
