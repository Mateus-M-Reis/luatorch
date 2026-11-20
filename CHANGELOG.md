# Changelog

All notable changes to luatorch are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- `DataLoader` cooperative Lua coroutine workers (`num_workers`), bounded
  `prefetch`, deterministic loader-local `seed`, explicit
  `iterator:close()`/`loader:close()`, and ordered worker-error propagation.
- `torch.fuse.linear` / `torch.fuse.conv2d` epilogue paths that fuse the bias
  and ReLU/GELU/sigmoid/tanh activation around ATen's existing
  `addmm`/`conv2d`, apply the activation in place (no activation temporary),
  preserve autograd, and fall back eagerly for unsupported shapes, dtypes, and
  callable activations.
- Exact `gelu` as a tensor op, a free function, `nn.functional.gelu`, and
  `nn.GELU`; `nn.Linear` and `nn.Conv2d` accept an `activation` option.

### Changed

- `DataLoader:iter()` now returns a callable iterator object; existing
  `for ... in loader:iter()` and `loader:iter()()` usage is unchanged.

### Documentation

- Document that workers are cooperative coroutines in the current Lua state,
  not OS threads or processes, and that they do not overlap with native ATen
  work.

### Tests

- Expanded the dependency-free suite to 313 tests, including prefetch ordering
  and bounds, exhausted epochs, worker errors, seeded deterministic shuffling
  across worker counts, and epilogue correctness/autograd/fallback coverage.

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
