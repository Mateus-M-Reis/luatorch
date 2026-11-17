# Changelog

All notable changes to luatorch are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

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
