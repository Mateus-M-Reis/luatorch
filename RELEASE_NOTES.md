# luatorch 0.8.0

LuaJIT tensor engine with an ATen backend and fusion compiler.

## Highlights

- PyTorch-style optimizer parameter groups with per-group hyperparameters;
- weighted/ignored/smoothed classification losses;
- single-layer RNN and LSTM with PyTorch-compatible parameter names;
- safetensors interchange for PyTorch state dictionaries;
- batched index-select in one native call;
- fused softmax/log-softmax over arbitrary dimensions;
- LuaRocks rockspec and `make install`.

## Requirements

- LuaJIT 2.1 with headers
- CMake ≥ 3.18 and a C++17 compiler
- libtorch 2.5.x CPU (or a compatible installation)
- Linux/macOS; a C compiler for the fusion backend

## Install

```sh
scripts/fetch_libtorch.sh
scripts/build.sh
make install                 # or: luarocks make luatorch-0.8.0-1.rockspec
```

The LuaRocks package contains the Lua surface; build the native shim with
`scripts/build.sh` and point `LUATORCH_LIB` at it (or install it next to
`torch/lib.lua`).

## Verification

- 295 dependency-free tests
- GCC and Clang CI
- LuaRocks install smoke test
