# luatorch 0.9.0

LuaJIT tensor engine with an ATen backend and fusion compiler.

## Highlights

- multi-layer RNN and LSTM stacks;
- bidirectional forward/reverse passes;
- inter-layer dropout, disabled in eval mode;
- PyTorch-compatible parameter names (`weight_ih_l1_reverse`, ...);
- explicit per-layer hidden/cell states.

The 0.8 feature set remains:

- optimizer parameter groups, weighted/ignored/smoothed losses;
- safetensors state-dict interchange, `.npz`-adjacent I/O through numlua;
- batched index-select, arbitrary-dimension fused softmax;
- LuaRocks packaging and `make install`.

## Requirements

- LuaJIT 2.1
- CMake ≥ 3.18 and C++17
- libtorch 2.5.x CPU
- LuaRocks for the optional package install

## Install

```sh
scripts/fetch_libtorch.sh
scripts/build.sh
make install
# or
luarocks --lua-version=5.1 make luatorch-0.9.0-1.rockspec
```

## Verification

- 303 dependency-free tests
- GCC and Clang CI
- LuaRocks install smoke test
