# luatorch 0.10.0

LuaJIT tensor engine with an ATen backend and fusion compiler.

## Highlights

- SIMD-friendly polynomial `exp`/`tanh`/`sigmoid` helpers in the fused C
  backend;
- `#pragma omp parallel for simd` on the elementwise kernel;
- the 0.9 multi-layer/bidirectional RNN and LSTM stack remains included.

## Accuracy and scope

The fast transcendental helpers are compiled into fused kernels only; eager
ATen operations are unchanged. They are intended for float32/float64 inference
and training paths where the existing fusion accuracy tolerances apply. `log`
still uses the system libm implementation. Linear/conv epilogue fusion is not
part of this release.

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
luarocks --lua-version=5.1 make luatorch-0.10.0-1.rockspec
```

## Verification

- 303 dependency-free tests
- GCC and Clang CI
- LuaRocks install smoke test
