# luatorch 0.11.0

LuaJIT tensor engine with an ATen backend and fusion compiler.

## Highlights

- cooperative `DataLoader` workers (`num_workers`) and bounded `prefetch`;
- deterministic loader-local shuffling with `seed`, independent of the global
  `torch` RNG;
- callable iterators with explicit `iterator:close()` and `loader:close()`;
- ordered error propagation, including `num_workers = 0` look-ahead;
- runnable `examples/dataloader.lua` and a full `docs/data.md` reference.

## DataLoader semantics

Workers are Lua coroutines in the current Lua state, not OS threads or
processes. They do not overlap with each other or with native ATen work, so
`num_workers` is about API compatibility and isolating batch preparation rather
than parallel speedup. At most `prefetch + 1` batches are held. An empty dataset
produces zero batches even with `shuffle = true`. A seeded loader repeats the
same order every epoch; without a seed, shuffling consumes the global RNG.

Errors are raised when the failing batch position is reached, after earlier
prepared batches, and close the iterator. `close()` is idempotent. See
[docs/data.md](docs/data.md) for the full contract and limits.

## Scope

CUDA and linear/conv epilogue fusion are not part of this release. The 0.10
SIMD-friendly polynomial `exp`/`tanh`/`sigmoid` helpers and the 0.9
multi-layer/bidirectional RNN and LSTM support remain included.

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
luarocks --lua-version=5.1 make luatorch-0.11.0-1.rockspec
```

## Verification

- 311 dependency-free tests (`scripts/test.sh`)
- `examples/dataloader.lua` runs standalone
- LuaRocks spec prepared; no tag, package, or release was published from this
  change
