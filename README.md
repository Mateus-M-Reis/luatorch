# luatorch

**LuaJIT bindings for PyTorch's ATen — Torch7 ergonomics on a modern tensor engine.**

Torch7 was a wonderful way to do numerical computing in Lua, but it stopped in
2018. PyTorch kept going, and its C++ core (`libtorch` / ATen) is now a fast,
well-tested tensor library. `luatorch` reconnects the two:

- native `at::Tensor` objects behind a small stable C ABI, loaded through LuaJIT FFI
- Torch7-style Lua ergonomics (1-based dimensions and indices, `narrow`, `select`,
  `:t()`, `torch.Tensor{...}`, `:add_` in-place methods)
- PyTorch semantics for gradients, dtypes, broadcasting, and serialization
- **luafun** (`fun.lua`) as the functional core — no Moses dependency
- autograd, dtype casts, Boolean masking, and PyTorch-compatible `.pt` files

> Status: **0.1.0 alpha**, CPU build. Core tensors, math, autograd, I/O, and
> the luafun bridge are implemented and covered by **120 tests**. `nn`, `optim`,
> CUDA, and the long tail of ATen operators are next (see [ROADMAP.md](ROADMAP.md)).

---

## Quick start

```lua
local torch = require("torch")

local x = torch.randn(2, 3)
print(x)
-- tensor([[-0.2273,  0.5088, -1.1477],
--         [ 0.1655,  3.0343, -0.1068]])

local w = torch.tensor({1, 2, 3}, {requires_grad = true})
local loss = (w * w):sum()
loss:backward()
print(w:grad())
-- tensor([2., 4., 6.])

local mask = w:gt(1)
print(w[mask])              -- tensor([2., 3.])
print((w + 1):sqrt():mean())
```

Functional pipelines use **luafun** directly, or through the tensor bridge:

```lua
local fun = require("fun")          -- == torch.fun

local squares = fun.totable(fun.iter({1, 2, 3, 4})
  :map(function(x) return x * x end)
  :filter(function(x) return x % 2 == 0 end))
-- {4, 16}

local data = torch.randn(1000000)
local total = 0
for value in data:iter() do total = total + value end

local doubled = torch.randn(4, 4):map(function(x) return 2 * x end)
local evens   = torch.arange(10):filter(function(x) return x % 2 == 0 end)
```

## Build

Requirements: Linux/macOS, CMake ≥ 3.18, Ninja or Make, a C++17 compiler,
LuaJIT 2.1 (with headers), `curl`, `unzip`. The native shim links against the
official prebuilt **libtorch** (CPU, C++11 ABI, 2.5.1).

```bash
scripts/fetch_libtorch.sh     # downloads ~170 MB into third_party/libtorch
scripts/build.sh              # builds build/libluatorch.so
scripts/test.sh               # 120 tests
scripts/bench.sh              # micro-benchmarks
scripts/repl.sh               # interactive REPL with torch preloaded
```

If you already have libtorch installed, skip the download:

```bash
Torch_DIR=/path/to/libtorch/share/cmake/Torch scripts/build.sh
```

Install the Lua sources and the shim side by side:

```bash
cmake --install build --prefix "$HOME/.local"
# then LUA_PATH="$HOME/.local/share/lua/5.1/?.lua;$HOME/.local/share/lua/5.1/?/init.lua;;"
```

## What is implemented

| Area | Covered |
| --- | --- |
| Creation | `tensor`, `zeros`, `ones`, `empty`, `full`, `rand`, `randn`, `randperm`, `eye`, `arange`, `linspace`, `*_like` |
| Dtypes | float32/float64/float16, int8/16/32/64, uint8, bool, `.to()`, Torch7 typed constructors |
| Math | elementwise `+ - * / ^ %`, `maximum/minimum`, broadcasting, ~25 unary ops, `matmul/mm/bmm/dot`, comparisons, `where`, `masked_fill`, `masked_select` |
| Shape | `reshape`, `view`, `transpose`, `t`, `permute`, `squeeze`, `unsqueeze`, `flatten`, `expand`, `cat`, `stack`, `slice`, `narrow`, `select`, `flip`, `index_select` |
| Reductions | `sum`, `mean`, `prod`, `max/min` (+ indices), `argmax/argmin`, `var/std`, `norm`, `cumsum/cumprod`, `all/any`, dims and `keepdim` |
| Autograd | `requires_grad`, `backward`, `grad`, `detach`, `zero_grad`, `no_grad`, `set_grad_enabled` |
| I/O | `torch.save`/`torch.load` using the PyTorch zip archive format |
| Functional | `Tensor:iter/map/filter/reduce/apply`, `Tensor:fun()`, and the vendored luafun |
| Print | PyTorch-style `tensor([...])`, dtype and `requires_grad` annotations |

## Deliberate design choices

- **1-based Lua surface.** Dimensions and indices are 1-based (`t:select(2, -1)`),
  matching Torch7. Reductions such as `argmax` also return 1-based indices.
  Native ATen stays 0-based behind the boundary.
- **Lua numbers are float32 tensors by default.** Lua cannot distinguish `1` from
  `1.0`, so table/scalar inference chooses float32 (the PyTorch default). Integer
  tensors are explicit: `torch.tensor({1, 2}, {dtype = torch.int64})` or
  `torch.LongTensor{1, 2}`.
- **Out-of-place by default.** `t:add(x)` returns a new tensor; `t:add_(x)` mutates
  and returns `t`. This mirrors PyTorch and keeps autograd tractable.
- **No Moses.** luafun is lazy, LuaJIT-friendly, MIT-licensed, and vendored as a
  single file (`lua/fun.lua`). See [docs/torch7-compat.md](docs/torch7-compat.md)
  for the Moses → luafun mapping.
- **Small C ABI.** Lua never touches C++ types directly. Every operation is a
  plain C function returning an opaque pointer; errors travel through a
  thread-local message. This keeps FFI fast and crash-resistant.

## Documentation

- [docs/api.md](docs/api.md) — the full Lua API
- [docs/design.md](docs/design.md) — architecture, memory, and error handling
- [docs/torch7-compat.md](docs/torch7-compat.md) — Torch7 ⇄ luatorch mapping, Moses → luafun
- [ROADMAP.md](ROADMAP.md) — where this is going

## Performance snapshot

Measured on an 8-core x86-64 CPU with the bundled CPU libtorch (median of
repeats, `scripts/bench.sh`):

| Workload | Result |
| --- | --- |
| add, 1M float32 tensors | ~4.3 ms (≈2.8 GB/s effective, single stream) |
| sum, 1M float32 | ~0.06 ms |
| matmul 512×512 float32 | ~11.9 ms (≈22.6 GFLOP/s) |
| matmul 128×128 float32 | ~0.06 ms |
| small tensor add (16 elements) | ~3 µs per call round-trip |
| luafun map+reduce over 200k | ~2.0 ms (plain Lua loop ~0.26 ms) |
| `torch.load` 100k float32 | ~0.2 ms |

The dispatch overhead is a few microseconds; real workloads are dominated by
ATen kernels. luafun is lazy and allocation-light, but a hand-written loop over
a flat Lua table is still the fastest option for element-by-element work — use
tensor ops when you can.

## Credits

- [PyTorch](https://pytorch.org) / ATen — BSD-style license.
- [LuaJIT](https://luajit.org) — MIT.
- [luafun](https://github.com/rtsisyk/luafun) by Roman Tsisyk — MIT.
- [Torch7](https://github.com/torch/torch7) — the ergonomics this project borrows
  and the community that made it fun.

See [LICENSE](LICENSE).
