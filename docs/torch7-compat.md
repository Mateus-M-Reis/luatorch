# Torch7 compatibility and the Moses → luafun switch

`luatorch` is not a drop-in replacement for Torch7. It keeps the ergonomics that
made Torch7 pleasant, but the engine and several semantics are PyTorch's. This
page lists what maps cleanly, what differs, and how to translate the functional
utilities Torch7 code often imported from Moses.

## Naming and semantics

| Torch7 | luatorch | Notes |
| --- | --- | --- |
| `torch.Tensor(2, 3)` | same | float32, uninitialised |
| `torch.Tensor{1, 2}` | same | float32 values |
| `torch.FloatTensor` / `DoubleTensor` / `LongTensor` / `ByteTensor` | same | plus Int/Short/Char/Bool/Half |
| `t:size(1)` | same | 1-based |
| `t:narrow(1, 2, 3)` | same | 1-based start |
| `t:select(1, -1)` | same | negative indices supported |
| `t:sub(1, 3, 2)` | `t:slice(1, 1, 3, 2)` | select a sub-range |
| `t:add(x)` | same | now **out-of-place** |
| `t:add(x)` mutating | `t:add_(x)` | in-place methods have `_` |
| `t:cmul(x)` | `t:cmul(x)` or `t:mul(x)` | alias provided |
| `t:cdiv(x)` | `t:cdiv(x)` or `t:div(x)` | alias provided |
| `t:t()` | same | |
| `t:mm(x)` | same | |
| `t:mv(x)` | *missing* | use `x:matmul(v)` for 1-D/2-D combinations |
| `t:sum(1)` / `t:sum()` | same | `sum()` returns a 0-dim tensor |
| `t:max(1)` | same | returns values plus 1-based indices |
| `t:mean()` | same | integer tensors are rejected, like PyTorch |
| `torch.save(path, t)` | `torch.save(t, path)` | argument order is PyTorch's |
| `torch.load(path)` | same | PyTorch zip format |
| `torch.type(t)` | `t:type()` | |
| `torch.typename(t)` | `t:type()` | |

### Things that intentionally differ

1. **Out-of-place is the default.** `t:add(x)` returns a new tensor. Torch7 code
   that relied on mutation must add underscores: `t:add_(x)`.
2. **Default dtype is float32, not double.** Numeric tables cannot distinguish
   `1` from `1.0` in Lua, so luatorch follows PyTorch's floating default.
   Use `torch.set_default_dtype(torch.float64)` (affects size-based
   constructors) or an explicit dtype for double precision.
3. **Reductions return 0-dim tensors**, not Lua numbers:
   `t:sum():item()` instead of `t:sum()`.
4. **`argmax`/`argmin` are 1-based** (like Torch7's max indices, unlike PyTorch).
5. **No `torch.setdefaulttensortype`**, no class system, no `nn`/`optim`/`image`
   in core. Those are roadmap items and will live in separate modules.
6. **CUDA is not built yet.** `device = "cuda"` raises a clear error.

## Moses → luafun

Torch7-era code often used [Moses](https://github.com/Yonaba/Moses) for
functional helpers. luatorch vendors [luafun](https://github.com/rtsisyk/luafun)
instead: it is a single MIT-licensed file, works well with LuaJIT, evaluates
lazily, and avoids intermediate tables.

| Moses | luafun / luatorch |
| --- | --- |
| `moses.map(t, f)` | `fun.totable(fun.iter(t):map(f))` or `tensor:map(f)` |
| `moses.filter(t, p)` | `fun.totable(fun.iter(t):filter(p))` or `tensor:filter(p)` |
| `moses.reduce(t, f, init)` | `fun.iter(t):reduce(f, init)` or `tensor:reduce(f, init)` |
| `moses.each(t, f)` | `for v in tensor:iter() do f(v) end` |
| `moses.range(a, b)` | `fun.range(a, b)` (end-exclusive) |
| `moses.zip(a, b)` | `fun.zip(fun.iter(a), fun.iter(b))` |
| `moses.chain(...)` | luafun method chains |
| `moses.take(t, n)` | `fun.totable(fun.iter(t):take_n(n))` |
| `moses.flatten(t)` | tensor: `t:flatten()`; tables: `fun.totable(fun.flatten(t))` |
| `moses.include(t, v)` | `fun.iter(t):any(function(x) return x == v end)` |
| `moses.count(t, v)` | `fun.iter(t):filter(function(x) return x == v end):length()` |
| `moses.sum(t)` | `fun.iter(t):reduce(function(a, b) return a + b end, 0)` |

### Why luafun

- **Lazy composition**: `fun.iter(x):map(f):filter(p)` walks the source once.
- **No table allocations per stage**, unlike Moses' eager array results.
- **MIT licensed** and a single file, so it ships inside `lua/fun.lua`.
- **Tensors expose the same model**: `t:iter()` is a Lua iterator, `t:map` /
  `t:filter` / `t:reduce` are thin tensor-aware conveniences, and `t:fun()`
  promotes a tensor to a luafun iterator.

Example migration:

```lua
-- Moses style
local moses = require "moses"
local data = {1, 2, 3, 4, 5, 6}
local evens_squared = moses.map(
  moses.filter(data, function(x) return x % 2 == 0 end),
  function(x) return x * x end)

-- luafun style
local fun = require "fun"
local evens_squared = fun.totable(
  fun.iter(data)
    :filter(function(x) return x % 2 == 0 end)
    :map(function(x) return x * x end))
-- {4, 16, 36}
```

```lua
-- Tensor style: stays in tensor-land
local t = torch.arange(1, 7)
local evens_squared = t:filter(function(x) return x % 2 == 0 end)
  :map(function(x) return x * x end)
-- tensor([ 4., 16., 36.])
```

Moses is not a dependency of luatorch; nothing in the runtime, tests, or
benchmarks requires it.
