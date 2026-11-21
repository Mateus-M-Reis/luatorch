# luatorch API reference (0.16.0)

All examples assume:

```lua
local torch = require("torch")
```

## Conventions

- **Dimensions are 1-based.** `t:size(1)` is the first dimension, `t:sum(2)`
  reduces the second dimension, `t:select(-1, 1)` selects the first element of
  the last dimension. Dimension `0` is an error.
- **Indices are 1-based.** Positive indices start at 1, negative indices count
  from the end (`-1` is the last element). `get`, `set`, `select`, `narrow`,
  `slice`, `cat`, and `index_select` follow this rule.
- **`argmax`/`argmin` return 1-based indices**, unlike Python PyTorch.
- **Shapes** can be passed as varargs (`torch.zeros(2, 3)`) or as a table
  (`torch.zeros({2, 3})`). `-1` is accepted by `reshape`/`view` as an inferred
  dimension.
- **Options tables** use `{dtype = ..., device = "cpu", requires_grad = true}`.
  `dtype` accepts the constants below or their string names.
- **Mutating methods** end in `_` and return the same tensor object. Everything
  else is out-of-place.
- **Reductions with `dim = nil`** reduce over every dimension and return a
  **0-dim tensor**; use `:item()` to obtain a Lua number.
- **Errors** are raised as Lua errors (`pcall` catches them) with the original
  ATen message.

## Fusion

`torch.fuse(fn, options?)` traces a pure elementwise function and compiles it
into a single kernel:

```lua
local fused = torch.fuse(function(a, b, c)
  return (a * b + c):relu()
end, {backend = "auto"})        -- "auto" | "c" | "lua"

local out = fused(a, b, c)
local buffer = torch.empty(a:sizes(), {dtype = a:dtype()})
fused:into(buffer, a, b, c)     -- reusable output, no allocation
```

Supported: `+ - * / ^`, unary `-`, `abs`, `relu`, `sigmoid`, `tanh`, `exp`, `log`,
`sqrt`, `clamp`. Tensor inputs must be float32/float64 with the same shape;
scalar arguments are runtime values. Anything else (reductions, broadcasting,
autograd inputs, unsupported ops) falls back to eager ATen.

The default C backend requires `gcc`; it caches compiled kernels in
`.luatorch-cache/` (`LUATORCH_CACHE` overrides). The LuaJIT backend is always
available. `fused.last_error` / `fused.last_c_error` explain fallbacks. See
[docs/fusion.md](fusion.md) for internals and benchmarks.

`torch.fuse.train(fn)` (alias `torch.fuse_train`) is the trainable variant. It
compiles a backward kernel and wires both directions into PyTorch's autograd
engine:

```lua
local train_fused = torch.fuse.train(function(a, b, c)
  return (a * b + c):relu()
end)

local output = train_fused(a, b, c)
output:sum():backward()         -- compiled backward kernel
```

It supports first-order derivatives of the elementwise operations above and
requires the C backend. Unsupported expressions, missing derivative rules, or
the absence of `gcc` fall back to eager autograd. `train_fused.last_error`
explains the fallback.

`torch.fuse.reduce(fn, reduction)` (alias `torch.fuse_reduce`) fuses an
elementwise expression directly into a scalar reduction, avoiding the
intermediate tensor. Supported reductions are `"sum"`, `"mean"` and
`"logsumexp"`:

```lua
local mse = torch.fuse.reduce(function(prediction, target)
  return (prediction - target) ^ 2
end, "mean")

local loss = mse(prediction, target)   -- 0-dim tensor
```

When inputs require grad, a compiled backward kernel runs through a C++
autograd Function, so the 0-dim loss can be
used normally in training. Unsupported expressions or a missing C compiler fall
back to eager expression + reduction. `reduce.last_error` explains the
fallback.

## Dtypes

| Constant | String | Bytes |
| --- | --- | --- |
| `torch.float32` / `torch.float` | `"float32"` | 4 |
| `torch.float64` / `torch.double` | `"float64"` | 8 |
| `torch.float16` / `torch.half` | `"float16"` | 2 |
| `torch.int64` / `torch.long` | `"int64"` | 8 |
| `torch.int32` / `torch.int` | `"int32"` | 4 |
| `torch.int16` / `torch.short` | `"int16"` | 2 |
| `torch.int8` / `torch.char` | `"int8"` | 1 |
| `torch.uint8` / `torch.byte` | `"uint8"` | 1 |
| `torch.bool` / `torch.boolean` | `"bool"` | 1 |

Lua numbers have no int/float distinction, so numeric tables and scalars infer
**float32**. Integer tensors are explicit:

```lua
torch.tensor({1, 2, 3}, {dtype = torch.int64})
torch.LongTensor({1, 2, 3})
```

`float16` cannot be built directly from Lua tables; build a float32 tensor and
cast: `torch.tensor({1.5}):to(torch.float16)`.

## Creation

```lua
torch.tensor(value, opts?)          -- table/number/boolean/Tensor
torch.zeros(shape..., opts?)
torch.ones(shape..., opts?)
torch.empty(shape..., opts?)
torch.rand(shape..., opts?)
torch.randn(shape..., opts?)
torch.full(shape..., value, opts?)
torch.eye(n, opts?)
torch.arange(stop, opts?)           -- or (start, stop[, step], opts?)
torch.linspace(start, stop, steps, opts?)
torch.randperm(n, opts?)
```

`torch.tensor` infers `bool` for boolean tables and `float32` otherwise. Passing
an existing tensor returns a clone.

Like-constructors:

```lua
torch.zeros_like(t, opts?)
torch.ones_like(t, opts?)
torch.full_like(t, value, opts?)
torch.rand_like(t)
torch.randn_like(t)
```

### `torch.from_blob(buffer, shape[, opts])`

Creates a tensor sharing memory with a Lua string or LuaJIT FFI buffer. The
tensor keeps the buffer alive through `tensor._owner`. A Lua string is immutable,
so tensors backed by strings must be treated as read-only. FFI array buffers
are mutable and changes are visible in both directions.

```lua
local ffi = require("ffi")
local buffer = ffi.new("float[4]", { 1, 2, 3, 4 })
local t = torch.from_blob(buffer, { 2, 2 }, { dtype = "float32" })
t:set(1, 1, 9)       -- buffer[0] is now 9
```

`shape` is a table of dimensions. `opts.dtype` defaults to `float32`.
Array cdata sizes are checked with `ffi.sizeof`; pointer cdata must provide
`opts.nbytes`. `opts.requires_grad` is honored. `float16` is not supported.

### Torch7-style constructors

```lua
torch.Tensor(2, 3)                  -- uninitialised float32
torch.Tensor({1, 2, 3})             -- float32 values
torch.FloatTensor(2, 3)
torch.DoubleTensor({1.5, 2.5})
torch.LongTensor({1, 2, 3})
torch.IntTensor(...)   torch.ShortTensor(...)   torch.CharTensor(...)
torch.ByteTensor(...)  torch.BoolTensor(...)    torch.HalfTensor(...)
```

## Tensor metadata

| Method | Description |
| --- | --- |
| `t:dim()` / `t:ndimension()` | number of dimensions |
| `t:numel()` | number of elements |
| `t:size(dim?)` | one dimension, or a table of all sizes |
| `t:sizes()` / `t:shape()` | table of sizes |
| `t:dtype()` | dtype string |
| `t:dtype_id()` | numeric dtype id |
| `t:device()` | `"cpu"` (CUDA is not built yet) |
| `t:is_cuda()` | `false` |
| `t:is_contiguous()` | whether the memory layout is contiguous |
| `t:is_floating_point()` | true for float16/32/64 |
| `t:requires_grad()` | autograd flag |
| `t:element_size()` / `t:nbytes()` | element and total byte size |
| `t:type()` | Torch7 type name, e.g. `"torch.FloatTensor"` |

## Dtype conversion and copies

```lua
t:to(dtype_or_device)   -- new tensor; also accepts "cpu"
t:type_as(other)
t:float()  t:double()  t:long()  t:int()  t:half()  t:bool()
t:clone()  t:contiguous()  t:detach()
t:copy_(src)            -- in-place copy with broadcasting
```

## Scalar access and indexing

```lua
t:get(i, j, ...)        -- Lua number / boolean
t:set(i, j, ..., value)
t:item()                -- the single element of a 1-element tensor
t[index]                -- selects dimension 1; t[1] == t:select(1, 1)
t[index] = number       -- 1-d scalar assignment
t[index] = tensor       -- copy into the selected slice
t[mask]                 -- boolean mask indexing -> 1-d tensor
```

Shape-preserving selection:

```lua
t:select(dim, index)
t:narrow(dim, start, length)
t:slice(dim, start, stop, step?)   -- start/stop are inclusive, 1-based
t:flip(dim)
t:masked_select(mask)
t:index_select(dim, indices)       -- indices is an integer tensor, 1-based
t:masked_fill(mask, value)
t:gather(dim, index)               -- 1-based index tensor
t:index_select_batch(dim, {i1,i2}) -- one C call for several index tensors
t:scatter(dim, index, source)      -- out-of-place, 1-based index tensor
t:nonzero()                        -- 1-based coordinates, shape (N, ndim)
t:sort(dim?, descending?)          -- values, 1-based indices
t:topk(k, dim?, largest?, sorted?) -- values, 1-based indices
t:take_along_dim(indices, dim?)    -- gather along a dimension
t:unique(sorted?, inverse?, counts?) -- values[, inverse, counts]; inverse is 1-based
t:median(dim?)                    -- median value or dimension median
t:quantile(q, dim?, keepdim?)     -- interpolated quantile
```

Examples:

```lua
local m = torch.arange(1, 13):reshape(3, 4)  -- 3x4, values 1..12
m:select(1, 2)          -- tensor([5, 6, 7, 8])
m:slice(2, 2, 3)        -- 3x2, second and third columns
m:slice(1, -2, -1)      -- last two rows
m[m:gt(6)]              -- tensor([7, 8, 9, 10, 11, 12])
```

## Elementwise math

Two-tensor and scalar variants:

```lua
t:add(x)  t:sub(x)  t:mul(x)  t:div(x)  t:pow(x)  t:remainder(x)
t:add_(x) ...                       -- in-place
t:maximum(other)  t:minimum(other)  -- tensors only
```

Operators: `+ - * / ^` and unary `-`, including scalar-left forms
(`2 - t`, `1 / t`, `2 ^ t`). `cmul`/`cdiv` are Torch7 aliases for
`mul`/`div` (and `cmul_`/`cdiv_`).

Comparisons return Boolean tensors: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
(tensor or scalar argument). Logical: `logical_and`, `logical_or`,
`logical_not`.

Unary methods (each has an `_` in-place version):

`neg abs exp log log1p sqrt rsqrt sin cos tan asin acos atan sinh cosh tanh
sigmoid relu floor ceil round trunc sign square reciprocal`

Plus:

```lua
t:clamp(min, max)     t:clamp_(min, max)
t:fill_(value)        t:zero_()
```

## Linear algebra

```lua
a:matmul(b)     -- batched matmul with broadcasting
a:mm(b)         -- 2-D matrix multiplication
a:bmm(b)        -- batched 3-D matmul
a:dot(b)        -- 1-D dot product
a:mv(v)         -- matrix-vector product
a:outer(v)      -- outer product
a:addmm(m1, m2, beta?, alpha?)    -- beta*a + alpha*(m1 @ m2)
a:baddbmm(m1, m2, beta?, alpha?)  -- batched addmm
```

### Linear algebra

```lua
a:inverse()
a:det()
a:solve(b)          -- solve a x = b (b may be a vector or matrix)
a:cholesky(upper?)  -- A = L L^T (or U^T U)
a:svd(full_matrices?)  -- returns U, S, Vh
a:qr(mode?)            -- returns Q, R; mode defaults to "reduced"

torch.linalg.inv/det/solve/cholesky/svd/qr   -- aliases
```

## Neural-network primitives

These back `torch.nn`; they are also useful when writing custom layers.

```lua
t:softmax(dim?)        t:log_softmax(dim?)     -- any dim; fused via layout move
t:uniform_(low?, high?)                       -- random in-place init
t:normal_(mean?, std?)      t:exponential_(lambda?)
t:bernoulli_(p?)            t:multinomial(n, replacement?)  -- 1-based samples

torch.softmax(t, dim)  torch.log_softmax(t, dim)
torch.conv2d(input, weight, bias, stride?, padding?, dilation?, groups?)
torch.max_pool2d(input, kernel, stride?, padding?, dilation?, ceil_mode?)
torch.avg_pool2d(input, kernel, stride?, padding?, ceil_mode?, count_include_pad?)
torch.embedding(weight, indices)                -- 1-based indices
torch.layer_norm(input, normalized_shape, weight?, bias?, eps?)
torch.cross_entropy(input, target, reduction?)  -- targets are 1-based
torch.cross_entropy(input, target, { weight = w, ignore_index = i,
  label_smoothing = s })                         -- Lua option path
torch.nll_loss(log_probs, target, reduction?)
torch.binary_cross_entropy_with_logits(input, target, reduction?)
torch.uniform_(t, low, high)
```

`reduction` is `"none"`, `"mean"` (default) or `"sum"`.

## Data loading

`DataLoader` options, iterator lifecycle, worker semantics, and limits are
documented in [docs/data.md](data.md).

```lua
local data = require("torch.utils.data")   -- or torch.utils.data

local dataset = data.TensorDataset(inputs, labels)  -- 1-based get()
local loader = data.DataLoader(dataset, {
  batch_size = 64,
  shuffle = true,
  drop_last = false,
  num_workers = 2,      -- cooperative coroutine workers (default 0)
  prefetch = 4,         -- batches prepared ahead (default 2*num_workers, or 0)
  seed = 42,            -- optional deterministic shuffle
})
for batch_inputs, batch_labels in loader:iter() do
  -- ...
end
```

Custom datasets implement `size()` and `get(index)`; an optional `batch(indices)`
can return a whole batch at once. The dataset contract is unchanged by workers
and prefetch.

`loader:iter()` returns a callable iterator. It is exhausted with `nil`, closes
its workers at the end of the epoch, and can be closed early with
`iterator:close()` (or `loader:close()` for every live iterator). Iterators that
are abandoned without closing are finalized by garbage collection.

### Worker and prefetch model

`num_workers` creates Lua coroutine workers in the current Lua state. These are
**cooperative, not OS threads or processes**: only one Lua coroutine runs at a
time, and a worker is resumed only while the loader is preparing batches. A
worker yields one complete batch per resume, so dataset access remains
single-threaded and needs no locking. This avoids thread/process start-up,
copied Lua states, and IPC, but it also means workers cannot run in parallel
with each other or with training math. In particular, a long-running ATen call
does not yield to Lua, so it does not overlap with prefetching work.

`prefetch` bounds how many batches may be prepared ahead of the batch currently
being consumed. The queue holds at most `prefetch + 1` batches; `0` computes each
batch only when requested. Workers and prefetch change neither the shuffle nor
the batch order: batches are always delivered in the same deterministic order
for a given seed and dataset size.

Without `seed`, shuffling uses the global `torch` RNG (`torch.manual_seed`).
With `seed`, the loader uses a private deterministic PRNG, does not consume the
global RNG, and produces the same order each epoch. Worker failures are re-raised
on the batch that failed, with worker context, and close the iterator. See
[docs/nn.md](nn.md#data-loading) for datasets, modules, losses and optimizers.

## Shape manipulation

```lua
t:reshape(2, 3)         t:reshape({2, 3})      t:reshape(-1)
t:view(2, 3)
t:transpose(dim1, dim2)
t:t()
t:permute(dim1, dim2, ...)
t:squeeze(dim?)         -- no dim: remove all size-1 dims
t:unsqueeze(dim)
t:flatten(start_dim?, end_dim?)
t:expand(sizes...)
torch.cat({t1, t2}, dim?)      -- default dim = 1
torch.stack({t1, t2}, dim?)    -- default dim = 1
```

`expand` follows PyTorch broadcasting rules (including `-1` to keep a size).

## Reductions

```lua
t:sum(dim?, keepdim?)
t:mean(dim?, keepdim?)
t:prod(dim?, keepdim?)
t:max(dim?, keepdim?)          -- values[, indices] when dim is given
t:min(dim?, keepdim?)          -- values[, indices] when dim is given
t:argmax(dim?, keepdim?)       -- 1-based indices
t:argmin(dim?, keepdim?)
t:var(dim?, unbiased?, keepdim?)
t:std(dim?, unbiased?, keepdim?)
t:norm(p?, dim?, keepdim?)     -- p defaults to 2; math.huge supported
t:logsumexp(dim?, keepdim?)
t:cumsum(dim?)                 -- dim defaults to 1
t:cumprod(dim?)
t:all(dim?, keepdim?)
t:any(dim?, keepdim?)
```

`max`/`min` return two values when `dim` is provided:

```lua
local values, indices = t:max(2)
```

## Autograd

```lua
local x = torch.tensor({1, 2, 3}, {requires_grad = true})
local loss = (x * x):sum()
loss:backward()
print(x:grad())                 -- tensor([2., 4., 6.])

x:zero_grad()
x:requires_grad_(true)
x:has_grad()
x:grad()                        -- nil before the first backward
x:detach()
x:backward(gradient?)           -- explicit upstream gradient
```

Global grad mode:

```lua
torch.is_grad_enabled()
torch.set_grad_enabled(false)
local y = torch.no_grad(function()
  return (x * 2):sum()          -- does not build a graph
end)
```

## Functional utilities and luafun

```lua
t:iter()                  -- iterator over elements (contiguous order)
t:map(fn)                 -- new tensor, same shape and dtype
t:filter(pred)            -- 1-d tensor of matching values
t:reduce(fn, initial?)    -- fold
t:apply(fn)               -- in-place map
t:fun()                   -- fun.iter(t:totable())
t:totable()               -- nested Lua table (or a scalar for 0-dim)
```

`torch.fun` (alias `torch.luafun`) is the vendored
[luafun](https://github.com/rtsisyk/luafun) module:

```lua
local fun = torch.fun
local total = fun.iter({1, 2, 3, 4}):reduce(function(a, b) return a + b end, 0)
local evens = fun.totable(fun.iter({1, 2, 3, 4}):filter(function(x) return x % 2 == 0 end))
```

## Comparison helpers

```lua
a:equal(b)              -- same shape and exactly equal values
a:allclose(b, rtol?, atol?)
torch.equal(a, b)
torch.allclose(a, b, rtol?, atol?)
```

## Serialization

```lua
t:save("tensor.pt")
torch.save(t, "tensor.pt")
local copy = torch.load("tensor.pt")
torch.print(t)          -- prints t:format()
```

Files use the PyTorch zip archive format (`torch::save` in C++), so tensors can
be exchanged with C++/Python tooling that understands that format.

Module state dictionaries can be exchanged as safetensors:

```lua
model:save_safetensors("model.safetensors")
model:load_safetensors("model.safetensors")
model:load_safetensors("model.safetensors", { strip_prefix = "model." })
```


## Module-level aliases

`torch.add`, `sub`, `mul`, `div`, `pow`, `matmul`, `mm`, `bmm`, `dot`, `sum`,
`mean`, `prod`, `max`, `min`, `argmax`, `argmin`, `transpose`, `reshape`,
`clamp`, `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`, `abs` are thin
wrappers around the corresponding tensor methods.

Other module helpers:

```lua
torch.is_tensor(x)
torch.manual_seed(seed)
torch.set_default_dtype(dtype)
torch.get_default_dtype()
torch.dtype_name(dtype)
torch.get_num_threads() / torch.set_num_threads(n)
torch.sort / topk / unique / median / quantile / bincount / histc / nonzero / gather / scatter / multinomial
torch.ger(a, b)                  -- outer product of two vectors
torch.linalg.matrix_norm(a, ord) -- fro, 2, -2, 1, -1, inf, -inf
torch.linalg.cond(a, p?)         -- 2, 1, inf, fro
torch.linalg.matrix_rank(a, tol?)
torch.linalg.vector_norm(a, ord?, dim?, keepdim?)
torch.mv / outer / addmm / baddbmm
torch.ALL_DIMS            -- -1000 sentinel (internal, exposed for clarity)
torch.profiler            -- opt-in tensor-method profiler (see docs/profiler.md)
torch.VERSION             -- luatorch version string ("0.16.0")
torch.version             -- libtorch version string
```
