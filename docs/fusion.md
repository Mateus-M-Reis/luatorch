# Fusion JIT

Eager tensor libraries launch one kernel per operator. A chain such as
`(a * b + c):relu()` allocates three temporaries and moves roughly 3× the data
it needs to. `torch.fuse` traces the Lua function, builds an expression tree,
and compiles it to a single loop that reads each input once and writes the
output once.

```lua
local fused = torch.fuse(function(a, b, c)
  return (a * b + c):relu()
end)

local out = fused(a, b, c)

-- avoid allocating a fresh output every call
local buffer = torch.empty(a:sizes(), {dtype = a:dtype()})
fused:into(buffer, a, b, c)

-- trainable variant: compiled forward and backward, wired into autograd
local train_fused = torch.fuse.train(function(a, b, c)
  return (a * b + c):relu()
end)
local y = train_fused(a, b, c)
y:sum():backward()
```

`torch.fuse(fn, options?)` returns a callable object. `options.backend` is
`"auto"` (default), `"c"` or `"lua"`. The default can be set with
`LUATORCH_FUSE_BACKEND`.

## Supported operations

The tracer currently understands pure elementwise expressions:

- binary: `+`, `-`, `*`, `/`, `^` (tensor or scalar operands)
- unary: unary `-`, `abs`, `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`,
  `clamp(min, max)`
- method spellings: `x:add(y)`, `x:mul(2)`, `x:relu()`, `x:sigmoid()`, etc.

Tensor inputs must:

- all be float32 or float64,
- share exactly the same shape,
- be contiguous (non-contiguous inputs are copied once).

For inference, inputs that require gradients make `torch.fuse` fall back to
eager. `torch.fuse.train` uses a different path that supports first-order
autograd.

Scalar arguments are runtime values; they are passed to the compiled kernel on
every call. Constants written inside the traced function are baked in.

Anything outside this subset — reductions, shape ops, broadcasting, multiple
return values, branches on tensor values — makes tracing fail and the call falls
back to eager ATen. Falling back is always correct; it just is not faster.

## Backends

### C backend (default when `gcc` is available)

The expression is emitted as a C function with a stable ABI:

```c
void luatorch_kernel(const void** ptrs, const double* scalars,
                     int64_t n, int threads);
```

It is compiled with:

```text
gcc -O3 -march=native -fPIC -shared -fopenmp -fno-math-errno -lm
```

and loaded through FFI. Kernels are cached by a hash of the generated source in
`.luatorch-cache/` (override with `LUATORCH_CACHE`). The first call for a new
expression pays the compile cost (typically 50–200 ms); later calls load the
shared object.

The loop is annotated with OpenMP:

```c
#pragma omp parallel for if(threads > 1 && n > 20000) num_threads(threads)
```

so large tensors use the same thread count as libtorch
(`torch.set_num_threads`). `relu`/`clamp` are emitted as ternaries so GCC
vectorises them; `fmaxf` is deliberately avoided because without `-ffast-math`
it turns into a libm call and is ~4× slower.

### LuaJIT backend

If no C compiler is available (`options.backend = "lua"`), the expression is
emitted as a LuaJIT loop over `float*`/`double*` pointers. It is portable and
still avoids temporaries, but scalar LuaJIT loops trail vectorised C by roughly
2x on large tensors.

Both backends share the same tracing and runtime wrapper; only code generation
and the kernel call differ.

## Output buffers and GC

`fused(a, b, c)` allocates a fresh output tensor. `fused:into(buffer, a, b, c)`
writes into a caller-supplied contiguous buffer and validates dtype/shape. This
matters because LuaJIT cannot know how many bytes live in C++ tensor storage.
`lua/torch/tensor.lua` tracks that external memory; when it crosses
`LUATORCH_GC_THRESHOLD_MB` (default 64 MB), a full collection runs so the C++
allocator can reuse storage. Without this, allocating one fresh 4 MB output per
call made out-of-place `add` ~3.5x slower than PyTorch.

`into` does not support autograd or non-contiguous outputs; use `fused(...)`
for those.

## Fused reductions

`torch.fuse.reduce(fn, "sum"|"mean"|"logsumexp")` emits the elementwise
expression and the reduction into one kernel:

```lua
local mse = torch.fuse.reduce(function(prediction, target)
  return (prediction - target) ^ 2
end, "mean")

local loss = mse(prediction, target)   -- 0-dim, no intermediate tensor
```

The generated C has an inlined expression helper and four independent
accumulators:

```c
static inline float luatorch_expr(const float* t1, ..., int64_t i) { ... }

a0 += luatorch_expr(..., i);
a1 += luatorch_expr(..., i + 1);
...
out[0] = total / n;               // mean
// logsumexp: max pass, then sum(exp(x - max)), then max + log(sum)
```

When any input requires grad, `torch.fuse.reduce` compiles a matching backward
kernel and runs it through a C++ reduction autograd Function. For `mean`, each
input gradient is `d(expression)/d(input) * grad_output / n`; for `sum` the
`/ n` is dropped. The 0-dim loss therefore connects to `nn.Linear`, optimizers,
and normal `loss:backward()`.

Unsupported expressions, missing derivative rules, or a missing C compiler fall
back to eager expression + `sum`/`mean`.

Measured on 1M floats:

| expression | fused | eager ATen | speedup |
| --- | ---: | ---: | ---: |
| `mean((a-b)^2)` forward | 0.45 ms | 1.74 ms | 3.87x |
| `mean(relu((a-b)^2 + c*0.5))` forward | 0.61 ms | 4.04 ms | 6.57x |
| MSE loss + backward | 4.36 ms | 11.55 ms | 2.65x |
| `logsumexp(a*b)` forward | 1.30 ms | 4.62 ms | 3.56x |
| `logsumexp(a*b)` + backward | 7.3 ms | 13.0 ms | 1.78x |

Reduction kernels are compiled with `-ffast-math` so GCC can vectorise the
four independent accumulators. That reassociates the sum: relative error is
about `1e-5` on 1M elements, which is fine for metrics and losses. Inference and
training kernels stay precise; only reductions opt into fast-math.

## Fused training

`torch.fuse.train(fn)` (also exposed as `torch.fuse_train`) compiles both
directions of an elementwise expression:

```lua
local fused = torch.fuse.train(function(a, b, c)
  return ((a * b + c):sigmoid() + a:abs()):tanh()
end)

local output = fused(a, b, c)   -- output has a grad_fn
output:backward(grad_output)    -- runs the compiled backward kernel
```

How it works:

1. The expression is traced exactly like inference.
2. It is differentiated symbolically with respect to every tensor argument.
   Supported rules: `+ - * / ^`, unary `-`, `exp`, `log`, `sqrt`, `abs`,
   `relu`, `sigmoid`, `tanh`, `clamp`.
3. Forward and backward are emitted into one C translation unit. The backward
   ABI is:

   ```c
   void luatorch_backward(const void** ptrs, const double* scalars,
                          int64_t n, int threads);
   // ptrs = [inputs..., grad_output, grad_inputs...]
   ```

4. A C++ `torch::autograd::Function` calls the forward kernel, saves the
   contiguous inputs and scalar arguments, and calls the backward kernel from
   `backward()`. Because it is a real autograd node, it connects to `nn`
   modules, `Linear`, losses, and the rest of the PyTorch graph.

The backward emitter is separate from the forward emitter and renders
expressions inline with a shared ReLU/clamp mask. That matters: a
temporary-per-node backward was ~6x slower because GCC refused to vectorise the
conditional; the current form runs at memory bandwidth.

Measured on 1M floats, `(a*b+c):relu()` forward + backward:

| path | time |
| --- | ---: |
| eager PyTorch | ~14.6 ms |
| fused training | ~11.6 ms (1.26x, best runs 1.5x) |

The win is smaller than inference fusion because the custom autograd node,
saved tensors, gradient allocations, and the autograd engine have fixed
overhead.

Limitations:

- first-order derivatives only; second-order gradients are not implemented;
- requires the C backend (falls back to eager if no compiler or no derivative
  rule);
- at most 8 tensor inputs;
- all tensor inputs must share dtype and shape;
- gradients are allocated for every tensor input, even ones that do not require
  grad (correct, but extra memory).

## Fallback rules

The eager function is called instead of the fused kernel when:

- an input requires grad and grad mode is enabled,
- tensor dtypes differ or are not float32/float64,
- shapes differ (broadcasting is left to ATen),
- an argument is not a tensor or number,
- tracing hits an unsupported operation, branch on tensor values, or multiple
  return values,
- C compilation fails (then the LuaJIT backend is tried; if that fails, eager).

`fused.last_error` records the last tracing error and `fused.last_c_error` the
last C compilation error, which helps when an expression unexpectedly falls back.

## Benchmarks

`bench/compare.py` runs both libraries with one thread and best-of-N timing:

```bash
./scripts/compare.sh --compile
```

Latest run on this machine (PyTorch 2.15.0+cpu, Ryzen 3 7320U):

| workload | luatorch | torch eager | speedup | torch.compile |
| --- | ---: | ---: | ---: | ---: |
| `add` out-of-place | 0.81 ms | 0.83 ms | 1.03x | — |
| `add_` in-place | 0.55 ms | 0.55 ms | 1.00x | — |
| `a*b+c` | 1.57 ms | 4.78 ms | 3.05x | — |
| `(a*b+c):relu()` eager | 2.17 ms | 5.53 ms | 2.55x | 1.05 ms |
| fused, into | 1.09 ms | 5.53 ms | 5.07x | 1.05 ms (0.96x) |
| fused, alloc | 1.03 ms | 5.53 ms | 5.36x | 1.05 ms (1.01x) |
| `sum` | 0.11 ms | 0.14 ms | 1.20x | — |
| `exp` | 3.02 ms | 3.02 ms | 1.00x | — |
| `matmul 512` | 7.66 ms | 7.67 ms | 1.00x | — |
| MLP Adam step | **0.224 ms** | 1.137 ms | **5.08x** | — |
| MLP train step | **3.97 ms** | 5.12 ms | **1.29x** | — |
| chain train step, fused | 11.6 ms | 14.6 ms | **1.26x** | — |
| `mean((a-b)^2)` fused | 0.45 ms | 1.74 ms | **3.87x** | — |
| `mean(relu((a-b)^2 + c*0.5))` fused | 0.61 ms | 4.04 ms | **6.57x** | — |
| MSE loss + backward fused | 4.36 ms | 11.55 ms | **2.65x** | — |
| `logsumexp(a*b)` fused | 1.30 ms | 4.62 ms | **3.56x** | — |
| `logsumexp` train fused | 7.3 ms | 13.0 ms | **1.78x** | — |

Caveats:

- This CPU throttles on sustained load; absolute numbers move by 10–20% between
  runs. The harness uses best-of-N and one thread.
- `torch.compile` also fuses elementwise chains. We are roughly at parity on
  this one kernel (about 1 ms for both); the honest claim is "competitive with
  compiled PyTorch, several times faster than eager on chains".
- Fused training lands at ~1.3x over eager PyTorch on a 1M-element chain; the
  compiled kernels are fast, the autograd bridge is the remaining overhead.
- Adam/AdamW/SGD use fused update kernels: the optimizer step is ~5x faster
  than PyTorch's, which lifts the full MLP training step to ~1.3x.
- Matrix multiplication uses the same class of BLAS kernels on both sides, so a
  tie is the expected result.

## Limitations and next steps

- Fused reductions cover `sum`/`mean`/`logsumexp` with first-order backward;
  softmax, layernorm, convolution, and matmul epilogues are not fused yet.
- Fused backward is first-order and C-backend only; second-order gradients and
  the LuaJIT-only fallback use eager autograd.
- Transcendental functions (`exp`, `tanh`) call scalar libm; vectorised libmvec
  or polynomial approximations would close the remaining gap on unary-heavy
  expressions.
- The tracer is explicit (`torch.fuse(function ... end)`), not transparent to
  arbitrary operator code. Lazy/traced tensors are future work.
- Each new expression costs one C compilation; long-running training loops
  amortise it, one-shot scripts may prefer the LuaJIT backend or eager ATen.
