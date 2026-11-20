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

Linear and convolution epilogues are separate, concrete entry points
(`torch.fuse.linear` / `torch.fuse.conv2d`) described below; they do not go
through the elementwise tracer.

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

## Fused softmax and log-softmax

`Tensor:softmax(dim)`/`nn.Softmax` and `Tensor:log_softmax(dim)`/`nn.LogSoftmax`
use fast-math kernels when the normalised dimension is the last one and the
input is float32/float64. Softmax uses a store-then-scale loop:

```c
exp(x - max) -> output, sum; output *= 1 / sum
```
and its backward is `y * (dy - sum(dy * y))`.

`log_softmax` computes max, `sum(exp(x - max))`, and `x - max - log(sum)` in one
call:

```lua
local log_probabilities = logits:log_softmax(2)
```

The kernel computes max, `sum(exp(x - max))`, and `x - max - log(sum)` in one
call. A C++ autograd Function saves the output and applies the standard
backward `dy - exp(y) * sum(dy)` in a second fast-math kernel. Other dimensions,
dtypes, or empty rows fall back to ATen.

| workload | fused | eager ATen | speedup |
| --- | ---: | ---: | ---: |
| log_softmax 4096×256 forward | 1.21 ms | 2.11 ms | 1.74x |
| log_softmax + backward | 3.50 ms | 4.78 ms | 1.37x |
| softmax 4096×256 forward | 1.20 ms | 1.63 ms | 1.36x |
| softmax + backward | 3.33 ms | 3.82 ms | 1.15x |

`cross_entropy` on 2D logits with mean/sum reduction reuses that kernel and a
fast softmax-minus-onehot backward; class counts above 4096 fall back to ATen.

The fast-math kernels live in `csrc/fast_ops.cpp`, compiled separately from the
precise ATen shim.

## Linear and convolution epilogues

The most common layer pattern is a matrix multiply or convolution, a bias add,
and one activation. `torch.fuse.linear` and `torch.fuse.conv2d` turn that pattern
into a single call. The bias stays in ATen's `addmm`/`conv2d` (so it is folded
into the existing GEMM/conv kernel), and the activation runs **in place** on the
freshly produced output through a small native helper. No activation temporary
is allocated, and the autograd graph still records the in-place node, so
training works without a fallback:

```lua
-- x @ W^T + b, then relu
local y = torch.fuse.linear(x, weight, bias, "relu")

-- conv2d + bias, then GELU
local y = torch.fuse.conv2d(input, weight, bias, {
  activation = "gelu",
  stride = 1,
  padding = 1,
  dilation = 1,
  groups = 1,
})

-- The nn modules expose the same path as an option.
local layer = nn.Linear(784, 256, { activation = "relu" })
local conv = nn.Conv2d(3, 16, 3, { padding = 1, activation = "gelu" })
```

Supported activations are `"relu"`, `"gelu"` (exact, matching ATen's default),
`"sigmoid"`, `"tanh"`, and `nil`/`"none"`/`"identity"`. `torch.fuse.conv2d`
accepts the usual positional stride/padding/dilation/groups arguments or an
options table as shown above; when the default geometry is enough, the
activation can also be passed directly as the fourth argument
(`torch.fuse.conv2d(x, w, b, "relu")`).

Supported envelope:

- `linear`: 2-D `float32`/`float64` input and weight; optional 1-D bias with one
  value per output feature. Higher-rank inputs and other dtypes use eager ATen.
- `conv2d`: 4-D NCHW `float32`/`float64` input and weight; optional 1-D bias with
  one value per output channel; `groups` supported. Other ranks/dtypes use eager
  ATen.
- A callable or `nn.Module` activation is applied eagerly after the linear or
  convolution, so custom activations keep working. Unknown activation strings
  raise an error instead of silently changing semantics.
- `torch.fuse.epilogue_stats` counts fused and eager-fallback calls, and
  `torch.fuse.epilogue_last_error()` reports a native fallback message.

This is an **epilogue over ATen's GEMM/conv**. It does not replace the GEMM or
convolution microkernel, so the achievable speedup depends on how much of the
layer is the activation pass. A 128×784 → 256 linear is GEMM-bound and is
therefore a near-tie (and occasionally slower); convolution with a ReLU
epilogue reaches about 1.2x because the output is larger relative to the
convolution cost. The backward direction is ATen's normal backward through
`addmm`/`conv2d` and the in-place activation node; no separate fused backward
kernel is generated, so training speedups are smaller than forward-only gains.

Measured on the same Ryzen 3 7320U, libtorch 2.5.1, float32, one thread,
autograd disabled, best of 200 linear / 100 convolution iterations after ten
warm-up calls:

| workload | fused | eager addmm/conv + activation | speedup |
| --- | ---: | ---: | ---: |
| Linear 128×784→256 + relu | 0.953 ms | 0.961 ms | 1.01x |
| Linear 128×784→256 + gelu | 1.067 ms | 1.068 ms | 1.00x |
| Linear 128×784→256 + sigmoid | 0.973 ms | 0.980 ms | 1.01x |
| Linear 128×784→256 + tanh | 1.129 ms | 0.986 ms | 0.87x |
| Conv2d 8×3×32×32→16 + relu | 0.250 ms | 0.308 ms | 1.23x |
| Conv2d 8×3×32×32→16 + gelu | 0.420 ms | 0.447 ms | 1.06x |
| Conv2d 8×3×32×32→16 + sigmoid | 0.321 ms | 0.376 ms | 1.17x |
| Conv2d 8×3×32×32→16 + tanh | 0.736 ms | 0.712 ms | 0.97x |

The losses are real and are not hidden: the linear GEMM dominates, and the
transcendental activations are close to parity or slower because the in-place
ATen kernel is not uniformly faster than the out-of-place one (linear + tanh
lost ~13% in this run). Repeated runs on this throttling CPU move the linear
ratio by about ±10% and the convolution ratio by about ±10%. The reliable win in
this envelope is convolution + ReLU (roughly 1.15–1.25x); treat the other rows as
parity rather than a speed claim. Use `scripts/bench.sh` on the target machine
for your own numbers.

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

Epilogue helpers add their own fallback: a linear input that is not 2-D, a
convolution input that is not 4-D NCHW, a dtype other than float32/float64, an
unsupported bias shape, or a callable/module activation runs the eager
`addmm`/`conv2d` + activation expression instead. Unknown activation strings are
rejected up front.

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
| log_softmax 4096×256 fused | 1.21 ms | 2.11 ms | **1.74x** | — |
| log_softmax + backward fused | 3.50 ms | 4.78 ms | **1.37x** | — |
| softmax 4096×256 fused | 1.20 ms | 1.63 ms | **1.36x** | — |
| softmax + backward fused | 3.33 ms | 3.82 ms | **1.15x** | — |

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
  last-dimension `softmax`/`log_softmax` and cross-entropy (≤4096 classes) are
  fused. Layernorm is not fused. Linear/conv epilogues apply the activation
  in-place over ATen's GEMM/conv, with the bias folded into ATen's
  `addmm`/`conv2d`; they do not replace the GEMM or convolution microkernel.
- Fused backward is first-order and C-backend only; second-order gradients and
  the LuaJIT-only fallback use eager autograd.
- Transcendental functions (`exp`, `tanh`) call scalar libm; vectorised libmvec
  or polynomial approximations would close the remaining gap on unary-heavy
  expressions.
- The tracer is explicit (`torch.fuse(function ... end)`), not transparent to
  arbitrary operator code. Lazy/traced tensors are future work.
- Each new expression costs one C compilation; long-running training loops
  amortise it, one-shot scripts may prefer the LuaJIT backend or eager ATen.
