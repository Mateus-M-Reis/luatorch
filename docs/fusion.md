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
- be contiguous (non-contiguous inputs are copied once),
- not require gradients (the fusion path is inference-only for now).

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

Caveats:

- This CPU throttles on sustained load; absolute numbers move by 10–20% between
  runs. The harness uses best-of-N and one thread.
- `torch.compile` also fuses elementwise chains. We are roughly at parity on
  this one kernel (about 1 ms for both); the honest claim is "competitive with
  compiled PyTorch, several times faster than eager on chains".
- Adam/AdamW/SGD use fused update kernels: the optimizer step is ~5x faster
  than PyTorch's, which lifts the full MLP training step to ~1.3x.
- Matrix multiplication uses the same class of BLAS kernels on both sides, so a
  tie is the expected result.

## Limitations and next steps

- No fused reductions, softmax, layernorm, convolution, or matmul epilogues yet.
- No autograd through fused kernels; training falls back to eager. Fusing
  forward and backward is the next major milestone.
- Transcendental functions (`exp`, `tanh`) call scalar libm; vectorised libmvec
  or polynomial approximations would close the remaining gap on unary-heavy
  expressions.
- The tracer is explicit (`torch.fuse(function ... end)`), not transparent to
  arbitrary operator code. Lazy/traced tensors are future work.
- Each new expression costs one C compilation; long-running training loops
  amortise it, one-shot scripts may prefer the LuaJIT backend or eager ATen.
