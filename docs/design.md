# Design notes

## Goals

1. Run tensor code at native ATen speed from plain LuaJIT.
2. Keep the Torch7 feel (1-based dims, `:add_`, `narrow`, typed constructors)
   without inheriting the old C backend.
3. Make autograd, dtypes, broadcasting, and serialization behave like PyTorch.
4. Make the functional layer luafun-based and Moses-free.
5. Stay small enough to audit and extend.

Non-goals for 0.1: full operator parity with ATen, `nn`/`optim`, CUDA, Windows.

## Layers

```
        Lua user code
              │
     lua/torch/init.lua        public module: constructors, aliases, no_grad
     lua/torch/tensor.lua      Tensor class: methods, operators, printing, luafun
     lua/torch/lib.lua         ffi.cdef declarations + shared library loader
              │
        LuaJIT FFI (C ABI only)
              │
     build/libluatorch.so      csrc/luatorch.cpp: exception-safe C wrappers
              │
     libtorch 2.5.1 (CPU)      at::Tensor, autograd, kernels, serialization
```

### Why a C shim instead of FFI to C++?

ATen's public API is C++: `at::Tensor` is a value type with refcounted internals,
operators are overloaded templates, and failures are exceptions. Calling those
symbols through FFI would require mangled names, hand-built STL objects, and
undefined behaviour around exceptions. The shim is ~1k lines of ordinary C++ and
presents:

- opaque `lt_tensor*` handles
- plain C types (`int64_t`, `double`, `void*`)
- status codes + a thread-local error string
- no Lua headers at all, so the native library can be built and tested
  independently of Lua

The FFI layer then costs one indirect call plus argument marshalling per op
(measured at roughly 3 µs for a tiny tensor).

## Memory and lifetimes

- A handle is a heap-allocated `struct lt_tensor { at::Tensor t; }`.
- `at::Tensor` itself is a small refcounted view over storage, so copying a
  handle's tensor is cheap and views (`select`, `transpose`, `slice`) share
  storage exactly as in PyTorch.
- Lua wraps every returned pointer with `ffi.gc(ptr, lt_free)`, so the handle is
  released when the Lua object is collected. Storage is freed when the last
  `at::Tensor` referring to it dies.
- In-place C functions mutate the existing tensor and return the *same* pointer.
  The Lua wrapper ignores that pointer and returns `self`, avoiding a second
  `ffi.gc` registration and a double free.
- A few functions (`grad`, `data_ptr`) return borrowed pointers. The Lua layer
  only uses `data_ptr` while holding a contiguous temporary alive, and `grad`
  wraps a new `at::Tensor` copy of the gradient.

## Errors

C++ exceptions never cross the FFI boundary. Every shim function runs inside a
`try` block; on failure it stores `what()` in a `thread_local std::string` and
returns `NULL` (tensor functions) or a non-zero status. Lua checks the result and
raises `error(last_error())`, preserving the original ATen message (including
shape mismatches, dtype errors, and autograd errors).

## Dtypes and scalar promotion

Dtype ids (`0..8`) are shared between the header, the shim, and `lib.lua`.
Lua has only doubles, which creates one interesting mismatch:

- `torch.tensor({1, 2, 3})` cannot know whether the user meant ints or floats.
  It chooses float32, the PyTorch default floating dtype.
- `long_tensor + 1` *should* stay integral. The shim inspects the tensor dtype
  and, when the value is a whole number, passes an integer `at::Scalar`, avoiding
  PyTorch's float promotion. Non-integral scalars promote as usual
  (`int_tensor / 2` becomes float32, matching true division).
- Integer tensors are explicit: `dtype = torch.int64` or `torch.LongTensor`.

## Autograd

Autograd is ATen's own engine: the shim just forwards `requires_grad`,
`backward`, `grad`, and `detach`. `torch.no_grad(fn)` flips
`at::GradMode::set_enabled` around the call and restores the previous value even
on error. Because grad mode is thread-local in ATen, this is safe if the native
library is ever used from multiple Lua states.

## Indexing model

ATen is 0-based; the Lua API is 1-based. Conversions live exclusively in
`tensor.lua`:

- positive index `i` → `i - 1`; negative index passes through and ATen counts
  from the end;
- index `0` is rejected with a clear message;
- dimensions use the same rule, and `nil` means "all dimensions" via the
  `LT_ALL_DIMS` sentinel;
- `slice(dim, start, stop, step)` has inclusive `start`/`stop` (Torch7 style),
  while ATen's end is exclusive;
- `argmax`/`argmin` add 1 to ATen's result.

## Printing

Printing is implemented in Lua from `totable()` data because formatting needs
Lua's control over precision, dtype suffixes and indentation. Tensors over 1000
elements print a compact summary. No printing happens in C++.

## Testing

`tests/run.lua` is a dependency-free harness. Test files cover construction,
math, shape operations, reductions, autograd, luafun integration, serialization,
printing, and error paths. Tests assert both values and metadata (shape/dtype),
and every error test checks a meaningful part of the message.

## Benchmarks

`bench/bench.lua` measures creation, elementwise throughput, matmul GFLOP/s,
per-call dispatch overhead, Lua-side iteration, luafun pipelines, and
serialization. Results are machine-dependent; the script is the source of truth.

## Adding an operator

1. Declare it in `csrc/luatorch.h` (plain C types only).
2. Implement it in `csrc/luatorch.cpp` inside `try`/`catch`, using `clear_error()`
   and the `LT_CATCH_NULL`/`LT_CATCH_INT` macros. Resolve negative dims with
   `resolve_dim` and indices with `resolve_index`.
3. Add the matching declaration to the `ffi.cdef` block in `lua/torch/lib.lua`.
4. Add a Lua method in `lua/torch/tensor.lua`, converting Lua 1-based
   dims/indices and scalar promotion as needed. Add an `_` variant if ATen
   provides an in-place kernel.
5. Add tests (value, shape, dtype, and at least one error path).
6. Document it in `docs/api.md`.

## Build and distribution

The shim is built out-of-tree with CMake and links the official libtorch
distribution. `third_party/libtorch` is downloaded, not vendored, keeping the
repository small. The build sets an RPATH to the bundled `libtorch/lib` so
`ffi.load` can find ATen's shared libraries without `LD_LIBRARY_PATH`.

Installing (`cmake --install`) copies the Lua tree to `share/lua/5.1` and places
`libluatorch.so` in `share/lua/5.1/torch/`, next to `lib.lua`, which is where the
loader looks last.
