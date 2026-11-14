# luatorch roadmap

The project's goal is a Torch7-class Lua tensor stack on top of PyTorch's ATen:
fast, scriptable, and pleasant for research code. This roadmap is intentionally
concrete: each milestone should leave `scripts/test.sh` green and documented.

## v0.1 — tensor foundation (done)

- [x] C ABI shim over `at::Tensor` with exception-safe error reporting
- [x] LuaJIT FFI loader and PyTorch-style `torch` module
- [x] Dtypes: float16/32/64, int8/16/32/64, uint8, bool
- [x] Creation, broadcasting math, ~25 unary ops, matmul family
- [x] Shape ops, slicing, Boolean masking, `index_select`, `cat`/`stack`
- [x] Reductions with dim/keepdim, 1-based indices
- [x] Autograd: `requires_grad`, `backward`, `grad`, `detach`, `no_grad`
- [x] PyTorch `.pt` serialization
- [x] luafun integration (`torch.fun`, `Tensor:iter/map/filter/reduce`, `t:fun()`)
- [x] 160-test suite, benchmarks, API/design docs
- [x] Moses-vs-luafun benchmark documenting the functional-layer choice

## v0.2 — neural networks (current)

- [x] `nn.Module` system: parameters, buffers, children, train/eval, dtype
  conversion, named parameters, `state_dict`/`load_state_dict`
- [x] Linear, Conv2d, MaxPool2d, AdaptiveAvgPool2d, Flatten, ReLU, LeakyReLU,
  Sigmoid, Tanh, Softmax, LogSoftmax, Dropout, BatchNorm2d
- [x] Sequential and ModuleList containers
- [x] Losses: MSE, L1, CrossEntropy, NLL, BCE, BCEWithLogits
- [x] 1-based class labels across `argmax` and classification losses
- [x] Optimizers: SGD (momentum, nesterov, weight decay), Adam, RMSprop, Adagrad
- [x] MLP and CNN training examples plus forward/backward/optimizer benchmarks
- [ ] `torch.utils.data`: Dataset, DataLoader, deterministic shuffling
- [x] On-disk module serialization (`nn.save` / `model:load`, luatorch format)
- [ ] PyTorch state-dict interop
- [ ] More layers: AvgPool2d, ConvTranspose2d, Embedding, LayerNorm, GroupNorm,
  RNN/LSTM
- [ ] Loss options: class weights, `ignore_index`, label smoothing
- [ ] Optimizer extras: AdamW, AMSGrad, gradient clipping, LR schedulers

## v0.3 — tensor completeness

- [ ] Richer indexing: tensor-of-indices selection, ranges, `gather`/`scatter`,
  `index_put_`, `take_along_dim`, `nonzero`
- [ ] `sort`, `topk`, `unique`, `bincount`, `histc`
- [ ] Linear algebra: `inv`, `det`, `solve`, `svd`, `qr`, `eig`, `cholesky`,
  `linalg.*` bindings
- [ ] Random distributions beyond uniform/normal: bernoulli, exponential,
  multinomial, in-place `normal_`/`uniform_`
- [ ] `einsum`, `outer`, `mv`, `ger`, `addmm`, `baddbmm`
- [ ] More reductions: `logsumexp`, `aminmax`, `median`, `quantile`
- [ ] Thread-count control (`torch.set_num_threads`)
- [ ] `float16` creation from Lua values and printing parity

## v0.4 — the Torch7 stack

- [x] `torch.nn` layers, losses, containers, functional forms, parameter iteration
- [x] `torch.optim`: SGD, momentum, Adam, RMSprop, Adagrad
- [ ] `torch.utils.data`: Dataset/DataLoader with deterministic shuffling
- [x] `torch.class`-style module system for user-defined layers (`nn.Module.class`)
- [x] Model serialization to disk (`nn.save` / `model:load`)
- [ ] PyTorch state-dict interop
- [ ] `luatorch` rockspec and system installation

## v0.5 — performance and ergonomics

- [ ] Zero-copy tensor views where safe; avoid redundant `contiguous()` calls
- [ ] Direct `at::from_blob` wrappers for sharing memory with Lua strings/buffers
- [ ] Batched indexing kernels in the shim to cut per-call overhead
- [ ] Fused optimizer updates (one C call per parameter instead of several ops)
- [ ] Profiling helpers (per-op timing, dispatch counters)
- [ ] Improved large-tensor printing and pretty tables
- [ ] `luacheck`/CI matrix across LuaJIT versions and Linux/macOS

## v1.0 — GPU and distribution

- [ ] CUDA build via the CUDA libtorch distribution (`device = "cuda"`,
  `.cuda()`, `.cpu()`, device-aware constructors)
- [ ] CPU/GPU parity tests and device transfer benchmarks
- [ ] Optional MKL/OpenBLAS tuning and thread controls
- [ ] Stable C ABI versioning and prebuilt binary releases
- [ ] nn/optim GPU support

## Explicit non-goals

- Running unmodified Torch7 C extensions.
- Reimplementing `nn`/`optim` in Lua on top of a slow backend; modules should
  call ATen kernels directly.
- Windows support before the core is stable (patches welcome).

## How to help

Pick a v0.3 operator or a v0.2 layer, follow the six-step checklist in
[docs/design.md](docs/design.md#adding-an-operator), and include value, shape,
dtype, and error-path tests. Benchmarks and compatibility notes are as valuable
as new ops.
