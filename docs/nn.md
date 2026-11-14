# Neural networks: `torch.nn` and `torch.optim`

Layers are plain Lua objects that call Tensor operations; gradients come from
ATen autograd, so there is no separate `backward()` method on modules. The
training loop is PyTorch's:

```lua
local torch = require("torch")
local nn = require("torch.nn")
local optim = require("torch.optim")

local model = nn.Sequential(
  nn.Linear(2, 32), nn.ReLU(),
  nn.Linear(32, 2))

local criterion = nn.CrossEntropyLoss()
local optimizer = optim.Adam(model:parameters(), { lr = 0.01 })

for epoch = 1, 100 do
  local logits = model(inputs)
  local loss = criterion(logits, labels)

  optimizer:zero_grad()
  loss:backward()
  optimizer:step()
end
```

`torch.nn` and `torch.optim` are also reachable as `torch.nn` / `torch.optim`
(lazy-loaded), but requiring them directly is the clearest style.

## Modules

Use `nn.Module` as the base class:

```lua
local nn = require("torch.nn")

local Scale = nn.Module.class("Scale")

function Scale.new(factor)
  local self = setmetatable(nn.Module.new(), Scale)
  self.factor = factor or 1
  return self
end

function Scale:forward(input)
  return input:mul(self.factor)
end
```

An instance is callable (`module(x)` calls `forward`) and supports:

| Method | Meaning |
| --- | --- |
| `module:forward(x)` | compute the output (autograd records the graph) |
| `module(x)` | same, via `__call` |
| `module:parameters()` | array of parameter tensors (recursive) |
| `module:named_parameters(prefix?)` | table `name -> tensor` |
| `module:named_buffers(prefix?)` | running statistics and other buffers |
| `module:zero_grad()` | zero all parameter gradients |
| `module:train(mode?)` / `module:eval()` | switch training mode (recursive) |
| `module:to(dtype)` | convert parameters/buffers, preserving `requires_grad` |
| `module:state_dict()` | snapshot `name -> tensor` (detached **clones**) |
| `module:load_state_dict(dict, strict?)` | copy values back; strict by default |
| `module:register_parameter(name, t)` | register a trainable tensor |
| `module:register_buffer(name, t)` | register a non-trainable tensor |
| `module:register_module(name, m)` | register a child module |

Parameters are registered with `requires_grad = true` (floating dtypes only).
In-place edits to parameters are rejected while they are leaves with gradients,
exactly like PyTorch; use `torch.no_grad(function() ... end)` for initialisation.

`state_dict` returns **clones**, not live views, so a snapshot is not corrupted
when training continues. This is slightly safer than PyTorch's shallow
references.

## Built-in modules

| Module | Signature |
| --- | --- |
| `nn.Linear(in, out, bias?)` | affine transform over the last dimension |
| `nn.Conv2d(in, out, kernel, opts?)` | 2-D convolution, NCHW, groups supported |
| `nn.MaxPool2d(kernel, stride?, padding?, dilation?, ceil_mode?)` | max pooling |
| `nn.AvgPool2d(kernel, stride?, padding?, ceil_mode?, count_include_pad?)` | average pooling |
| `nn.AdaptiveAvgPool2d(1)` | global average pooling |
| `nn.BatchNorm2d(features, opts?)` | `eps`, `momentum`, `affine`, `track_running_stats` |
| `nn.LayerNorm(normalized_shape, opts?)` | `eps`, `elementwise_affine`; normalizes trailing dims |
| `nn.Embedding(num_embeddings, dim)` | 1-based index lookup; `padding_idx` not supported yet |
| `nn.Dropout(p?, inplace?)` | inverted dropout |
| `nn.Dropout2d(p?, inplace?)` | elementwise dropout with a 2-D name |
| `nn.ReLU(inplace?)` | rectifier |
| `nn.LeakyReLU(slope?, inplace?)` | leaky rectifier |
| `nn.Sigmoid()` / `nn.Tanh()` | elementwise activations |
| `nn.Softmax(dim?)` / `nn.LogSoftmax(dim?)` | default dim is the last one |
| `nn.Flatten(start_dim?, end_dim?)` | defaults `start_dim = 2`, `end_dim = -1` |
| `nn.Identity()` | pass-through |

Containers:

```lua
local model = nn.Sequential(
  nn.Conv2d(1, 8, 3, { padding = 1 }),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Flatten(),
  nn.Linear(8 * 4 * 4, 10))

model:add(nn.Dropout(0.2))     -- append a module
```

`nn.ModuleList({m1, m2})` stores modules for manual iteration
(`list:get(1)`, `#list:parameters()`).

`Conv2d` options can be a table (`{stride = 2, padding = 1, dilation = 1,
groups = 1, bias = true}`) or positional arguments. Kernel, stride, padding and
dilation accept a number or `{height, width}`.

## Data loading

`torch.utils.data` ships an in-memory dataset and a batcher:

```lua
local data = require("torch.utils.data")

local dataset = data.TensorDataset(inputs, labels)   -- same first dimension
local loader = data.DataLoader(dataset, {
  batch_size = 64,
  shuffle = true,
  drop_last = false,
})

for batch_inputs, batch_labels in loader:iter() do
  local loss = criterion(model(batch_inputs), batch_labels)
  optimizer:zero_grad()
  loss:backward()
  optimizer:step()
end
```

- `dataset:size()` and `dataset:get(index)` (1-based) define a custom dataset.
- Datasets may implement `dataset:batch(indices)` to return a whole batch
  (TensorDataset uses `index_select`, avoiding per-sample overhead).
- Shuffling uses the global RNG; call `torch.manual_seed` for reproducibility.
- `drop_last` discards an incomplete final batch; `loader:num_batches()` gives
  the number of batches per epoch.
- There are no built-in image/text datasets yet; build tensors and wrap them.

## Losses

| Module | Notes |
| --- | --- |
| `nn.MSELoss(reduction?)` | mean over all elements by default |
| `nn.L1Loss(reduction?)` | |
| `nn.CrossEntropyLoss(reduction?)` | log-softmax + NLL; targets are **1-based** |
| `nn.NLLLoss(reduction?)` | input is already log-probabilities; 1-based targets |
| `nn.BCEWithLogitsLoss(reduction?)` | numerically stable binary cross entropy |
| `nn.BCELoss(reduction?)` | probabilities, clamped to avoid logs of zero |

`reduction` is `"mean"` (default), `"sum"` or `"none"`.

`nn.Softmax` and `nn.LogSoftmax` over the last dimension use fused
fast-math kernels (float32/float64); other dimensions fall back to ATen. `CrossEntropyLoss` on 2D
float32/64 logits with mean/sum reduction uses a fused fast-math path when the
class count is at most 4096; larger classifiers fall back to ATen.

`MSELoss` and `L1Loss` use the fused one-pass reductions when input and target
have the same shape and float32/float64 dtype (and reduction is not `"none"`).
Broadcasting or unsupported layouts fall back to the ATen loss. This makes the
common regression losses trainable without materialising the elementwise error
tensor.

### Class labels are 1-based

`argmax` returns 1-based indices, and cross entropy expects 1-based labels:

```lua
local target = torch.LongTensor({1, 3, 2})
local loss = nn.CrossEntropyLoss()(logits, target)
local predicted = logits:argmax(2)          -- also 1-based
```

Internally the labels are shifted by one before calling ATen. This matches
Torch7's `ClassNLLCriterion` and keeps Lua code free of off-by-one conversions.

## Functional API

`nn.functional` exposes the stateless versions:

```lua
local F = require("torch.nn").functional

F.linear(x, weight, bias)
F.relu(x, inplace)
F.leaky_relu(x, slope, inplace)
F.sigmoid(x)   F.tanh(x)
F.softmax(x, dim)   F.log_softmax(x, dim)
F.conv2d(x, weight, bias, stride, padding, dilation, groups)
F.max_pool2d(x, kernel, stride, padding, dilation, ceil_mode)
F.adaptive_avg_pool2d(x, 1)
F.mse_loss(x, target, reduction)
F.l1_loss(x, target, reduction)
F.cross_entropy(x, target, reduction)
F.nll_loss(x, target, reduction)
F.binary_cross_entropy_with_logits(x, target, reduction)
F.binary_cross_entropy(x, target, reduction)   -- probabilities
F.dropout(x, p, training, inplace)
```

## Optimizers

All optimizers take `(parameters, options)` and expose `step()` and
`zero_grad()`. Updates run under `torch.no_grad`.

| Optimizer | Options |
| --- | --- |
| `optim.SGD` | `lr`, `momentum`, `dampening`, `weight_decay`, `nesterov` |
| `optim.Adam` | `lr`, `betas = {b1, b2}`, `eps`, `weight_decay` |
| `optim.AdamW` | Adam with decoupled weight decay |
| `optim.RMSprop` | `lr`, `alpha`, `eps`, `weight_decay`, `momentum`, `centered` |
| `optim.Adagrad` | `lr`, `lr_decay`, `weight_decay`, `eps`, `initial_accumulator_value` |

Passing a number as options means `{lr = number}`. Per-parameter state is keyed
by tensor identity.

### Linear

`nn.Linear` uses a single `addmm` for 2D inputs, folding the bias into the
matrix multiply. Higher-rank inputs are flattened to 2D, projected, and
reshaped back. This avoids a separate bias kernel and temporary.

### Fused update kernels

Adam, AdamW, SGD, RMSprop and Adagrad execute their update in a single C kernel
over contiguous float32/float64 storage, instead of a chain of temporary tensor
ops. On a 784→256→10 MLP this reduces the Adam step from ~1.14 ms (PyTorch) to
~0.22 ms, and the complete training step from ~5.1 ms to ~4.0 ms.

Parameters, gradients and optimizer state must be contiguous and float32 or
float64; non-contiguous parameters raise an error rather than silently copying.

### Gradient clipping

Gradient-norm and gradient-value clipping use fused C kernels when every
gradient is contiguous float32/float64; other layouts fall back to the Lua
tensor path.

```lua
optim.clip_grad_norm_(model:parameters(), 1.0)        -- global L2 norm
optim.clip_grad_norm_(model:parameters(), 1.0, math.huge)
optim.clip_grad_value_(model:parameters(), 0.5)       -- elementwise clamp
```

### Learning-rate schedulers

Call `scheduler:step()` once per epoch, after `optimizer:step()`:

```lua
local scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
local scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, {30, 60}, 0.1)
local scheduler3 = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
local scheduler4 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
local scheduler5 = optim.lr_scheduler.LambdaLR(optimizer, function(epoch)
  return 1 / epoch
end)
```

Schedulers mutate `optimizer.lr`; they do not own parameter state.

Create the optimizer **after** converting the model dtype, because
`module:to(dtype)` replaces parameter objects:

```lua
model:to(torch.float64)
local optimizer = optim.Adam(model:parameters(), { lr = 1e-3 })
```

## Saving and loading models

```lua
model:save("classifier.luatorch")
model:load("classifier.luatorch")

-- functional equivalents
nn.save(model, "classifier.luatorch")
nn.load(model, "classifier.luatorch")
```

The file contains the state dictionary (parameters plus buffers) in a compact
luatorch-specific format: a small text header per tensor followed by contiguous
bytes. Loading requires the same architecture; values are copied with
`load_state_dict`, so dtypes are cast when needed. This format is **not** the
PyTorch zip format — `Tensor:save`/`torch.load` still handles individual tensors
in that format, and state-dict interop with PyTorch is on the roadmap.

## Training recipe

```lua
local loader = data.DataLoader(dataset, {batch_size = 64, shuffle = true})
local scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)

for epoch = 1, epochs do
  model:train()
  for inputs, targets in loader:iter() do
    local loss = criterion(model(inputs), targets)
    optimizer:zero_grad()
    loss:backward()
    optimizer:step()
  end
  scheduler:step()
end

model:eval()
local accuracy = model(test_inputs):argmax(2):eq(test_targets)
  :to(torch.float32):mean():item()
```

- Call `zero_grad()` before `backward()` each iteration.
- Wrap evaluation in `torch.no_grad(function() ... end)` if you want to skip
  graph construction.
- `BatchNorm2d` uses batch statistics in training mode (updating running
  estimates) and running statistics in `eval()`.
- `Dropout` is identity in `eval()` and scales survivors by `1/(1-p)` in
  training.

## Current limitations

- No built-in datasets; `DataLoader` works on in-memory tensors and custom
  dataset objects.
- `nn.save`/`nn.load` use a luatorch-specific format; PyTorch state-dict
  interop is on the roadmap.
- `ConvTranspose2d`, GroupNorm, RNN/LSTM and weight-norm layers are not
  implemented.
- `Embedding` does not support `padding_idx` yet.
- Cross entropy supports plain class targets and `mean`/`sum`/`none`; class
  weights, `ignore_index` and label smoothing are rejected explicitly.
- Modules are CPU-only until the CUDA libtorch build lands.
