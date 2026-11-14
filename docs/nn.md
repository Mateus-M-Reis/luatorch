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
| `nn.AdaptiveAvgPool2d(1)` | global average pooling |
| `nn.BatchNorm2d(features, opts?)` | `eps`, `momentum`, `affine`, `track_running_stats` |
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
| `optim.RMSprop` | `lr`, `alpha`, `eps`, `weight_decay`, `momentum`, `centered` |
| `optim.Adagrad` | `lr`, `lr_decay`, `weight_decay`, `eps`, `initial_accumulator_value` |

Passing a number as options means `{lr = number}`. Per-parameter state is keyed
by tensor identity. Gradient clipping and learning-rate schedulers are not
implemented yet.

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
for epoch = 1, epochs do
  model:train()
  local output = model(inputs)
  local loss = criterion(output, targets)
  optimizer:zero_grad()
  loss:backward()
  optimizer:step()
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

- No `DataLoader` / datasets yet; feed tensors directly.
- `nn.save`/`nn.load` use a luatorch-specific format; PyTorch state-dict
  interop is on the roadmap.
- `ConvTranspose2d`, `AvgPool2d`, embeddings, RNNs, and weight-norm layers are
  not implemented.
- Cross entropy supports plain class targets and `mean`/`sum`/`none`; class
  weights, `ignore_index` and label smoothing are rejected explicitly.
- Modules are CPU-only until the CUDA libtorch build lands.
