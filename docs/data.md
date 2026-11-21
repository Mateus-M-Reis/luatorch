# Data loading

`torch.utils.data` provides an in-memory dataset interface and a deterministic
batcher. It is deliberately small: there are no built-in image datasets, no
samplers, and no OS-level worker processes.

```lua
local torch = require("torch")
local data = require("torch.utils.data")

local dataset = data.TensorDataset(inputs, labels)   -- 1-based get()
local loader = data.DataLoader(dataset, {
  batch_size = 64,
  shuffle = true,
  drop_last = false,
  num_workers = 2,
  prefetch = 4,
  seed = 42,
})

for batch_inputs, batch_labels in loader:iter() do
  -- batch_inputs is inputs[index_set], stacked along dimension 1
end
```

See [`examples/dataloader.lua`](../examples/dataloader.lua) for a runnable
walkthrough.

## Dataset protocol

A dataset is any table with:

| Method | Contract |
| --- | --- |
| `dataset:size()` | number of samples, a non-negative integer |
| `dataset:get(index)` | sample at 1-based `index`; returns one or more Tensors |

An optional `dataset:batch(indices)` receives a Lua array of 1-based indices and
returns a whole batch. `TensorDataset` implements it with `index_select`, which
avoids per-sample Lua overhead. When `batch` is absent, the loader calls
`get(index)` for every sample, requires every sample to return the same number of
Tensors, and stacks each column along a new first dimension.

`TensorDataset(t1, t2, ...)` accepts Tensors that share a first dimension. It
rejects non-Tensors, scalars, and mismatched sample counts at construction.
`TensorDataset({t1, t2})` is equivalent to the vararg form.

## Options

| Option | Default | Accepted values |
| --- | --- | --- |
| `batch_size` | `1` | positive integer |
| `shuffle` | `false` | any truthy value |
| `drop_last` | `false` | any truthy value |
| `num_workers` | `0` | non-negative integer |
| `prefetch` | `2 * num_workers`, or `0` | non-negative integer |
| `seed` | `nil` | integer |

Invalid numeric options raise at `DataLoader(dataset, options)` time. `shuffle`
and `drop_last` follow Lua truthiness; `shuffle = false` is the default.

`loader:num_batches()` reports the number of batches in one epoch:

- `floor(size / batch_size)` with `drop_last = true`;
- `ceil(size / batch_size)` otherwise.

A dataset smaller than `batch_size` therefore produces one partial batch, or no
batches with `drop_last = true`. An empty dataset always produces zero batches,
including with `shuffle = true`; the loader does not build an empty permutation.

## Iteration lifecycle

`loader:iter()` starts a new epoch and returns a **callable iterator**:

```lua
local iterator = loader:iter()
local first = iterator()          -- same as iterator:next()
for batch in iterator do ... end  -- __call makes this work
```

The iterator returns every batch once, in order, then `nil`. It closes itself
automatically when the last batch is returned, when an error is raised, or when
the iterator is garbage-collected. Closing releases queued batches, worker
coroutines, the shuffle order, and the dataset reference.

- `iterator:close()` is idempotent and may be called before the first batch.
- After close, `iterator()` returns `nil` forever.
- `loader:close()` closes every iterator that is still alive for that loader.
- A loader itself remains reusable: the next `loader:iter()` starts a fresh
  epoch, including a fresh shuffle and fresh workers.

Call `iterator:close()` (or `loader:close()`) when leaving a loop early. Lua does
not run finalizers at `break` time, and `__gc` timing is not guaranteed, so
relying on collection can retain a dataset and its worker coroutines longer than
expected.

## Shuffling

Indices are 1-based throughout. The loader builds a permutation of
`1 .. dataset:size()` for every epoch when `shuffle = true`.

- Without `seed`, the permutation comes from `torch.randperm` and consumes the
  global `torch` RNG at `loader:iter()` time. Call `torch.manual_seed` before
  creating the iterator for reproducibility.
- With `seed`, the loader uses a private LCG-based permutation. It neither reads
  nor advances the global RNG, and it produces the **same order on every
  epoch**. This differs from PyTorch's `generator`, which advances between
  epochs. Vary the seed yourself, or use `shuffle = false`, if epochs must
  differ.

`num_workers` and `prefetch` never change batch composition or order. The same
seed and dataset size produce the same batches for any worker count.

## Workers and prefetch

`num_workers` creates Lua coroutines in the current Lua state. They are
**cooperative, not OS threads or processes**:

- Only one coroutine runs at a time, and only while the loader is preparing a
  batch. Dataset access stays single-threaded and needs no locking.
- Each worker yields exactly one completed batch per resume. Worker `w` handles
  batch sequences `w, w + num_workers, w + 2 * num_workers, ...`, and the
  iterator reassembles them in order.
- `num_workers` is clamped to the number of batches in the epoch. An empty
  dataset creates no coroutines.
- `prefetch` bounds look-ahead. At most `prefetch + 1` batches are held: the one
  about to be returned plus `prefetch` prepared successors. `prefetch = 0`
  prepares only the requested batch.
- With `num_workers = 0`, preparation happens in the calling coroutine. With
  `prefetch > 0`, that path still computes batches ahead of time, but it cannot
  overlap with training math.

### Error propagation

Dataset and collation errors are re-raised in batch order. Batches that were
already prepared before the failing batch are still delivered; the error appears
when its batch position is reached, and the iterator closes immediately.

- A worker failure is reported as
  `DataLoader: worker N failed while preparing batch M: <cause>`.
- A `num_workers = 0` failure is reported as
  `DataLoader: failed while preparing batch M: <cause>`.

Errors detected while prefetching are remembered but not raised early. If the
caller abandons the iterator before reaching the failing batch, the error is
discarded by `close()`.

## Limits

- **No parallel loading.** Cooperative coroutines cannot overlap with each other,
  with training math, or with a long native ATen call. `num_workers` exists for
  API compatibility and partitions batch preparation across coroutines; it does
  not reduce total work or make loading faster by itself. Prefetch front-loads
  work into `next()` calls.
- **In-memory data only.** The dataset lives in the current Lua state. There is
  no `IterableDataset`, no multiprocessing, no shared memory, no pinned memory,
  no worker initialization hook, no custom `collate_fn`, and no CUDA transfer.
- **No samplers.** `shuffle`, `drop_last`, and sequential order are the only
  epoch policies. Weighted or replacement sampling must be implemented in a
  custom dataset.
- **Shuffling is O(size) Lua memory** for the permutation, plus the index tensor
  used by `TensorDataset:batch`.
- **Prefetch trades memory for look-ahead.** Each queued batch is a full set of
  Tensors; a large `batch_size * (prefetch + 1)` can be significant.
- **Workers are per-iterator.** A new `loader:iter()` creates new coroutines.
  There are no persistent worker pools.
- **Dataset methods must be safe to call from the loader coroutine.** They run in
  the same Lua state, and a worker is suspended between batches, never
  mid-sample. Avoid relying on global mutable state that changes while an epoch
  is in flight.
- **CPU only.** Tensors and `DataLoader` are CPU-only in this release.

## Custom dataset example

```lua
local Dataset = data.Dataset
local Squares = setmetatable({}, { __index = Dataset })
Squares.__index = Squares

function Squares.new(count)
  local self = setmetatable(Dataset.new(), Squares)
  self.count = count
  return self
end

function Squares:size()
  return self.count
end

function Squares:get(index)
  return torch.tensor({ index, index * index })
end

local loader = data.DataLoader(Squares.new(10), {
  batch_size = 4, shuffle = true, seed = 3, num_workers = 2,
})
assert(loader:num_batches() == 3)
```
