--
-- torch.utils.data
--
-- Dataset protocol:
--   dataset:size()          number of samples
--   dataset:get(index)      one sample, 1-based index
--
-- Datasets may also implement dataset:batch(indices) to return a whole batch
-- at once (TensorDataset does this with index_select).
--
local torch = require("torch")

local unpack = table.unpack or unpack

local function pack(...)
  return { n = select("#", ...), ... }
end

-- Dataset --------------------------------------------------------------------

local Dataset = {}
Dataset.__index = Dataset

function Dataset.new()
  return setmetatable({}, Dataset)
end

function Dataset:size()
  error("Dataset:size() is not implemented", 2)
end

function Dataset:get()
  error("Dataset:get(index) is not implemented", 2)
end

setmetatable(Dataset, {
  __call = function(c, ...)
    return c.new(...)
  end,
})

-- TensorDataset --------------------------------------------------------------

local TensorDataset = setmetatable({}, {
  __index = Dataset,
  __call = function(c, ...)
    return c.new(...)
  end,
})
TensorDataset.__index = TensorDataset
TensorDataset.__name = "TensorDataset"

function TensorDataset.new(...)
  local tensors = { ... }
  if #tensors == 1 and type(tensors[1]) == "table" and not torch.is_tensor(tensors[1]) then
    tensors = tensors[1]
  end
  if #tensors == 0 then
    error("TensorDataset expects at least one tensor", 3)
  end

  local count
  for i, tensor in ipairs(tensors) do
    if not torch.is_tensor(tensor) then
      error("TensorDataset argument " .. i .. " is not a Tensor", 3)
    end
    if tensor:dim() == 0 then
      error("TensorDataset tensors must have a batch dimension", 3)
    end
    if count == nil then
      count = tensor:size(1)
    elseif tensor:size(1) ~= count then
      error("TensorDataset tensors must share the same first dimension", 3)
    end
  end

  local self = setmetatable(Dataset.new(), TensorDataset)
  self.tensors = tensors
  self.count = count
  return self
end

function TensorDataset:size()
  return self.count
end

function TensorDataset:get(index)
  local values = {}
  for i, tensor in ipairs(self.tensors) do
    values[i] = tensor:select(1, index)
  end
  return unpack(values, 1, #values)
end

function TensorDataset:batch(indices)
  local index_tensor = torch.LongTensor(indices)
  local values = {}
  for i, tensor in ipairs(self.tensors) do
    values[i] = tensor:index_select(1, index_tensor)
  end
  return unpack(values, 1, #values)
end

function TensorDataset:__tostring()
  local parts = {}
  for i, tensor in ipairs(self.tensors) do
    parts[i] = table.concat(tensor:sizes(), "x")
  end
  return "TensorDataset(" .. self.count .. " samples: " ..
    table.concat(parts, ", ") .. ")"
end

-- DataLoader -----------------------------------------------------------------

local DataLoader = {}
DataLoader.__index = DataLoader
DataLoader.__name = "DataLoader"

local RNG_MODULUS = 4294967296
local STOP = {}

-- A small, self-contained LCG keeps seeded shuffles independent of the global
-- torch RNG. Only the high bits are used so permutations do not inherit the
-- weak low bits of a linear-congruential generator.
local function seeded_order(count, seed)
  local order = {}
  for i = 1, count do
    order[i] = i
  end
  local state = seed % RNG_MODULUS
  for i = count, 2, -1 do
    state = (1664525 * state + 1013904223) % RNG_MODULUS
    local j = math.floor((state / RNG_MODULUS) * i) + 1
    order[i], order[j] = order[j], order[i]
  end
  return order
end

local function batch_indices(seq, batch_size, total, order)
  local first = (seq - 1) * batch_size + 1
  local size = math.min(batch_size, total - first + 1)
  local indices = {}
  for i = 1, size do
    local position = first + i - 1
    indices[i] = order and order[position] or position
  end
  return indices
end

local function collate(dataset, indices)
  if dataset.batch then
    return pack(dataset:batch(indices))
  end

  local columns = nil
  for i = 1, #indices do
    local sample = pack(dataset:get(indices[i]))
    if columns == nil then
      columns = {}
      for j = 1, sample.n do
        columns[j] = {}
      end
    elseif sample.n ~= #columns then
      error("DataLoader: dataset returned an inconsistent number of values", 2)
    end
    for j = 1, sample.n do
      columns[j][#columns[j] + 1] = sample[j]
    end
  end

  local result = {}
  for j = 1, #columns do
    if not torch.is_tensor(columns[j][1]) then
      error("DataLoader: dataset values must be Tensors", 2)
    end
    result[j] = torch.stack(columns[j], 1)
  end
  return pack(unpack(result, 1, #result))
end

local function make_batch(dataset, seq, batch_size, total, order)
  return collate(dataset, batch_indices(seq, batch_size, total, order))
end

-- Each worker is a Lua coroutine. It yields exactly one prepared batch per
-- resume, so the main iterator can keep an ordered, bounded queue without any
-- OS threads or processes.
local function run_worker(worker_id, num_workers, dataset, total, total_batches,
  batch_size, order)
  for seq = worker_id, total_batches, num_workers do
    local ok, result = pcall(make_batch, dataset, seq, batch_size, total, order)
    local signal
    if ok then
      signal = coroutine.yield(seq, nil, result)
    else
      signal = coroutine.yield(seq, tostring(result))
    end
    if signal == STOP then
      return
    end
  end
end

local Iterator = {}
Iterator.__index = Iterator
Iterator.__name = "DataLoaderIterator"

Iterator.__call = function(iterator)
  return iterator:next()
end

-- Prepare batches through next_seq + prefetch. The queue therefore holds at
-- most prefetch + 1 batches (the batch about to be returned plus prefetch ahead).
function Iterator:_fill()
  local target = math.min(self.total_batches, self.next_seq + self.prefetch)
  if self.first_error then
    target = math.min(target, self.first_error - 1)
  end
  while self.produce_seq <= target do
    local seq = self.produce_seq

    if self.num_workers == 0 then
      local ok, result = pcall(
        make_batch, self.dataset, seq, self.batch_size, self.total, self.order)
      if not ok then
        local message = string.format(
          "DataLoader: failed while preparing batch %d: %s",
          seq, tostring(result))
        self.errors[seq] = message
        self.first_error = self.first_error or seq
        self.produce_seq = seq + 1
        break
      end
      self.ready[seq] = result
      self.produce_seq = seq + 1
    else
      local worker_id = (seq - 1) % self.num_workers + 1
      local co = self.coroutines[worker_id]
      local previous_error = self.worker_errors[worker_id]
      if previous_error or coroutine.status(co) ~= "suspended" then
        local message = previous_error or string.format(
          "DataLoader: worker %d exited before preparing batch %d",
          worker_id, seq)
        self.errors[seq] = message
        self.first_error = self.first_error or seq
        self.produce_seq = seq + 1
        break
      end

      self.started[worker_id] = true
      local ok, got_seq, worker_error, result = coroutine.resume(co)
      if not ok then
        local message = string.format(
          "DataLoader: worker %d failed while preparing batch %d: %s",
          worker_id, seq, tostring(got_seq))
        self.worker_errors[worker_id] = message
        self.errors[seq] = message
        self.first_error = self.first_error or seq
        self.produce_seq = seq + 1
        break
      elseif got_seq ~= seq then
        local message = string.format(
          "DataLoader: worker %d returned batch %s while batch %d was expected",
          worker_id, tostring(got_seq), seq)
        self.errors[seq] = message
        self.first_error = self.first_error or seq
        self.produce_seq = seq + 1
        break
      elseif worker_error ~= nil then
        local message = string.format(
          "DataLoader: worker %d failed while preparing batch %d: %s",
          worker_id, seq, worker_error)
        self.errors[seq] = message
        self.first_error = self.first_error or seq
        self.produce_seq = seq + 1
        break
      else
        self.ready[seq] = result
        self.produce_seq = seq + 1
      end
    end
  end
end

function Iterator:next()
  if self.closed then
    return nil
  end

  if self.errors[self.next_seq] then
    local message = self.errors[self.next_seq]
    self:close()
    error(message, 0)
  end

  if self.next_seq > self.total_batches then
    self:close()
    return nil
  end

  local ok, err = pcall(self._fill, self)
  if not ok then
    self:close()
    error(err, 0)
  end

  if self.errors[self.next_seq] then
    local message = self.errors[self.next_seq]
    self:close()
    error(message, 0)
  end

  local result = self.ready[self.next_seq]
  if result == nil then
    local message = string.format(
      "DataLoader: no batch was produced for position %d", self.next_seq)
    self:close()
    error(message, 0)
  end

  self.ready[self.next_seq] = nil
  self.next_seq = self.next_seq + 1
  if self.next_seq > self.total_batches then
    self:close()
  end
  return unpack(result, 1, result.n)
end

-- Stop worker coroutines and release queued tensors. Called automatically when
-- an epoch is exhausted; call it (or loader:close()) when leaving a loop early.
function Iterator:close()
  if self.closed then
    return
  end
  self.closed = true

  if self.loader then
    self.loader._iterators[self] = nil
  end

  for worker_id, co in ipairs(self.coroutines or {}) do
    if self.started[worker_id] and coroutine.status(co) == "suspended" then
      pcall(coroutine.resume, co, STOP)
    end
  end

  self.coroutines = nil
  self.started = nil
  self.worker_errors = nil
  self.ready = nil
  self.errors = nil
  self.first_error = nil
  self.order = nil
  self.dataset = nil
  self.loader = nil
end

function Iterator:__gc()
  pcall(self.close, self)
end

function DataLoader.new(dataset, options)
  if not (type(dataset) == "table" and dataset.size and dataset.get) then
    error("DataLoader expects a Dataset", 3)
  end
  options = options or {}
  local self = setmetatable({}, DataLoader)
  self.dataset = dataset
  self.batch_size = options.batch_size or 1
  self.shuffle = options.shuffle and true or false
  self.drop_last = options.drop_last and true or false
  self.num_workers = options.num_workers or 0
  self.seed = options.seed
  if type(self.batch_size) ~= "number" or self.batch_size < 1 or
    self.batch_size % 1 ~= 0 then
    error("DataLoader: batch_size must be a positive integer", 3)
  end
  if type(self.num_workers) ~= "number" or self.num_workers < 0 or
    self.num_workers % 1 ~= 0 then
    error("DataLoader: num_workers must be a non-negative integer", 3)
  end
  if self.seed ~= nil and (type(self.seed) ~= "number" or
    self.seed % 1 ~= 0) then
    error("DataLoader: seed must be an integer", 3)
  end
  self.prefetch = options.prefetch
  if self.prefetch == nil then
    self.prefetch = self.num_workers > 0 and self.num_workers * 2 or 0
  end
  if type(self.prefetch) ~= "number" or self.prefetch < 0 or
    self.prefetch % 1 ~= 0 then
    error("DataLoader: prefetch must be a non-negative integer", 3)
  end
  self._iterators = setmetatable({}, { __mode = "k" })
  return self
end

setmetatable(DataLoader, {
  __call = function(c, ...)
    return c.new(...)
  end,
})

function DataLoader:num_batches()
  local count = self.dataset:size()
  if self.drop_last then
    return math.floor(count / self.batch_size)
  end
  return math.ceil(count / self.batch_size)
end

-- Returns a callable iterator over batches:
--   for inputs, targets in loader:iter() do ...
--   local first = loader:iter()()
function DataLoader:iter()
  local dataset = self.dataset
  local total = dataset:size()
  local total_batches = self.drop_last and math.floor(total / self.batch_size) or
    math.ceil(total / self.batch_size)

  local order = nil
  if self.shuffle and total > 0 then
    if self.seed ~= nil then
      order = seeded_order(total, self.seed)
    else
      -- torch.randperm yields 0-based values; dataset indices are 1-based.
      order = torch.randperm(total):add(1):totable()
    end
  end

  local num_workers = math.min(self.num_workers, total_batches)
  local iterator = setmetatable({
    loader = self,
    dataset = dataset,
    total = total,
    total_batches = total_batches,
    batch_size = self.batch_size,
    num_workers = num_workers,
    prefetch = self.prefetch,
    order = order,
    next_seq = 1,
    produce_seq = 1,
    ready = {},
    errors = {},
    first_error = nil,
    coroutines = {},
    started = {},
    worker_errors = {},
    closed = false,
  }, Iterator)

  local batch_size = self.batch_size
  for worker_id = 1, num_workers do
    iterator.coroutines[worker_id] = coroutine.create(function()
      return run_worker(worker_id, num_workers, dataset, total, total_batches,
        batch_size, order)
    end)
  end

  self._iterators[iterator] = true
  return iterator
end

-- Close every live iterator created by this loader. Iterators also close
-- themselves at the end of an epoch or on the first error.
function DataLoader:close()
  local active = {}
  for iterator in pairs(self._iterators) do
    active[#active + 1] = iterator
  end
  for _, iterator in ipairs(active) do
    pcall(iterator.close, iterator)
  end
end

function DataLoader:__tostring()
  return string.format(
    "DataLoader(%d samples, batch_size=%d, shuffle=%s, num_workers=%d, prefetch=%d)",
    self.dataset:size(), self.batch_size, tostring(self.shuffle),
    self.num_workers, self.prefetch)
end

return {
  Dataset = Dataset,
  TensorDataset = TensorDataset,
  DataLoader = DataLoader,
}
