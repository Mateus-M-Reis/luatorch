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
  if type(self.batch_size) ~= "number" or self.batch_size < 1 or
    self.batch_size % 1 ~= 0 then
    error("DataLoader: batch_size must be a positive integer", 3)
  end
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

-- Returns an iterator over batches:
--   for inputs, targets in loader:iter() do ...
function DataLoader:iter()
  local dataset = self.dataset
  local total = dataset:size()
  local order = nil
  if self.shuffle then
    -- torch.randperm yields 0-based values; dataset indices are 1-based.
    order = torch.randperm(total):add(1):totable()
  end

  local position = 0
  local function next_batch()
    if position >= total then
      return nil
    end
    local size = math.min(self.batch_size, total - position)
    if size < self.batch_size and self.drop_last then
      position = total
      return nil
    end

    local indices = {}
    for i = 1, size do
      local index = position + i
      indices[i] = order and order[index] or index
    end
    position = position + size

    if dataset.batch then
      return dataset:batch(indices)
    end

    local columns = nil
    for i = 1, size do
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
    return unpack(result, 1, #result)
  end

  return next_batch
end

function DataLoader:__tostring()
  return string.format("DataLoader(%d samples, batch_size=%d, shuffle=%s)",
    self.dataset:size(), self.batch_size, tostring(self.shuffle))
end

return {
  Dataset = Dataset,
  TensorDataset = TensorDataset,
  DataLoader = DataLoader,
}
