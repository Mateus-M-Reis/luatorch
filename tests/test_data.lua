local torch = require("torch")
local data = require("torch.utils.data")

test("TensorDataset basics", function()
  local inputs = torch.randn(10, 3)
  local targets = torch.LongTensor(10):fill_(1)
  local dataset = data.TensorDataset(inputs, targets)

  assert_eq(dataset:size(), 10)
  local sample_input, sample_target = dataset:get(4)
  assert_shape(sample_input, { 3 })
  assert_shape(sample_target, {})
  assert_true(sample_input:equal(inputs:select(1, 4)))
  assert_eq(sample_target:item(), 1)
end)

test("TensorDataset validates shapes", function()
  assert_error(function()
    data.TensorDataset(torch.zeros(5, 2), torch.zeros(4))
  end, "first dimension")
  assert_error(function()
    data.TensorDataset(torch.tensor(1))
  end, "batch dimension")
  assert_error(function()
    data.TensorDataset()
  end, "at least one")
end)

test("DataLoader batches", function()
  local inputs = torch.randn(10, 2)
  local targets = torch.LongTensor(10)
  for i = 1, 10 do
    targets:set(i, i)
  end
  local loader = data.DataLoader(data.TensorDataset(inputs, targets),
    { batch_size = 4 })

  local sizes = {}
  local totals = 0
  for batch_inputs, batch_targets in loader:iter() do
    sizes[#sizes + 1] = batch_inputs:size(1)
    assert_eq(batch_targets:dim(), 1)
    totals = totals + batch_targets:sum():item()
  end
  assert_eq(sizes, { 4, 4, 2 })
  assert_eq(totals, 55)
  assert_eq(loader:num_batches(), 3)
end)

test("DataLoader drop_last", function()
  local loader = data.DataLoader(
    data.TensorDataset(torch.randn(10, 2)), { batch_size = 4, drop_last = true })
  assert_eq(loader:num_batches(), 2)
  local seen = 0
  for batch in loader:iter() do
    seen = seen + batch:size(1)
  end
  assert_eq(seen, 8)
end)

test("DataLoader shuffles deterministically", function()
  local targets = torch.LongTensor(20)
  for i = 1, 20 do
    targets:set(i, i)
  end
  local dataset = data.TensorDataset(targets)

  torch.manual_seed(99)
  local first = data.DataLoader(dataset, { batch_size = 20, shuffle = true })
  torch.manual_seed(99)
  local second = data.DataLoader(dataset, { batch_size = 20, shuffle = true })

  torch.manual_seed(99)
  local first_batch = first:iter()()
  torch.manual_seed(99)
  local second_batch = second:iter()()
  assert_true(first_batch:equal(second_batch))

  local sequential = data.DataLoader(dataset, { batch_size = 20 }):iter()()
  assert_false(first_batch:equal(sequential))

  local seen = {}
  for batch in data.DataLoader(dataset, { batch_size = 3, shuffle = true }):iter() do
    for i = 1, batch:size(1) do
      local value = batch:get(i)
      assert_false(seen[value] or false, "duplicate sample " .. value)
      seen[value] = true
    end
  end
  for i = 1, 20 do
    assert_true(seen[i], "missing sample " .. i)
  end
end)

test("DataLoader works with custom datasets", function()
  local Dataset = data.Dataset
  local Custom = setmetatable({}, { __index = Dataset })
  Custom.__index = Custom

  function Custom.new(count)
    local self = setmetatable(Dataset.new(), Custom)
    self.count = count
    return self
  end

  function Custom:size()
    return self.count
  end

  function Custom:get(index)
    return torch.tensor({ index, index * 2 }), torch.tensor(index)
  end

  local loader = data.DataLoader(Custom.new(5), { batch_size = 2 })
  local batches = 0
  for inputs, targets in loader:iter() do
    batches = batches + 1
    assert_eq(inputs:size(2), 2)
    assert_eq(inputs:size(1), targets:size(1))
  end
  assert_eq(batches, 3)
end)

test("DataLoader validates arguments", function()
  assert_error(function()
    data.DataLoader({}, {})
  end, "Dataset")
  assert_error(function()
    data.DataLoader(data.TensorDataset(torch.zeros(2)), { batch_size = 0 })
  end, "batch_size")
end)

test("empty dataset produces no batches", function()
  local loader = data.DataLoader(data.TensorDataset(torch.zeros(0, 2)),
    { batch_size = 4 })
  assert_eq(loader:num_batches(), 0)
  local batches = 0
  for _ in loader:iter() do
    batches = batches + 1
  end
  assert_eq(batches, 0)
end)

test("torch.utils.data is lazily exposed", function()
  local torch = require("torch")
  assert_true(torch.utils ~= nil)
  assert_true(torch.utils.data.DataLoader ~= nil)
end)

test("TensorDataset tostring", function()
  local dataset = data.TensorDataset(torch.zeros(7, 3), torch.zeros(7))
  assert_true(tostring(dataset):find("7 samples", 1, true) ~= nil)
end)
