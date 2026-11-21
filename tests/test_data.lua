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

test("DataLoader prefetch prepares ahead and preserves order", function()
  local Dataset = data.Dataset
  local Recording = setmetatable({}, { __index = Dataset })
  Recording.__index = Recording

  function Recording.new(count)
    local self = setmetatable(Dataset.new(), Recording)
    self.count = count
    self.gets = 0
    return self
  end

  function Recording:size()
    return self.count
  end

  function Recording:get(index)
    self.gets = self.gets + 1
    return torch.tensor(index)
  end

  local dataset = Recording.new(8)
  local loader = data.DataLoader(dataset,
    { batch_size = 1, num_workers = 2, prefetch = 2 })
  local iterator = loader:iter()
  local seen = {}
  local consumed = 0

  for batch in iterator do
    consumed = consumed + 1
    if consumed == 1 then
      assert_true(dataset.gets >= 3, "prefetch did not prepare ahead")
    end
    assert_true(dataset.gets - consumed <= 2,
      "prefetch exceeded its configured bound")
    seen[#seen + 1] = batch:get(1)
  end

  assert_eq(seen, { 1, 2, 3, 4, 5, 6, 7, 8 })
  assert_eq(dataset.gets, 8)
  assert_eq(next(loader._iterators), nil)
end)

test("DataLoader prefetch=0 does not look ahead", function()
  local Dataset = data.Dataset
  local Counting = setmetatable({}, { __index = Dataset })
  Counting.__index = Counting

  function Counting.new(count)
    local self = setmetatable(Dataset.new(), Counting)
    self.count = count
    self.gets = 0
    return self
  end

  function Counting:size()
    return self.count
  end

  function Counting:get(index)
    self.gets = self.gets + 1
    return torch.tensor(index)
  end

  local dataset = Counting.new(3)
  local iterator = data.DataLoader(dataset,
    { batch_size = 1, num_workers = 2, prefetch = 0 }):iter()
  assert_eq(iterator():get(1), 1)
  assert_eq(dataset.gets, 1)
  assert_eq(iterator():get(1), 2)
  assert_eq(dataset.gets, 2)
  assert_eq(iterator():get(1), 3)
  assert_eq(iterator(), nil)
end)

test("DataLoader seeded shuffle is deterministic with workers", function()
  local targets = torch.LongTensor(23)
  for i = 1, 23 do
    targets:set(i, i)
  end
  local dataset = data.TensorDataset(targets)

  local function collect(options)
    local values = {}
    for batch in data.DataLoader(dataset, options):iter() do
      for i = 1, batch:size(1) do
        values[#values + 1] = batch:get(i)
      end
    end
    return values
  end

  local expected = collect({ batch_size = 5, shuffle = true, seed = 123 })
  local with_workers = collect({
    batch_size = 5, shuffle = true, seed = 123,
    num_workers = 3, prefetch = 2,
  })
  assert_eq(with_workers, expected)

  local different = collect({ batch_size = 5, shuffle = true, seed = 124 })
  assert_true(table.concat(different, ",") ~= table.concat(expected, ","),
    "different seeds produced the same order")

  local seen = {}
  for _, value in ipairs(with_workers) do
    assert_false(seen[value] or false, "duplicate sample " .. value)
    seen[value] = true
  end
  for i = 1, 23 do
    assert_true(seen[i], "missing sample " .. i)
  end

  local loader = data.DataLoader(dataset, {
    batch_size = 5, shuffle = true, seed = 123, num_workers = 2,
  })
  assert_eq(collect(loader), expected)
  assert_eq(collect(loader), expected)

  -- A loader seed must not consume or depend on the global torch RNG.
  torch.manual_seed(31337)
  local baseline = torch.rand(4):totable()
  torch.manual_seed(31337)
  collect(loader)
  assert_eq(torch.rand(4):totable(), baseline)
end)

test("DataLoader exhausted epochs release workers", function()
  local targets = torch.LongTensor(7)
  for i = 1, 7 do
    targets:set(i, i)
  end
  local loader = data.DataLoader(data.TensorDataset(targets), {
    batch_size = 3, shuffle = true, seed = 5, num_workers = 3, prefetch = 3,
  })

  for _ = 1, 20 do
    local iterator = loader:iter()
    local count = 0
    for batch in iterator do
      count = count + batch:size(1)
    end
    assert_eq(count, 7)
    assert_eq(iterator(), nil)
    assert_eq(iterator(), nil)
    assert_eq(next(loader._iterators), nil)
    collectgarbage()
  end

  local first = loader:iter()
  local second = loader:iter()
  first()
  second()
  loader:close()
  assert_eq(first(), nil)
  assert_eq(second(), nil)
  assert_eq(next(loader._iterators), nil)
end)

test("DataLoader propagates worker errors and closes the epoch", function()
  local Dataset = data.Dataset
  local Boom = setmetatable({}, { __index = Dataset })
  Boom.__index = Boom

  function Boom.new(failing_index)
    local self = setmetatable(Dataset.new(), Boom)
    self.failing_index = failing_index
    return self
  end

  function Boom:size()
    return 4
  end

  function Boom:get(index)
    if index == self.failing_index then
      error("boom from dataset", 0)
    end
    return torch.tensor(index)
  end

  local loader = data.DataLoader(Boom.new(2),
    { batch_size = 1, num_workers = 2, prefetch = 2 })
  local iterator = loader:iter()
  assert_eq(iterator():get(1), 1)

  local ok, err = pcall(iterator)
  assert_false(ok)
  assert_true(tostring(err):find("boom from dataset", 1, true) ~= nil,
    "worker error was not preserved")
  assert_true(tostring(err):find("worker", 1, true) ~= nil,
    "worker error lacked context")
  assert_eq(iterator(), nil)
  assert_eq(next(loader._iterators), nil)

  local first_error = data.DataLoader(Boom.new(1),
    { batch_size = 1, num_workers = 2, prefetch = 2 }):iter()
  local first_ok, first_err = pcall(first_error)
  assert_false(first_ok)
  assert_true(tostring(first_err):find("boom from dataset", 1, true) ~= nil)
  assert_eq(first_error(), nil)
end)

test("DataLoader defers main-thread prefetch errors to their batch", function()
  local Dataset = data.Dataset
  local Failing = setmetatable({}, { __index = Dataset })
  Failing.__index = Failing

  function Failing.new(failing_index)
    local self = setmetatable(Dataset.new(), Failing)
    self.failing_index = failing_index
    return self
  end

  function Failing:size()
    return 6
  end

  function Failing:get(index)
    if index == self.failing_index then
      error("main-thread boom", 0)
    end
    return torch.tensor(index)
  end

  -- A prefetch error on batch 2 must not discard the already prepared batch 1.
  local loader = data.DataLoader(Failing.new(2),
    { batch_size = 1, num_workers = 0, prefetch = 3 })
  local iterator = loader:iter()
  assert_eq(iterator():get(1), 1)

  local ok, err = pcall(iterator)
  assert_false(ok)
  assert_true(tostring(err):find("failed while preparing batch 2", 1, true) ~= nil,
    "main-thread error lacked the batch number")
  assert_true(tostring(err):find("main-thread boom", 1, true) ~= nil,
    "main-thread error was not preserved")
  assert_eq(iterator(), nil)
  assert_eq(next(loader._iterators), nil)
end)

test("DataLoader shuffles empty datasets without crashing", function()
  local dataset = data.TensorDataset(torch.zeros(0, 2))
  local configs = {
    { batch_size = 4, shuffle = true },
    { batch_size = 4, shuffle = true, drop_last = true },
    { batch_size = 4, shuffle = true, seed = 11 },
    { batch_size = 4, shuffle = true, num_workers = 3, prefetch = 3 },
  }

  for _, options in ipairs(configs) do
    local loader = data.DataLoader(dataset, options)
    assert_eq(loader:num_batches(), 0)
    local iterator = loader:iter()
    local batches = 0
    for _ in iterator do
      batches = batches + 1
    end
    assert_eq(batches, 0)
    assert_eq(iterator(), nil)
    assert_eq(iterator(), nil)
    assert_eq(next(loader._iterators), nil)
    loader:close()
  end
end)

test("DataLoader validates arguments", function()
  assert_error(function()
    data.DataLoader({}, {})
  end, "Dataset")
  assert_error(function()
    data.DataLoader(data.TensorDataset(torch.zeros(2)), { batch_size = 0 })
  end, "batch_size")
  assert_error(function()
    data.DataLoader(data.TensorDataset(torch.zeros(2)), { num_workers = -1 })
  end, "num_workers")
  assert_error(function()
    data.DataLoader(data.TensorDataset(torch.zeros(2)), { num_workers = 1.5 })
  end, "num_workers")
  assert_error(function()
    data.DataLoader(data.TensorDataset(torch.zeros(2)), { prefetch = -1 })
  end, "prefetch")
  assert_error(function()
    data.DataLoader(data.TensorDataset(torch.zeros(2)), { prefetch = 1.5 })
  end, "prefetch")
  assert_error(function()
    data.DataLoader(data.TensorDataset(torch.zeros(2)), { seed = 1.5 })
  end, "seed")
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
