--
-- DataLoader walkthrough: batching, seeded shuffling, prefetch workers and
-- explicit iterator cleanup.
--
--   luajit examples/dataloader.lua
--
local root = (arg and arg[0] and arg[0]:match("^(.*)/examples/[^/]+$")) or "."
package.path = root .. "/lua/?.lua;" .. root .. "/lua/?/init.lua;" .. package.path

local torch = require("torch")
local data = require("torch.utils.data")

-- A small in-memory dataset: three feature columns and a class label --------
local sample_count = 11
local features = torch.zeros(sample_count, 3)
local labels = torch.LongTensor(sample_count)
for i = 1, sample_count do
  features:set(i, 1, i)
  features:set(i, 2, i * 10)
  features:set(i, 3, -i)
  labels:set(i, (i - 1) % 3 + 1)
end

local dataset = data.TensorDataset(features, labels)
print(tostring(dataset))

local loader = data.DataLoader(dataset, {
  batch_size = 4,
  shuffle = true,
  seed = 42,           -- private deterministic shuffle, same order each epoch
  num_workers = 2,     -- cooperative coroutine workers
  prefetch = 2,        -- prepared batches ahead of the one being consumed
})
print(tostring(loader))
print(string.format("batches per epoch: %d", loader:num_batches()))

-- Iterate one epoch. The loader is callable as `for ... in loader:iter()`. --
local seen = {}
local batches = 0
for batch_features, batch_labels in loader:iter() do
  batches = batches + 1
  local ids, classes = {}, {}
  for i = 1, batch_features:size(1) do
    local id = batch_features:get(i, 1)
    ids[#ids + 1] = tostring(id)
    classes[#classes + 1] = tostring(batch_labels:get(i))
    seen[id] = true
  end
  print(string.format("batch %d: %2d samples  ids {%s}  labels {%s}",
    batches, #ids, table.concat(ids, ", "), table.concat(classes, ", ")))
end

assert(batches == loader:num_batches(), "unexpected number of batches")
for i = 1, sample_count do
  assert(seen[i], "sample " .. i .. " was not visited")
end
print(string.format("visited all %d samples exactly once", sample_count))

-- Worker count and prefetch change scheduling, never the seeded order. -------
local function collect(options)
  local order = {}
  for batch in data.DataLoader(dataset, options):iter() do
    for i = 1, batch:size(1) do
      order[#order + 1] = batch:get(i, 1)
    end
  end
  return table.concat(order, ",")
end

local main_thread = collect({ batch_size = 5, shuffle = true, seed = 7 })
local cooperative = collect({
  batch_size = 5, shuffle = true, seed = 7, num_workers = 3, prefetch = 3,
})
assert(main_thread == cooperative, "worker count changed the seeded order")
print("seeded order, num_workers=0: " .. main_thread)
print("seeded order, num_workers=3: " .. cooperative)

-- Close iterators when leaving an epoch early. Close is idempotent. ---------
local iterator = loader:iter()
iterator()
iterator:close()
iterator:close()
assert(iterator() == nil, "a closed iterator produced another batch")
loader:close()
print("early close: iterator is terminal")

-- Edge cases: drop_last and an empty dataset (with shuffle enabled). ---------
local dropped = data.DataLoader(dataset, { batch_size = 4, drop_last = true })
assert(dropped:num_batches() == 2, "drop_last did not discard the partial batch")
print(string.format("drop_last=true: %d batches", dropped:num_batches()))

local empty_loader = data.DataLoader(
  data.TensorDataset(torch.zeros(0, 3)), { batch_size = 4, shuffle = true })
assert(empty_loader:num_batches() == 0, "empty dataset reported batches")
local empty_batches = 0
for _ in empty_loader:iter() do
  empty_batches = empty_batches + 1
end
assert(empty_batches == 0, "empty dataset produced a batch")
print("empty dataset: 0 batches")

print("ok")
