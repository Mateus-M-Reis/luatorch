--
-- Train a small MLP on three Gaussian clusters.
--
--   luajit examples/mlp_classification.lua
--
local root = (arg and arg[0] and arg[0]:match("^(.*)/examples/[^/]+$")) or "."
package.path = root .. "/lua/?.lua;" .. root .. "/lua/?/init.lua;" .. package.path

local torch = require("torch")
local nn = require("torch.nn")
local optim = require("torch.optim")

torch.manual_seed(42)

local per_class = 200
local centers = { { 2, 2 }, { -2, 2 }, { 0, -2.5 } }

local inputs = torch.zeros(#centers * per_class, 2)
local labels = torch.LongTensor(#centers * per_class)

for class = 1, #centers do
  local first = (class - 1) * per_class + 1
  local last = class * per_class
  local cluster = torch.randn(per_class, 2):add(torch.tensor(centers[class]))
  inputs:slice(1, first, last):copy_(cluster)
  labels:slice(1, first, last):fill_(class)
end

local model = nn.Sequential(
  nn.Linear(2, 32),
  nn.ReLU(),
  nn.Linear(32, 32),
  nn.ReLU(),
  nn.Linear(32, #centers))

local criterion = nn.CrossEntropyLoss()
local optimizer = optim.Adam(model:parameters(), { lr = 0.05 })

local function accuracy()
  local predictions = model(inputs):argmax(2):reshape(-1)
  return predictions:eq(labels):to(torch.float32):mean():item()
end

print(string.format("training on %d samples, %d classes", inputs:size(1), #centers))

for epoch = 1, 300 do
  local logits = model(inputs)
  local loss = criterion(logits, labels)

  optimizer:zero_grad()
  loss:backward()
  optimizer:step()

  if epoch % 50 == 0 then
    print(string.format("epoch %3d  loss %.5f  accuracy %.3f",
      epoch, loss:item(), accuracy()))
  end
end

local final_accuracy = accuracy()
print(string.format("final accuracy: %.3f", final_accuracy))
assert(final_accuracy > 0.95, "the MLP failed to separate the clusters")
print("converged")
