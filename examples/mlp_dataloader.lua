--
-- Mini-batch training with TensorDataset and DataLoader.
--
--   luajit examples/mlp_dataloader.lua
--
local root = (arg and arg[0] and arg[0]:match("^(.*)/examples/[^/]+$")) or "."
package.path = root .. "/lua/?.lua;" .. root .. "/lua/?/init.lua;" .. package.path

local torch = require("torch")
local nn = require("torch.nn")
local optim = require("torch.optim")
local data = require("torch.utils.data")

torch.manual_seed(42)

-- Three Gaussian clusters ----------------------------------------------------
local per_class = 300
local centers = { { 2, 2 }, { -2, 2 }, { 0, -2.5 } }
local count = #centers * per_class

local inputs = torch.zeros(count, 2)
local labels = torch.LongTensor(count)

for class = 1, #centers do
  local first = (class - 1) * per_class + 1
  local last = class * per_class
  inputs:slice(1, first, last):copy_(
    torch.randn(per_class, 2):add(torch.tensor(centers[class])))
  labels:slice(1, first, last):fill_(class)
end

local dataset = data.TensorDataset(inputs, labels)
local loader = data.DataLoader(dataset, { batch_size = 64, shuffle = true })

print(string.format("dataset: %s", tostring(dataset)))
print(string.format("batches per epoch: %d", loader:num_batches()))

local model = nn.Sequential(
  nn.Linear(2, 32), nn.ReLU(),
  nn.Linear(32, 32), nn.ReLU(),
  nn.Linear(32, #centers))
local criterion = nn.CrossEntropyLoss()
local optimizer = optim.Adam(model:parameters(), { lr = 0.05 })
local scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.5)

local function evaluate()
  local predictions = model(inputs):argmax(2):reshape(-1)
  return predictions:eq(labels):to(torch.float32):mean():item()
end

for epoch = 1, 60 do
  local total_loss = 0
  local batches = 0
  for batch_inputs, batch_labels in loader:iter() do
    local loss = criterion(model(batch_inputs), batch_labels)
    optimizer:zero_grad()
    loss:backward()
    optimizer:step()
    total_loss = total_loss + loss:item()
    batches = batches + 1
  end
  scheduler:step()

  if epoch % 10 == 0 then
    print(string.format(
      "epoch %3d  loss %.5f  accuracy %.3f  lr %.4f",
      epoch, total_loss / batches, evaluate(), optimizer.lr))
  end
end

local final_accuracy = evaluate()
print(string.format("final accuracy: %.3f", final_accuracy))
assert(final_accuracy > 0.95, "mini-batch training did not converge")
print("converged")
