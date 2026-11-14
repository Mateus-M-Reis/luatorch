--
-- Train a small CNN to distinguish horizontal from vertical bars.
--
--   luajit examples/cnn_classification.lua
--
local root = (arg and arg[0] and arg[0]:match("^(.*)/examples/[^/]+$")) or "."
package.path = root .. "/lua/?.lua;" .. root .. "/lua/?/init.lua;" .. package.path

local torch = require("torch")
local nn = require("torch.nn")
local optim = require("torch.optim")

torch.manual_seed(11)

local samples = 400
local size = 8
local inputs = torch.zeros(samples, 1, size, size)
local labels = torch.LongTensor(samples)

for sample = 1, samples do
  local horizontal = sample % 2 == 0
  local position = math.random(1, size)
  if horizontal then
    for column = 1, size do
      inputs:set(sample, 1, position, column, 1)
    end
    labels:set(sample, 1)
  else
    for row = 1, size do
      inputs:set(sample, 1, row, position, 1)
    end
    labels:set(sample, 2)
  end
end
-- A little noise so the task is not perfectly trivial.
inputs:add_(torch.randn(samples, 1, size, size):mul(0.1))

local model = nn.Sequential(
  nn.Conv2d(1, 8, 3, { padding = 1 }),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Conv2d(8, 16, 3, { padding = 1 }),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Flatten(),
  nn.Linear(16 * 2 * 2, 2))

local criterion = nn.CrossEntropyLoss()
local optimizer = optim.Adam(model:parameters(), { lr = 0.01 })

local function accuracy()
  local predictions = model(inputs):argmax(2):reshape(-1)
  return predictions:eq(labels):to(torch.float32):mean():item()
end

print(string.format("training on %d %dx%d images", samples, size, size))

for epoch = 1, 120 do
  local logits = model(inputs)
  local loss = criterion(logits, labels)

  optimizer:zero_grad()
  loss:backward()
  optimizer:step()

  if epoch % 20 == 0 then
    print(string.format("epoch %3d  loss %.5f  accuracy %.3f",
      epoch, loss:item(), accuracy()))
  end
end

local final_accuracy = accuracy()
print(string.format("final accuracy: %.3f", final_accuracy))
assert(final_accuracy > 0.95, "the CNN failed to learn the bars")
print("converged")
