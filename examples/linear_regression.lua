--
-- Fit y = w * x + b with autograd and a hand-written SGD loop.
--
--   luajit examples/linear_regression.lua
--
local root = (arg and arg[0] and arg[0]:match("^(.*)/examples/[^/]+$")) or "."
package.path = root .. "/lua/?.lua;" .. root .. "/lua/?/init.lua;" .. package.path

local torch = require("torch")

torch.manual_seed(7)

-- Synthetic data: y = 2x - 1 + noise.
local samples = 200
local x = torch.randn(samples, 1)
local true_w, true_b = 2.0, -1.0
local y = x * true_w + true_b + torch.randn(samples, 1) * 0.2

-- Scalars (0-dim tensors) are valid parameters.
local w = torch.tensor(0.0, { requires_grad = true })
local b = torch.tensor(0.0, { requires_grad = true })
local learning_rate = 0.2

print(string.format("target: y = %.2f x + %.2f", true_w, true_b))

for epoch = 1, 300 do
  local prediction = x * w + b
  local loss = (prediction - y):pow(2):mean()
  loss:backward()

  -- Parameter update must not be recorded by autograd.
  torch.no_grad(function()
    w:add_(-learning_rate * w:grad():item())
    b:add_(-learning_rate * b:grad():item())
  end)
  w:zero_grad()
  b:zero_grad()

  if epoch % 50 == 0 or epoch == 1 then
    print(string.format("epoch %3d  loss %.6f  w %.4f  b %.4f",
      epoch, loss:item(), w:item(), b:item()))
  end
end

print(string.format("learned: y = %.4f x + %.4f", w:item(), b:item()))

local error_w = math.abs(w:item() - true_w)
local error_b = math.abs(b:item() - true_b)
assert(error_w < 0.1, "w did not converge")
assert(error_b < 0.1, "b did not converge")
print("converged")
