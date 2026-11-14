--
-- Fused forward+backward through the C++ autograd bridge.
--

local torch = require("torch")

local function expression(a, b, c)
  return ((a * b + c):sigmoid() + a:abs() + (b ^ 2) * 0.1):tanh()
end

test("fused train matches eager values and gradients", function()
  local fused = torch.fuse_train(expression)
  local values = { torch.randn(64), torch.randn(64), torch.randn(64) }
  local fused_args, eager_args = {}, {}
  for index = 1, 3 do
    fused_args[index] = values[index]:clone():requires_grad_(true)
    eager_args[index] = values[index]:clone():requires_grad_(true)
  end

  local fused_output = fused(fused_args[1], fused_args[2], fused_args[3])
  local eager_output = expression(eager_args[1], eager_args[2], eager_args[3])
  assert_true(next(fused.kernels) ~= nil, "expected a compiled train kernel")
  assert_true(fused_output:allclose(eager_output, 1e-5, 1e-6))
  assert_true(fused_output:requires_grad())

  local upstream = torch.randn(fused_output:sizes())
  fused_output:backward(upstream)
  eager_output:backward(upstream)

  for index = 1, 3 do
    local fused_grad = fused_args[index]:grad()
    local eager_grad = eager_args[index]:grad()
    assert_true(fused_grad ~= nil)
    assert_true(eager_grad ~= nil)
    assert_true(fused_grad:allclose(eager_grad, 1e-5, 1e-6))
  end
end)

test("fused train supports scalar arguments", function()
  local fused = torch.fuse_train(function(x)
    return (x * 2 + 1):relu()
  end)
  local input = (torch.rand(32) + 0.5):requires_grad_(true)
  local output = fused(input)
  output:sum():backward()
  assert_true(input:grad():allclose(torch.ones_like(input) * 2, 1e-6, 1e-6))
end)

test("fused train supports float64", function()
  local fused = torch.fuse_train(function(x)
    return (x * 3 - 1):tanh()
  end)
  local input = torch.randn(32, { dtype = torch.float64 }):requires_grad_(true)
  local output = fused(input)
  assert_true(output:dtype() == "float64")
  output:sum():backward()
  assert_true(input:grad() ~= nil)
  assert_true(input:grad():dtype() == "float64")
end)

test("fused train connects to nn modules", function()
  local nn = require("torch.nn")
  local layer = nn.Linear(4, 3)
  local fused = torch.fuse_train(function(x)
    return (x * 2 + 1):relu()
  end)
  local input = torch.randn(8, 4)
  local output = fused(layer(input))
  assert_true(output:requires_grad())
  local loss = (output * output):mean()
  loss:backward()
  assert_true(layer.weight:grad() ~= nil)
  assert_true(layer.bias:grad() ~= nil)
  local total = layer.weight:grad():abs():sum():item() +
    layer.bias:grad():abs():sum():item()
  assert_true(total > 0)
end)

test("fused train falls back for unsupported expressions", function()
  local fused = torch.fuse_train(function(x)
    return x:sum()
  end)
  local input = torch.randn(8):requires_grad_(true)
  local output = fused(input)
  assert_true(output:requires_grad())
  output:backward()
  assert_true(input:grad():allclose(torch.ones(8), 1e-6, 1e-6))
end)

test("fused train respects no_grad", function()
  local fused = torch.fuse_train(function(x)
    return (x * x):relu()
  end)
  local input = torch.randn(8):requires_grad_(true)
  local output
  torch.no_grad(function()
    output = fused(input)
  end)
  assert_true(not output:requires_grad())
end)
