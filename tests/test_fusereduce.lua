--
-- Fused reductions: expression and reduction in one pass.
--

local torch = require("torch")

test("fused mean matches eager for an elementwise chain", function()
  local expression = function(a, b, c)
    return ((a - b) ^ 2 + c * 0.5):relu()
  end
  local fused = torch.fuse_reduce(expression, "mean")
  local a = torch.randn(256)
  local b = torch.randn(256)
  local c = torch.randn(256)
  local output = fused(a, b, c)
  assert_eq(output:sizes(), {})
  assert_true(next(fused.kernels) ~= nil, "expected a compiled reduce kernel")
  assert_true(output:allclose(expression(a, b, c):mean(), 1e-5, 1e-5))
end)

test("fused sum matches eager", function()
  local fused = torch.fuse_reduce(function(a, b, c)
    return a * b + c
  end, "sum")
  local a = torch.randn(1000)
  local b = torch.randn(1000)
  local c = torch.randn(1000)
  assert_true(fused(a, b, c):allclose((a * b + c):sum(), 1e-4, 1e-5))
end)

test("fused reductions support float64 and scalar arguments", function()
  local fused = torch.fuse_reduce(function(a)
    return (a * 2 + 1):relu()
  end, "mean")
  local input = torch.randn(128, { dtype = torch.float64 })
  local output = fused(input)
  assert_eq(output:dtype(), "float64")
  assert_true(output:allclose(((input * 2 + 1):relu()):mean(), 1e-5, 1e-6))
end)

test("fused mean reduction backpropagates through autograd", function()
  local fused = torch.fuse_reduce(function(a, b)
    return (a - b) ^ 2
  end, "mean")
  local a = torch.randn(64):requires_grad_(true)
  local b = torch.randn(64):requires_grad_(true)
  local eager_a = a:detach():clone():requires_grad_(true)
  local eager_b = b:detach():clone():requires_grad_(true)

  local output = fused(a, b)
  assert_true(output:requires_grad())
  assert_true(next(fused.train_kernels) ~= nil,
    "expected a compiled reduce backward kernel")
  output:backward()

  local reference = ((eager_a - eager_b) ^ 2):mean()
  reference:backward()
  assert_true(a:grad():allclose(eager_a:grad(), 1e-5, 1e-6))
  assert_true(b:grad():allclose(eager_b:grad(), 1e-5, 1e-6))
end)

test("fused sum reduction backpropagates", function()
  local fused = torch.fuse_reduce(function(a, b)
    return a * b
  end, "sum")
  local a = torch.randn(32):requires_grad_(true)
  local b = torch.randn(32):requires_grad_(true)
  local eager_a = a:detach():clone():requires_grad_(true)
  local eager_b = b:detach():clone():requires_grad_(true)
  local fused_loss = fused(a, b)
  fused_loss:backward()
  local eager_loss = (eager_a * eager_b):sum()
  eager_loss:backward()
  assert_true(a:grad():allclose(eager_a:grad(), 1e-5, 1e-6))
  assert_true(b:grad():allclose(eager_b:grad(), 1e-5, 1e-6))
end)

test("fused reductions fall back for unsupported expressions", function()
  local fused = torch.fuse_reduce(function(a)
    return a:sum()
  end, "sum")
  local input = torch.randn(8)
  assert_true(fused(input):allclose(input:sum(), 1e-6, 1e-6))
end)
