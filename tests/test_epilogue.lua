--
-- Linear and convolution epilogue fusion.
--

local torch = require("torch")
local nn = require("torch.nn")

local activations = { "relu", "gelu", "sigmoid", "tanh" }

local function activate(tensor, name)
  if name == "relu" then return tensor:relu() end
  if name == "gelu" then return tensor:gelu() end
  if name == "sigmoid" then return tensor:sigmoid() end
  if name == "tanh" then return tensor:tanh() end
  error("unknown activation in test: " .. tostring(name))
end

local function tolerance(dtype)
  return dtype == torch.float64 and 1e-10 or 1e-5
end

local function clone_with_grad(tensor)
  return tensor:clone():requires_grad_(true)
end

local function tensor_dtype(dtype)
  return dtype == torch.float64 and "float64" or "float32"
end

test("fused linear epilogue matches eager ATen", function()
  for _, dtype in ipairs({ torch.float32, torch.float64 }) do
    local x = torch.randn(17, 9, { dtype = dtype })
    local weight = torch.randn(6, 9, { dtype = dtype })
    local bias = torch.randn(6, { dtype = dtype })
    local epsilon = tolerance(dtype)

    for _, name in ipairs(activations) do
      local expected = activate(bias:addmm(x, weight:t()), name)
      local fused = torch.fuse.linear(x, weight, bias, name)
      assert_true(fused:allclose(expected, epsilon, epsilon),
        "linear + " .. name .. " (" .. tensor_dtype(dtype) .. ")")
    end

    -- No bias is also part of the supported envelope.
    local expected = activate(x:matmul(weight:t()), "gelu")
    local fused = torch.fuse.linear(x, weight, nil, "gelu")
    assert_true(fused:allclose(expected, epsilon, epsilon), "linear without bias")
  end
end)

test("fused linear epilogue keeps autograd", function()
  for _, name in ipairs(activations) do
    local x_values = torch.randn(8, 5)
    local weight_values = torch.randn(3, 5)
    local bias_values = torch.randn(3)

    local x = clone_with_grad(x_values)
    local weight = clone_with_grad(weight_values)
    local bias = clone_with_grad(bias_values)
    local eager_x = clone_with_grad(x_values)
    local eager_weight = clone_with_grad(weight_values)
    local eager_bias = clone_with_grad(bias_values)

    local fused = torch.fuse.linear(x, weight, bias, name)
    local eager = activate(eager_bias:addmm(eager_x, eager_weight:t()), name)
    assert_true(fused:allclose(eager, 1e-5, 1e-5), "linear + " .. name)

    local upstream = torch.randn(fused:sizes())
    fused:backward(upstream)
    eager:backward(upstream)

    assert_true(x:grad():allclose(eager_x:grad(), 1e-5, 1e-5),
      "input gradient for " .. name)
    assert_true(weight:grad():allclose(eager_weight:grad(), 1e-5, 1e-5),
      "weight gradient for " .. name)
    assert_true(bias:grad():allclose(eager_bias:grad(), 1e-5, 1e-5),
      "bias gradient for " .. name)
  end
end)

test("fused linear epilogue falls back eagerly", function()
  local stats = torch.fuse.epilogue_stats
  local before_fused = stats.fused
  local before_fallback = stats.fallback

  local x = torch.randn(2, 3, 5)
  local weight = torch.randn(4, 5)
  local bias = torch.randn(4)
  local expected = bias:addmm(x:reshape(-1, 5), weight:t())
    :reshape(2, 3, 4):relu()
  local output = torch.fuse.linear(x, weight, bias, "relu")
  assert_true(output:allclose(expected, 1e-5, 1e-5), "rank-3 fallback")
  assert_eq(stats.fallback, before_fallback + 1, "rank-3 path fell back")

  local callable = nn.ReLU()
  local x2 = torch.randn(6, 5)
  local expected2 = bias:addmm(x2, weight:t()):relu()
  local output2 = torch.fuse.linear(x2, weight, bias, callable)
  assert_true(output2:allclose(expected2, 1e-5, 1e-5), "module fallback")
  assert_eq(stats.fallback, before_fallback + 2, "callable path fell back")

  local half_x = torch.randn(4, 5, { dtype = torch.float16 })
  local half_weight = torch.randn(4, 5, { dtype = torch.float16 })
  local half_bias = torch.randn(4, { dtype = torch.float16 })
  local half_output = torch.fuse.linear(
    half_x, half_weight, half_bias, "relu")
  assert_dtype(half_output, "float16")
  assert_eq(stats.fallback, before_fallback + 3, "float16 path fell back")

  assert_true(stats.fused >= before_fused)
  assert_error(function()
    torch.fuse.linear(x2, weight, bias, "not-an-activation")
  end, "unsupported epilogue activation")
end)

test("fused conv2d epilogue matches eager ATen", function()
  for _, dtype in ipairs({ torch.float32, torch.float64 }) do
    local x = torch.randn(2, 3, 8, 8, { dtype = dtype })
    local weight = torch.randn(5, 3, 3, 3, { dtype = dtype })
    local bias = torch.randn(5, { dtype = dtype })
    local epsilon = tolerance(dtype)

    for _, name in ipairs(activations) do
      local expected = activate(
        torch.conv2d(x, weight, bias, 2, 1, 1, 1), name)
      local fused = torch.fuse.conv2d(x, weight, bias, {
        activation = name,
        stride = 2,
        padding = 1,
      })
      assert_true(fused:allclose(expected, epsilon, epsilon),
        "conv2d + " .. name .. " (" .. tensor_dtype(dtype) .. ")")
    end
  end
end)

test("fused conv2d epilogue supports groups", function()
  local x = torch.randn(1, 4, 7, 7)
  local weight = torch.randn(6, 2, 3, 3)
  local bias = torch.randn(6)
  local expected = torch.conv2d(x, weight, bias, 1, 1, 1, 2):tanh()
  local fused = torch.fuse.conv2d(x, weight, bias, {
    activation = "tanh",
    padding = 1,
    groups = 2,
  })
  assert_true(fused:allclose(expected, 1e-5, 1e-5), "grouped conv2d")
end)

test("fused conv2d accepts the activation shorthand", function()
  local x = torch.randn(1, 2, 6, 6)
  local weight = torch.randn(3, 2, 3, 3)
  local bias = torch.randn(3)
  local expected = torch.conv2d(x, weight, bias, 1, 0, 1, 1):relu()
  local fused = torch.fuse.conv2d(x, weight, bias, "relu")
  assert_true(fused:allclose(expected, 1e-5, 1e-5), "conv2d shorthand")
end)

test("fused conv2d epilogue keeps autograd", function()
  local x_values = torch.randn(2, 3, 5, 5)
  local weight_values = torch.randn(4, 3, 3, 3)
  local bias_values = torch.randn(4)

  for _, name in ipairs(activations) do
    local x = clone_with_grad(x_values)
    local weight = clone_with_grad(weight_values)
    local bias = clone_with_grad(bias_values)
    local eager_x = clone_with_grad(x_values)
    local eager_weight = clone_with_grad(weight_values)
    local eager_bias = clone_with_grad(bias_values)

    local fused = torch.fuse.conv2d(x, weight, bias, {
      activation = name,
      padding = 1,
    })
    local eager = activate(
      torch.conv2d(eager_x, eager_weight, eager_bias, 1, 1, 1, 1), name)
    assert_true(fused:allclose(eager, 1e-5, 1e-5), "conv2d + " .. name)

    local upstream = torch.randn(fused:sizes())
    fused:backward(upstream)
    eager:backward(upstream)

    assert_true(x:grad():allclose(eager_x:grad(), 1e-5, 1e-5),
      "conv input gradient for " .. name)
    assert_true(weight:grad():allclose(eager_weight:grad(), 1e-5, 1e-5),
      "conv weight gradient for " .. name)
    assert_true(bias:grad():allclose(eager_bias:grad(), 1e-5, 1e-5),
      "conv bias gradient for " .. name)
  end
end)

test("fused conv2d epilogue falls back for callable activations", function()
  local stats = torch.fuse.epilogue_stats
  local before_fallback = stats.fallback
  local x = torch.randn(1, 2, 6, 6)
  local weight = torch.randn(3, 2, 3, 3)
  local bias = torch.randn(3)
  local expected = torch.conv2d(x, weight, bias, 1, 0, 1, 1):sigmoid()
  local output = torch.fuse.conv2d(x, weight, bias, {
    activation = function(value) return value:sigmoid() end,
  })
  assert_true(output:allclose(expected, 1e-5, 1e-5), "callable fallback")
  assert_eq(stats.fallback, before_fallback + 1, "callable conv fell back")
end)

test("nn.Linear and nn.Conv2d expose activation epilogues", function()
  local x = torch.randn(9, 7)
  local linear = nn.Linear(7, 5, { activation = "relu" })
  local expected_linear = linear.bias:addmm(x, linear.weight:t()):relu()
  assert_true(linear(x):allclose(expected_linear, 1e-5, 1e-5),
    "nn.Linear activation")

  local images = torch.randn(2, 3, 8, 8)
  local conv = nn.Conv2d(3, 4, 3, { padding = 1, activation = "gelu" })
  local expected_conv = torch.conv2d(
    images, conv.weight, conv.bias, 1, 1, 1, 1):gelu()
  assert_true(conv(images):allclose(expected_conv, 1e-5, 1e-5),
    "nn.Conv2d activation")

  local loss = linear(x):sum()
  loss:backward()
  assert_true(linear.weight:grad() ~= nil)
  assert_true(linear.bias:grad() ~= nil)

  local conv_loss = conv(images):sum()
  conv_loss:backward()
  assert_true(conv.weight:grad() ~= nil)
  assert_true(conv.bias:grad() ~= nil)
end)

test("GELU is available as a tensor op and module", function()
  local x = torch.randn(32)
  assert_true(x:gelu():allclose(torch.gelu(x), 1e-6, 1e-6))
  local module = nn.GELU()
  assert_true(module(x):allclose(x:gelu(), 1e-6, 1e-6))
  local functional = nn.functional.gelu(x)
  assert_true(functional:allclose(x:gelu(), 1e-6, 1e-6))
end)
