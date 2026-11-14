local torch = require("torch")
local nn = require("torch.nn")
local optim = require("torch.optim")

test("AdamW minimizes a quadratic", function()
  local parameter = torch.tensor({ 5, -3 }, { requires_grad = true })
  local optimizer = optim.AdamW({ parameter }, { lr = 0.1 })
  for _ = 1, 200 do
    local loss = parameter:pow(2):sum()
    optimizer:zero_grad()
    loss:backward()
    optimizer:step()
  end
  assert_true(parameter:abs():max():item() < 1e-3)
end)

test("AdamW decouples weight decay", function()
  local parameter = torch.tensor({ 1, 2 }, { requires_grad = true })
  local optimizer = optim.AdamW({ parameter }, { lr = 0.1, weight_decay = 0.5 })
  -- Zero gradient: only the decoupled decay should shrink the parameter.
  parameter:mul(0):sum():backward()
  optimizer:step()
  assert_near(parameter:totable(), { 0.95, 1.9 }, 1e-6)
end)

test("clip_grad_norm_ scales gradients", function()
  local parameter = torch.tensor({ 3, 4 }, { requires_grad = true })
  parameter:sum():backward()
  local total = optim.clip_grad_norm_({ parameter }, 1)
  assert_near(total, math.sqrt(2), 1e-6)
  assert_near(parameter:grad():totable(),
    { 1 / math.sqrt(2), 1 / math.sqrt(2) }, 1e-6)
end)

test("clip_grad_norm_ handles infinity norm", function()
  local parameter = torch.tensor({ 3, 4 }, { requires_grad = true })
  parameter:sum():backward()
  local total = optim.clip_grad_norm_({ parameter }, 1, math.huge)
  assert_near(total, 1, 1e-6)
  assert_near(parameter:grad():totable(), { 1, 1 }, 1e-5)
end)

test("clip_grad_value_ clamps gradients", function()
  local parameter = torch.tensor({ 1, 2, 3 }, { requires_grad = true })
  local weights = torch.tensor({ 5, -5, 1 })
  parameter:mul(weights):sum():backward()
  optim.clip_grad_value_({ parameter }, 1)
  assert_near(parameter:grad():totable(), { 1, -1, 1 }, 1e-6)
end)

test("StepLR", function()
  local optimizer = optim.SGD({ torch.zeros(1, { requires_grad = true }) },
    { lr = 1 })
  local scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 0.5)
  scheduler:step()
  assert_near(optimizer.lr, 1)
  scheduler:step()
  assert_near(optimizer.lr, 0.5)
  scheduler:step()
  assert_near(optimizer.lr, 0.5)
  scheduler:step()
  assert_near(optimizer.lr, 0.25)
end)

test("MultiStepLR", function()
  local optimizer = optim.SGD({ torch.zeros(1, { requires_grad = true }) },
    { lr = 1 })
  local scheduler = optim.lr_scheduler.MultiStepLR(optimizer, { 2, 5 }, 0.1)
  for _ = 1, 2 do scheduler:step() end
  assert_near(optimizer.lr, 0.1)
  for _ = 3, 5 do scheduler:step() end
  assert_near(optimizer.lr, 0.01)
end)

test("ExponentialLR", function()
  local optimizer = optim.SGD({ torch.zeros(1, { requires_grad = true }) },
    { lr = 1 })
  local scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
  scheduler:step()
  assert_near(optimizer.lr, 0.5)
  scheduler:step()
  assert_near(optimizer.lr, 0.25)
end)

test("CosineAnnealingLR", function()
  local optimizer = optim.SGD({ torch.zeros(1, { requires_grad = true }) },
    { lr = 1 })
  local scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 4, 0)
  local minimum = 1
  for _ = 1, 4 do
    scheduler:step()
    minimum = math.min(minimum, optimizer.lr)
  end
  assert_near(optimizer.lr, 0, 1e-9)
  assert_true(minimum >= 0)
end)

test("LambdaLR", function()
  local optimizer = optim.SGD({ torch.zeros(1, { requires_grad = true }) },
    { lr = 2 })
  local scheduler = optim.lr_scheduler.LambdaLR(optimizer,
    function(epoch) return 1 / epoch end)
  scheduler:step()
  assert_near(optimizer.lr, 2)
  scheduler:step()
  assert_near(optimizer.lr, 1)
  scheduler:step()
  assert_near(optimizer.lr, 2 / 3)
end)

test("scheduler toString", function()
  local optimizer = optim.SGD({ torch.zeros(1, { requires_grad = true }) },
    { lr = 0.1 })
  local scheduler = optim.lr_scheduler.StepLR(optimizer, 3, 0.5)
  assert_true(tostring(scheduler):find("StepLR", 1, true) ~= nil)
end)

test("fused Adam matches a reference implementation", function()
  local lr, beta1, beta2, eps = 0.01, 0.9, 0.999, 1e-8
  local parameter = torch.tensor({ 1.0, -2.0, 0.5 }, { requires_grad = true })
  local reference = parameter:detach():clone()
  local first_moment = torch.zeros_like(reference)
  local second_moment = torch.zeros_like(reference)
  local optimizer = optim.Adam({ parameter },
    { lr = lr, betas = { beta1, beta2 }, eps = eps })

  for step = 1, 5 do
    local loss = (parameter * parameter):sum()
    optimizer:zero_grad()
    loss:backward()
    optimizer:step()

    local gradient = reference * 2
    first_moment = first_moment * beta1 + gradient * (1 - beta1)
    second_moment = second_moment * beta2 +
      (gradient * gradient) * (1 - beta2)
    local corrected_first = first_moment / (1 - beta1 ^ step)
    local corrected_second = second_moment / (1 - beta2 ^ step)
    reference = reference -
      (corrected_first / (corrected_second:sqrt() + eps)) * lr
  end

  assert_true(parameter:allclose(reference, 1e-5, 1e-6))
  local state = optimizer.state[parameter]
  assert_true(state.exp_avg:allclose(first_moment, 1e-5, 1e-6))
  assert_true(state.exp_avg_sq:allclose(second_moment, 1e-5, 1e-6))
end)

test("fused SGD momentum and nesterov match the equations", function()
  local parameter = torch.tensor({ 1.0, 2.0 }, { requires_grad = true })
  local optimizer = optim.SGD({ parameter },
    { lr = 0.1, momentum = 0.9, nesterov = true })

  parameter:sum():backward()
  optimizer:step()
  -- buffer = grad = 1; update = grad + momentum * buffer = 1.9
  assert_near(parameter:totable(), { 0.81, 1.81 }, 1e-6)
  local state = optimizer.state[parameter]
  assert_near(state.momentum_buffer:totable(), { 1, 1 }, 1e-6)

  optimizer:zero_grad()
  parameter:sum():backward()
  optimizer:step()
  -- buffer = 0.9 * 1 + 1 * 1 = 1.9; update = 1 + 0.9 * 1.9 = 2.71
  assert_near(parameter:totable(), { 0.539, 1.539 }, 1e-6)
end)

test("fused SGD dampening", function()
  local parameter = torch.tensor({ 1.0 }, { requires_grad = true })
  local optimizer = optim.SGD({ parameter },
    { lr = 0.1, momentum = 0.9, dampening = 0.5 })
  parameter:sum():backward()
  optimizer:step()
  assert_near(parameter:totable()[1], 0.9, 1e-6)
  optimizer:zero_grad()
  parameter:sum():backward()
  optimizer:step()
  -- buffer = 0.9 * 1 + 0.5 * 1 = 1.4
  assert_near(parameter:totable()[1], 0.76, 1e-6)
end)

test("fused optimizers reject non-contiguous parameters", function()
  local parameter = torch.zeros(2, 2):t()
  parameter:requires_grad_(true)
  local optimizer = optim.SGD({ parameter }, { lr = 0.1 })
  parameter:sum():backward()
  assert_error(function()
    optimizer:step()
  end, "contiguous")
end)

test("fused RMSprop matches a reference implementation", function()
  local lr, alpha, eps = 0.01, 0.9, 1e-8
  local momentum = 0.5
  local parameter = torch.tensor({ 1.0, -2.0 }, { requires_grad = true })
  local reference = parameter:detach():clone()
  local square_avg = torch.zeros_like(reference)
  local grad_avg = torch.zeros_like(reference)
  local buffer = torch.zeros_like(reference)
  local optimizer = optim.RMSprop({ parameter }, {
    lr = lr, alpha = alpha, eps = eps,
    momentum = momentum, centered = true,
  })

  for _ = 1, 4 do
    optimizer:zero_grad()
    parameter:sum():backward()
    optimizer:step()

    local gradient = torch.ones_like(reference)
    square_avg = square_avg * alpha + gradient * gradient * (1 - alpha)
    grad_avg = grad_avg * alpha + gradient * (1 - alpha)
    local average = square_avg - grad_avg * grad_avg
    local denominator = average:sqrt() + eps
    buffer = buffer * momentum + gradient / denominator
    reference = reference - buffer * lr
  end

  assert_true(parameter:allclose(reference, 1e-5, 1e-6))
  local state = optimizer.state[parameter]
  assert_true(state.square_avg:allclose(square_avg, 1e-5, 1e-6))
  assert_true(state.momentum_buffer:allclose(buffer, 1e-5, 1e-6))
end)

test("fused Adagrad matches a reference implementation", function()
  local lr, lr_decay, eps = 0.1, 0.01, 1e-10
  local parameter = torch.tensor({ 1.0, -2.0 }, { requires_grad = true })
  local reference = parameter:detach():clone()
  local sum = torch.zeros_like(reference)
  local optimizer = optim.Adagrad({ parameter }, {
    lr = lr, lr_decay = lr_decay, eps = eps,
  })

  for step = 1, 4 do
    optimizer:zero_grad()
    parameter:sum():backward()
    optimizer:step()

    local gradient = torch.ones_like(reference)
    sum = sum + gradient * gradient
    local learning_rate = lr / (1 + (step - 1) * lr_decay)
    reference = reference - (gradient / (sum:sqrt() + eps)) * learning_rate
  end

  assert_true(parameter:allclose(reference, 1e-5, 1e-6))
  assert_true(optimizer.state[parameter].sum:allclose(sum, 1e-5, 1e-6))
end)
