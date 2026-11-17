local torch = require("torch")
local nn = require("torch.nn")
local optim = require("torch.optim")

local function minimize(OptimizerClass, options, steps)
  local parameter = torch.tensor({ 5, -3 }, { requires_grad = true })
  local optimizer = OptimizerClass({ parameter }, options)
  for _ = 1, steps do
    local loss = parameter:pow(2):sum()
    optimizer:zero_grad()
    loss:backward()
    optimizer:step()
  end
  return parameter:abs():max():item(), optimizer
end

test("SGD minimizes a quadratic", function()
  local maximum = minimize(optim.SGD, { lr = 0.1 }, 60)
  assert_true(maximum < 1e-3, "SGD did not converge: " .. maximum)
end)

test("SGD accepts a plain learning rate", function()
  local maximum = minimize(optim.SGD, 0.1, 60)
  assert_true(maximum < 1e-3)
end)

test("SGD with momentum keeps a buffer", function()
  local maximum, optimizer = minimize(
    optim.SGD, { lr = 0.02, momentum = 0.8 }, 300)
  assert_true(maximum < 1e-3, "momentum SGD did not converge: " .. maximum)
  local parameter = optimizer.params[1]
  assert_true(optimizer.state[parameter].momentum_buffer ~= nil)
end)

test("SGD validates nesterov options", function()
  assert_error(function()
    optim.SGD({ torch.zeros(2, { requires_grad = true }) }, { nesterov = true })
  end, "momentum")
end)

test("Adam minimizes a quadratic", function()
  local maximum = minimize(optim.Adam, { lr = 0.1 }, 200)
  assert_true(maximum < 1e-3, "Adam did not converge: " .. maximum)
end)

test("Adam keeps first and second moments", function()
  local parameter = torch.tensor({ 1, 2 }, { requires_grad = true })
  local optimizer = optim.Adam({ parameter }, { lr = 0.01 })
  parameter:pow(2):sum():backward()
  optimizer:step()
  local state = optimizer.state[parameter]
  assert_true(state.exp_avg ~= nil)
  assert_true(state.exp_avg_sq ~= nil)
  assert_eq(state.step, 1)
end)

test("RMSprop minimizes a quadratic", function()
  local maximum = minimize(
    optim.RMSprop, { lr = 0.02, alpha = 0.9 }, 400)
  assert_true(maximum < 0.1, "RMSprop did not converge: " .. maximum)
end)

test("Adagrad minimizes a quadratic", function()
  local maximum = minimize(optim.Adagrad, { lr = 0.5 }, 200)
  assert_true(maximum < 0.1, "Adagrad did not converge: " .. maximum)
end)

test("zero_grad clears gradients", function()
  local parameter = torch.tensor({ 1, 2 }, { requires_grad = true })
  parameter:pow(2):sum():backward()
  assert_true(parameter:grad() ~= nil)
  local optimizer = optim.SGD({ parameter }, { lr = 0.1 })
  optimizer:zero_grad()
  assert_tensor(parameter:grad(), { 0, 0 })
end)

test("step without gradients is a no-op", function()
  local parameter = torch.tensor({ 1, 2 })
  local optimizer = optim.SGD({ parameter }, { lr = 0.1 })
  optimizer:step()
  assert_tensor(parameter, { 1, 2 })
end)

test("weight decay shrinks parameters", function()
  local parameter = torch.tensor({ 1, 2 }, { requires_grad = true })
  local before = parameter:clone()
  local optimizer = optim.SGD({ parameter }, { lr = 0.1, weight_decay = 0.5 })
  -- Constant loss: the decay term adds 0.5 * parameter to the gradient.
  parameter:sum():backward()
  optimizer:step()
  assert_true(parameter:get(1) < before:get(1))
  assert_true(parameter:get(2) < before:get(2))
end)

test("optimizers expose parameters and state", function()
  local parameters = { torch.zeros(2, { requires_grad = true }) }
  local optimizer = optim.Adam(parameters, { lr = 0.01 })
  assert_eq(#optimizer.params, 1)
  assert_eq(optimizer:format(), "Adam(1 parameters)")
end)

test("XOR training end to end", function()
  torch.manual_seed(7)
  local input = torch.tensor({ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } })
  local target = torch.LongTensor({ 1, 2, 2, 1 })

  local model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Linear(8, 2))
  local optimizer = optim.Adam(model:parameters(), { lr = 0.1 })
  local criterion = nn.CrossEntropyLoss()

  local final_loss
  for _ = 1, 500 do
    local output = model(input)
    final_loss = criterion(output, target)
    optimizer:zero_grad()
    final_loss:backward()
    optimizer:step()
  end

  assert_true(final_loss:item() < 0.01,
    "XOR loss did not converge: " .. final_loss:item())

  local predictions = model(input):argmax(2):reshape(4)
  assert_tensor(predictions, { 1, 2, 2, 1 })
end)

test("parameter groups use independent learning rates", function()
  local first = torch.tensor({ 1, 1 }, { requires_grad = true })
  local second = torch.tensor({ 1, 1 }, { requires_grad = true })
  local optimizer = optim.SGD({
    { params = { first }, lr = 0.1 },
    { params = { second }, lr = 0.01 },
  }, { lr = 0.5 })
  assert_eq(#optimizer.param_groups, 2)
  assert_eq(#optimizer.params, 2)
  first:sum():backward()
  second:sum():backward()
  optimizer:step()
  assert_near(first:get(1), 0.9, 1e-6, "first group update")
  assert_near(second:get(1), 0.99, 1e-6, "second group update")
  assert_near(optimizer.lr, 0.1, 1e-6, "root lr follows first group")
end)

test("add_param_group appends a group and is trainable", function()
  local first = torch.tensor({ 2 }, { requires_grad = true })
  local second = torch.tensor({ 2 }, { requires_grad = true })
  local optimizer = optim.SGD({ first }, { lr = 0.1 })
  optimizer:add_param_group({ params = { second }, lr = 0.01 })
  assert_eq(#optimizer.param_groups, 2)
  assert_eq(#optimizer.params, 2)
  first:sum():backward()
  second:sum():backward()
  optimizer:step()
  assert_near(first:get(1), 1.9, 1e-6)
  assert_near(second:get(1), 1.99, 1e-6)
end)

test("learning-rate schedulers update every parameter group", function()
  local first = torch.tensor({ 1 }, { requires_grad = true })
  local second = torch.tensor({ 1 }, { requires_grad = true })
  local optimizer = optim.SGD({
    { params = { first }, lr = 0.2 },
    { params = { second }, lr = 0.02 },
  }, { lr = 0.2 })
  local scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.5)
  scheduler:step()
  assert_near(optimizer.param_groups[1].lr, 0.1, 1e-6)
  assert_near(optimizer.param_groups[2].lr, 0.01, 1e-6)
  assert_near(optimizer.lr, 0.1, 1e-6)
end)
