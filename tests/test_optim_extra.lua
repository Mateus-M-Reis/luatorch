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
