local torch = require("torch")
local nn = require("torch.nn")

test("Linear forward, parameters and gradients", function()
  local layer = nn.Linear(3, 2)
  assert_eq(#layer:parameters(), 2)
  assert_true(layer.weight:requires_grad())
  assert_true(layer.bias:requires_grad())
  assert_shape(layer.weight, { 2, 3 })
  assert_shape(layer.bias, { 2 })

  local input = torch.randn(4, 3)
  local output = layer(input)
  assert_shape(output, { 4, 2 })
  assert_true(output:requires_grad())

  output:sum():backward()
  assert_shape(layer.weight:grad(), { 2, 3 })
  assert_shape(layer.bias:grad(), { 2 })
end)

test("Linear without bias", function()
  local layer = nn.Linear(2, 5, false)
  assert_eq(#layer:parameters(), 1)
  assert_eq(layer.bias, nil)
  assert_shape(layer(torch.zeros(3, 2)), { 3, 5 })
end)

test("Linear handles 1-d input", function()
  local layer = nn.Linear(3, 2)
  assert_shape(layer(torch.zeros(3)), { 2 })
end)

test("Sequential chains modules and names parameters", function()
  local model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1))
  assert_eq(#model:parameters(), 4)

  local named = model:named_parameters()
  assert_true(named["1.weight"] ~= nil)
  assert_true(named["3.bias"] ~= nil)

  local output = model(torch.randn(5, 2))
  assert_shape(output, { 5, 1 })

  local loss = output:sum()
  loss:backward()
  assert_true(named["1.weight"]:grad() ~= nil)
  assert_true(named["3.weight"]:grad() ~= nil)
end)

test("train and eval propagate", function()
  local model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
  assert_true(model.training)
  model:eval()
  assert_false(model.training)
  assert_false(model._modules["2"].training)
  model:train()
  assert_true(model.training)
  assert_true(model._modules["2"].training)
end)

test("activation modules", function()
  local values = torch.tensor({ -1, 0, 2 })
  assert_tensor(nn.ReLU()(values), { 0, 0, 2 })
  assert_near(nn.Sigmoid()(torch.tensor(0)):item(), 0.5)
  assert_near(nn.Tanh()(torch.tensor(0)):item(), 0)
  assert_tensor(nn.LeakyReLU(0.1)(values), { -0.1, 0, 2 })

  local probabilities = nn.Softmax(2)(torch.tensor({ { 1, 2, 3 }, { 1, 1, 1 } }))
  assert_shape(probabilities, { 2, 3 })
  assert_near(probabilities:sum(2):totable(), { 1, 1 }, 1e-6)

  local log_probabilities = nn.LogSoftmax(2)(torch.zeros(2, 3))
  assert_near(math.exp(log_probabilities:get(1, 1)), 1 / 3, 1e-6)
end)

test("Flatten", function()
  local flatten = nn.Flatten()
  assert_shape(flatten(torch.zeros(2, 3, 4)), { 2, 12 })
  assert_shape(nn.Flatten(2, 3)(torch.zeros(2, 3, 4)), { 2, 12 })
end)

test("MSE and L1 losses", function()
  local input = torch.tensor({ 1, 2 })
  local target = torch.tensor({ 1, 1 })
  assert_near(nn.MSELoss()(input, target):item(), 0.5)
  assert_near(nn.MSELoss("sum")(input, target):item(), 1)
  assert_shape(nn.MSELoss("none")(input, target), { 2 })
  assert_near(nn.L1Loss()(input, target):item(), 0.5)
end)

test("cross entropy uses 1-based class labels", function()
  local logits = torch.zeros(2, 3)
  local target = torch.LongTensor({ 1, 2 })
  local loss = nn.CrossEntropyLoss()(logits, target)
  assert_near(loss:item(), math.log(3), 1e-6)

  local log_probabilities = nn.LogSoftmax(2)(logits)
  assert_near(nn.NLLLoss()(log_probabilities, target):item(), math.log(3), 1e-6)

  assert_error(function()
    nn.CrossEntropyLoss()(logits, torch.LongTensor({ 4, 1 }))
  end)
end)

test("binary cross entropy losses", function()
  local input = torch.zeros(2)
  local target = torch.tensor({ 0, 1 })
  assert_near(nn.BCEWithLogitsLoss()(input, target):item(), math.log(2), 1e-6)
  assert_near(nn.BCELoss()(torch.tensor({ 0.5, 0.5 }), target):item(),
    math.log(2), 1e-6)
end)

test("Conv2d computes a known convolution", function()
  local conv = nn.Conv2d(1, 1, 2, { bias = false })
  torch.no_grad(function()
    conv.weight:fill_(1)
  end)
  local input = torch.tensor({ { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } } })
  local output = conv(input)
  assert_shape(output, { 1, 1, 2, 2 })
  assert_tensor(output, { { { { 12, 16 }, { 24, 28 } } } })

  local padded = nn.Conv2d(1, 1, 2, { padding = 1, bias = false })
  torch.no_grad(function()
    padded.weight:fill_(1)
  end)
  assert_shape(padded(input), { 1, 1, 4, 4 })
end)

test("Conv2d parameters and gradients", function()
  local conv = nn.Conv2d(3, 4, 3, { stride = 2, padding = 1 })
  assert_eq(#conv:parameters(), 2)
  assert_shape(conv.weight, { 4, 3, 3, 3 })
  assert_shape(conv.bias, { 4 })

  local output = conv(torch.randn(2, 3, 8, 8))
  assert_shape(output, { 2, 4, 4, 4 })
  output:sum():backward()
  assert_shape(conv.weight:grad(), { 4, 3, 3, 3 })
end)

test("MaxPool2d", function()
  local pool = nn.MaxPool2d(2)
  local input = torch.arange(1, 17):reshape(1, 1, 4, 4):to(torch.float32)
  local output = pool(input)
  assert_tensor(output, { { { { 6, 8 }, { 14, 16 } } } })
  assert_shape(output, { 1, 1, 2, 2 })
end)

test("AdaptiveAvgPool2d", function()
  local pool = nn.AdaptiveAvgPool2d(1)
  assert_tensor(pool(torch.ones(2, 3, 4, 5)), {
    { { { 1 } }, { { 1 } }, { { 1 } } },
    { { { 1 } }, { { 1 } }, { { 1 } } },
  })
end)

test("BatchNorm2d normalizes during training", function()
  local normalization = nn.BatchNorm2d(2)
  local input = torch.randn(4, 2, 3, 3):mul(2):add(5)
  local output = normalization(input)
  assert_shape(output, { 4, 2, 3, 3 })

  local mean = output:mean(4, true):mean(3, true):mean(1, true)
  assert_true(mean:abs():max():item() < 1e-4)
  local second = output:pow(2):mean(4, true):mean(3, true):mean(1, true)
  assert_true((second:sub(1)):abs():max():item() < 1e-3)

  assert_true(normalization.running_mean:abs():max():item() > 0)
end)

test("BatchNorm2d eval uses running statistics", function()
  local normalization = nn.BatchNorm2d(1)
  normalization.running_mean:fill_(10)
  normalization.running_var:fill_(4)
  normalization:eval()
  local input = torch.tensor({
    { { { 8, 12 }, { 8, 12 } } },
    { { { 8, 12 }, { 8, 12 } } },
  })
  local output = normalization(input)
  assert_near(output:mean():item(), 0, 1e-4)
  assert_near(output:var(nil, false):item(), 1, 1e-3)
end)

test("Dropout behaviour", function()
  local dropout = nn.Dropout(0.5)
  local input = torch.ones(100, 100)
  dropout:eval()
  assert_true(dropout(input):equal(input))

  dropout:train()
  local output = dropout(input)
  assert_false(output:equal(input))
  local zeros = output:eq(0):sum():item()
  local ratio = zeros / input:numel()
  assert_true(ratio > 0.45 and ratio < 0.55, "unexpected dropout ratio " .. ratio)
  assert_near(output:max():item(), 2, 1e-6)

  local identity = nn.Dropout(0)
  assert_true(identity(input):equal(input))
end)

test("state_dict roundtrip", function()
  local model = nn.Linear(2, 3)
  local state = model:state_dict()
  local original_weight = model.weight:clone()
  local original_bias = model.bias:clone()

  torch.no_grad(function()
    model.weight:fill_(0)
    model.bias:fill_(0)
  end)
  model:load_state_dict(state)

  assert_true(model.weight:equal(original_weight))
  assert_true(model.bias:equal(original_bias))
  assert_error(function()
    model:load_state_dict({})
  end, "missing key")
  assert_error(function()
    model:load_state_dict({ weight = state.weight, bias = state.bias, extra = state.weight })
  end, "unexpected key")
end)

test("module dtype conversion keeps parameter aliases", function()
  local model = nn.Linear(2, 2)
  model:to(torch.float64)
  assert_eq(model.weight:dtype(), "float64")
  assert_true(model.weight:requires_grad())
  assert_true(model.weight == model._parameters.weight)

  local output = model(torch.zeros(3, 2, { dtype = torch.float64 }))
  assert_eq(output:dtype(), "float64")

  local normalization = nn.BatchNorm2d(2)
  normalization:to(torch.float64)
  local value = normalization(torch.zeros(2, 2, 2, 2, { dtype = torch.float64 }))
  assert_eq(value:dtype(), "float64")
  assert_true(normalization.running_mean == normalization._buffers.running_mean)
end)

test("ModuleList", function()
  local list = nn.ModuleList({ nn.Linear(2, 2), nn.ReLU() })
  assert_true(list:get(1) ~= nil)
  assert_eq(#list:parameters(), 2)
end)

test("functional API", function()
  local x = torch.tensor({ { 1, 2 }, { 3, 4 } })
  local weight = torch.ones(2, 2)
  local bias = torch.zeros(2)
  assert_tensor(nn.functional.linear(x, weight, bias), { { 3, 3 }, { 7, 7 } })
  assert_tensor(nn.functional.relu(x:sub(10)), { { 0, 0 }, { 0, 0 } })

  local probabilities = nn.functional.softmax(torch.ones(1, 4), 2)
  assert_near(probabilities:sum():item(), 1, 1e-6)

  assert_near(nn.functional.mse_loss(x, x):item(), 0)
  assert_near(nn.functional.cross_entropy(torch.zeros(1, 4),
    torch.LongTensor({ 1 })):item(), math.log(4), 1e-6)
end)

test("module toString", function()
  local model = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
  local text = tostring(model)
  assert_true(text:find("Sequential", 1, true) ~= nil)
  assert_true(text:find("Linear", 1, true) ~= nil)
end)

test("model save/load roundtrip", function()
  local model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3))
  torch.no_grad(function()
    model._modules["1"].weight:uniform_(-1, 1)
    model._modules["3"].bias:uniform_(-1, 1)
  end)

  local path = os.tmpname() .. ".luatorch"
  model:save(path)

  local restored = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3))
  restored:load(path)
  os.remove(path)

  local original_state = model:state_dict()
  local restored_state = restored:state_dict()
  for name, tensor in pairs(original_state) do
    assert_true(restored_state[name]:equal(tensor), "mismatch for " .. name)
  end
end)

test("model save/load preserves buffers", function()
  local normalization = nn.BatchNorm2d(2)
  normalization(torch.randn(3, 2, 2, 2))

  local path = os.tmpname() .. ".luatorch"
  normalization:save(path)

  local restored = nn.BatchNorm2d(2)
  restored:load(path)
  os.remove(path)

  assert_true(restored.running_mean:equal(normalization.running_mean))
  assert_true(restored.running_var:equal(normalization.running_var))
end)

test("loading a corrupted model fails", function()
  local path = os.tmpname()
  local file = assert(io.open(path, "wb"))
  file:write("not a model")
  file:close()
  local model = nn.Linear(2, 2)
  assert_error(function()
    model:load(path)
  end, "not a luatorch model file")
  os.remove(path)
end)

test("nn.save and nn.load", function()
  local model = nn.Linear(3, 2)
  local path = os.tmpname() .. ".luatorch"
  nn.save(model, path)
  local restored = nn.Linear(3, 2)
  nn.load(restored, path)
  os.remove(path)
  local state = model:state_dict()
  assert_true(restored:state_dict().weight:equal(state.weight))
end)

test("torch.nn and torch.optim load lazily", function()
  local torch = require("torch")
  assert_true(torch.nn ~= nil)
  assert_true(torch.optim ~= nil)
  assert_true(torch.nn.Linear == nn.Linear)
end)
