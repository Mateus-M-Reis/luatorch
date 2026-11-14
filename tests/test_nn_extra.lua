local torch = require("torch")
local nn = require("torch.nn")

test("AvgPool2d", function()
  local pool = nn.AvgPool2d(2)
  local input = torch.arange(1, 17):reshape(1, 1, 4, 4):to(torch.float32)
  local output = pool(input)
  assert_shape(output, { 1, 1, 2, 2 })
  assert_tensor(output, { { { { 3.5, 5.5 }, { 11.5, 13.5 } } } })
end)

test("Embedding uses 1-based indices", function()
  local embedding = nn.Embedding(5, 3)
  assert_shape(embedding.weight, { 5, 3 })

  local indices = torch.LongTensor({ 1, 3, 5 })
  local output = embedding(indices)
  assert_shape(output, { 3, 3 })
  for i, index in ipairs({ 1, 3, 5 }) do
    assert_true(output:select(1, i):equal(embedding.weight:select(1, index)))
  end

  local batched = embedding(torch.LongTensor({ { 1, 2 }, { 3, 4 } }))
  assert_shape(batched, { 2, 2, 3 })
end)

test("Embedding gradients and errors", function()
  local embedding = nn.Embedding(4, 2)
  embedding(torch.LongTensor({ 1, 1, 3 })):sum():backward()
  assert_shape(embedding.weight:grad(), { 4, 2 })
  assert_true(embedding.weight:grad():get(1, 1) ~= 0)
  assert_eq(embedding.weight:grad():get(2, 1), 0)

  assert_error(function()
    embedding(torch.LongTensor({ 0 }))
  end)
  assert_error(function()
    embedding(torch.LongTensor({ 5 }))
  end)
end)

test("LayerNorm normalizes the last dimensions", function()
  local normalization = nn.LayerNorm(4)
  assert_eq(#normalization:parameters(), 2)
  assert_shape(normalization.weight, { 4 })
  assert_shape(normalization.bias, { 4 })

  local input = torch.randn(3, 4):mul(3):add(7)
  local output = normalization(input)
  assert_shape(output, { 3, 4 })

  local means = output:mean(2)
  assert_true(means:abs():max():item() < 1e-4)
  local variances = output:var(2, false)
  assert_near(variances:totable(), { 1, 1, 1 }, 1e-3)
end)

test("LayerNorm supports multi-dimensional normalized shapes", function()
  local normalization = nn.LayerNorm({ 2, 3 })
  assert_shape(normalization.weight, { 2, 3 })
  local input = torch.randn(2, 2, 3)
  local output = normalization(input)
  assert_shape(output, { 2, 2, 3 })

  local means = output:mean(3):mean(2)
  assert_true(means:abs():max():item() < 1e-4)
end)

test("LayerNorm gradients and affine toggle", function()
  local normalization = nn.LayerNorm(3)
  local input = torch.randn(2, 3, { requires_grad = true })
  normalization(input):sum():backward()
  assert_true(normalization.weight:grad() ~= nil)
  assert_true(input:grad() ~= nil)

  local plain = nn.LayerNorm(3, { elementwise_affine = false })
  assert_eq(#plain:parameters(), 0)
  assert_shape(plain(torch.randn(2, 3)), { 2, 3 })
end)

test("embedding and layernorm module tostring", function()
  assert_true(tostring(nn.Embedding(4, 2)):find("Embedding", 1, true) ~= nil)
  assert_true(tostring(nn.LayerNorm(4)):find("LayerNorm", 1, true) ~= nil)
  assert_true(tostring(nn.AvgPool2d(2)):find("AvgPool2d", 1, true) ~= nil)
end)

test("new layers survive state_dict roundtrips", function()
  local model = nn.Sequential(
    nn.LayerNorm(4),
    nn.Linear(4, 3))
  local path = os.tmpname() .. ".luatorch"
  model:save(path)
  local restored = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 3))
  restored:load(path)
  os.remove(path)
  local original = model:state_dict()
  for name, tensor in pairs(original) do
    assert_true(restored:state_dict()[name]:equal(tensor), "mismatch for " .. name)
  end
end)

test("Linear supports 3D inputs", function()
  local nn = require("torch.nn")
  local layer = nn.Linear(5, 3)
  local input = torch.randn(2, 4, 5)
  local output = layer(input)
  assert_eq(output:sizes(), { 2, 4, 3 })
  local reference = input:matmul(layer.weight:t()):add(layer.bias)
  assert_true(output:allclose(reference, 1e-5, 1e-6))
end)
