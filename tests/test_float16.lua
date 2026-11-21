local torch = require("torch")

test("float16 table creation", function()
  local t = torch.tensor({ 1.5, 2.5, -4.5 }, { dtype = torch.float16 })
  assert_dtype(t, "float16")
  assert_eq(t:element_size(), 2)
  assert_near(t:to(torch.float32):totable(), { 1.5, 2.5, -4.5 })
end)

test("float16 scalar creation", function()
  local t = torch.tensor(1.5, { dtype = torch.float16 })
  assert_dtype(t, "float16")
  assert_eq(t:dim(), 0)
  assert_near(t:item(), 1.5)
end)

test("float16 nested table creation", function()
  local t = torch.tensor({ { 1.5, 2.5 }, { 3.5, 4.5 } },
    { dtype = torch.float16 })
  assert_dtype(t, "float16")
  assert_shape(t, { 2, 2 })
  assert_near(t:to(torch.float32):totable(),
    { { 1.5, 2.5 }, { 3.5, 4.5 } })
end)

test("Torch7-style HalfTensor", function()
  local t = torch.HalfTensor({ 1.5, -2.25 })
  assert_dtype(t, "float16")
  assert_near(t:to(torch.float32):totable(), { 1.5, -2.25 })
end)

test("float16 staging rounds like ATen", function()
  local t = torch.tensor({ 0.1 }, { dtype = torch.float16 })
  -- nearest half to 0.1 is 0.0999755859375 (0x2E66)
  assert_near(t:to(torch.float32):item(), 0.0999755859375, 1e-12)
end)

test("float16 overflow becomes infinity", function()
  local t = torch.tensor({ 1e5, -1e5 }, { dtype = torch.float16 })
  local values = t:to(torch.float32):totable()
  assert_eq(values[1], math.huge)
  assert_eq(values[2], -math.huge)
end)

test("float16 printing", function()
  assert_eq(tostring(torch.tensor({ 1.5, 2.5 }, { dtype = torch.float16 })),
    "tensor([1.5, 2.5], dtype=torch.float16)")
  assert_eq(tostring(torch.tensor(1.5, { dtype = torch.float16 })),
    "tensor(1.5, dtype=torch.float16)")
  assert_eq(tostring(torch.tensor({ { 1.5, 2.5 }, { 3.5, 4.5 } },
    { dtype = torch.float16 })),
    "tensor([[1.5, 2.5],\n        [3.5, 4.5]], dtype=torch.float16)")
end)

test("float16 printing of special values", function()
  local t = torch.tensor({ math.huge, -math.huge, 0 / 0 },
    { dtype = torch.float16 })
  assert_eq(tostring(t), "tensor([inf, -inf, nan], dtype=torch.float16)")
end)

test("float16 empty printing", function()
  assert_eq(tostring(torch.zeros(0, { dtype = torch.float16 })),
    "tensor([], dtype=torch.float16)")
end)

test("float16 creators keep their dtype", function()
  assert_dtype(torch.zeros(2, { dtype = torch.float16 }), "float16")
  assert_dtype(torch.full(2, 0.5, { dtype = torch.float16 }), "float16")
  assert_dtype(torch.arange(3, { dtype = torch.float16 }), "float16")
  assert_near(torch.full(2, 0.1, { dtype = torch.float16 })
    :to(torch.float32):totable(), { 0.0999755859375, 0.0999755859375 }, 1e-12)
end)

test("float16 autograd keeps the half dtype", function()
  local x = torch.tensor({ { 1, 2 }, { 3, 4 } },
    { dtype = torch.float16, requires_grad = true })
  assert_true(x:requires_grad())
  local loss = x:mul(x):sum()
  assert_dtype(loss, "float16")
  loss:backward()
  assert_dtype(x:grad(), "float16")
  assert_near(x:grad():to(torch.float32):totable(),
    { { 2, 4 }, { 6, 8 } }, 1e-3)

  local y = torch.tensor({ 1, 2 }, { dtype = torch.float16, requires_grad = true })
  y:mul(2):sum():backward()
  y:mul(3):sum():backward()
  assert_tensor(y:grad():to(torch.float32), { 5, 5 })
  y:zero_grad()
  assert_tensor(y:grad(), { 0, 0 })
end)

test("float16 casts round-trip gradients", function()
  local x = torch.tensor({ 1, 2, 3 }, { requires_grad = true })
  local half = x:half()
  assert_dtype(half, "float16")
  assert_true(half:requires_grad())
  half:mul(2):float():sum():backward()
  assert_dtype(x:grad(), "float32")
  assert_tensor(x:grad(), { 2, 2, 2 })

  local h = torch.tensor({ 1, 2, 3 },
    { dtype = torch.float16, requires_grad = true })
  h:float():sum():backward()
  assert_dtype(h:grad(), "float16")
end)

test("float16 linear layer forward and backward", function()
  local nn = require("torch.nn")
  local layer = nn.Linear(3, 2):to(torch.float16)
  assert_dtype(layer.weight, "float16")
  assert_dtype(layer.bias, "float16")
  local x = torch.randn(2, 3, { dtype = torch.float16, requires_grad = true })
  local output = layer:forward(x)
  assert_dtype(output, "float16")
  output:sum():backward()
  assert_dtype(x:grad(), "float16")
  assert_dtype(layer.weight:grad(), "float16")
  assert_dtype(layer.bias:grad(), "float16")
end)

test("float16 parameters update under no_grad", function()
  local p = torch.tensor({ 1, 2, 3 },
    { dtype = torch.float16, requires_grad = true })
  p:mul(2):sum():backward()
  torch.no_grad(function()
    p:copy_(p:sub(p:grad():mul(0.1)))
  end)
  assert_dtype(p, "float16")
  assert_near(p:to(torch.float32):totable(),
    { 0.7998, 1.7998, 2.8008 }, 1e-3)
end)

test("float16 linalg reports an unsupported dtype", function()
  local a = torch.randn(3, 3, { dtype = torch.float16 })
  assert_error(function()
    a:inverse()
  end, "Half")
  assert_error(function()
    a:det()
  end, "Half")
end)
