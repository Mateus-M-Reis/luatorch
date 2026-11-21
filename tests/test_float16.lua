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
