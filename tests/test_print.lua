local torch = require("torch")

test("0-dim float", function()
  assert_eq(tostring(torch.tensor(2.5)), "tensor(2.5)")
end)

test("1-d float", function()
  assert_eq(tostring(torch.tensor({ 1, 2, 3 })), "tensor([1., 2., 3.])")
end)

test("2-d float", function()
  assert_eq(tostring(torch.tensor({ { 1, 2 }, { 3, 4 } })),
    "tensor([[1., 2.],\n        [3., 4.]])")
end)

test("integers", function()
  assert_eq(tostring(torch.tensor({ 1, 2 }, { dtype = torch.int64 })), "tensor([1, 2])")
end)

test("booleans", function()
  assert_eq(tostring(torch.tensor({ true, false })), "tensor([true, false])")
end)

test("float64 suffix", function()
  assert_eq(tostring(torch.tensor({ 1, 2 }, { dtype = torch.float64 })),
    "tensor([1., 2.], dtype=torch.float64)")
end)

test("requires_grad suffix", function()
  local t = torch.tensor({ 1, 2 }, { requires_grad = true })
  assert_eq(tostring(t), "tensor([1., 2.], requires_grad=True)")
end)

test("empty tensor", function()
  local t = torch.zeros(0)
  assert_eq(tostring(t), "tensor([])")
end)

test("large 2-d tensor preview", function()
  local t = torch.zeros(100, 20)
  local text = tostring(t)
  assert_true(text:find("shape=[100, 20]", 1, true) ~= nil)
  assert_true(text:find("...", 1, true) ~= nil)
  assert_true(#text < 2000)
end)

test("large 1-d tensor preview", function()
  local t = torch.zeros(2000)
  local text = tostring(t)
  assert_true(text:find("shape=[2000]", 1, true) ~= nil)
  assert_true(text:find("...", 1, true) ~= nil)
end)

test("large 3-d tensor stays a shape summary", function()
  local t = torch.zeros(20, 20, 20)
  assert_eq(tostring(t),
    "tensor(shape=[20, 20, 20], dtype=torch.float32, numel=8000)")
end)

test("tostring never crashes", function()
  local tensors = {
    torch.zeros(1, 1),
    torch.zeros(1, 2, 3),
    torch.LongTensor({ -1, 0, 1 }),
    torch.tensor({ 1e-6, 1e6 }),
    torch.tensor({ 0 / 0, math.huge, -math.huge }),
  }
  for _, t in ipairs(tensors) do
    assert_true(type(tostring(t)) == "string")
  end
end)
