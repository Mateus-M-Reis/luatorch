local torch = require("torch")

local function temp_path()
  return os.tmpname() .. ".pt"
end

test("save and load a matrix", function()
  local path = temp_path()
  local t = torch.tensor({ { 1, 2 }, { 3, 4 } })
  t:save(path)
  local loaded = torch.load(path)
  os.remove(path)
  assert_dtype(loaded, "float32")
  assert_shape(loaded, { 2, 2 })
  assert_tensor(loaded, { { 1, 2 }, { 3, 4 } })
end)

test("roundtrip preserves dtypes", function()
  local path = temp_path()
  local tensors = {
    torch.LongTensor({ 1, 2, 3 }),
    torch.BoolTensor({ true, false }),
    torch.tensor({ 1.5, 2.5 }, { dtype = torch.float64 }),
  }
  for i, t in ipairs(tensors) do
    t:save(path)
    local loaded = torch.load(path)
    assert_dtype(loaded, t:dtype())
    assert_tensor(loaded, t:totable())
  end
  os.remove(path)
end)

test("scalar roundtrip", function()
  local path = temp_path()
  torch.tensor(3.25):save(path)
  local loaded = torch.load(path)
  os.remove(path)
  assert_eq(loaded:dim(), 0)
  assert_near(loaded:item(), 3.25)
end)

test("load missing file raises", function()
  assert_error(function()
    torch.load("/nonexistent/path/to/tensor.pt")
  end)
end)
