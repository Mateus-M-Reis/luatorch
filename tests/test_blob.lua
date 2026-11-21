local ffi = require("ffi")
local torch = require("torch")

test("from_blob shares FFI array memory", function()
  local buffer = ffi.new("float[4]", { 1, 2, 3, 4 })
  local tensor = torch.from_blob(buffer, { 2, 2 }, { dtype = "float32" })
  assert_shape(tensor, { 2, 2 })
  assert_tensor(tensor, { { 1, 2 }, { 3, 4 } })
  assert_true(tensor._owner == buffer)
  tensor:set(1, 1, 9)
  assert_true(buffer[0] == 9)
  buffer[1] = 8
  assert_true(tensor:get(1, 2) == 8)
end)

test("from_blob accepts immutable string backing", function()
  local tensor = torch.from_blob("abcd", { 2, 2 }, { dtype = "uint8" })
  assert_tensor(tensor, { { 97, 98 }, { 99, 100 } })
end)

test("from_blob shares float16 buffers", function()
  -- IEEE-754 binary16 bit patterns: 1.0, 2.0, 3.0, 0.5
  local buffer = ffi.new("uint16_t[4]", { 0x3C00, 0x4000, 0x4200, 0x3800 })
  local tensor = torch.from_blob(buffer, { 2, 2 }, { dtype = torch.float16 })
  assert_dtype(tensor, "float16")
  assert_eq(tensor:element_size(), 2)
  assert_eq(tensor:nbytes(), 8)
  assert_near(tensor:to(torch.float32):totable(),
    { { 1, 2 }, { 3, 0.5 } }, 0)
  assert_true(tensor._owner == buffer)
  buffer[0] = 0x4000
  assert_eq(tensor:get(1, 1), 2)
  tensor:set(2, 2, -2)
  assert_eq(tonumber(buffer[3]), 0xC000)
end)

test("from_blob float16 special values", function()
  local buffer = ffi.new("uint16_t[6]",
    { 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x8000, 0x3C00 })
  local tensor = torch.from_blob(buffer, { 6 }, { dtype = "float16" })
  local values = tensor:to(torch.float32):totable()
  assert_eq(values[1], math.huge)
  assert_eq(values[2], -math.huge)
  assert_true(values[3] ~= values[3])
  assert_near(values[4], 5.960464477539063e-08, 1e-20)
  assert_eq(1 / values[5], -math.huge)  -- negative zero
  assert_eq(values[6], 1)
end)

test("from_blob accepts float16 strings and pointers", function()
  local buffer = ffi.new("uint16_t[2]", { 0x3C00, 0xC000 })
  local from_string = torch.from_blob(ffi.string(buffer, 4), { 2 },
    { dtype = "float16" })
  assert_near(from_string:to(torch.float32):totable(), { 1, -2 }, 0)
  local pointer = ffi.cast("uint16_t*", buffer)
  local from_pointer = torch.from_blob(pointer, { 2 },
    { dtype = "float16", nbytes = 4 })
  assert_near(from_pointer:to(torch.float32):totable(), { 1, -2 }, 0)
end)

test("from_blob honors requires_grad for float16", function()
  local buffer = ffi.new("uint16_t[2]", { 0x3C00, 0x4000 })
  local tensor = torch.from_blob(buffer, { 2 },
    { dtype = "float16", requires_grad = true })
  assert_true(tensor:requires_grad())
end)

test("from_blob accepts a pointer when nbytes is supplied", function()
  local buffer = ffi.new("double[3]", { 1.5, 2.5, 3.5 })
  local pointer = ffi.cast("double*", buffer)
  local tensor = torch.from_blob(pointer, { 3 },
    { dtype = "float64", nbytes = 24 })
  assert_near(tensor:totable(), { 1.5, 2.5, 3.5 }, 0)
end)

test("from_blob sizes arrays that match a pointer's byte width", function()
  local buffer = ffi.new("float[2]", { 1.5, 2.5 })
  local tensor = torch.from_blob(buffer, { 2 }, { dtype = "float32" })
  assert_near(tensor:totable(), { 1.5, 2.5 }, 0)
end)

test("from_blob rejects short buffers and bad arguments", function()
  local buffer = ffi.new("float[2]")
  assert_error(function()
    torch.from_blob(buffer, { 3 }, { dtype = "float32" })
  end, "buffer has")
  assert_error(function()
    torch.from_blob(42, { 1 })
  end, "string or FFI cdata")
  assert_error(function()
    torch.from_blob(buffer, { -1 })
  end, "non-negative")
  assert_error(function()
    torch.from_blob(ffi.new("uint16_t[1]"), { 2 }, { dtype = "float16" })
  end, "buffer has 2 bytes, need 4")
end)
