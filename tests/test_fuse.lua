local torch = require("torch")
local fusec = require("torch.fusec")

local has_c = fusec.available()

local function values(tensor)
  return tensor:totable()
end

test("fused chain, Lua backend", function()
  local fused = torch.fuse(function(a, b, c)
    return (a * b + c):relu()
  end, { backend = "lua" })

  local a = torch.randn(128)
  local b = torch.randn(128)
  local c = torch.randn(128)
  local output = fused(a, b, c)
  local expected = (a * b + c):relu()
  assert_true(output:allclose(expected, 1e-4, 1e-4))
  assert_dtype(output, "float32")
  assert_eq(fused.kernels[next(fused.kernels)].kind, "lua")
end)

test("fused chain, C backend", function()
  if not has_c then
    return
  end
  local fused = torch.fuse(function(a, b, c)
    return (a * b + c):relu()
  end, { backend = "c" })

  local a = torch.randn(1024)
  local b = torch.randn(1024)
  local c = torch.randn(1024)
  local output = fused(a, b, c)
  assert_true(output:allclose((a * b + c):relu(), 1e-5, 1e-5))
  assert_eq(fused.kernels[next(fused.kernels)].kind, "c")
end)

test("fused expression with many unary ops", function()
  for _, backend in ipairs(has_c and { "lua", "c" } or { "lua" }) do
    local fused = torch.fuse(function(x)
      return ((x * 1.5 + 0.25):abs():sigmoid() + x:exp():clamp(0, 2))
        :tanh()
    end, { backend = backend })

    local x = torch.randn(512)
    local output = fused(x)
    local expected = ((x * 1.5 + 0.25):abs():sigmoid() +
      x:exp():clamp(0, 2)):tanh()
    assert_true(output:allclose(expected, 1e-4, 1e-4),
      "backend " .. backend .. " mismatch")
  end
end)

test("fused scalar arguments are runtime values", function()
  for _, backend in ipairs(has_c and { "lua", "c" } or { "lua" }) do
    local fused = torch.fuse(function(x, scale, offset)
      return x * scale + offset
    end, { backend = backend })

    local x = torch.randn(64)
    assert_true(fused(x, 2, 1):allclose(x * 2 + 1, 1e-5, 1e-5))
    assert_true(fused(x, 3, -1):allclose(x * 3 - 1, 1e-5, 1e-5))
  end
end)

test("fused float64", function()
  local fused = torch.fuse(function(x, y)
    return (x * y + x):sqrt()
  end, { backend = has_c and "c" or "lua" })

  local x = torch.randn(128, { dtype = torch.float64 }):abs():add(1)
  local y = torch.randn(128, { dtype = torch.float64 }):abs()
  local output = fused(x, y)
  assert_dtype(output, "float64")
  assert_true(output:allclose((x * y + x):sqrt(), 1e-10, 1e-10))
end)

test("fused non-contiguous inputs", function()
  local fused = torch.fuse(function(x, y)
    return x + y
  end, { backend = has_c and "c" or "lua" })

  local matrix = torch.randn(8, 8):t()
  local other = torch.randn(8, 8)
  local output = fused(matrix, other)
  assert_true(output:allclose(matrix + other, 1e-5, 1e-5))
end)

test("fused into reuses the output buffer", function()
  local fused = torch.fuse(function(x, y)
    return (x * y):relu()
  end, { backend = has_c and "c" or "lua" })

  local x = torch.randn(256)
  local y = torch.randn(256)
  local buffer = torch.empty(256)
  local returned = fused:into(buffer, x, y)
  assert_true(returned == buffer)
  assert_true(buffer:allclose((x * y):relu(), 1e-5, 1e-5))
end)

test("fused into validates the output buffer", function()
  local fused = torch.fuse(function(x, y)
    return x + y
  end, { backend = "lua" })
  local x = torch.randn(4, 4)
  local y = torch.randn(4, 4)

  assert_error(function()
    fused:into(torch.empty(16), x, y)
  end, "shape")
  assert_error(function()
    fused:into(torch.empty(4, 4, { dtype = torch.float64 }), x, y)
  end, "dtype")
  assert_error(function()
    fused:into(torch.zeros(4, 4):t(), x, y)
  end, "contiguous")
  assert_error(function()
    fused:into(torch.empty(4, 4, { requires_grad = true }), x, y)
  end, "grad")
end)

test("fused autograd falls back to eager", function()
  local fused = torch.fuse(function(x, y)
    return x * y
  end)
  local x = torch.randn(8, { requires_grad = true })
  local y = torch.randn(8)
  local output = fused(x, y)
  assert_true(output:requires_grad())
  output:sum():backward()
  assert_true(x:grad() ~= nil)
  assert_true(x:grad():allclose(y, 1e-6, 1e-6))
end)

test("fused unsupported operations fall back", function()
  local fused = torch.fuse(function(x)
    return x:sum()
  end)
  local x = torch.randn(16)
  assert_near(fused(x):item(), x:sum():item(), 1e-5)
  assert_true(fused.last_error ~= nil)
end)

test("fused shape mismatch falls back to broadcasting", function()
  local fused = torch.fuse(function(x, y)
    return x + y
  end)
  local x = torch.randn(8, 1)
  local y = torch.randn(1, 8)
  local output = fused(x, y)
  assert_shape(output, { 8, 8 })
  assert_true(output:allclose(x + y, 1e-5, 1e-5))
end)

test("fused callable handles changing shapes", function()
  local fused = torch.fuse(function(x)
    return x * 2 + 1
  end, { backend = has_c and "c" or "lua" })
  local first = fused(torch.randn(10))
  local second = fused(torch.randn(1000))
  assert_shape(first, { 10 })
  assert_shape(second, { 1000 })
end)

test("fused backend option validation", function()
  assert_error(function()
    torch.fuse(function(x) return x end, { backend = "nope" })
  end, "backend")
  assert_error(function()
    torch.fuse(42)
  end, "function")
end)

test("fused toString", function()
  local fused = torch.fuse(function(x) return x + 1 end, { backend = "lua" })
  fused(torch.randn(4))
  assert_true(tostring(fused):find("Fused", 1, true) ~= nil)
end)
