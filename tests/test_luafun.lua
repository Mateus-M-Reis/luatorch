local torch = require("torch")
local fun = require("fun")

test("luafun map/filter/reduce", function()
  local mapped = fun.totable(fun.iter({ 1, 2, 3, 4 }):map(function(x) return x * 2 end))
  assert_eq(mapped, { 2, 4, 6, 8 })

  local filtered = fun.totable(fun.iter({ 1, 2, 3, 4 }):filter(function(x) return x % 2 == 0 end))
  assert_eq(filtered, { 2, 4 })

  local total = fun.iter({ 1, 2, 3, 4 }):reduce(function(a, b) return a + b end, 0)
  assert_eq(total, 10)
end)

test("torch.fun exposes luafun", function()
  assert_true(torch.fun == fun)
  assert_true(torch.luafun == fun)
end)

test("tensor:iter", function()
  local t = torch.tensor({ 1, 2, 3, 4 })
  local total = 0
  local count = 0
  for value in t:iter() do
    total = total + value
    count = count + 1
  end
  assert_eq(count, 4)
  assert_near(total, 10)
end)

test("tensor:iter over 2-d tensor", function()
  local t = torch.tensor({ { 1, 2 }, { 3, 4 } })
  local values = {}
  for value in t:iter() do
    values[#values + 1] = value
  end
  assert_eq(values, { 1, 2, 3, 4 })
end)

test("tensor:map", function()
  local t = torch.tensor({ { 1, 2 }, { 3, 4 } })
  local doubled = t:map(function(x) return x * 2 end)
  assert_shape(doubled, { 2, 2 })
  assert_dtype(doubled, "float32")
  assert_tensor(doubled, { { 2, 4 }, { 6, 8 } })
end)

test("tensor:filter", function()
  local t = torch.tensor({ 1, 2, 3, 4, 5, 6 })
  local even = t:filter(function(x) return x % 2 == 0 end)
  assert_shape(even, { 3 })
  assert_tensor(even, { 2, 4, 6 })
end)

test("tensor:reduce", function()
  local t = torch.tensor({ 1, 2, 3, 4 })
  assert_eq(t:reduce(function(a, b) return a + b end, 0), 10)
  assert_eq(t:reduce(function(a, b) return a + b end), 10)
end)

test("tensor:apply", function()
  local t = torch.tensor({ 1, 2, 3 })
  local returned = t:apply(function(x) return x + 10 end)
  assert_true(returned == t)
  assert_tensor(t, { 11, 12, 13 })
end)

test("tensor:fun bridges to luafun", function()
  local t = torch.tensor({ 1, 2, 3 })
  local doubled = fun.totable(t:fun():map(function(x) return x * 2 end))
  assert_eq(doubled, { 2, 4, 6 })

  local total = t:fun():reduce(function(a, b) return a + b end, 0)
  assert_eq(total, 6)
end)

test("luafun range and zip", function()
  assert_eq(fun.totable(fun.range(3)), { 1, 2, 3 })

  local sums = fun.totable(fun.zip(fun.iter({ 1, 2, 3 }), fun.iter({ 10, 11, 12 }))
    :map(function(a, b) return a + b end))
  assert_eq(sums, { 11, 13, 15 })
end)
