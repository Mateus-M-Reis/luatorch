local torch = require("torch")

local function float_matrix(rows)
  return torch.tensor(rows, { dtype = torch.float64 })
end

test("det and inverse", function()
  local matrix = float_matrix({ { 4, 7 }, { 2, 6 } })
  assert_near(matrix:det():item(), 10, 1e-10)
  assert_near(matrix:inverse():totable(), {
    { 0.6, -0.7 },
    { -0.2, 0.4 },
  }, 1e-10)

  local identity = matrix:matmul(matrix:inverse())
  assert_near(identity:totable(), { { 1, 0 }, { 0, 1 } }, 1e-10)
end)

test("linalg aliases", function()
  local matrix = float_matrix({ { 4, 7 }, { 2, 6 } })
  assert_near(torch.linalg.det(matrix):item(), 10, 1e-10)
  assert_near(torch.linalg.inv(matrix):get(1, 1), 0.6, 1e-10)
end)

test("solve", function()
  local a = float_matrix({ { 3, 1 }, { 1, 2 } })
  local b = torch.tensor({ 9, 8 }, { dtype = torch.float64 })
  assert_near(a:solve(b):totable(), { 2, 3 }, 1e-10)

  local multiple = float_matrix({ { 9, 5 }, { 8, 5 } })
  assert_near(a:solve(multiple):totable(), {
    { 2, 1 },
    { 3, 2 },
  }, 1e-9)
end)

test("cholesky", function()
  local matrix = float_matrix({ { 4, 2 }, { 2, 3 } })
  local lower = matrix:cholesky()
  assert_near(lower:matmul(lower:t()):totable(), matrix:totable(), 1e-9)

  local upper = matrix:cholesky(true)
  assert_near(upper:t():matmul(upper):totable(), matrix:totable(), 1e-9)
end)

test("svd reconstruction", function()
  local matrix = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local u, s, vh = matrix:svd(false)
  assert_shape(u, { 3, 2 })
  assert_shape(s, { 2 })
  assert_shape(vh, { 2, 2 })

  local diagonal = torch.zeros(2, 2, { dtype = torch.float64 })
  for i = 1, 2 do
    diagonal:set(i, i, s:get(i))
  end
  assert_near(u:matmul(diagonal):matmul(vh):totable(), matrix:totable(), 1e-9)
end)

test("qr reconstruction", function()
  local matrix = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local q, r = matrix:qr()
  assert_shape(q, { 3, 2 })
  assert_shape(r, { 2, 2 })
  assert_near(q:matmul(r):totable(), matrix:totable(), 1e-9)

  local gram = q:t():matmul(q)
  assert_near(gram:totable(), { { 1, 0 }, { 0, 1 } }, 1e-9)
end)

test("mv and outer", function()
  local matrix = torch.tensor({ { 1, 2, 3 }, { 4, 5, 6 } })
  local vector = torch.tensor({ 1, 1, 1 })
  assert_tensor(matrix:mv(vector), { 6, 15 })

  assert_tensor(torch.tensor({ 1, 2 }):outer(torch.tensor({ 3, 4, 5 })), {
    { 3, 4, 5 },
    { 6, 8, 10 },
  })
end)

test("addmm and baddbmm", function()
  local mat1 = torch.ones(2, 3)
  local mat2 = torch.ones(3, 2)
  local input = torch.ones(2, 2)
  assert_tensor(input:addmm(mat1, mat2), { { 4, 4 }, { 4, 4 } })
  assert_tensor(input:addmm(mat1, mat2, 2, 3), { { 11, 11 }, { 11, 11 } })

  local batch_input = torch.zeros(2, 2, 2)
  local batch1 = torch.ones(2, 2, 3)
  local batch2 = torch.ones(2, 3, 2)
  assert_tensor(batch_input:baddbmm(batch1, batch2), {
    { { 3, 3 }, { 3, 3 } },
    { { 3, 3 }, { 3, 3 } },
  })
end)

test("linalg errors", function()
  assert_error(function()
    torch.zeros(2, 3):inverse()
  end)
  assert_error(function()
    torch.zeros(2, 3):det()
  end)
  assert_error(function()
    torch.ones(2, 2):cholesky()
  end)
end)
