local torch = require("torch")

test("take_along_dim selects values along a dimension", function()
  local input = torch.tensor({ { 1, 2, 3 }, { 4, 5, 6 } })
  local index = torch.LongTensor({ { 1, 1, 3 }, { 2, 2, 2 } })
  local output = input:take_along_dim(2, index)
  assert_shape(output, { 2, 3 })
  assert_tensor(output, { { 1, 1, 3 }, { 5, 5, 5 } })

  local column_index = torch.LongTensor({ { 2 }, { 1 } })
  assert_tensor(input:take_along_dim(1, column_index),
    { { 4, 5, 6 }, { 1, 2, 3 } })
end)

test("take_along_dim accepts broadcast-compatible index shapes", function()
  local input = torch.tensor({ { 1, 2, 3 }, { 4, 5, 6 } })
  local row = torch.LongTensor({ { 1, 3 } })
  assert_shape(input:take_along_dim(2, row), { 2, 2 })
  assert_tensor(input:take_along_dim(2, row), { { 1, 3 }, { 4, 6 } })

  local column = torch.LongTensor({ { 2 }, { 1 } })
  assert_shape(input:take_along_dim(2, column), { 2, 1 })
  assert_tensor(input:take_along_dim(2, column), { { 2 }, { 4 } })
end)

test("take_along_dim uses 1-based dimensions and PyTorch argument order", function()
  local input = torch.tensor({ { 1, 2, 3 }, { 4, 5, 6 } })
  local index = torch.LongTensor({ { 1, 3 } })
  assert_tensor(input:take_along_dim(-1, index), { { 1, 3 }, { 4, 6 } })
  assert_tensor(input:take_along_dim(index, 2), { { 1, 3 }, { 4, 6 } })
  assert_tensor(torch.take_along_dim(input, index, 2),
    { { 1, 3 }, { 4, 6 } })
  assert_tensor(torch.take_along_dim(input, 2, index),
    { { 1, 3 }, { 4, 6 } })

  -- With no dimension, ATen treats the input as flattened.
  assert_tensor(input:take_along_dim(torch.LongTensor({ 2, 4, 1, 3 })),
    { 2, 4, 1, 3 })
end)

test("take_along_dim gradients", function()
  local input = torch.tensor({ { 1, 2, 3 }, { 4, 5, 6 } },
    { requires_grad = true })
  local index = torch.LongTensor({ { 1, 1, 3 }, { 2, 2, 2 } })
  input:take_along_dim(2, index):sum():backward()
  assert_tensor(input:grad(), { { 2, 0, 1 }, { 0, 3, 0 } })
end)

test("take_along_dim error paths", function()
  local input = torch.tensor({ { 1, 2 }, { 3, 4 } })
  assert_error(function()
    input:take_along_dim(2, torch.LongTensor({ { 0, 1 } }))
  end)
  assert_error(function()
    input:take_along_dim(2, torch.BoolTensor({ { true, false } }))
  end, "integer")
  assert_error(function()
    input:take_along_dim(0, torch.LongTensor({ { 1, 1 } }))
  end, "1-based")
  assert_error(function()
    input:take_along_dim(2, torch.LongTensor({ { 1, 2, 3, 4 } }))
  end)
end)

test("index_put_ with a boolean mask", function()
  local input = torch.zeros(2, 3)
  local mask = torch.tensor({ { true, false, true }, { false, true, false } })
  local values = torch.tensor({ 10, 20, 30 })
  local returned = input:index_put_(mask, values)
  assert_true(returned == input)
  assert_shape(input, { 2, 3 })
  assert_tensor(input, { { 10, 0, 20 }, { 0, 30, 0 } })

  local filled = torch.zeros(2, 2)
  filled:index_put_(torch.tensor({ { true, false }, { true, true } }), 7)
  assert_tensor(filled, { { 7, 0 }, { 7, 7 } })

  local scalar = torch.zeros(2, 1)
  scalar:index_put_(torch.tensor({ { true }, { false } }), torch.tensor(9.0))
  assert_tensor(scalar, { { 9 }, { 0 } })
end)

test("index_put_ with tensor-of-indices and duplicate behavior", function()
  local index = torch.LongTensor({ 1, 1, 2 })
  local values = torch.tensor({ { 1, 2 }, { 3, 4 }, { 5, 6 } })

  local replaced = torch.zeros(3, 2)
  replaced:index_put_(index, values)
  -- Without accumulate, CPU ATen applies duplicates in order: the last wins.
  assert_tensor(replaced, { { 3, 4 }, { 5, 6 }, { 0, 0 } })

  local accumulated = torch.zeros(3, 2)
  accumulated:index_put_(index, values, true)
  assert_tensor(accumulated, { { 4, 6 }, { 5, 6 }, { 0, 0 } })

  -- ATen also accepts a multi-dimensional tensor of row indices.
  local matrix_index = torch.LongTensor({ { 1, 3 }, { 2, 1 } })
  local matrix_values = torch.tensor({
    { { 1, 2 }, { 3, 4 } },
    { { 5, 6 }, { 7, 8 } },
  })
  local matrix = torch.zeros(3, 2)
  matrix:index_put_(matrix_index, matrix_values)
  assert_tensor(matrix, { { 7, 8 }, { 5, 6 }, { 3, 4 } })
end)

test("index_put_ converts 1-based indices", function()
  local input = torch.zeros(3, 2)
  input:index_put_(torch.LongTensor({ 3, 1 }),
    torch.tensor({ { 1, 2 }, { 3, 4 } }))
  assert_tensor(input, { { 3, 4 }, { 0, 0 }, { 1, 2 } })
end)

test("index_put_ gradients through a non-leaf tensor", function()
  local input = torch.tensor({ { 1, 2, 3 }, { 4, 5, 6 } },
    { requires_grad = true })
  local output = input:mul(2)
  local values = torch.tensor({ 10, 20, 30 }, { requires_grad = true })
  local mask = torch.tensor({ { true, false, true }, { false, true, false } })
  output:index_put_(mask, values)
  output:sum():backward()
  assert_tensor(input:grad(), { { 0, 2, 0 }, { 2, 0, 2 } })
  assert_tensor(values:grad(), { 1, 1, 1 })
end)

test("index_put_ error paths", function()
  local input = torch.zeros(2, 2)
  assert_error(function()
    input:index_put_(torch.LongTensor({ 0, 1 }), torch.ones(2, 2))
  end)
  assert_error(function()
    input:index_put_(torch.tensor({ { true, false } }), torch.ones(2))
  end)
  assert_error(function()
    input:index_put_(torch.LongTensor({ 1, 2 }), torch.ones(3, 2))
  end)
  assert_error(function()
    input:index_put_("nope", torch.ones(2, 2))
  end, "Tensor")
end)

test("tensor-of-indices selection and assignment", function()
  local input = torch.tensor({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local rows = torch.LongTensor({ 3, 1 })
  assert_tensor(input:index(1, rows), { { 5, 6 }, { 1, 2 } })
  assert_tensor(input:index(rows), { { 5, 6 }, { 1, 2 } })
  assert_tensor(input[rows], { { 5, 6 }, { 1, 2 } })
  assert_tensor(input:index(2, torch.LongTensor({ 2, 1 })),
    { { 2, 1 }, { 4, 3 }, { 6, 5 } })

  local target = torch.zeros(3, 2)
  target[rows] = torch.tensor({ { 7, 8 }, { 9, 10 } })
  assert_tensor(target, { { 9, 10 }, { 0, 0 }, { 7, 8 } })

  local filled = torch.zeros(2, 2)
  filled[torch.tensor({ { true, false }, { false, true } })] = 5
  assert_tensor(filled, { { 5, 0 }, { 0, 5 } })
end)

test("tensor-of-indices selection error paths", function()
  local input = torch.tensor({ { 1, 2 }, { 3, 4 } })
  assert_error(function()
    input:index(1, torch.LongTensor({ 0, 1 }))
  end)
  assert_error(function()
    local _ = input[torch.LongTensor({ 0, 1 })]
  end)
  assert_error(function()
    input:index(1, torch.BoolTensor({ true, false }))
  end, "integer")
end)

test("module-level advanced indexing helpers", function()
  local input = torch.tensor({ { 1, 2 }, { 3, 4 } })
  assert_tensor(torch.index_select(input, 1, torch.LongTensor({ 2 })),
    { { 3, 4 } })
  assert_tensor(torch.index(input, torch.LongTensor({ 2 })),
    { { 3, 4 } })

  local index = torch.LongTensor({ { 1, 2 } })
  assert_tensor(torch.take_along_dim(input, index, 2),
    { { 1, 2 }, { 3, 4 } })

  local target = torch.zeros(2, 2)
  assert_true(torch.index_put_(target, torch.LongTensor({ 2 }),
    torch.tensor({ { 8, 9 } })) == target)
  assert_tensor(target, { { 0, 0 }, { 8, 9 } })
end)
