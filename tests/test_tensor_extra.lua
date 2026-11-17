local torch = require("torch")

test("sort 1-d", function()
  local input = torch.tensor({ 3, 1, 2 })
  local values, indices = input:sort(1)
  assert_tensor(values, { 1, 2, 3 })
  assert_tensor(indices, { 2, 3, 1 })
  assert_dtype(indices, "int64")

  local descending_values, descending_indices = input:sort(1, true)
  assert_tensor(descending_values, { 3, 2, 1 })
  assert_tensor(descending_indices, { 1, 3, 2 })
end)

test("sort 2-d along each dimension", function()
  local input = torch.tensor({ { 3, 1, 2 }, { 6, 4, 5 } })
  local values, indices = input:sort(2)
  assert_tensor(values, { { 1, 2, 3 }, { 4, 5, 6 } })
  assert_tensor(indices, { { 2, 3, 1 }, { 2, 3, 1 } })

  local columns, column_indices = input:sort(1)
  assert_tensor(columns, { { 3, 1, 2 }, { 6, 4, 5 } })
  assert_tensor(column_indices, { { 1, 1, 1 }, { 2, 2, 2 } })

  local shape = input:sort(-1)
  assert_shape(shape, { 2, 3 })
end)

test("topk", function()
  local input = torch.tensor({ 5, 1, 4, 2, 3 })
  local values, indices = input:topk(2)
  assert_tensor(values, { 5, 4 })
  assert_tensor(indices, { 1, 3 })

  local smallest, smallest_indices = input:topk(2, 1, false)
  assert_tensor(smallest, { 1, 2 })
  assert_tensor(smallest_indices, { 2, 4 })

  local matrix = torch.tensor({ { 1, 5, 3 }, { 6, 2, 4 } })
  local matrix_values = matrix:topk(2, 2)
  assert_tensor(matrix_values, { { 5, 3 }, { 6, 4 } })
end)

test("unique with inverse and counts", function()
  local input = torch.tensor({ 3, 1, 3, 2 })
  local values, inverse, counts = input:unique(true, true, true)
  assert_tensor(values, { 1, 2, 3 })
  assert_tensor(inverse, { 3, 1, 3, 2 })
  assert_tensor(counts, { 1, 1, 2 })

  -- inverse is 1-based and reconstructs the input.
  assert_true(values:index_select(1, inverse):equal(input))

  local only_values = input:unique()
  assert_tensor(only_values, { 1, 2, 3 })
end)

test("nonzero returns 1-based coordinates", function()
  local input = torch.tensor({ { 0, 1 }, { 2, 0 } })
  local coordinates = input:nonzero()
  assert_shape(coordinates, { 2, 2 })
  assert_tensor(coordinates, { { 1, 2 }, { 2, 1 } })

  local vector = torch.tensor({ 0, 5, 0, 7 })
  assert_shape(vector:nonzero(), { 2, 1 })
  assert_eq(vector:nonzero():get(1, 1), 2)
  assert_eq(vector:nonzero():get(2, 1), 4)
end)

test("gather", function()
  local input = torch.tensor({ { 1, 2, 3 }, { 4, 5, 6 } })
  local index = torch.LongTensor({ { 1, 1 }, { 3, 3 } })
  assert_tensor(input:gather(2, index), { { 1, 1 }, { 6, 6 } })

  local row_index = torch.LongTensor({ { 2, 2, 2 } })
  assert_tensor(input:gather(1, row_index), { { 4, 5, 6 } })
end)

test("scatter", function()
  local input = torch.zeros(2, 3)
  local index = torch.LongTensor({ { 1, 3 }, { 2, 1 } })
  local source = torch.ones(2, 2)
  local output = input:scatter(2, index, source)
  assert_tensor(output, { { 1, 0, 1 }, { 1, 1, 0 } })
  assert_tensor(input, { { 0, 0, 0 }, { 0, 0, 0 } })
end)

test("gather and scatter reject index zero", function()
  local input = torch.ones(2, 2)
  local index = torch.LongTensor({ { 0, 1 }, { 1, 1 } })
  assert_error(function()
    input:gather(2, index)
  end)
  assert_error(function()
    torch.zeros(2, 2):scatter(2, index, torch.ones(2, 2))
  end)
end)

test("logsumexp", function()
  assert_near(torch.tensor({ 0, 0 }):logsumexp(1):item(), math.log(2), 1e-6)
  assert_near(torch.tensor({ 0, 0 }):logsumexp():item(), math.log(2), 1e-6)

  local matrix = torch.tensor({ { 0, 0 }, { 1, 1 } })
  assert_near(matrix:logsumexp(2):totable(), { math.log(2), 1 + math.log(2) }, 1e-6)
  assert_shape(matrix:logsumexp(2, true), { 2, 1 })
end)

test("module-level sorting helpers", function()
  local values, indices = torch.sort(torch.tensor({ 3, 1, 2 }))
  assert_tensor(values, { 1, 2, 3 })
  assert_tensor(indices, { 2, 3, 1 })

  local top_values = torch.topk(torch.tensor({ 1, 3, 2 }), 2)
  assert_tensor(top_values, { 3, 2 })

  assert_tensor(torch.nonzero(torch.tensor({ 0, 1 })), { { 2 } })
end)

test("batched index_select matches individual calls", function()
  local input = torch.arange(12):reshape(3, 4)
  local first = torch.LongTensor({ 1, 3 })
  local second = torch.LongTensor({ 2 })
  local results = input:index_select_batch(1, { first, second })
  assert_eq(#results, 2)
  assert_tensor(results[1], input:index_select(1, first):totable())
  assert_tensor(results[2], input:index_select(1, second):totable())
  assert_error(function() input:index_select_batch(1, {}) end,
    "empty")
  assert_error(function() input:index_select_batch(1, { 1 }) end,
    "not a Tensor")
end)

test("softmax and log_softmax normalize arbitrary dimensions", function()
  torch.manual_seed(64)
  local input = torch.randn(2, 3, 4)
  local soft = input:softmax(1)
  local sums = soft:sum(1)
  for i = 1, sums:numel() do
    assert_near(sums:reshape(-1):get(i), 1, 1e-5)
  end
  local logits = input:log_softmax(1)
  local expected = input - input:logsumexp(1, true)
  assert_near((logits - expected):abs():max():item(), 0, 1e-5)
end)
