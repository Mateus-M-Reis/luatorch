local torch = require("torch")

test("normal_ statistics and reproducibility", function()
  torch.manual_seed(123)
  local sample = torch.empty(200000):normal_(0, 1)
  assert_near(sample:mean():item(), 0, 0.02)
  assert_near(sample:std():item(), 1, 0.02)

  torch.manual_seed(7)
  local first = torch.empty(1000):normal_(3, 2)
  torch.manual_seed(7)
  local second = torch.empty(1000):normal_(3, 2)
  assert_true(first:equal(second))
  assert_near(first:mean():item(), 3, 0.2)
end)

test("exponential_ statistics", function()
  torch.manual_seed(11)
  local sample = torch.empty(200000):exponential_(2)
  assert_near(sample:mean():item(), 0.5, 0.01)
  assert_true(sample:min():item() >= 0)
end)

test("bernoulli_ statistics", function()
  torch.manual_seed(13)
  local sample = torch.empty(100000):bernoulli_(0.3)
  assert_near(sample:mean():item(), 0.3, 0.01)
  assert_true(sample:max():item() <= 1)
end)

test("multinomial with replacement", function()
  torch.manual_seed(17)
  local probabilities = torch.tensor({ 0.2, 0.3, 0.5 }, { dtype = torch.float64 })
  local samples = probabilities:multinomial(30000, true)
  assert_shape(samples, { 30000 })
  assert_dtype(samples, "int64")

  local counts = { 0, 0, 0 }
  for value in samples:iter() do
    counts[value] = counts[value] + 1
  end
  assert_near(counts[1] / 30000, 0.2, 0.02)
  assert_near(counts[2] / 30000, 0.3, 0.02)
  assert_near(counts[3] / 30000, 0.5, 0.02)
end)

test("multinomial without replacement", function()
  local probabilities = torch.tensor({ 0.1, 0.2, 0.3, 0.4 }, { dtype = torch.float64 })
  local samples = probabilities:multinomial(4, false)
  local seen = {}
  for value in samples:iter() do
    assert_false(seen[value] or false, "duplicate sample " .. value)
    seen[value] = true
  end
  assert_eq(samples:size(1), 4)
end)

test("torch.multinomial helper", function()
  local samples = torch.multinomial(
    torch.tensor({ 0.5, 0.5 }, { dtype = torch.float64 }), 8, true)
  assert_shape(samples, { 8 })
end)

test("thread count control", function()
  local original = torch.get_num_threads()
  assert_true(original >= 1)
  torch.set_num_threads(2)
  assert_eq(torch.get_num_threads(), 2)
  torch.set_num_threads(original)
  assert_eq(torch.get_num_threads(), original)
  assert_error(function()
    torch.set_num_threads(0)
  end, "positive")
end)
