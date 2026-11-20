--
-- luatorch micro-benchmarks.
--
--   luajit bench/bench.lua
--
-- Numbers are wall-clock-ish (os.clock, single-threaded) and meant for
-- relative comparisons on the same machine, not as authoritative results.
--
local root = (arg and arg[0] and arg[0]:match("^(.*)/bench/[^/]+$")) or "."
package.path = root .. "/lua/?.lua;" .. root .. "/lua/?/init.lua;" .. package.path

local torch = require("torch")
local nn = require("torch.nn")
local optim = require("torch.optim")
local fun = require("fun")

local results = {}

local function median(times)
  table.sort(times)
  return times[math.floor(#times / 2) + 1]
end

local function bench(name, iterations, fn, note)
  fn() -- warmup
  collectgarbage("collect")
  local times = {}
  for i = 1, iterations do
    local started = os.clock()
    fn()
    times[i] = os.clock() - started
  end
  local seconds = median(times)
  results[#results + 1] = { name = name, seconds = seconds, note = note }
  return seconds
end

-- Best-of variant for the epilogue rows: the additive dispatch/memory saving
-- is small relative to a BLAS call, so the minimum is less noisy than the
-- median on a throttling CPU.  Conditions are printed by the caller.
local function bench_best(name, iterations, fn, note)
  for _ = 1, 10 do
    fn()
  end
  collectgarbage("collect")
  local best = math.huge
  for _ = 1, iterations do
    local started = os.clock()
    fn()
    local elapsed = os.clock() - started
    if elapsed < best then
      best = elapsed
    end
  end
  results[#results + 1] = { name = name, seconds = best, note = note }
  return best
end

local function format_row(row)
  local milliseconds = row.seconds * 1000
  local per_iteration = row.note or ""
  return string.format("%-34s %12.3f ms  %s", row.name, milliseconds, per_iteration)
end

print(string.format("luatorch %s benchmarks (libtorch %s)", torch.VERSION or "0.7.0", torch.version))
print(string.rep("-", 78))

torch.manual_seed(42)

-- creation --------------------------------------------------------------
bench("zeros(1000, 1000)", 20, function()
  torch.zeros(1000, 1000)
end, "8 MB")

bench("randn(1000, 1000)", 20, function()
  torch.randn(1000, 1000)
end, "8 MB")

-- elementwise -----------------------------------------------------------
local a = torch.randn(1000000)
local b = torch.randn(1000000)

local add_seconds = bench("add: 1M float32 tensors", 30, function()
  local c = a + b
end, "12 MB traffic")
print(string.format("%-34s %12.1f GB/s", "  effective bandwidth",
  (3 * 4 * 1000000) / add_seconds / 1e9))

bench("add scalar: 1M float32", 30, function()
  local c = a + 1.5
end)

bench("sum: 1M float32", 30, function()
  local s = a:sum()
end)

bench("relu: 1M float32", 30, function()
  local c = a:relu()
end)

-- linear algebra --------------------------------------------------------
for _, size in ipairs({ 128, 256, 512 }) do
  local x = torch.randn(size, size)
  local y = torch.randn(size, size)
  local seconds = bench(string.format("matmul: %dx%d float32", size, size), 10, function()
    local z = x:mm(y)
  end)
  local gflops = (2 * size ^ 3) / seconds / 1e9
  print(string.format("%-34s %12.1f GFLOP/s", "  throughput", gflops))
end

-- sorting and linear algebra ------------------------------------------
local million = torch.randn(1000000)
bench("sort 1M float32", 5, function()
  local values, indices = million:sort(1)
end)
bench("topk 1M, k=10", 10, function()
  local values, indices = million:topk(10)
end)

local square = torch.randn(128, 128)
bench("svd 128x128 float32", 5, function()
  local u, s, vh = square:svd(false)
end)

-- per-call overhead -----------------------------------------------------
local tiny = torch.randn(16)
bench("small tensor add (16 elements)", 20000, function()
  local c = tiny + tiny
end, "per call")

bench("scalar item read", 20000, function()
  local value = tiny:get(1)
end, "per call")

-- Lua-side iteration ----------------------------------------------------
local values = torch.randn(200000):totable()

local plain_seconds = bench("plain Lua loop over 200k", 10, function()
  local total = 0
  for i = 1, #values do
    total = total + values[i]
  end
end)

local fun_seconds = bench("luafun map+reduce over 200k", 10, function()
  local total = fun.iter(values)
    :map(function(x) return x * 2 end)
    :reduce(function(acc, x) return acc + x end, 0)
end)

print(string.format("%-34s %12.2fx", "  luafun vs plain loop", fun_seconds / plain_seconds))

local tensor = torch.randn(200000)

bench("tensor:iter sum 200k", 10, function()
  local total = 0
  for value in tensor:iter() do
    total = total + value
  end
end)

bench("tensor:totable 200k", 10, function()
  local t = tensor:totable()
end, "allocates a Lua table")

-- serialization ---------------------------------------------------------
local path = os.tmpname() .. ".pt"
local saved = torch.randn(1000, 100)
bench("torch.save 100k floats", 10, function()
  saved:save(path)
end, "PyTorch zip format")
bench("torch.load 100k floats", 10, function()
  torch.load(path)
end)
os.remove(path)

-- neural networks -------------------------------------------------------
local mlp = nn.Sequential(
  nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
local batch = torch.randn(128, 784)
local labels = torch.LongTensor(128):fill_(1)
local criterion = nn.CrossEntropyLoss()
local adam = optim.Adam(mlp:parameters(), { lr = 1e-3 })

bench("MLP forward 128x784", 50, function()
  local output = mlp(batch)
end)

bench("MLP forward+backward 128x784", 30, function()
  local output = mlp(batch)
  local loss = criterion(output, labels)
  mlp:zero_grad()
  loss:backward()
end)

bench("MLP Adam step 128x784", 30, function()
  local output = mlp(batch)
  local loss = criterion(output, labels)
  adam:zero_grad()
  loss:backward()
  adam:step()
end)

local convnet = nn.Sequential(
  nn.Conv2d(3, 16, 3, { padding = 1 }),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Conv2d(16, 32, 3, { padding = 1 }),
  nn.ReLU(),
  nn.MaxPool2d(2))
local images = torch.randn(8, 3, 32, 32)

bench("ConvNet forward 8x3x32x32", 20, function()
  local output = convnet(images)
end)

-- Linear / convolution epilogues ----------------------------------------
-- Compare one fused call against the existing addmm/conv2d + activation
-- sequence.  Inference conditions: float32, no autograd, same BLAS/ATen
-- kernels on both sides.
local function eager_activation(tensor, name)
  if name == "relu" then return tensor:relu() end
  if name == "gelu" then return tensor:gelu() end
  if name == "sigmoid" then return tensor:sigmoid() end
  if name == "tanh" then return tensor:tanh() end
  error("unknown activation: " .. tostring(name))
end

local linear_input = torch.randn(128, 784)
local linear_weight = torch.randn(256, 784)
local linear_bias = torch.randn(256)

local previous_threads = torch.get_num_threads()
torch.set_num_threads(1)
print("epilogue benchmarks: float32, no autograd, 1 thread, best-of")

torch.no_grad(function()
  for _, name in ipairs({ "relu", "gelu", "sigmoid", "tanh" }) do
    local eager_seconds = bench_best(
      "Linear 128x784->256 eager+" .. name, 200, function()
        eager_activation(
          linear_bias:addmm(linear_input, linear_weight:t()), name)
      end, "best-of")
    local fused_seconds = bench_best(
      "Linear 128x784->256 fused+" .. name, 200, function()
        torch.fuse.linear(
          linear_input, linear_weight, linear_bias, name)
      end, "best-of")
    print(string.format("%-34s %12.2fx", "  fused vs eager " .. name,
      eager_seconds / fused_seconds))
  end
end)

local conv_input = torch.randn(8, 3, 32, 32)
local conv_weight = torch.randn(16, 3, 3, 3)
local conv_bias = torch.randn(16)

torch.no_grad(function()
  for _, name in ipairs({ "relu", "gelu", "sigmoid", "tanh" }) do
    local eager_seconds = bench_best(
      "Conv2d 8x3x32x32->16 eager+" .. name, 100, function()
        eager_activation(
          torch.conv2d(conv_input, conv_weight, conv_bias, 1, 1, 1, 1),
          name)
      end, "best-of")
    local fused_seconds = bench_best(
      "Conv2d 8x3x32x32->16 fused+" .. name, 100, function()
        torch.fuse.conv2d(conv_input, conv_weight, conv_bias, {
          activation = name,
          padding = 1,
        })
      end, "best-of")
    print(string.format("%-34s %12.2fx", "  fused vs eager " .. name,
      eager_seconds / fused_seconds))
  end
end)

torch.set_num_threads(previous_threads)

print(string.rep("-", 78))
print(string.format("%-34s %12s", "benchmark", "time"))

for _, row in ipairs(results) do
  print(format_row(row))
end
