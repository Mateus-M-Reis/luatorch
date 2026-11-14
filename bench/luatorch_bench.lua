--
-- luatorch benchmark suite used by bench/compare.py.
-- Prints "name<TAB>best_ms" lines.
--
local ffi = require("ffi")
ffi.cdef[[
struct timespec { long tv_sec; long tv_nsec; };
int clock_gettime(int clk_id, struct timespec* tp);
]]
local timespec = ffi.new("struct timespec")
local function now()
  ffi.C.clock_gettime(1, timespec)
  return tonumber(timespec.tv_sec) + tonumber(timespec.tv_nsec) * 1e-9
end

local torch = require("torch")
local nn = require("torch.nn")
local optim = require("torch.optim")
torch.set_num_threads(1)
torch.manual_seed(0)

local function best(name, iterations, fn)
  for _ = 1, 3 do
    fn()
  end
  local best_time = math.huge
  for _ = 1, iterations do
    local started = now()
    fn()
    local elapsed = now() - started
    if elapsed < best_time then
      best_time = elapsed
    end
  end
  print(string.format("%s\t%.6f", name, best_time * 1000))
  io.flush()
end

local n = 1000000
local a = torch.randn(n)
local b = torch.randn(n)
local c = torch.randn(n)

best("add", 30, function()
  local result = a + b
end)

best("add_inplace", 30, function()
  a:add_(b)
end)

best("chain", 30, function()
  local result = a * b + c
end)

best("chain_relu", 30, function()
  local result = (a * b + c):relu()
end)

best("sum", 30, function()
  local result = a:sum()
end)

best("exp", 20, function()
  local result = a:exp()
end)

local x = torch.randn(512, 512)
local y = torch.randn(512, 512)
best("matmul512", 5, function()
  local z = x:mm(y)
end)

local tiny = torch.randn(16)
local tiny_b = torch.randn(16)
local tiny_c = torch.randn(16)
best("tiny_add", 20000, function()
  local result = tiny + tiny_b
end)

best("tiny_chain", 20000, function()
  local result = (tiny * tiny_b + tiny_c):relu()
end)

-- MLP training -------------------------------------------------------------
local mlp = nn.Sequential(
  nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
local mlp_criterion = nn.CrossEntropyLoss()
local mlp_optimizer = optim.Adam(mlp:parameters(), { lr = 1e-3 })
local mlp_inputs = torch.randn(128, 784)
local mlp_labels = torch.LongTensor(128):fill_(1)

local function mlp_forward()
  return mlp(mlp_inputs)
end

local function mlp_forward_backward()
  local loss = mlp_criterion(mlp(mlp_inputs), mlp_labels)
  mlp_optimizer:zero_grad()
  loss:backward()
end

local function mlp_step()
  local loss = mlp_criterion(mlp(mlp_inputs), mlp_labels)
  mlp_optimizer:zero_grad()
  loss:backward()
  mlp_optimizer:step()
end

for _ = 1, 3 do
  mlp_forward_backward()
  mlp_optimizer:step()
end

best("mlp_forward", 50, mlp_forward)
best("mlp_forward_backward", 50, mlp_forward_backward)
best("mlp_adam_step", 50, function()
  mlp_optimizer:step()
end)
best("mlp_train_step", 50, mlp_step)

-- Fused autograd training --------------------------------------------------
local train_x = torch.randn(n, { requires_grad = true })
local train_y = torch.randn(n, { requires_grad = true })
local train_z = torch.randn(n, { requires_grad = true })
local fused_train = torch.fuse_train(function(a, b, c)
  return (a * b + c):relu()
end)

local function chain_train_eager()
  train_x:zero_grad()
  train_y:zero_grad()
  train_z:zero_grad()
  local output = (train_x * train_y + train_z):relu()
  output:sum():backward()
end

local function chain_train_fused()
  train_x:zero_grad()
  train_y:zero_grad()
  train_z:zero_grad()
  local output = fused_train(train_x, train_y, train_z)
  output:sum():backward()
end

best("chain_train", 15, chain_train_eager)
best("chain_train_fused", 15, chain_train_fused)

-- Fusion ------------------------------------------------------------------
local fused_chain = torch.fuse(function(p, q, r)
  return (p * q + r):relu()
end, { backend = "c" })

local fused_into = torch.empty(n)
local fused_tiny = torch.fuse(function(p, q, r)
  return (p * q + r):relu()
end, { backend = "c" })
local tiny_out = torch.empty(16)

-- Warm up compilation plus buffer reuse.
fused_chain:into(fused_into, a, b, c)
fused_tiny:into(tiny_out, tiny, tiny_b, tiny_c)

best("chain_fused", 30, function()
  fused_chain:into(fused_into, a, b, c)
end)

best("chain_fused_alloc", 30, function()
  local result = fused_chain(a, b, c)
end)

best("tiny_chain_fused", 20000, function()
  fused_tiny:into(tiny_out, tiny, tiny_b, tiny_c)
end)
