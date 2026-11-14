--
-- luafun as the functional layer: lazy pipelines over plain Lua tables and
-- over tensors.  No Moses involved.
--
--   luajit examples/luafun_pipeline.lua
--
local root = (arg and arg[0] and arg[0]:match("^(.*)/examples/[^/]+$")) or "."
package.path = root .. "/lua/?.lua;" .. root .. "/lua/?/init.lua;" .. package.path

local torch = require("torch")
local fun = require("fun")

-- 1. Lazy luafun pipeline over a table.
local numbers = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }
local pipeline = fun.iter(numbers)
  :filter(function(x) return x % 2 == 0 end)
  :map(function(x) return x * x end)
  :take_n(3)
print("first three even squares:", table.concat(fun.totable(pipeline), ", "))

-- 2. Fold to a single value.
local total = fun.iter(numbers):reduce(function(acc, x) return acc + x end, 0)
print("sum 1..10:", total)

-- 3. The same idea on a tensor, with tensor-aware helpers.
local t = torch.arange(1, 11)
local even_squares = t
  :filter(function(x) return x % 2 == 0 end)
  :map(function(x) return x * x end)
print("tensor even squares:", even_squares)

-- 4. Tensor iteration as a luafun source.
local tensor_sum = t:fun():reduce(function(acc, x) return acc + x end, 0)
print("tensor sum via t:fun():", tensor_sum)

-- 5. Iterate values without materialising a Lua table.
local manual_sum = 0
for value in t:iter() do
  manual_sum = manual_sum + value
end
print("tensor sum via t:iter():", manual_sum)

-- 6. A tiny word-count style pipeline to show luafun's generality.
local words = { "lua", "torch", "lua", "fun", "tensor", "lua" }
local counts = fun.tomap(
  fun.iter(words)
    :map(function(word) return word, 1 end))
print("word map contains 'lua':", counts.lua ~= nil)
