--
-- luafun vs Moses: the benchmark behind the functional-layer choice.
--
--   luajit bench/compare_moses.lua
--
-- Moses is vendored under bench/vendor only for this comparison; it is not a
-- dependency of luatorch.  Moses is eager (every stage allocates), luafun is
-- lazy (one pass, intermediate values are not stored).
--
local root = (arg and arg[0] and arg[0]:match("^(.*)/bench/[^/]+$")) or "."
package.path = root .. "/lua/?.lua;" .. root .. "/lua/?/init.lua;" ..
  root .. "/bench/vendor/?.lua;;"

local fun = require("fun")
local moses = require("moses")

local count = 200000
local values = {}
for i = 1, count do
  values[i] = (i * 37) % 1000 / 10
end

local function median(times)
  table.sort(times)
  return times[math.floor(#times / 2) + 1]
end

local function measure(iterations, fn)
  fn()
  collectgarbage("collect")
  local times = {}
  for i = 1, iterations do
    local started = os.clock()
    fn()
    times[i] = os.clock() - started
  end
  return median(times)
end

local function double(x) return x * 2 end
local function is_even(x) return x % 2 == 0 end
local function add(a, b) return a + b end

local function moses_pipeline()
  local doubled = moses.map(values, double)
  local filtered = moses.filter(doubled, is_even)
  return moses.reduce(filtered, add, 0)
end

local function luafun_pipeline()
  return fun.iter(values)
    :map(double)
    :filter(is_even)
    :reduce(add, 0)
end

local function moses_map()
  return moses.map(values, double)
end

local function luafun_map()
  return fun.totable(fun.iter(values):map(double))
end

print(string.format("Moses vs luafun, %d numbers, median of 10 runs", count))
print(string.rep("-", 62))

local moses_time = measure(10, moses_pipeline)
local luafun_time = measure(10, luafun_pipeline)

print(string.format("%-28s %10.2f ms", "Moses map+filter+reduce", moses_time * 1000))
print(string.format("%-28s %10.2f ms", "luafun map+filter+reduce", luafun_time * 1000))
print(string.format("%-28s %10.2fx", "luafun speedup", moses_time / luafun_time))

local moses_map_time = measure(10, moses_map)
local luafun_map_time = measure(10, luafun_map)
print(string.format("%-28s %10.2f ms", "Moses map", moses_map_time * 1000))
print(string.format("%-28s %10.2f ms", "luafun map+totable", luafun_map_time * 1000))
print(string.format("%-28s %10.2fx", "luafun map speedup",
  moses_map_time / luafun_map_time))

assert(moses_pipeline() == luafun_pipeline(), "pipelines disagree")
