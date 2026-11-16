--
-- Opt-in profiler for tensor method dispatch.
--
--   torch.profiler.start()
--   local y = a:matmul(b)
--   torch.profiler.stop()
--   print(torch.profiler.format())
--
-- Methods called with `Tensor.foo(t, ...)` bypass the per-instance metatable
-- and are not counted.  The profiler adds a wrapper for each unique tensor
-- method name only while enabled.
--
local profiler = {}
local enabled = false
local stats = {}
local order = {}

local function pack(...)
  return { n = select("#", ...), ... }
end

local function record(name, elapsed)
  local entry = stats[name]
  if entry == nil then
    entry = { name = name, count = 0, total = 0, max = 0 }
    stats[name] = entry
    order[#order + 1] = name
  end
  entry.count = entry.count + 1
  entry.total = entry.total + elapsed
  if elapsed > entry.max then entry.max = elapsed end
end

function profiler.is_enabled()
  return enabled
end

function profiler.start()
  for name in pairs(stats) do stats[name] = nil end
  for i = #order, 1, -1 do order[i] = nil end
  enabled = true
end

function profiler.stop()
  enabled = false
end

function profiler.reset()
  for name in pairs(stats) do stats[name] = nil end
  for i = #order, 1, -1 do order[i] = nil end
end

-- Called by the tensor metatable wrapper.
function profiler.call(name, fn, ...)
  local started = os.clock()
  local result = pack(pcall(fn, ...))
  record(name, os.clock() - started)
  if not result[1] then error(result[2], 0) end
  return unpack(result, 2, result.n)
end

function profiler.report()
  local rows = {}
  for _, name in ipairs(order) do
    local entry = stats[name]
    rows[#rows + 1] = {
      name = entry.name,
      count = entry.count,
      total = entry.total,
      average = entry.total / entry.count,
      max = entry.max,
    }
  end
  table.sort(rows, function(a, b)
    if a.total == b.total then return a.name < b.name end
    return a.total > b.total
  end)
  return rows
end

function profiler.format()
  local rows = profiler.report()
  if #rows == 0 then return "profiler: no tensor method calls recorded" end
  local lines = { "profiler: tensor method calls (CPU time)" }
  for _, row in ipairs(rows) do
    lines[#lines + 1] = string.format(
      "  %-18s %8d calls  %9.3f ms total  %8.3f ms avg  %8.3f ms max",
      row.name, row.count, row.total * 1000, row.average * 1000,
      row.max * 1000)
  end
  return table.concat(lines, "\n")
end

return profiler
