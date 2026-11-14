--
-- luatorch / init.lua
--
-- Public entry point.  Mirrors the ergonomics of both PyTorch and Torch7:
--
--   local torch = require("torch")
--   local x = torch.randn(2, 3)
--   local y = x:matmul(torch.ones(3, 4))
--   print(y)
--
-- Functional utilities come from luafun (fun.lua) instead of Moses:
--
--   local fun = require("fun")          -- or torch.fun
--   fun.iter({1, 2, 3}):map(function(x) return x * x end):totable()
--
local libmod = require("torch.lib")
local ffi = libmod.ffi
local lib = libmod.lib
local Tensor = require("torch.tensor")
local fun = require("fun")

local torch = {}

torch.version = libmod.version
torch.__version = libmod.version
torch.Tensor = Tensor
torch.is_tensor = Tensor.is_tensor
torch.ALL_DIMS = Tensor.ALL_DIMS
torch.fun = fun
torch.luafun = fun

-- dtype ids (also accepted as strings everywhere) ---------------------------
torch.float32 = 0
torch.float = 0
torch.float64 = 1
torch.double = 1
torch.int64 = 2
torch.long = 2
torch.int32 = 3
torch.int = 3
torch.int16 = 4
torch.short = 4
torch.int8 = 5
torch.char = 5
torch.uint8 = 6
torch.byte = 6
torch.bool = 7
torch.boolean = 7
torch.float16 = 8
torch.half = 8

local default_dtype = 0

local function with_default(opts)
  opts = opts or {}
  if opts.dtype ~= nil then
    return opts
  end
  local copy = {}
  for key, value in pairs(opts) do
    copy[key] = value
  end
  copy.dtype = default_dtype
  return copy
end

-- creation ------------------------------------------------------------------

function torch.tensor(value, opts)
  opts = opts or {}
  if Tensor.is_tensor(value) then
    return value:clone()
  end
  if type(value) == "table" then
    return Tensor.from_table(value, opts)
  end
  if type(value) == "number" or type(value) == "boolean" then
    return Tensor.from_scalar(value, opts)
  end
  error("torch.tensor expects a table, number, boolean or Tensor", 2)
end

local function size_creator(kind)
  return function(...)
    local shape, opts = Tensor._parse_shape_opts(...)
    return Tensor._create(kind, shape, with_default(opts))
  end
end

torch.zeros = size_creator("zeros")
torch.ones = size_creator("ones")
torch.empty = size_creator("empty")
torch.rand = size_creator("rand")
torch.randn = size_creator("randn")

function torch.full(...)
  local args = { ... }
  local opts = {}
  if #args > 0 and Tensor._is_options(args[#args]) then
    opts = table.remove(args)
  end
  local value = table.remove(args)
  if type(value) ~= "number" then
    error("torch.full expects a numeric fill value", 2)
  end
  local shape
  if #args == 1 and type(args[1]) == "table" then
    shape = args[1]
  else
    shape = args
  end
  if #shape == 0 then
    error("torch.full expects a tensor shape", 2)
  end
  return Tensor._create("full", shape, with_default(opts), value)
end

function torch.eye(n, opts)
  opts = with_default(opts)
  return Tensor._wrap(lib.lt_eye(n, Tensor.resolve_dtype(opts.dtype)))
end

function torch.arange(...)
  local args = { ... }
  local opts = {}
  if #args > 0 and Tensor._is_options(args[#args]) then
    opts = table.remove(args)
  end
  if #args < 1 or #args > 3 then
    error("torch.arange expects start, stop[, step]", 2)
  end
  local start, stop, step
  if #args == 1 then
    start, stop, step = 0, args[1], 1
  elseif #args == 2 then
    start, stop, step = args[1], args[2], 1
  else
    start, stop, step = args[1], args[2], args[3]
  end
  local dtype = opts.dtype
  if dtype == nil then
    dtype = (start % 1 == 0 and stop % 1 == 0 and step % 1 == 0) and 2 or 0
  end
  return Tensor._wrap(lib.lt_arange(start, stop, step, Tensor.resolve_dtype(dtype)))
end

function torch.linspace(start, stop, steps, opts)
  steps = steps or 100
  opts = with_default(opts)
  return Tensor._wrap(lib.lt_linspace(start, stop, steps, Tensor.resolve_dtype(opts.dtype)))
end

function torch.randperm(n, opts)
  opts = with_default(opts)
  return Tensor._wrap(lib.lt_randperm(n, Tensor.resolve_dtype(opts.dtype)))
end

-- like-constructors ----------------------------------------------------------

function torch.zeros_like(t, opts)
  local out = t:detach():clone()
  if opts and opts.dtype then
    out = out:to(opts.dtype)
  end
  return out:zero_()
end

function torch.ones_like(t, opts)
  return torch.zeros_like(t, opts):fill_(1)
end

function torch.full_like(t, value, opts)
  return torch.zeros_like(t, opts):fill_(value)
end

local function random_like(t, opts, generator)
  opts = opts or {}
  local dtype = opts.dtype or t:dtype()
  local sizes = t:sizes()
  if #sizes == 0 then
    return generator(1, { dtype = dtype }):reshape({})
  end
  return generator(sizes, { dtype = dtype })
end

function torch.rand_like(t, opts)
  return random_like(t, opts, torch.rand)
end

function torch.randn_like(t, opts)
  return random_like(t, opts, torch.randn)
end

-- joining --------------------------------------------------------------------

local function tensor_array(tensors, what)
  if type(tensors) ~= "table" or #tensors == 0 then
    error(what .. " expects a non-empty table of tensors", 3)
  end
  local n = #tensors
  local array = ffi.new("lt_tensor*[?]", n)
  for i = 1, n do
    local t = tensors[i]
    if not Tensor.is_tensor(t) then
      error(what .. " expects tensors, got " .. type(t) .. " at position " .. i, 3)
    end
    array[i - 1] = t.h
  end
  return array, n
end

function torch.cat(tensors, dim)
  local array, n = tensor_array(tensors, "torch.cat")
  dim = dim or 1
  local cdim = dim > 0 and dim - 1 or dim
  return Tensor._wrap(lib.lt_cat(array, n, cdim))
end

function torch.stack(tensors, dim)
  local array, n = tensor_array(tensors, "torch.stack")
  dim = dim or 1
  local cdim = dim > 0 and dim - 1 or dim
  return Tensor._wrap(lib.lt_stack(array, n, cdim))
end

function torch.where(condition, a, b)
  if not (Tensor.is_tensor(condition) and Tensor.is_tensor(a) and Tensor.is_tensor(b)) then
    error("torch.where expects three tensors", 2)
  end
  return Tensor._wrap(lib.lt_where(condition.h, a.h, b.h))
end

-- Torch7-style typed constructors -------------------------------------------

local function typed_constructor(dtype)
  return function(...)
    local args = { ... }
    if #args == 1 and type(args[1]) == "table" then
      return Tensor.from_table(args[1], { dtype = dtype })
    end
    if #args == 1 and (type(args[1]) == "number" or type(args[1]) == "boolean") then
      return Tensor.from_scalar(args[1], { dtype = dtype })
    end
    local shape = Tensor._parse_shape_opts(...)
    return Tensor._create("empty", shape, { dtype = dtype })
  end
end

torch.FloatTensor = typed_constructor(0)
torch.DoubleTensor = typed_constructor(1)
torch.LongTensor = typed_constructor(2)
torch.IntTensor = typed_constructor(3)
torch.ShortTensor = typed_constructor(4)
torch.CharTensor = typed_constructor(5)
torch.ByteTensor = typed_constructor(6)
torch.BoolTensor = typed_constructor(7)
torch.HalfTensor = typed_constructor(8)

-- autograd context -----------------------------------------------------------

function torch.set_grad_enabled(enabled)
  lib.lt_set_grad_enabled(enabled and 1 or 0)
end

function torch.is_grad_enabled()
  return lib.lt_is_grad_enabled() == 1
end

function torch.no_grad(fn)
  local previous = lib.lt_is_grad_enabled()
  lib.lt_set_grad_enabled(0)
  local ok, result = pcall(fn)
  lib.lt_set_grad_enabled(previous)
  if not ok then
    error(result, 0)
  end
  return result
end

-- free-function aliases ------------------------------------------------------

torch.add = function(a, b) return a:add(b) end
torch.sub = function(a, b) return a:sub(b) end
torch.mul = function(a, b) return a:mul(b) end
torch.div = function(a, b) return a:div(b) end
torch.pow = function(a, b) return a:pow(b) end
torch.matmul = function(a, b) return a:matmul(b) end
torch.mm = function(a, b) return a:mm(b) end
torch.bmm = function(a, b) return a:bmm(b) end
torch.dot = function(a, b) return a:dot(b) end
torch.sum = function(t, dim, keepdim) return t:sum(dim, keepdim) end
torch.mean = function(t, dim, keepdim) return t:mean(dim, keepdim) end
torch.prod = function(t, dim, keepdim) return t:prod(dim, keepdim) end
torch.max = function(t, dim, keepdim) return t:max(dim, keepdim) end
torch.min = function(t, dim, keepdim) return t:min(dim, keepdim) end
torch.argmax = function(t, dim, keepdim) return t:argmax(dim, keepdim) end
torch.argmin = function(t, dim, keepdim) return t:argmin(dim, keepdim) end
torch.transpose = function(t, d1, d2) return t:transpose(d1, d2) end
torch.reshape = function(t, ...) return t:reshape(...) end
torch.clamp = function(t, lo, hi) return t:clamp(lo, hi) end
torch.relu = function(t) return t:relu() end
torch.sigmoid = function(t) return t:sigmoid() end
torch.tanh = function(t) return t:tanh() end
torch.exp = function(t) return t:exp() end
torch.log = function(t) return t:log() end
torch.sqrt = function(t) return t:sqrt() end
torch.abs = function(t) return t:abs() end

function torch.equal(a, b)
  return a:equal(b)
end

function torch.allclose(a, b, rtol, atol)
  return a:allclose(b, rtol, atol)
end

-- utilities ------------------------------------------------------------------

function torch.manual_seed(seed)
  lib.lt_manual_seed(seed)
  return seed
end

function torch.set_default_dtype(dtype)
  default_dtype = Tensor.resolve_dtype(dtype)
end

function torch.get_default_dtype()
  return default_dtype
end

function torch.save(t, path)
  if not Tensor.is_tensor(t) then
    error("torch.save expects a tensor", 2)
  end
  return t:save(path)
end

function torch.load(path)
  return Tensor._wrap(lib.lt_load(path))
end

function torch.print(t)
  if not Tensor.is_tensor(t) then
    error("torch.print expects a tensor", 2)
  end
  print(t:format())
end

function torch.dtype_name(dtype)
  return Tensor.dtype_names[Tensor.resolve_dtype(dtype)]
end

return torch
