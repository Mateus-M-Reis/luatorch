--
-- torch.fuse
--
-- Trace a pure elementwise function and compile it to a single LuaJIT loop.
-- The generated kernel avoids intermediate tensors, which is where eager
-- PyTorch pays for chains like `(a * b + c):relu()`.
--
--   local fused = torch.fuse(function(a, b, c)
--     return (a * b + c):relu()
--   end)
--   local out = fused(a, b, c)
--
-- Supported operations: + - * / ^, unary -, relu, abs, neg, clamp, exp, log,
-- sqrt, tanh, sigmoid.  Tensor inputs must have the same shape and dtype
-- (float32 or float64); scalars may be mixed in.
--
-- If tracing fails, the original function is called eagerly (correct, slower).
-- If any input requires grad, the eager function is used so autograd works;
-- fused kernels are currently inference-only.
--
local ffi = require("ffi")
local libmod = require("torch.lib")
local lib = libmod.lib
local Tensor = require("torch.tensor")
local fusec = require("torch.fusec")

ffi.cdef[[
float expf(float); float logf(float); float sqrtf(float); float tanhf(float); float powf(float, float);
double exp(double); double log(double); double sqrt(double); double tanh(double); double pow(double, double);
]]

local unpack = table.unpack or unpack

local function pack(...)
  return { n = select("#", ...), ... }
end

-- Symbolic expression nodes -------------------------------------------------

local Symbol = {}
Symbol.__index = Symbol
Symbol.__name = "FuseSymbol"

local function node(op, args)
  return setmetatable({ op = op, args = args }, Symbol)
end

local function input_node(index)
  return setmetatable({ op = "tensor_arg", arg = index }, Symbol)
end

local function scalar_node(index)
  return setmetatable({ op = "scalar_arg", arg = index }, Symbol)
end

local function is_symbol(value)
  return type(value) == "table" and getmetatable(value) == Symbol
end

Symbol.__add = function(a, b) return node("add", { a, b }) end
Symbol.__sub = function(a, b) return node("sub", { a, b }) end
Symbol.__mul = function(a, b) return node("mul", { a, b }) end
Symbol.__div = function(a, b) return node("div", { a, b }) end
Symbol.__pow = function(a, b) return node("pow", { a, b }) end
Symbol.__unm = function(a) return node("neg", { a }) end

function Symbol:add(other) return self + other end
function Symbol:sub(other) return self - other end
function Symbol:mul(other) return self * other end
function Symbol:div(other) return self / other end
function Symbol:pow(other) return self ^ other end
function Symbol:neg() return node("neg", { self }) end
function Symbol:abs() return node("abs", { self }) end
function Symbol:relu() return node("relu", { self }) end
function Symbol:exp() return node("exp", { self }) end
function Symbol:log() return node("log", { self }) end
function Symbol:sqrt() return node("sqrt", { self }) end
function Symbol:tanh() return node("tanh", { self }) end
function Symbol:sigmoid() return node("sigmoid", { self }) end
function Symbol:clamp(low, high) return node("clamp", { self, low, high }) end

-- Code generation -----------------------------------------------------------

local math_functions = {
  [0] = { -- float32
    exp = "ffi.C.expf",
    log = "ffi.C.logf",
    sqrt = "ffi.C.sqrtf",
    tanh = "ffi.C.tanhf",
    pow = "ffi.C.powf",
    ctype = "float",
  },
  [1] = { -- float64
    exp = "ffi.C.exp",
    log = "ffi.C.log",
    sqrt = "ffi.C.sqrt",
    tanh = "ffi.C.tanh",
    pow = "ffi.C.pow",
    ctype = "double",
  },
}

local function format_number(value)
  if value ~= value then
    return "(0/0)"
  elseif value == math.huge then
    return "math.huge"
  elseif value == -math.huge then
    return "(-math.huge)"
  end
  return string.format("%.17g", value)
end

local function new_temporary(context, expression)
  context.counter = context.counter + 1
  local variable = "v" .. context.counter
  context.lines[#context.lines + 1] =
    string.format("local %s = %s", variable, expression)
  return variable
end

local function emit(node, context)
  if type(node) == "number" then
    return format_number(node)
  end
  if type(node) ~= "table" or getmetatable(node) ~= Symbol then
    error("fuse: invalid expression node", 0)
  end

  local memo = context.memo[node]
  if memo ~= nil then
    return memo
  end

  local op = node.op
  local value

  if op == "tensor_arg" then
    value = new_temporary(context,
      string.format("t%d[i]", node.arg))
  elseif op == "scalar_arg" then
    value = "s" .. node.arg
  else
    local args = node.args
    local first = args and emit(args[1], context) or nil

    if op == "neg" then
      value = new_temporary(context, "(-" .. first .. ")")
    elseif op == "abs" then
      value = new_temporary(context, "math.abs(" .. first .. ")")
    elseif op == "relu" then
      -- Branchless: a conditional `and/or` defeats LuaJIT's JIT here and is
      -- several times slower than math.max for random data.
      value = new_temporary(context, "math.max(" .. first .. ", 0)")
    elseif op == "exp" then
      value = new_temporary(context, context.math.exp .. "(" .. first .. ")")
    elseif op == "log" then
      value = new_temporary(context, context.math.log .. "(" .. first .. ")")
    elseif op == "sqrt" then
      value = new_temporary(context, context.math.sqrt .. "(" .. first .. ")")
    elseif op == "tanh" then
      value = new_temporary(context, context.math.tanh .. "(" .. first .. ")")
    elseif op == "sigmoid" then
      value = new_temporary(context,
        "(1 / (1 + " .. context.math.exp .. "(-" .. first .. ")))")
    elseif op == "clamp" then
      local low = emit(args[2], context)
      local high = emit(args[3], context)
      value = new_temporary(context,
        "math.min(math.max(" .. first .. ", " .. low .. "), " .. high .. ")")
    elseif op == "add" or op == "sub" or op == "mul" or op == "div" then
      local second = emit(args[2], context)
      local symbol = ({ add = "+", sub = "-", mul = "*", div = "/" })[op]
      value = new_temporary(context,
        "(" .. first .. " " .. symbol .. " " .. second .. ")")
    elseif op == "pow" then
      local exponent = args[2]
      if type(exponent) == "number" then
        if exponent == 2 then
          value = new_temporary(context, "(" .. first .. " * " .. first .. ")")
        elseif exponent == 3 then
          value = new_temporary(context,
            "(" .. first .. " * " .. first .. " * " .. first .. ")")
        elseif exponent == 0.5 then
          value = new_temporary(context, context.math.sqrt .. "(" .. first .. ")")
        elseif exponent == -1 then
          value = new_temporary(context, "(1 / " .. first .. ")")
        else
          value = new_temporary(context,
            context.math.pow .. "(" .. first .. ", " .. format_number(exponent) .. ")")
        end
      else
        local second = emit(exponent, context)
        value = new_temporary(context,
          context.math.pow .. "(" .. first .. ", " .. second .. ")")
      end
    else
      error("fuse: unsupported operation '" .. tostring(op) .. "'", 0)
    end
  end

  context.memo[node] = value
  return value
end

local function generate(expression, signature, dtype)
  local math = math_functions[dtype]
  if math == nil then
    return nil, "unsupported dtype for fusion"
  end

  local context = {
    lines = {},
    counter = 0,
    memo = {},
    math = math,
  }

  local result
  local ok, error_message = pcall(function()
    result = emit(expression, context)
  end)
  if not ok then
    return nil, error_message
  end

  local parameters = {}
  local prelude = { "local out = ffi.cast(\"" .. math.ctype .. "*\", out_ptr)" }
  for index = 1, #signature do
    parameters[index] = "p" .. index
    if signature:sub(index, index) == "T" then
      prelude[#prelude + 1] = string.format(
        "local t%d = ffi.cast(\"%s*\", p%d)", index, math.ctype, index)
    else
      prelude[#prelude + 1] = string.format("local s%d = p%d", index, index)
    end
  end

  local source = table.concat({
    "local ffi = require(\"ffi\")",
    "return function(out_ptr, n, " .. table.concat(parameters, ", ") .. ")",
    "  " .. table.concat(prelude, "\n  "),
    "  for i = 0, n - 1 do",
    "    " .. table.concat(context.lines, "\n    "),
    "    out[i] = " .. result,
    "  end",
    "end",
  }, "\n")

  return source
end

-- Tracing -------------------------------------------------------------------

local function trace(function_value, args)
  local count = args.n
  local symbolic = {}
  for index = 1, count do
    local value = args[index]
    if Tensor.is_tensor(value) then
      symbolic[index] = input_node(index)
    elseif type(value) == "number" then
      symbolic[index] = scalar_node(index)
    else
      return nil, "unsupported argument type: " .. type(value)
    end
  end

  local ok, result, extra = pcall(function_value, unpack(symbolic, 1, count))
  if not ok then
    return nil, tostring(result)
  end
  if extra ~= nil then
    return nil, "fusion only supports a single return value"
  end
  if not is_symbol(result) then
    return nil, "fusion function did not return a tensor expression"
  end
  return result
end

-- Fused callable ------------------------------------------------------------

local Fused = {}
Fused.__index = Fused

function Fused.new(function_value, options)
  if type(function_value) ~= "function" then
    error("torch.fuse expects a function", 2)
  end
  options = options or {}
  local backend = options.backend
  if backend == nil then
    backend = os.getenv("LUATORCH_FUSE_BACKEND") or "auto"
  end
  if backend ~= "auto" and backend ~= "c" and backend ~= "lua" then
    error("torch.fuse: backend must be 'auto', 'c' or 'lua'", 3)
  end
  return setmetatable({
    fn = function_value,
    backend = backend,
    traces = {},
    kernels = {},
    last_error = nil,
    last_c_error = nil,
    fallbacks = 0,
  }, Fused)
end

function Fused:_execute(given_output, ...)
  local count = select("#", ...)
  local args = pack(...)

  local function fallback()
    local result = self.fn(unpack(args, 1, count))
    if given_output ~= nil then
      if not Tensor.is_tensor(result) then
        error("fused:into requires a function that returns one tensor", 3)
      end
      given_output:copy_(result)
      return given_output
    end
    return result
  end

  if given_output ~= nil and not Tensor.is_tensor(given_output) then
    error("fused:into expects a Tensor output buffer", 3)
  end

  local signature_parts = {}
  local dtype = nil
  local shape = nil
  local requires_grad = false
  local grad_enabled = lib.lt_is_grad_enabled() == 1

  for index = 1, count do
    local value = args[index]
    if Tensor.is_tensor(value) then
      signature_parts[index] = "T"
      if grad_enabled and value:requires_grad() then
        requires_grad = true
      end
      local value_dtype = value:dtype_id()
      if dtype == nil then
        dtype = value_dtype
        dtype_name = value:dtype()
        shape = value:sizes()
      elseif value_dtype ~= dtype then
        return fallback()
      end
      local sizes = value:sizes()
      if #sizes ~= #shape then
        return fallback()
      end
      for dimension = 1, #sizes do
        if sizes[dimension] ~= shape[dimension] then
          return fallback()
        end
      end
    elseif type(value) == "number" then
      signature_parts[index] = "S"
    else
      return fallback()
    end
  end

  if shape == nil or (dtype ~= 0 and dtype ~= 1) then
    return fallback()
  end

  if given_output ~= nil then
    if given_output:dtype_id() ~= dtype then
      error("fused:into output dtype does not match the inputs", 3)
    end
    local output_sizes = given_output:sizes()
    if #output_sizes ~= #shape then
      error("fused:into output shape does not match the inputs", 3)
    end
    for dimension = 1, #shape do
      if output_sizes[dimension] ~= shape[dimension] then
        error("fused:into output shape does not match the inputs", 3)
      end
    end
    if not given_output:is_contiguous() then
      error("fused:into expects a contiguous output buffer", 3)
    end
    if given_output:requires_grad() then
      error("fused:into does not support output tensors requiring grad", 3)
    end
  end

  if requires_grad then
    -- Fused kernels are inference-only for now; eager keeps autograd intact.
    if given_output ~= nil then
      error("fused:into does not support autograd inputs", 3)
    end
    return fallback()
  end

  local signature = table.concat(signature_parts)
  local expression = self.traces[signature]
  if expression == nil then
    local traced, error_message = trace(self.fn, args)
    if traced == nil then
      self.traces[signature] = false
      self.last_error = error_message
      return fallback()
    end
    expression = traced
    self.traces[signature] = expression
  elseif expression == false then
    return fallback()
  end

  local key = signature .. "|" .. dtype
  local kernel = self.kernels[key]
  if kernel == nil then
    if self.backend ~= "lua" then
      local c_kernel, c_error = fusec.compile(expression, signature, dtype)
      if c_kernel ~= nil then
        kernel = { kind = "c", fn = c_kernel }
        self.kernels[key] = kernel
      else
        self.last_c_error = c_error
      end
    end
    if kernel == nil then
      local source, error_message = generate(expression, signature, dtype)
      if source == nil then
        self.traces[signature] = false
        self.last_error = error_message
        return fallback()
      end
      local chunk, load_error = loadstring(source, "=torch.fuse")
      if chunk == nil then
        self.traces[signature] = false
        self.last_error = load_error
        return fallback()
      end
      kernel = { kind = "lua", fn = chunk() }
      self.last_source = source
      self.kernels[key] = kernel
    end
  end

  local output = given_output
  if output == nil then
    output = Tensor._create("empty", shape, { dtype = dtype })
  end
  local elements = output:numel()
  if elements == 0 then
    return output
  end

  local contiguous = {}
  local pointers = {}
  for index = 1, count do
    if signature_parts[index] == "T" then
      local copy = args[index]:contiguous()
      contiguous[index] = copy
      local pointer = lib.lt_data_ptr(copy.h)
      if pointer == nil then
        return fallback()
      end
      pointers[index] = pointer
    end
  end

  local output_pointer = lib.lt_data_ptr(output.h)
  if output_pointer == nil then
    return fallback()
  end

  if kernel.kind == "c" then
    local tensor_count, scalar_count = 0, 0
    for index = 1, count do
      if signature_parts[index] == "T" then
        tensor_count = tensor_count + 1
      else
        scalar_count = scalar_count + 1
      end
    end

    local pointer_array = ffi.new("const void*[?]", tensor_count + 1)
    local scalar_array = ffi.new("double[?]", math.max(scalar_count, 1))
    local tensor_index, scalar_index = 0, 0
    for index = 1, count do
      if signature_parts[index] == "T" then
        pointer_array[tensor_index] = pointers[index]
        tensor_index = tensor_index + 1
      else
        scalar_array[scalar_index] = args[index]
        scalar_index = scalar_index + 1
      end
    end
    pointer_array[tensor_index] = output_pointer
    kernel.fn(pointer_array, scalar_array, elements,
      lib.lt_get_num_threads())
  else
    local call_arguments = { output_pointer, elements }
    for index = 1, count do
      if signature_parts[index] == "T" then
        call_arguments[#call_arguments + 1] = pointers[index]
      else
        call_arguments[#call_arguments + 1] = args[index]
      end
    end
    kernel.fn(unpack(call_arguments, 1, #call_arguments))
  end
  return output
end

function Fused:run(...)
  return self:_execute(nil, ...)
end

-- Write into a caller-provided contiguous buffer to avoid allocating a new
-- output tensor (and waiting for LuaJIT's GC to reclaim the previous one).
function Fused:into(output, ...)
  return self:_execute(output, ...)
end

Fused.__call = function(self, ...)
  return self:run(...)
end

Fused.__tostring = function(self)
  local compiled = 0
  for _ in pairs(self.kernels) do
    compiled = compiled + 1
  end
  return string.format("Fused(%d kernels)", compiled)
end

-- Fused reductions ----------------------------------------------------------

local FusedReduce = {}
FusedReduce.__index = FusedReduce

function FusedReduce.new(function_value, reduction, options)
  if type(function_value) ~= "function" then
    error("torch.fuse.reduce expects a function", 2)
  end
  if reduction ~= "sum" and reduction ~= "mean" and
    reduction ~= "logsumexp" then
    error("torch.fuse.reduce expects 'sum', 'mean' or 'logsumexp'", 2)
  end
  return setmetatable({
    fn = function_value,
    reduction = reduction,
    options = options or {},
    traces = {},
    kernels = {},
    train_kernels = {},
    last_error = nil,
  }, FusedReduce)
end

function FusedReduce:run(...)
  local count = select("#", ...)
  local args = pack(...)

  local function fallback()
    local result = self.fn(unpack(args, 1, count))
    if self.reduction == "mean" then
      return result:mean()
    elseif self.reduction == "logsumexp" then
      return result:logsumexp()
    end
    return result:sum()
  end

  local signature_parts = {}
  local dtype = nil
  local shape = nil
  local tensor_count, scalar_count = 0, 0
  local needs_grad = false

  for index = 1, count do
    local value = args[index]
    if Tensor.is_tensor(value) then
      if value:requires_grad() then
        needs_grad = true
      end
      signature_parts[index] = "T"
      tensor_count = tensor_count + 1
      local value_dtype = value:dtype_id()
      if dtype == nil then
        dtype = value_dtype
        shape = value:sizes()
      elseif value_dtype ~= dtype then
        return fallback()
      end
      local sizes = value:sizes()
      if #sizes ~= #shape then
        return fallback()
      end
      for dimension = 1, #sizes do
        if sizes[dimension] ~= shape[dimension] then
          return fallback()
        end
      end
    elseif type(value) == "number" then
      signature_parts[index] = "S"
      scalar_count = scalar_count + 1
    else
      return fallback()
    end
  end

  if shape == nil or tensor_count == 0 or (dtype ~= 0 and dtype ~= 1) then
    return fallback()
  end

  local signature = table.concat(signature_parts)
  local expression = self.traces[signature]
  if expression == nil then
    local traced, error_message = trace(self.fn, args)
    if traced == nil then
      self.traces[signature] = false
      self.last_error = error_message
      return fallback()
    end
    expression = traced
    self.traces[signature] = expression
  elseif expression == false then
    return fallback()
  end

  local key = signature .. "|" .. dtype

  if needs_grad then
    local train_kernel = self.train_kernels[key]
    if train_kernel == nil then
      local compiled, error_message = fusec.compile_reduce_train(
        expression, signature, dtype, self.reduction)
      if compiled == nil then
        self.traces[signature] = false
        self.last_error = error_message
        return fallback()
      end
      train_kernel = compiled
      self.train_kernels[key] = train_kernel
    end

    local input_array = ffi.new("lt_tensor*[?]", tensor_count)
    local scalars = ffi.new("double[?]", math.max(scalar_count, 1))
    local tensor_index, scalar_index = 0, 0
    for index = 1, count do
      if signature_parts[index] == "T" then
        input_array[tensor_index] = args[index].h
        tensor_index = tensor_index + 1
      else
        scalars[scalar_index] = args[index]
        scalar_index = scalar_index + 1
      end
    end

    local pointer = lib.lt_fused_reduce_apply(
      input_array, tensor_count,
      ffi.cast("void*", train_kernel.forward),
      ffi.cast("void*", train_kernel.backward),
      scalar_count > 0 and scalars or nil,
      scalar_count)
    if pointer == nil then
      self.last_error = ffi.string(lib.lt_last_error())
      return fallback()
    end
    return Tensor._wrap(pointer)
  end

  local kernel = self.kernels[key]
  if kernel == nil then
    local compiled, error_message = fusec.compile_reduce(
      expression, signature, dtype, self.reduction)
    if compiled == nil then
      self.traces[signature] = false
      self.last_error = error_message
      return fallback()
    end
    kernel = compiled
    self.kernels[key] = kernel
  end

  local output = Tensor._create("empty", {}, { dtype = dtype })
  local pointers = ffi.new("const void*[?]", tensor_count + 1)
  local scalars = ffi.new("double[?]", math.max(scalar_count, 1))
  local contiguous_inputs = {}
  local tensor_index, scalar_index = 0, 0
  for index = 1, count do
    if signature_parts[index] == "T" then
      local contiguous = args[index]:contiguous()
      contiguous_inputs[index] = contiguous
      pointers[tensor_index] = lib.lt_data_ptr(contiguous.h)
      tensor_index = tensor_index + 1
    else
      scalars[scalar_index] = args[index]
      scalar_index = scalar_index + 1
    end
  end
  pointers[tensor_count] = lib.lt_data_ptr(output.h)

  local elements = 0
  for index = 1, count do
    if signature_parts[index] == "T" then
      elements = args[index]:numel()
      break
    end
  end

  kernel(pointers, scalars, elements, lib.lt_get_num_threads())
  return output
end

FusedReduce.__call = function(self, ...)
  return self:run(...)
end

FusedReduce.__tostring = function(self)
  return string.format("FusedReduce(%s)", self.reduction)
end

-- Fused forward + backward through the C++ autograd bridge -----------------

local FusedTrain = {}
FusedTrain.__index = FusedTrain

function FusedTrain.new(function_value, options)
  if type(function_value) ~= "function" then
    error("torch.fuse.train expects a function", 2)
  end
  options = options or {}
  return setmetatable({
    fn = function_value,
    traces = {},
    kernels = {},
    last_error = nil,
  }, FusedTrain)
end

function FusedTrain:run(...)
  local count = select("#", ...)
  local args = pack(...)

  local function fallback()
    return self.fn(unpack(args, 1, count))
  end

  local signature_parts = {}
  local dtype = nil
  local dtype_name = nil
  local shape = nil
  local tensor_count, scalar_count = 0, 0

  for index = 1, count do
    local value = args[index]
    if Tensor.is_tensor(value) then
      signature_parts[index] = "T"
      tensor_count = tensor_count + 1
      local value_dtype = value:dtype_id()
      if dtype == nil then
        dtype = value_dtype
        dtype_name = value:dtype()
        shape = value:sizes()
      elseif value_dtype ~= dtype then
        return fallback()
      end
      local sizes = value:sizes()
      if #sizes ~= #shape then
        return fallback()
      end
      for dimension = 1, #sizes do
        if sizes[dimension] ~= shape[dimension] then
          return fallback()
        end
      end
    elseif type(value) == "number" then
      signature_parts[index] = "S"
      scalar_count = scalar_count + 1
    else
      return fallback()
    end
  end

  if shape == nil or tensor_count == 0 or tensor_count > 8 or
    (dtype ~= 0 and dtype ~= 1) then
    return fallback()
  end

  local signature = table.concat(signature_parts)
  local expression = self.traces[signature]
  if expression == nil then
    local traced, error_message = trace(self.fn, args)
    if traced == nil then
      self.traces[signature] = false
      self.last_error = error_message
      return fallback()
    end
    expression = traced
    self.traces[signature] = expression
  elseif expression == false then
    return fallback()
  end

  local key = signature .. "|" .. dtype
  local kernel = self.kernels[key]
  if kernel == nil then
    local compiled, error_message = fusec.compile_train(
      expression, signature, dtype)
    if compiled == nil then
      self.traces[signature] = false
      self.last_error = error_message
      return fallback()
    end
    kernel = compiled
    self.kernels[key] = kernel
  end

  local input_array = ffi.new("lt_tensor*[?]", tensor_count)
  local scalars = ffi.new("double[?]", math.max(scalar_count, 1))
  local tensor_index, scalar_index = 0, 0
  for index = 1, count do
    if signature_parts[index] == "T" then
      input_array[tensor_index] = args[index].h
      tensor_index = tensor_index + 1
    else
      scalars[scalar_index] = args[index]
      scalar_index = scalar_index + 1
    end
  end

  local pointer = lib.lt_fused_apply(
    input_array, tensor_count,
    ffi.cast("void*", kernel.forward),
    ffi.cast("void*", kernel.backward),
    scalar_count > 0 and scalars or nil,
    scalar_count)
  if pointer == nil then
    self.last_error = ffi.string(lib.lt_last_error())
    return fallback()
  end
  return Tensor._wrap(pointer)
end

FusedTrain.__call = function(self, ...)
  return self:run(...)
end

FusedTrain.__tostring = function(self)
  local compiled = 0
  for _ in pairs(self.kernels) do
    compiled = compiled + 1
  end
  return string.format("FusedTrain(%d kernels)", compiled)
end

-- Linear / convolution epilogues -------------------------------------------
--
-- These are deliberately small and concrete: the common inference/training
-- pattern is addmm (or conv2d) with the bias already folded in, followed by a
-- single activation.  The native helper applies the activation in place to the
-- freshly produced output, so the fused path does not allocate an activation
-- temporary and ATen still records the in-place autograd node.  Anything
-- outside the supported envelope uses the eager expression below.

local epilogue_activation_codes = {
  none = 0,
  identity = 0,
  relu = 1,
  gelu = 2,
  sigmoid = 3,
  tanh = 4,
}

local epilogue_stats = { fused = 0, fallback = 0 }
local epilogue_last_error = nil

local function normalize_epilogue_activation(value)
  if value == nil or value == false then
    return "none", 0
  end
  if type(value) == "string" then
    local name = value:lower()
    local code = epilogue_activation_codes[name]
    if code ~= nil then
      return name, code
    end
    return nil, nil, "unsupported epilogue activation '" .. value .. "'"
  end
  if type(value) == "function" then
    return value, nil
  end
  if type(value) == "table" and type(value.forward) == "function" then
    return value, nil
  end
  return nil, nil, "activation must be a string, function or module"
end

local function apply_eager_activation(output, activation, code)
  if code == 0 then
    return output
  end
  if type(activation) == "string" then
    if activation == "gelu" then
      return output:gelu()
    end
    local method = Tensor[activation]
    if method == nil then
      error("torch.fuse: unsupported activation '" .. activation .. "'", 0)
    end
    return method(output)
  end
  return activation(output)
end

local function eager_linear(input, weight, bias, activation, code)
  local output
  if bias ~= nil then
    if input:dim() == 2 then
      output = bias:addmm(input, weight:t())
    else
      output = input:matmul(weight:t()):add(bias)
    end
  else
    output = input:matmul(weight:t())
  end
  return apply_eager_activation(output, activation, code)
end

local function epilogue_linear(input, weight, bias, activation)
  local name, code, message = normalize_epilogue_activation(activation)
  if message ~= nil then
    error("torch.fuse.linear: " .. message, 3)
  end
  if bias == false then
    bias = nil
  end
  if not Tensor.is_tensor(input) or not Tensor.is_tensor(weight) then
    error("torch.fuse.linear expects input and weight tensors", 3)
  end
  if bias ~= nil and not Tensor.is_tensor(bias) then
    error("torch.fuse.linear: bias must be a Tensor or nil", 3)
  end

  local dtype = input:dtype_id()
  local fused = code ~= nil and
    input:dim() == 2 and weight:dim() == 2 and
    weight:dtype_id() == dtype and (dtype == 0 or dtype == 1)
  if fused then
    local input_sizes = input:sizes()
    local weight_sizes = weight:sizes()
    fused = input_sizes[2] == weight_sizes[2]
    if fused and bias ~= nil then
      local bias_sizes = bias:sizes()
      fused = bias:dim() == 1 and bias_sizes[1] == weight_sizes[1] and
        bias:dtype_id() == dtype
    end
  end

  if fused then
    local pointer = lib.lt_linear_epilogue(
      input.h, weight.h, bias and bias.h or nil, code)
    if pointer ~= nil then
      epilogue_stats.fused = epilogue_stats.fused + 1
      epilogue_last_error = nil
      return Tensor._wrap(pointer)
    end
    epilogue_last_error = ffi.string(lib.lt_last_error())
  end

  epilogue_stats.fallback = epilogue_stats.fallback + 1
  return eager_linear(input, weight, bias, name, code)
end

local function epilogue_pair(value, default, name)
  if value == nil then
    return default, default
  end
  if type(value) == "number" then
    return value, value
  end
  if type(value) == "table" and #value == 2 then
    return value[1], value[2]
  end
  error(name .. " must be a number or a {height, width} table", 3)
end

local function parse_conv_options(stride, padding, dilation, groups, activation)
  if type(stride) == "function" or type(stride) == "string" or
      (type(stride) == "table" and type(stride.forward) == "function") then
    -- Shorthand: torch.fuse.conv2d(x, w, b, "relu") with default geometry.
    return nil, padding, dilation, groups, stride
  end
  if type(stride) == "table" and type(stride[1]) ~= "number" then
    local options = stride
    return options.stride, options.padding, options.dilation,
      options.groups, options.activation
  end
  return stride, padding, dilation, groups, activation
end

local function eager_conv2d(input, weight, bias, stride_h, stride_w,
    pad_h, pad_w, dilation_h, dilation_w, groups, activation, code)
  local bias_pointer = nil
  if bias ~= nil then
    bias_pointer = bias.h
  end
  local output = Tensor._wrap(lib.lt_conv2d(
    input.h, weight.h, bias_pointer,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups))
  return apply_eager_activation(output, activation, code)
end

local function epilogue_conv2d(input, weight, bias, stride, padding,
    dilation, groups, activation)
  stride, padding, dilation, groups, activation =
    parse_conv_options(stride, padding, dilation, groups, activation)

  local name, code, message = normalize_epilogue_activation(activation)
  if message ~= nil then
    error("torch.fuse.conv2d: " .. message, 3)
  end
  if bias == false then
    bias = nil
  end
  if not Tensor.is_tensor(input) or not Tensor.is_tensor(weight) then
    error("torch.fuse.conv2d expects input and weight tensors", 3)
  end
  if bias ~= nil and not Tensor.is_tensor(bias) then
    error("torch.fuse.conv2d: bias must be a Tensor or nil", 3)
  end
  if groups == nil then
    groups = 1
  end
  if type(groups) ~= "number" or groups < 1 or groups % 1 ~= 0 then
    error("torch.fuse.conv2d: groups must be a positive integer", 3)
  end

  local stride_h, stride_w = epilogue_pair(stride, 1, "stride")
  local pad_h, pad_w = epilogue_pair(padding, 0, "padding")
  local dilation_h, dilation_w = epilogue_pair(dilation, 1, "dilation")

  local dtype = input:dtype_id()
  local fused = code ~= nil and
    input:dim() == 4 and weight:dim() == 4 and
    weight:dtype_id() == dtype and (dtype == 0 or dtype == 1)
  if fused then
    local input_sizes = input:sizes()
    local weight_sizes = weight:sizes()
    fused = input_sizes[2] == weight_sizes[2] * groups
    if fused and bias ~= nil then
      local bias_sizes = bias:sizes()
      fused = bias:dim() == 1 and bias_sizes[1] == weight_sizes[1] and
        bias:dtype_id() == dtype
    end
  end

  if fused then
    local pointer = lib.lt_conv2d_epilogue(
      input.h, weight.h, bias and bias.h or nil,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
      groups, code)
    if pointer ~= nil then
      epilogue_stats.fused = epilogue_stats.fused + 1
      epilogue_last_error = nil
      return Tensor._wrap(pointer)
    end
    epilogue_last_error = ffi.string(lib.lt_last_error())
  end

  epilogue_stats.fallback = epilogue_stats.fallback + 1
  return eager_conv2d(input, weight, bias, stride_h, stride_w,
    pad_h, pad_w, dilation_h, dilation_w, groups, name, code)
end

local function fuse(function_value, options)
  return Fused.new(function_value, options)
end

local fuse_module = setmetatable({}, {
  __call = function(_, function_value, options)
    return fuse(function_value, options)
  end,
})
fuse_module.train = function(function_value, options)
  return FusedTrain.new(function_value, options)
end
fuse_module.reduce = function(function_value, reduction, options)
  return FusedReduce.new(function_value, reduction, options)
end
fuse_module.linear = epilogue_linear
fuse_module.conv2d = epilogue_conv2d
fuse_module.epilogue_stats = epilogue_stats
fuse_module.epilogue_last_error = function()
  return epilogue_last_error
end

return fuse_module
