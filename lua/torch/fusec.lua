--
-- torch.fusec
--
-- C backend for torch.fuse.  The traced expression is turned into a small C
-- translation unit, compiled with the system C compiler, cached on disk, and
-- loaded through FFI.  Modern compilers vectorise these loop bodies, so this
-- is the backend that can compete with torch.compile on large tensors.
--
-- The generated kernel has a stable ABI:
--   void luatorch_kernel(const void** ptrs, const double* scalars,
--                        int64_t n, int threads)
-- pointers are [tensor inputs..., output], scalars are argument-order doubles.
--
local ffi = require("ffi")
local bit = require("bit")

ffi.cdef[[
void luatorch_kernel(const void** ptrs, const double* scalars, int64_t n, int threads);
void luatorch_backward(const void** ptrs, const double* scalars, int64_t n, int threads);
]]

local M = {}

local type_info = {
  [0] = {
    ctype = "float",
    zero = "0.0f",
    abs = "fabsf",
    max = "fmaxf",
    min = "fminf",
    exp = "expf",
    log = "logf",
    sqrt = "sqrtf",
    tanh = "tanhf",
    pow = "powf",
    suffix = "f",
  },
  [1] = {
    ctype = "double",
    zero = "0.0",
    abs = "fabs",
    max = "fmax",
    min = "fmin",
    exp = "exp",
    log = "log",
    sqrt = "sqrt",
    tanh = "tanh",
    pow = "pow",
    suffix = "",
  },
}

local function format_number(value, info)
  if value ~= value then
    return "NAN"
  elseif value == math.huge then
    return "INFINITY"
  elseif value == -math.huge then
    return "(-INFINITY)"
  end
  local text = string.format("%.17g", value)
  if not text:find("[%.eE]") then
    text = text .. ".0"
  end
  return text .. info.suffix
end

local function new_temporary(context, ctype, expression)
  context.counter = context.counter + 1
  local variable = "v" .. context.counter
  context.lines[#context.lines + 1] =
    string.format("    const %s %s = %s;", ctype, variable, expression)
  return variable
end

local function emit(node, context)
  if type(node) == "number" then
    return format_number(node, context.info)
  end
  if type(node) ~= "table" or getmetatable(node).__name ~= "FuseSymbol" then
    error("fuse: invalid expression node", 0)
  end

  local memo = context.memo[node]
  if memo ~= nil then
    return memo
  end

  local op = node.op
  local info = context.info
  local ctype = info.ctype
  local value

  if op == "tensor_arg" then
    value = new_temporary(context, ctype,
      string.format("t%d[i]", node.arg))
  elseif op == "scalar_arg" then
    value = "s" .. node.arg
  else
    local args = node.args
    local first = args and emit(args[1], context) or nil

    if op == "neg" then
      value = new_temporary(context, ctype, "(-" .. first .. ")")
    elseif op == "abs" then
      value = new_temporary(context, ctype, info.abs .. "(" .. first .. ")")
    elseif op == "relu" then
      -- A ternary keeps NaN semantics and vectorises; fmaxf calls libm
      -- without -ffast-math and is ~4x slower.
      value = new_temporary(context, ctype,
        "(" .. first .. " < 0 ? 0 : " .. first .. ")")
    elseif op == "exp" then
      value = new_temporary(context, ctype, info.exp .. "(" .. first .. ")")
    elseif op == "log" then
      value = new_temporary(context, ctype, info.log .. "(" .. first .. ")")
    elseif op == "sqrt" then
      value = new_temporary(context, ctype, info.sqrt .. "(" .. first .. ")")
    elseif op == "tanh" then
      value = new_temporary(context, ctype, info.tanh .. "(" .. first .. ")")
    elseif op == "sigmoid" then
      value = new_temporary(context, ctype,
        "(1.0" .. info.suffix .. " / (1.0" .. info.suffix .. " + " ..
        info.exp .. "(-" .. first .. ")))")
    elseif op == "clamp" then
      local low = emit(args[2], context)
      local high = emit(args[3], context)
      value = new_temporary(context, ctype,
        "(" .. first .. " < " .. low .. " ? " .. low ..
        " : (" .. first .. " > " .. high .. " ? " .. high ..
        " : " .. first .. "))")
    elseif op == "grad_output" then
      value = new_temporary(context, ctype, "go[i]")
    elseif op == "relu_mask" then
      value = new_temporary(context, ctype,
        "((" .. first .. " > 0) ? 1.0" .. info.suffix .. " : 0.0" .. info.suffix .. ")")
    elseif op == "abs_sign" then
      value = new_temporary(context, ctype,
        "((" .. first .. " > 0) ? 1.0" .. info.suffix ..
        " : ((" .. first .. " < 0) ? -1.0" .. info.suffix .. " : 0.0" .. info.suffix .. "))")
    elseif op == "clamp_mask" then
      local low = emit(args[2], context)
      local high = emit(args[3], context)
      value = new_temporary(context, ctype,
        "(((" .. first .. " >= " .. low .. ") && (" .. first .. " <= " .. high ..
        ")) ? 1.0" .. info.suffix .. " : 0.0" .. info.suffix .. ")")
    elseif op == "relu_backward" then
      local delta = emit(args[2], context)
      value = new_temporary(context, ctype,
        "(" .. first .. " > 0 ? " .. delta .. " : 0)")
    elseif op == "abs_backward" then
      local delta = emit(args[2], context)
      value = new_temporary(context, ctype,
        "(" .. first .. " > 0 ? " .. delta ..
        " : (" .. first .. " < 0 ? -" .. delta .. " : 0))")
    elseif op == "clamp_backward" then
      local low = emit(args[2], context)
      local high = emit(args[3], context)
      local delta = emit(args[4], context)
      value = new_temporary(context, ctype,
        "((" .. first .. " >= " .. low .. " && " .. first .. " <= " .. high ..
        ") ? " .. delta .. " : 0)")
    elseif op == "sigmoid_backward" then
      local delta = emit(args[2], context)
      local sigmoid_value = new_temporary(context, ctype,
        "(1.0" .. info.suffix .. " / (1.0" .. info.suffix .. " + " ..
        info.exp .. "(-" .. first .. ")))")
      value = new_temporary(context, ctype,
        "(" .. sigmoid_value .. " * (1.0" .. info.suffix .. " - " ..
        sigmoid_value .. ") * " .. delta .. ")")
    elseif op == "tanh_backward" then
      local delta = emit(args[2], context)
      local tanh_value = new_temporary(context, ctype,
        info.tanh .. "(" .. first .. ")")
      value = new_temporary(context, ctype,
        "((1.0" .. info.suffix .. " - " .. tanh_value .. " * " .. tanh_value ..
        ") * " .. delta .. ")")
    elseif op == "add" or op == "sub" or op == "mul" or op == "div" then
      local second = emit(args[2], context)
      local symbol = ({ add = "+", sub = "-", mul = "*", div = "/" })[op]
      value = new_temporary(context, ctype,
        "(" .. first .. " " .. symbol .. " " .. second .. ")")
    elseif op == "pow" then
      local exponent = args[2]
      if type(exponent) == "number" then
        if exponent == 2 then
          value = new_temporary(context, ctype,
            "(" .. first .. " * " .. first .. ")")
        elseif exponent == 3 then
          value = new_temporary(context, ctype,
            "(" .. first .. " * " .. first .. " * " .. first .. ")")
        elseif exponent == 0.5 then
          value = new_temporary(context, ctype,
            info.sqrt .. "(" .. first .. ")")
        elseif exponent == -1 then
          value = new_temporary(context, ctype,
            "(1.0" .. info.suffix .. " / " .. first .. ")")
        else
          value = new_temporary(context, ctype,
            info.pow .. "(" .. first .. ", " ..
            format_number(exponent, info) .. ")")
        end
      else
        local second = emit(exponent, context)
        value = new_temporary(context, ctype,
          info.pow .. "(" .. first .. ", " .. second .. ")")
      end
    else
      error("fuse: unsupported operation '" .. tostring(op) .. "'", 0)
    end
  end

  context.memo[node] = value
  return value
end

local function generate(expression, signature, dtype)
  local info = type_info[dtype]
  if info == nil then
    return nil, "unsupported dtype for C fusion"
  end

  local context = {
    lines = {},
    counter = 0,
    memo = {},
    info = info,
  }
  local ok, error_message = pcall(function()
    context.result = emit(expression, context)
  end)
  if not ok then
    return nil, error_message
  end

  local prelude = {}
  local tensor_index = 0
  local scalar_index = 0
  for index = 1, #signature do
    if signature:sub(index, index) == "T" then
      prelude[#prelude + 1] = string.format(
        "  const %s* __restrict t%d = (const %s*)ptrs[%d];",
        info.ctype, index, info.ctype, tensor_index)
      tensor_index = tensor_index + 1
    else
      prelude[#prelude + 1] = string.format(
        "  const %s s%d = (%s)scalars[%d];",
        info.ctype, index, info.ctype, scalar_index)
      scalar_index = scalar_index + 1
    end
  end
  prelude[#prelude + 1] = string.format(
    "  %s* __restrict out = (%s*)ptrs[%d];", info.ctype, info.ctype, tensor_index)

  local source = table.concat({
    "#include <stdint.h>",
    "#include <math.h>",
    "#ifdef _OPENMP",
    "#include <omp.h>",
    "#endif",
    "",
    "void luatorch_kernel(const void** ptrs, const double* scalars, int64_t n, int threads) {",
    table.concat(prelude, "\n"),
    "#ifdef _OPENMP",
    "#pragma omp parallel for if(threads > 1 && n > 20000) num_threads(threads)",
    "#endif",
    "  for (int64_t i = 0; i < n; ++i) {",
    table.concat(context.lines, "\n"),
    "    out[i] = " .. context.result .. ";",
    "  }",
    "}",
    "",
  }, "\n")

  return source
end

-- Symbolic differentiation and backward C code -----------------------------

local function make_derivative(metatable)
  local function make(op, args)
    return setmetatable({ op = op, args = args }, metatable)
  end
  local function is_number(value)
    return type(value) == "number"
  end
  local function is_zero(value)
    return is_number(value) and value == 0
  end
  local function add(a, b)
    if is_number(a) and is_number(b) then return a + b end
    if is_number(a) and a == 0 then return b end
    if is_number(b) and b == 0 then return a end
    return make("add", { a, b })
  end
  local neg
  local function sub(a, b)
    if is_number(a) and is_number(b) then return a - b end
    if is_number(b) and b == 0 then return a end
    if is_number(a) and a == 0 then return neg(b) end
    return make("sub", { a, b })
  end
  local function mul(a, b)
    if is_number(a) and is_number(b) then return a * b end
    if (is_number(a) and a == 0) or (is_number(b) and b == 0) then return 0 end
    if is_number(a) and a == 1 then return b end
    if is_number(b) and b == 1 then return a end
    return make("mul", { a, b })
  end
  local function div(a, b)
    if is_number(a) and is_number(b) then return a / b end
    if is_number(a) and a == 0 then return 0 end
    if is_number(b) and b == 1 then return a end
    return make("div", { a, b })
  end
  neg = function(a)
    if is_number(a) then return -a end
    return make("neg", { a })
  end
  local function exp_node(a) return make("exp", { a }) end
  local function log_node(a) return make("log", { a }) end
  local function sqrt_node(a) return make("sqrt", { a }) end
  local function pow_node(a, b)
    if is_number(b) then
      if b == 0 then return 1 end
      if b == 1 then return a end
    end
    return make("pow", { a, b })
  end

  local relu_masks = {}
  local abs_masks = {}
  local clamp_masks = {}

  local derive -- forward declaration

  derive = function(value, target)
    if is_number(value) then
      return 0
    end
    if type(value) ~= "table" or getmetatable(value) ~= metatable then
      error("fuse: cannot differentiate a non-expression value", 0)
    end

    local op = value.op
    if op == "tensor_arg" then
      return value.arg == target and 1 or 0
    elseif op == "scalar_arg" then
      return 0
    end

    local args = value.args
    local a = args and args[1]
    local b = args and args[2]
    local da = a ~= nil and derive(a, target) or 0

    if op == "add" then
      return add(da, derive(b, target))
    elseif op == "sub" then
      return sub(da, derive(b, target))
    elseif op == "mul" then
      return add(mul(da, b), mul(a, derive(b, target)))
    elseif op == "div" then
      local db = derive(b, target)
      return div(sub(mul(da, b), mul(a, db)), mul(b, b))
    elseif op == "neg" then
      return neg(da)
    elseif op == "exp" then
      return mul(exp_node(a), da)
    elseif op == "log" then
      return div(da, a)
    elseif op == "sqrt" then
      if is_zero(da) then return 0 end
      return div(da, mul(2, sqrt_node(a)))
    elseif op == "relu" then
      if is_zero(da) then return 0 end
      local mask = relu_masks[value]
      if mask == nil then
        mask = make("relu_mask", { a })
        relu_masks[value] = mask
      end
      return mul(mask, da)
    elseif op == "abs" then
      if is_zero(da) then return 0 end
      local mask = abs_masks[value]
      if mask == nil then
        mask = make("abs_sign", { a })
        abs_masks[value] = mask
      end
      return mul(mask, da)
    elseif op == "sigmoid" then
      if is_zero(da) then return 0 end
      return make("sigmoid_backward", { a, da })
    elseif op == "tanh" then
      if is_zero(da) then return 0 end
      return make("tanh_backward", { a, da })
    elseif op == "clamp" then
      if is_zero(da) then return 0 end
      local mask = clamp_masks[value]
      if mask == nil then
        mask = make("clamp_mask", { a, args[2], args[3] })
        clamp_masks[value] = mask
      end
      return mul(mask, da)
    elseif op == "pow" then
      local db = derive(b, target)
      if is_zero(da) and is_zero(db) then
        return 0
      end
      if is_zero(db) then
        return mul(mul(b, pow_node(a, sub(b, 1))), da)
      elseif is_zero(da) then
        return mul(mul(value, log_node(a)), db)
      end
      return mul(value, add(mul(db, log_node(a)), div(mul(b, da), a)))
    end

    error("fuse: no derivative rule for '" .. tostring(op) .. "'", 0)
  end

  return derive
end

local function generate_backward(expression, signature, dtype)
  local info = type_info[dtype]
  if info == nil then
    return nil, "unsupported dtype for fused autograd"
  end
  local metatable = getmetatable(expression)
  local derive = make_derivative(metatable)
  local grad_output = setmetatable({ op = "grad_output", args = { 1 } }, metatable)

  local tensor_positions = {}
  for index = 1, #signature do
    if signature:sub(index, index) == "T" then
      tensor_positions[#tensor_positions + 1] = index
    end
  end

  -- Backward code is rendered inline rather than through the temporary-heavy
  -- forward emitter.  GCC refuses to vectorise the loop when every load and
  -- intermediate gets its own C temporary; the inline form produces one
  -- if-converted mask and vectorises to AVX2.
  local context = {
    counter = 0,
    masks = {},
    values = {},
    lines = {},
  }
  local function value_temporary(expression)
    context.counter = context.counter + 1
    local name = "v" .. context.counter
    context.lines[#context.lines + 1] = string.format(
      "    const %s %s = %s;", info.ctype, name, expression)
    return name
  end
  local function mask_temporary(expression)
    context.counter = context.counter + 1
    local name = "m" .. context.counter
    context.lines[#context.lines + 1] = string.format(
      "    const %s %s = %s;", info.ctype, name, expression)
    return name
  end

  local function factor_priority(value)
    if type(value) == "table" and getmetatable(value) == metatable then
      local op = value.op
      if op == "relu_mask" or op == "abs_sign" or op == "clamp_mask" then
        return 1
      elseif op == "grad_output" then
        return 2
      end
    elseif type(value) == "number" then
      return 3
    end
    return 4
  end

  local render
  render = function(value)
    if type(value) == "number" then
      return format_number(value, info)
    end
    if type(value) ~= "table" or getmetatable(value) ~= metatable then
      error("fuse: cannot generate backward code for this expression", 0)
    end
    local op = value.op
    local args = value.args or {}
    local function arg(index)
      return render(args[index])
    end

    if op == "tensor_arg" then
      return "t" .. value.arg .. "[i]"
    elseif op == "scalar_arg" then
      return "s" .. value.arg
    elseif op == "grad_output" then
      return "go[i]"
    elseif op == "add" then
      return "(" .. arg(1) .. " + " .. arg(2) .. ")"
    elseif op == "sub" then
      return "(" .. arg(1) .. " - " .. arg(2) .. ")"
    elseif op == "mul" then
      local factors = {}
      local function collect(item)
        if type(item) == "table" and getmetatable(item) == metatable and
          item.op == "mul" then
          collect(item.args[1])
          collect(item.args[2])
        else
          factors[#factors + 1] = item
        end
      end
      collect(value)
      if #factors == 1 then
        return render(factors[1])
      end
      table.sort(factors, function(left, right)
        return factor_priority(left) < factor_priority(right)
      end)
      local text = render(factors[1])
      for index = 2, #factors do
        text = "(" .. text .. " * " .. render(factors[index]) .. ")"
      end
      return text
    elseif op == "div" then
      return "(" .. arg(1) .. " / " .. arg(2) .. ")"
    elseif op == "neg" then
      return "(-" .. arg(1) .. ")"
    elseif op == "exp" then
      return info.exp .. "(" .. arg(1) .. ")"
    elseif op == "log" then
      return info.log .. "(" .. arg(1) .. ")"
    elseif op == "sqrt" then
      return info.sqrt .. "(" .. arg(1) .. ")"
    elseif op == "abs" then
      return info.abs .. "(" .. arg(1) .. ")"
    elseif op == "relu" then
      local a = arg(1)
      return "((" .. a .. " > 0) ? " .. a .. " : 0.0" .. info.suffix .. ")"
    elseif op == "sigmoid" then
      local a = arg(1)
      return "(1.0" .. info.suffix .. " / (1.0" .. info.suffix .. " + " ..
        info.exp .. "(-" .. a .. ")))"
    elseif op == "tanh" then
      return info.tanh .. "(" .. arg(1) .. ")"
    elseif op == "clamp" then
      local a, low, high = arg(1), arg(2), arg(3)
      return "((" .. a .. " < " .. low .. ") ? " .. low ..
        " : ((" .. a .. " > " .. high .. ") ? " .. high .. " : " .. a .. "))"
    elseif op == "pow" then
      return info.pow .. "(" .. arg(1) .. ", " .. arg(2) .. ")"
    elseif op == "relu_mask" then
      local cached = context.masks[value]
      if cached then return cached end
      local a = context.values[value.args[1]]
      if a == nil then
        a = value_temporary(render(value.args[1]))
        context.values[value.args[1]] = a
      end
      local name = mask_temporary(
        "((" .. a .. " > 0) ? 1.0" .. info.suffix .. " : 0.0" .. info.suffix .. ")")
      context.masks[value] = name
      return name
    elseif op == "abs_sign" then
      local cached = context.masks[value]
      if cached then return cached end
      local a = context.values[value.args[1]]
      if a == nil then
        a = value_temporary(render(value.args[1]))
        context.values[value.args[1]] = a
      end
      local name = mask_temporary(
        "((" .. a .. " > 0) ? 1.0" .. info.suffix ..
        " : ((" .. a .. " < 0) ? -1.0" .. info.suffix .. " : 0.0" .. info.suffix .. "))")
      context.masks[value] = name
      return name
    elseif op == "clamp_mask" then
      local cached = context.masks[value]
      if cached then return cached end
      local a = context.values[value.args[1]]
      if a == nil then
        a = value_temporary(render(value.args[1]))
        context.values[value.args[1]] = a
      end
      local low, high = arg(2), arg(3)
      local name = mask_temporary(
        "(((" .. a .. " >= " .. low .. ") && (" .. a .. " <= " .. high ..
        ")) ? 1.0" .. info.suffix .. " : 0.0" .. info.suffix .. ")")
      context.masks[value] = name
      return name
    elseif op == "relu_backward" then
      local a, d = arg(1), arg(2)
      return "((" .. a .. " > 0) ? " .. d .. " : 0.0" .. info.suffix .. ")"
    elseif op == "abs_backward" then
      local a, d = arg(1), arg(2)
      return "((" .. a .. " > 0) ? " .. d ..
        " : ((" .. a .. " < 0) ? -" .. d .. " : 0.0" .. info.suffix .. "))"
    elseif op == "clamp_backward" then
      local a, low, high, d = arg(1), arg(2), arg(3), arg(4)
      return "(((" .. a .. " >= " .. low .. ") && (" .. a .. " <= " .. high ..
        ")) ? " .. d .. " : 0.0" .. info.suffix .. ")"
    elseif op == "sigmoid_backward" then
      local a, d = arg(1), arg(2)
      local s = "(1.0" .. info.suffix .. " / (1.0" .. info.suffix .. " + " ..
        info.exp .. "(-" .. a .. ")))"
      return "(" .. s .. " * (1.0" .. info.suffix .. " - " .. s .. ") * " .. d .. ")"
    elseif op == "tanh_backward" then
      local a, d = arg(1), arg(2)
      local t = info.tanh .. "(" .. a .. ")"
      return "((1.0" .. info.suffix .. " - " .. t .. " * " .. t .. ") * " .. d .. ")"
    end
    error("fuse: no backward code generator for '" .. tostring(op) .. "'", 0)
  end

  local gradients = {}
  local ok, error_message = pcall(function()
    for ordinal, argument in ipairs(tensor_positions) do
      local derivative = derive(expression, argument)
      if type(derivative) == "number" and derivative == 0 then
        gradients[ordinal] = 0
      else
        gradients[ordinal] = setmetatable(
          { op = "mul", args = { grad_output, derivative } }, metatable)
      end
    end
  end)
  if not ok then
    return nil, error_message
  end

  local prelude = {}
  local tensor_index = 0
  local scalar_index = 0
  for index = 1, #signature do
    if signature:sub(index, index) == "T" then
      prelude[#prelude + 1] = string.format(
        "  const %s* __restrict t%d = (const %s*)ptrs[%d];",
        info.ctype, index, info.ctype, tensor_index)
      tensor_index = tensor_index + 1
    else
      prelude[#prelude + 1] = string.format(
        "  const %s s%d = (%s)scalars[%d];",
        info.ctype, index, info.ctype, scalar_index)
      scalar_index = scalar_index + 1
    end
  end
  prelude[#prelude + 1] = string.format(
    "  const %s* __restrict go = (const %s*)ptrs[%d];",
    info.ctype, info.ctype, tensor_index)
  for ordinal = 1, #tensor_positions do
    prelude[#prelude + 1] = string.format(
      "  %s* __restrict g%d = (%s*)ptrs[%d];",
      info.ctype, ordinal, info.ctype, tensor_index + ordinal)
  end

  local writes = {}
  for ordinal, gradient in ipairs(gradients) do
    writes[#writes + 1] = string.format("    g%d[i] = %s;", ordinal, render(gradient))
  end

  local source = table.concat({
    "void luatorch_backward(const void** ptrs, const double* scalars, int64_t n, int threads) {",
    table.concat(prelude, "\n"),
    "#ifdef _OPENMP",
    "#pragma omp parallel for if(threads > 1 && n > 20000) num_threads(threads)",
    "#endif",
    "  for (int64_t i = 0; i < n; ++i) {",
    table.concat(context.lines, "\n"),
    table.concat(writes, "\n"),
    "  }",
    "}",
    "",
  }, "\n")
  return source
end

local function hash(text)
  local value = 2166136261
  for index = 1, #text do
    value = bit.bxor(value, text:byte(index))
    value = bit.tobit(value * 16777619)
  end
  return string.format("%08x", value)
end

function M.available()
  if M._available ~= nil then
    return M._available
  end
  if os.getenv("LUATORCH_DISABLE_C_FUSE") == "1" then
    M._available = false
    return false
  end
  local status = os.execute("gcc --version >/dev/null 2>&1")
  M._available = status == 0 or status == true
  return M._available
end

local function compile_source(source)
  if not M.available() then
    return nil, "no C compiler available"
  end

  local cache_dir = os.getenv("LUATORCH_CACHE") or ".luatorch-cache"
  os.execute("mkdir -p '" .. cache_dir .. "' 2>/dev/null")

  local key = hash(source)
  local c_path = cache_dir .. "/" .. key .. ".c"
  local so_path = cache_dir .. "/" .. key .. ".so"
  local log_path = cache_dir .. "/" .. key .. ".log"

  local file = io.open(so_path, "rb")
  if file then
    file:close()
  else
    file = assert(io.open(c_path, "wb"))
    file:write(source)
    file:close()

    local command = string.format(
      "gcc -O3 -march=native -fPIC -shared -fopenmp -fno-math-errno " ..
      "-o '%s' '%s' -lm 2> '%s'", so_path, c_path, log_path)
    local status = os.execute(command)
    if status ~= 0 and status ~= true then
      local log = io.open(log_path, "rb")
      local text = log and log:read("*a") or "compilation failed"
      if log then log:close() end
      return nil, text
    end
  end

  local ok, library = pcall(ffi.load, so_path)
  if not ok then
    return nil, library
  end
  M._libraries = M._libraries or {}
  M._libraries[so_path] = library
  return library
end

function M.compile(expression, signature, dtype)
  local source, error_message = generate(expression, signature, dtype)
  if source == nil then
    return nil, error_message
  end
  local library, compile_error = compile_source(source)
  if library == nil then
    return nil, compile_error
  end
  return library.luatorch_kernel
end

function M.compile_train(expression, signature, dtype)
  local forward, forward_error = generate(expression, signature, dtype)
  if forward == nil then
    return nil, forward_error
  end
  local backward, backward_error = generate_backward(expression, signature, dtype)
  if backward == nil then
    return nil, backward_error
  end
  local library, compile_error = compile_source(forward .. "\n" .. backward)
  if library == nil then
    return nil, compile_error
  end
  return {
    forward = library.luatorch_kernel,
    backward = library.luatorch_backward,
    library = library,
  }
end

return M
