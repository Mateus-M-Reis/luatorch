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

function M.compile(expression, signature, dtype)
  if not M.available() then
    return nil, "no C compiler available"
  end

  local source, error_message = generate(expression, signature, dtype)
  if source == nil then
    return nil, error_message
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
  -- Keep the library handle alive for the lifetime of the process; otherwise
  -- LuaJIT may unload the shared object and leave a dangling function pointer.
  M._libraries = M._libraries or {}
  M._libraries[so_path] = library
  return library.luatorch_kernel
end

return M
