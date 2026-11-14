--
-- torch.nn.serialization
--
-- A compact single-file format for module state dictionaries.  It is not the
-- PyTorch zip format; it stores a small text header per tensor followed by the
-- raw contiguous bytes:
--
--   LUATORCH-NN 1
--   <number of tensors>
--   <key> <dtype> <ndim> <size...> <nbytes>
--   <nbytes of little-endian data>
--   ...
--
-- Keys are parameter/buffer names and may not contain whitespace.  Values are
-- copied into detached tensors and then loaded with load_state_dict().
--
local ffi = require("ffi")
local libmod = require("torch.lib")
local lib = libmod.lib
local Tensor = require("torch.tensor")

local Serialization = {}

local MAGIC = "LUATORCH-NN 1"

local dtype_sizes = {
  [0] = 4, -- float32
  [1] = 8, -- float64
  [2] = 8, -- int64
  [3] = 4, -- int32
  [4] = 2, -- int16
  [5] = 1, -- int8
  [6] = 1, -- uint8
  [7] = 1, -- bool
  [8] = 2, -- float16
}

local function sorted_keys(dictionary)
  local keys = {}
  for key in pairs(dictionary) do
    keys[#keys + 1] = key
  end
  table.sort(keys)
  return keys
end

function Serialization.save(module, path)
  local state = module:state_dict()
  local keys = sorted_keys(state)

  local file = assert(io.open(path, "wb"))
  file:write(MAGIC .. "\n")
  file:write(#keys .. "\n")

  for _, key in ipairs(keys) do
    if key:find("%s") then
      file:close()
      error("cannot serialize a tensor name containing whitespace: '" .. key .. "'", 2)
    end
    local tensor = state[key]:contiguous()
    local dtype = tensor:dtype_id()
    local sizes = tensor:sizes()
    local nbytes = tensor:nbytes()

    local header = { key, tostring(dtype), tostring(#sizes) }
    for i = 1, #sizes do
      header[#header + 1] = tostring(sizes[i])
    end
    header[#header + 1] = tostring(nbytes)
    file:write(table.concat(header, " ") .. "\n")

    if nbytes > 0 then
      local pointer = lib.lt_data_ptr(tensor.h)
      if pointer == nil then
        file:close()
        error(ffi.string(lib.lt_last_error()), 2)
      end
      file:write(ffi.string(pointer, nbytes))
    end
  end

  assert(file:close())
  return path
end

function Serialization.load(module, path)
  local file = assert(io.open(path, "rb"))
  local magic = file:read("*l")
  if magic ~= MAGIC then
    file:close()
    error("not a luatorch model file: " .. tostring(path), 2)
  end

  local count = tonumber(file:read("*l"))
  if count == nil or count < 0 then
    file:close()
    error("corrupt luatorch model file: bad tensor count", 2)
  end

  local state = {}
  for _ = 1, count do
    local line = file:read("*l")
    if line == nil then
      file:close()
      error("truncated luatorch model file", 2)
    end
    local tokens = {}
    for token in line:gmatch("%S+") do
      tokens[#tokens + 1] = token
    end
    if #tokens < 4 then
      file:close()
      error("corrupt luatorch model header", 2)
    end

    local key = tokens[1]
    local dtype = tonumber(tokens[2])
    local ndim = tonumber(tokens[3])
    if dtype == nil or dtype_sizes[dtype] == nil or ndim == nil or
      #tokens ~= 4 + ndim then
      file:close()
      error("corrupt luatorch model header for '" .. key .. "'", 2)
    end

    local sizes = {}
    for dimension = 1, ndim do
      sizes[dimension] = tonumber(tokens[3 + dimension])
      if sizes[dimension] == nil then
        file:close()
        error("corrupt shape for '" .. key .. "'", 2)
      end
    end
    local nbytes = tonumber(tokens[4 + ndim])
    if nbytes == nil then
      file:close()
      error("corrupt payload size for '" .. key .. "'", 2)
    end

    local data
    if nbytes > 0 then
      data = file:read(nbytes)
      if data == nil or #data ~= nbytes then
        file:close()
        error("truncated tensor data for '" .. key .. "'", 2)
      end
    else
      data = ""
    end

    local buffer = ffi.new("uint8_t[?]", math.max(nbytes, 1))
    if nbytes > 0 then
      ffi.copy(buffer, data, nbytes)
    end
    local shape = ffi.new("int64_t[?]", math.max(ndim, 1))
    for dimension = 1, ndim do
      shape[dimension - 1] = sizes[dimension]
    end
    local pointer = lib.lt_from_data(buffer, ndim,
      ndim > 0 and shape or nil, dtype)
    state[key] = Tensor._wrap(pointer)
  end

  file:close()
  module:load_state_dict(state, true)
  return module
end

return Serialization
