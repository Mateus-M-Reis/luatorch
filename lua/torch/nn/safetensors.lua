--
-- Safetensors state-dictionary interchange.
--
-- The format is the standard safetensors container: an 8-byte little-endian
-- header length, a JSON header padded to an 8-byte boundary, and raw
-- little-endian tensor data. Tensor names are the module state-dict names, so
-- PyTorch parameter names can be mapped directly (see docs/nn.md).
--
local ffi = require("ffi")
local libmod = require("torch.lib")
local lib = libmod.lib
local Tensor = require("torch.tensor")

local M = {}

local DTYPES = {
  [0] = { name = "F32", size = 4 },
  [1] = { name = "F64", size = 8 },
  [2] = { name = "I64", size = 8 },
  [3] = { name = "I32", size = 4 },
  [4] = { name = "I16", size = 2 },
  [5] = { name = "I8", size = 1 },
  [6] = { name = "U8", size = 1 },
  [7] = { name = "BOOL", size = 1 },
  [8] = { name = "F16", size = 2 },
}

local BY_NAME = {}
for id, info in pairs(DTYPES) do BY_NAME[info.name] = id end

local function json_escape(value)
  return '"' .. tostring(value):gsub('[%z\1-\31\\"]', function(ch)
    if ch == '"' or ch == "\\" then return "\\" .. ch end
    if ch == "\n" then return "\\n" end
    if ch == "\r" then return "\\r" end
    if ch == "\t" then return "\\t" end
    return string.format("\\u%04x", ch:byte())
  end) .. '"'
end

local function parse_json(text)
  local index = 1
  local function skip()
    while true do
      local ch = text:sub(index, index)
      if ch == " " or ch == "\n" or ch == "\r" or ch == "\t" then index = index + 1
      else break end
    end
  end
  local parse_value
  local function parse_string()
    index = index + 1
    local out = {}
    while true do
      local ch = text:sub(index, index)
      if ch == "" then error("safetensors: unterminated JSON string", 3) end
      index = index + 1
      if ch == '"' then break end
      if ch == "\\" then
        local esc = text:sub(index, index)
        index = index + 1
        if esc == "n" then out[#out + 1] = "\n"
        elseif esc == "r" then out[#out + 1] = "\r"
        elseif esc == "t" then out[#out + 1] = "\t"
        elseif esc == "b" then out[#out + 1] = "\b"
        elseif esc == "f" then out[#out + 1] = "\f"
        elseif esc == '"' or esc == "\\" or esc == "/" then out[#out + 1] = esc
        elseif esc == "u" then
          local hex = text:sub(index, index + 3)
          index = index + 4
          local code = tonumber(hex, 16)
          if code == nil then error("safetensors: invalid JSON escape", 3) end
          out[#out + 1] = string.char(code % 256)
        else error("safetensors: invalid JSON escape", 3) end
      else
        out[#out + 1] = ch
      end
    end
    return table.concat(out)
  end
  local function parse_number()
    local start = index
    while text:sub(index, index):match("[%d%.eE%+%-]") do index = index + 1 end
    local value = tonumber(text:sub(start, index - 1))
    if value == nil then error("safetensors: invalid JSON number", 3) end
    return value
  end
  local function parse_array()
    index = index + 1
    local result = {}
    skip()
    if text:sub(index, index) == "]" then index = index + 1; return result end
    while true do
      result[#result + 1] = parse_value()
      skip()
      local ch = text:sub(index, index)
      if ch == "]" then index = index + 1; break end
      if ch ~= "," then error("safetensors: malformed JSON array", 3) end
      index = index + 1
      skip()
    end
    return result
  end
  local function parse_object()
    index = index + 1
    local result = {}
    skip()
    if text:sub(index, index) == "}" then index = index + 1; return result end
    while true do
      skip()
      if text:sub(index, index) ~= '"' then
        error("safetensors: JSON object key must be a string", 3)
      end
      local key = parse_string()
      skip()
      if text:sub(index, index) ~= ":" then
        error("safetensors: expected ':' in JSON object", 3)
      end
      index = index + 1
      result[key] = parse_value()
      skip()
      local ch = text:sub(index, index)
      if ch == "}" then index = index + 1; break end
      if ch ~= "," then error("safetensors: malformed JSON object", 3) end
      index = index + 1
    end
    return result
  end
  function parse_value()
    skip()
    local ch = text:sub(index, index)
    if ch == '"' then return parse_string() end
    if ch == "[" then return parse_array() end
    if ch == "{" then return parse_object() end
    if ch == "t" and text:sub(index, index + 3) == "true" then index = index + 4; return true end
    if ch == "f" and text:sub(index, index + 4) == "false" then index = index + 5; return false end
    if ch == "n" and text:sub(index, index + 3) == "null" then index = index + 4; return nil end
    return parse_number()
  end
  local result = parse_value()
  skip()
  if index <= #text then error("safetensors: trailing JSON data", 3) end
  return result
end

local function encode_u64(value)
  local bytes = {}
  for index = 1, 8 do
    bytes[index] = string.char(value % 256)
    value = math.floor(value / 256)
  end
  return table.concat(bytes)
end

local function decode_u64(bytes)
  local value = 0
  local factor = 1
  for index = 1, 8 do
    value = value + bytes:byte(index) * factor
    factor = factor * 256
  end
  return value
end

local function sorted_keys(state)
  local keys = {}
  for key in pairs(state) do keys[#keys + 1] = key end
  table.sort(keys)
  return keys
end

function M.save(module, path, options)
  options = options or {}
  local state = module:state_dict()
  local keys = sorted_keys(state)
  local entries = {}
  local payload = {}
  local offset = 0
  for _, key in ipairs(keys) do
    local tensor = state[key]:contiguous()
    local dtype = tensor:dtype_id()
    local info = DTYPES[dtype]
    if info == nil then error("safetensors: unsupported dtype " .. tostring(dtype), 3) end
    local nbytes = tensor:nbytes()
    local pointer = lib.lt_data_ptr(tensor.h)
    if pointer == nil then error(ffi.string(lib.lt_last_error()), 3) end
    local shape = tensor:sizes()
    entries[#entries + 1] = string.format(
      '%s:{"dtype":"%s","shape":[%s],"data_offsets":[%d,%d]}',
      json_escape(key), info.name, table.concat(shape, ","), offset, offset + nbytes)
    if nbytes > 0 then payload[#payload + 1] = ffi.string(pointer, nbytes) end
    offset = offset + nbytes
  end
  local header = "{" .. table.concat(entries, ",") .. "}"
  local padding = (8 - (#header % 8)) % 8
  header = header .. string.rep(" ", padding)
  local file = assert(io.open(path, "wb"))
  file:write(encode_u64(#header))
  file:write(header)
  for _, chunk in ipairs(payload) do file:write(chunk) end
  assert(file:close())
  return path
end

function M.load(module, path, options)
  options = options or {}
  local file = assert(io.open(path, "rb"))
  local length = file:read(8)
  if length == nil or #length ~= 8 then file:close(); error("safetensors: truncated header", 3) end
  local header_length = decode_u64(length)
  local header = file:read(header_length)
  local data = file:read("*a") or ""
  file:close()
  if header == nil or #header ~= header_length then
    error("safetensors: truncated JSON header", 3)
  end
  local metadata = parse_json(header)
  local prefix = options.prefix or ""
  local strip = options.strip_prefix or ""
  local state = {}
  for name, entry in pairs(metadata) do
    if name == "__metadata__" then
      -- Reserved global metadata is not a tensor.
    else
      local dtype = BY_NAME[entry.dtype]
      if dtype == nil then error("safetensors: unknown dtype " .. tostring(entry.dtype), 3) end
      if dtype == 8 then
        error("safetensors: float16 tensors are not supported by from_blob", 3)
      end
      local offsets = entry.data_offsets
      if type(offsets) ~= "table" or #offsets ~= 2 then
        error("safetensors: invalid offsets for " .. name, 3)
      end
      local bytes = data:sub(offsets[1] + 1, offsets[2])
      local shape = entry.shape
      local tensor
      if #shape == 0 then
        tensor = Tensor.from_blob(bytes, { 1 }, { dtype = dtype }):reshape({})
      else
        tensor = Tensor.from_blob(bytes, shape, { dtype = dtype }):clone()
      end
      local key = name
      if strip ~= "" and key:sub(1, #strip) == strip then key = key:sub(#strip + 1) end
      if prefix ~= "" then key = prefix .. key end
      state[key] = tensor
    end
  end
  module:load_state_dict(state, options.strict ~= false)
  return module
end

return M
