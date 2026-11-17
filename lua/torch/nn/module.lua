--
-- torch.nn.module
--
-- A small, Torch7-flavoured module system built on Tensor autograd.
-- Modules only implement forward(); gradients come from Tensor:backward().
--
local torch = require("torch")

local Module = {}
Module.__index = Module
Module.__name = "Module"

function Module.new()
  return setmetatable({
    training = true,
    _parameters = {},
    _parameter_order = {},
    _buffers = {},
    _buffer_order = {},
    _modules = {},
    _module_order = {},
  }, Module)
end

-- Build a subclass of Module.  Instances get metamethods copied onto the
-- class itself because Lua looks up metamethods without following __index.
local function class(name)
  local cls = {}
  cls.__index = cls
  cls.__name = name
  cls.__call = function(self, ...)
    return self:forward(...)
  end
  cls.__tostring = function(self)
    return self:format()
  end
  setmetatable(cls, {
    __index = Module,
    __call = function(c, ...)
      return c.new(...)
    end,
  })
  return cls
end

Module.class = class

function Module:forward()
  error("forward() is not implemented for " .. self:_get_name(), 2)
end

function Module:__call(...)
  return self:forward(...)
end

function Module:_get_name()
  local mt = getmetatable(self)
  return (mt and mt.__name) or "Module"
end

function Module:format()
  return self:_get_name() .. "()"
end

function Module:register_parameter(name, tensor)
  if not torch.is_tensor(tensor) then
    error("register_parameter expects a Tensor", 3)
  end
  if self._parameters[name] ~= nil then
    error("parameter '" .. name .. "' is already registered", 3)
  end
  if tensor:is_floating_point() then
    tensor:requires_grad_(true)
  end
  self._parameters[name] = tensor
  self._parameter_order[#self._parameter_order + 1] = name
  return tensor
end

function Module:register_buffer(name, tensor)
  if not torch.is_tensor(tensor) then
    error("register_buffer expects a Tensor", 3)
  end
  self._buffers[name] = tensor
  self._buffer_order[#self._buffer_order + 1] = name
  return tensor
end

function Module:register_module(name, child)
  if not (type(child) == "table" and child.forward) then
    error("register_module expects a Module", 3)
  end
  self._modules[name] = child
  self._module_order[#self._module_order + 1] = name
  return child
end

function Module:_collect_parameters(out)
  for _, name in ipairs(self._parameter_order) do
    out[#out + 1] = self._parameters[name]
  end
  for _, name in ipairs(self._module_order) do
    self._modules[name]:_collect_parameters(out)
  end
  return out
end

function Module:parameters()
  return self:_collect_parameters({})
end

function Module:named_parameters(prefix)
  prefix = prefix or ""
  local result = {}
  for _, name in ipairs(self._parameter_order) do
    result[prefix .. name] = self._parameters[name]
  end
  for _, name in ipairs(self._module_order) do
    local child = self._modules[name]:named_parameters(prefix .. name .. ".")
    for key, value in pairs(child) do
      result[key] = value
    end
  end
  return result
end

function Module:named_buffers(prefix)
  prefix = prefix or ""
  local result = {}
  for _, name in ipairs(self._buffer_order) do
    result[prefix .. name] = self._buffers[name]
  end
  for _, name in ipairs(self._module_order) do
    local child = self._modules[name]:named_buffers(prefix .. name .. ".")
    for key, value in pairs(child) do
      result[key] = value
    end
  end
  return result
end

function Module:zero_grad()
  for _, parameter in ipairs(self:parameters()) do
    parameter:zero_grad()
  end
  return self
end

function Module:train(mode)
  if mode == nil then
    mode = true
  end
  self.training = mode and true or false
  for _, name in ipairs(self._module_order) do
    self._modules[name]:train(self.training)
  end
  return self
end

function Module:eval()
  return self:train(false)
end

function Module:to(dtype)
  torch.no_grad(function()
    for _, name in ipairs(self._parameter_order) do
      local converted = self._parameters[name]:to(dtype)
      if converted:is_floating_point() then
        converted:requires_grad_(true)
      end
      self._parameters[name] = converted
    end
    for _, name in ipairs(self._buffer_order) do
      self._buffers[name] = self._buffers[name]:to(dtype)
    end
    for _, name in ipairs(self._module_order) do
      self._modules[name]:to(dtype)
    end
  end)
  self:_refresh_aliases()
  return self
end

-- Subclasses that expose `self.weight` / `self.bias` conveniences override this
-- to refresh them after dtype conversion replaces parameter objects.
function Module:_refresh_aliases() end

function Module:state_dict()
  local result = {}
  -- Detached clones: mutating the model afterwards must not silently
  -- change a snapshot the caller intends to save or reload.
  for _, name in ipairs(self._parameter_order) do
    result[name] = self._parameters[name]:detach():clone()
  end
  for _, name in ipairs(self._buffer_order) do
    result[name] = self._buffers[name]:detach():clone()
  end
  for _, name in ipairs(self._module_order) do
    local child = self._modules[name]:state_dict()
    for key, value in pairs(child) do
      result[name .. "." .. key] = value
    end
  end
  return result
end

function Module:save(path)
  require("torch.nn.serialization").save(self, path)
  return self
end

function Module:load(path)
  require("torch.nn.serialization").load(self, path)
  return self
end

function Module:save_safetensors(path, options)
  require("torch.nn.safetensors").save(self, path, options)
  return self
end

function Module:load_safetensors(path, options)
  require("torch.nn.safetensors").load(self, path, options)
  return self
end

function Module:load_state_dict(source, strict)
  if strict == nil then
    strict = true
  end
  local expected = self:named_parameters()
  for key, value in pairs(self:named_buffers()) do
    expected[key] = value
  end

  local seen = {}
  for name, destination in pairs(expected) do
    local value = source[name]
    if value == nil then
      if strict then
        error("missing key in state_dict: " .. name, 3)
      end
    else
      if not torch.is_tensor(value) then
        error("state_dict entry '" .. name .. "' is not a Tensor", 3)
      end
      if #destination:sizes() ~= #value:sizes() then
        error("size mismatch for '" .. name .. "'", 3)
      end
      local destination_sizes = destination:sizes()
      local value_sizes = value:sizes()
      for i = 1, #destination_sizes do
        if destination_sizes[i] ~= value_sizes[i] then
          error("size mismatch for '" .. name .. "'", 3)
        end
      end
      torch.no_grad(function()
        destination:copy_(value)
      end)
      seen[name] = true
    end
  end

  if strict then
    for name in pairs(source) do
      if not seen[name] and expected[name] == nil then
        error("unexpected key in state_dict: " .. name, 3)
      end
    end
  end
  return self
end

return Module
