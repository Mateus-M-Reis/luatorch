--
-- torch.nn.dropout
--
local Module = require("torch.nn.module")
local F = require("torch.nn.functional")

local Dropout = Module.class("Dropout")

function Dropout.new(p, inplace)
  local self = setmetatable(Module.new(), Dropout)
  self.p = p or 0.5
  self.inplace = inplace and true or false
  return self
end

function Dropout:forward(input)
  return F.dropout(input, self.p, self.training, self.inplace)
end

function Dropout:format()
  return string.format("Dropout(p=%g, inplace=%s)", self.p, tostring(self.inplace))
end

local Dropout2d = Module.class("Dropout2d")

function Dropout2d.new(p, inplace)
  local self = setmetatable(Module.new(), Dropout2d)
  self.p = p or 0.5
  self.inplace = inplace and true or false
  return self
end

function Dropout2d:forward(input)
  return F.dropout(input, self.p, self.training, self.inplace)
end

return {
  Dropout = Dropout,
  Dropout2d = Dropout2d,
}
