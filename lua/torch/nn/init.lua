--
-- torch.nn
--
--   local nn = require("torch.nn")
--   local model = nn.Sequential(
--     nn.Linear(2, 16), nn.ReLU(),
--     nn.Linear(16, 2))
--
-- Modules implement forward() only; training uses Tensor autograd:
--
--   local loss = nn.CrossEntropyLoss()(model(x), y)
--   optimizer:zero_grad()
--   loss:backward()
--   optimizer:step()
--
local torch = require("torch")
local Module = require("torch.nn.module")
local functional = require("torch.nn.functional")
local serialization = require("torch.nn.serialization")

local nn = {
  Module = Module,
  functional = functional,
}

local groups = {
  require("torch.nn.basic"),
  require("torch.nn.loss"),
  require("torch.nn.conv"),
  require("torch.nn.pooling"),
  require("torch.nn.normalization"),
  require("torch.nn.dropout"),
}

for _, group in ipairs(groups) do
  for name, value in pairs(group) do
    if nn[name] ~= nil then
      error("duplicate nn export: " .. name)
    end
    nn[name] = value
  end
end

-- Weight initialisation helpers ---------------------------------------------

nn.save = serialization.save
nn.load = serialization.load

nn.init = {
  uniform_ = function(tensor, low, high)
    low = low or 0
    high = high or 1
    torch.no_grad(function()
      tensor:uniform_(low, high)
    end)
    return tensor
  end,
  constant_ = function(tensor, value)
    torch.no_grad(function()
      tensor:fill_(value)
    end)
    return tensor
  end,
  zeros_ = function(tensor)
    return nn.init.constant_(tensor, 0)
  end,
  ones_ = function(tensor)
    return nn.init.constant_(tensor, 1)
  end,
}

return nn
