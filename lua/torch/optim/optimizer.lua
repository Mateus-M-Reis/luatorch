--
-- torch.optim.optimizer
--
local torch = require("torch")

local Optimizer = {}
Optimizer.__index = Optimizer

function Optimizer.new(params, defaults)
  if type(params) ~= "table" then
    error("Optimizer expects a table of parameters", 3)
  end
  local list = {}
  for i = 1, #params do
    local parameter = params[i]
    if not torch.is_tensor(parameter) then
      error("optimizer parameter " .. i .. " is not a Tensor", 3)
    end
    list[#list + 1] = parameter
  end
  local self = setmetatable({
    params = list,
    defaults = defaults or {},
    state = {},
  }, Optimizer)
  return self
end

function Optimizer:zero_grad()
  for _, parameter in ipairs(self.params) do
    parameter:zero_grad()
  end
  return self
end

function Optimizer:state_for(parameter)
  local state = self.state[parameter]
  if state == nil then
    state = {}
    self.state[parameter] = state
  end
  return state
end

function Optimizer:step()
  error("step() is not implemented for this optimizer", 2)
end

function Optimizer:format()
  local name = getmetatable(self).__name or "Optimizer"
  return name .. "(" .. #self.params .. " parameters)"
end

Optimizer.__tostring = Optimizer.format

return Optimizer
