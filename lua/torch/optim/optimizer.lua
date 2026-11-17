--
-- torch.optim.optimizer
--
local torch = require("torch")

local Optimizer = {}
Optimizer.__index = Optimizer

function Optimizer.new(params, defaults)
  if type(params) ~= "table" then
    error("Optimizer expects a table of parameters or parameter groups", 3)
  end
  defaults = defaults or {}
  local groups = {}
  local list = {}
  local grouped = false
  for index = 1, #params do
    local entry = params[index]
    if not torch.is_tensor(entry) and type(entry) == "table" and
        entry.params ~= nil then
      grouped = true
      break
    end
  end
  if grouped then
    for group_index = 1, #params do
      local source = params[group_index]
      if type(source) ~= "table" or type(source.params) ~= "table" then
        error("optimizer group " .. group_index ..
          " must be a table with a params list", 3)
      end
      local group = {}
      for key, value in pairs(defaults) do group[key] = value end
      for key, value in pairs(source) do
        if key ~= "params" then group[key] = value end
      end
      group.params = {}
      for parameter_index = 1, #source.params do
        local parameter = source.params[parameter_index]
        if not torch.is_tensor(parameter) then
          error("optimizer group " .. group_index .. " parameter " ..
            parameter_index .. " is not a Tensor", 3)
        end
        group.params[#group.params + 1] = parameter
        list[#list + 1] = parameter
      end
      if #group.params == 0 then
        error("optimizer group " .. group_index .. " has no parameters", 3)
      end
      groups[#groups + 1] = group
    end
  else
    local group = { params = {} }
    for key, value in pairs(defaults) do group[key] = value end
    for parameter_index = 1, #params do
      local parameter = params[parameter_index]
      if not torch.is_tensor(parameter) then
        error("optimizer parameter " .. parameter_index .. " is not a Tensor", 3)
      end
      group.params[#group.params + 1] = parameter
      list[#list + 1] = parameter
    end
    groups[1] = group
  end
  if #list == 0 then
    error("Optimizer expects at least one parameter", 3)
  end
  local self = setmetatable({
    params = list,
    param_groups = groups,
    defaults = defaults,
    state = {},
  }, Optimizer)
  return self
end

function Optimizer:add_param_group(source)
  if type(source) ~= "table" or type(source.params) ~= "table" then
    error("add_param_group expects a table with a params list", 2)
  end
  local group = {}
  for key, value in pairs(self.defaults) do group[key] = value end
  for key, value in pairs(source) do
    if key ~= "params" then group[key] = value end
  end
  group.params = {}
  for index = 1, #source.params do
    local parameter = source.params[index]
    if not torch.is_tensor(parameter) then
      error("add_param_group parameter " .. index .. " is not a Tensor", 2)
    end
    group.params[#group.params + 1] = parameter
    self.params[#self.params + 1] = parameter
  end
  if #group.params == 0 then error("add_param_group has no parameters", 2) end
  self.param_groups[#self.param_groups + 1] = group
  return self
end

function Optimizer:group_lr(group)
  return group.lr or self.lr
end

function Optimizer:sync_lr()
  local group = self.param_groups[1]
  if group and group.lr ~= nil then self.lr = group.lr end
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
