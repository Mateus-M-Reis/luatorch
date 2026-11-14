--
-- torch.optim.sgd
--
local torch = require("torch")
local Optimizer = require("torch.optim.optimizer")

local SGD = setmetatable({}, {
  __index = Optimizer,
  __call = function(c, ...) return c.new(...) end,
})
SGD.__index = SGD
SGD.__name = "SGD"
SGD.__tostring = function(self) return self:format() end

function SGD.new(params, options)
  if type(options) == "number" then
    options = { lr = options }
  end
  options = options or {}
  local self = setmetatable(Optimizer.new(params, options), SGD)
  self.lr = options.lr or 0.01
  self.momentum = options.momentum or 0
  self.dampening = options.dampening or 0
  self.weight_decay = options.weight_decay or 0
  self.nesterov = options.nesterov and true or false
  if self.nesterov then
    if self.momentum <= 0 then
      error("SGD: nesterov momentum requires momentum > 0", 3)
    end
    if self.dampening ~= 0 then
      error("SGD: nesterov momentum requires dampening == 0", 3)
    end
  end
  return self
end

function SGD:step()
  torch.no_grad(function()
    for _, parameter in ipairs(self.params) do
      local gradient = parameter:grad()
      if gradient then
        if self.weight_decay ~= 0 then
          gradient = gradient:add(parameter:mul(self.weight_decay))
        end

        if self.momentum ~= 0 then
          local state = self:state_for(parameter)
          local buffer = state.momentum_buffer
          if buffer == nil then
            buffer = gradient:clone()
            state.momentum_buffer = buffer
          else
            buffer:mul_(self.momentum):add_(gradient:mul(1 - self.dampening))
          end
          if self.nesterov then
            gradient = gradient:add(buffer:mul(self.momentum))
          else
            gradient = buffer
          end
        end

        parameter:add_(gradient:mul(-self.lr))
      end
    end
  end)
  return self
end

return SGD
