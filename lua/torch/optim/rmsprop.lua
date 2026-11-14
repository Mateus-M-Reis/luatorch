--
-- torch.optim.rmsprop
--
local torch = require("torch")
local Optimizer = require("torch.optim.optimizer")

local RMSprop = setmetatable({}, {
  __index = Optimizer,
  __call = function(c, ...) return c.new(...) end,
})
RMSprop.__index = RMSprop
RMSprop.__name = "RMSprop"
RMSprop.__tostring = function(self) return self:format() end

function RMSprop.new(params, options)
  if type(options) == "number" then
    options = { lr = options }
  end
  options = options or {}
  local self = setmetatable(Optimizer.new(params, options), RMSprop)
  self.lr = options.lr or 1e-2
  self.alpha = options.alpha or 0.99
  self.eps = options.eps or 1e-8
  self.weight_decay = options.weight_decay or 0
  self.momentum = options.momentum or 0
  self.centered = options.centered and true or false
  return self
end

function RMSprop:step()
  torch.no_grad(function()
    for _, parameter in ipairs(self.params) do
      local gradient = parameter:grad()
      if gradient then
        local state = self:state_for(parameter)
        if state.square_avg == nil then
          state.square_avg = torch.zeros_like(parameter)
          state.step = 0
          if self.momentum > 0 then
            state.momentum_buffer = torch.zeros_like(parameter)
          end
          if self.centered then
            state.grad_avg = torch.zeros_like(parameter)
          end
        end
        state.step = state.step + 1

        if self.weight_decay ~= 0 then
          gradient = gradient:add(parameter:mul(self.weight_decay))
        end

        state.square_avg:mul_(self.alpha)
          :add_(gradient:mul(gradient):mul(1 - self.alpha))

        local average
        if self.centered then
          state.grad_avg:mul_(self.alpha):add_(gradient:mul(1 - self.alpha))
          average = state.square_avg:sub(state.grad_avg:pow(2))
        else
          average = state.square_avg
        end

        local denominator = average:sqrt():add_(self.eps)
        if self.momentum > 0 then
          state.momentum_buffer:mul_(self.momentum)
            :add_(gradient:div(denominator))
          parameter:add_(state.momentum_buffer:mul(-self.lr))
        else
          parameter:add_(gradient:div(denominator):mul(-self.lr))
        end
      end
    end
  end)
  return self
end

return RMSprop
