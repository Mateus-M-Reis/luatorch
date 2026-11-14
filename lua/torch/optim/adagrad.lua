--
-- torch.optim.adagrad
--
local torch = require("torch")
local Optimizer = require("torch.optim.optimizer")

local Adagrad = setmetatable({}, {
  __index = Optimizer,
  __call = function(c, ...) return c.new(...) end,
})
Adagrad.__index = Adagrad
Adagrad.__name = "Adagrad"
Adagrad.__tostring = function(self) return self:format() end

function Adagrad.new(params, options)
  if type(options) == "number" then
    options = { lr = options }
  end
  options = options or {}
  local self = setmetatable(Optimizer.new(params, options), Adagrad)
  self.lr = options.lr or 1e-2
  self.lr_decay = options.lr_decay or 0
  self.weight_decay = options.weight_decay or 0
  self.eps = options.eps or 1e-10
  self.initial_accumulator_value = options.initial_accumulator_value or 0
  return self
end

function Adagrad:step()
  torch.no_grad(function()
    for _, parameter in ipairs(self.params) do
      local gradient = parameter:grad()
      if gradient then
        local state = self:state_for(parameter)
        if state.sum == nil then
          state.sum = torch.full_like(parameter, self.initial_accumulator_value)
          state.step = 0
        end
        state.step = state.step + 1

        if self.weight_decay ~= 0 then
          gradient = gradient:add(parameter:mul(self.weight_decay))
        end

        state.sum:add_(gradient:mul(gradient))
        local learning_rate = self.lr /
          (1 + (state.step - 1) * self.lr_decay)
        local update = gradient:div(state.sum:sqrt():add_(self.eps))
        parameter:add_(update:mul(-learning_rate))
      end
    end
  end)
  return self
end

return Adagrad
