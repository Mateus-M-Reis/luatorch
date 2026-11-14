--
-- torch.optim.adam
--
local torch = require("torch")
local Optimizer = require("torch.optim.optimizer")

local Adam = setmetatable({}, {
  __index = Optimizer,
  __call = function(c, ...) return c.new(...) end,
})
Adam.__index = Adam
Adam.__name = "Adam"
Adam.__tostring = function(self) return self:format() end

function Adam.new(params, options)
  if type(options) == "number" then
    options = { lr = options }
  end
  options = options or {}
  local self = setmetatable(Optimizer.new(params, options), Adam)
  self.lr = options.lr or 1e-3
  local betas = options.betas or { 0.9, 0.999 }
  self.beta1 = betas[1]
  self.beta2 = betas[2]
  self.eps = options.eps or 1e-8
  self.weight_decay = options.weight_decay or 0
  return self
end

function Adam:step()
  torch.no_grad(function()
    for _, parameter in ipairs(self.params) do
      local gradient = parameter:grad()
      if gradient then
        local state = self:state_for(parameter)
        if state.step == nil then
          state.step = 0
          state.exp_avg = torch.zeros_like(parameter)
          state.exp_avg_sq = torch.zeros_like(parameter)
        end
        state.step = state.step + 1

        if self.weight_decay ~= 0 then
          gradient = gradient:add(parameter:mul(self.weight_decay))
        end

        state.exp_avg:mul_(self.beta1):add_(gradient:mul(1 - self.beta1))
        state.exp_avg_sq:mul_(self.beta2)
          :add_(gradient:mul(gradient):mul(1 - self.beta2))

        local bias_correction1 = 1 - self.beta1 ^ state.step
        local bias_correction2 = 1 - self.beta2 ^ state.step
        local step_size = self.lr / bias_correction1
        local denominator = state.exp_avg_sq:sqrt()
          :div(math.sqrt(bias_correction2))
          :add_(self.eps)

        parameter:add_(state.exp_avg:div(denominator):mul(-step_size))
      end
    end
  end)
  return self
end

return Adam
