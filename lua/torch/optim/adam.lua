--
-- torch.optim.adam
--
local torch = require("torch")
local libmod = require("torch.lib")
local lib = libmod.lib
local ffi = libmod.ffi
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
  self.decoupled_weight_decay = false
  self.amsgrad = options.amsgrad or false
  return self
end

function Adam:step()
  for _, parameter in ipairs(self.params) do
    local gradient = parameter:grad()
    if gradient then
      local state = self:state_for(parameter)
      if state.step == nil then
        state.step = 0
        state.exp_avg = torch.zeros_like(parameter)
        state.exp_avg_sq = torch.zeros_like(parameter)
        if self.amsgrad then
          state.max_exp_avg_sq = torch.zeros_like(parameter)
        end
      end
      state.step = state.step + 1

      if not gradient:is_contiguous() then
        gradient = gradient:contiguous()
      end

      local status
      if self.amsgrad then
        status = lib.lt_adam_amsgrad_step(
          parameter.h, gradient.h, state.exp_avg.h, state.exp_avg_sq.h,
          state.max_exp_avg_sq.h, self.lr, self.beta1, self.beta2, self.eps,
          self.weight_decay, state.step,
          self.decoupled_weight_decay and 1 or 0)
      else
        status = lib.lt_adam_step(
          parameter.h, gradient.h, state.exp_avg.h, state.exp_avg_sq.h,
          self.lr, self.beta1, self.beta2, self.eps,
          self.weight_decay, state.step,
          self.decoupled_weight_decay and 1 or 0)
      end
      if status ~= 0 then
        error(ffi.string(lib.lt_last_error()), 3)
      end
    end
  end
  return self
end

return Adam
