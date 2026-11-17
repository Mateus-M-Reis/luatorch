--
-- torch.optim.adagrad
--
local torch = require("torch")
local libmod = require("torch.lib")
local lib = libmod.lib
local ffi = libmod.ffi
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
  for _, group in ipairs(self.param_groups) do
    local lr = group.lr or self.lr
    local lr_decay = group.lr_decay or self.lr_decay
    local weight_decay = group.weight_decay or self.weight_decay
    local eps = group.eps or self.eps
    local initial = group.initial_accumulator_value
    if initial == nil then initial = self.initial_accumulator_value end
    for _, parameter in ipairs(group.params) do
      local gradient = parameter:grad()
      if gradient then
        local state = self:state_for(parameter)
        if state.sum == nil then
          state.sum = torch.full_like(parameter, initial)
          state.step = 0
        end
        state.step = state.step + 1

        if not gradient:is_contiguous() then
          gradient = gradient:contiguous()
        end

        local status = lib.lt_adagrad_step(
          parameter.h, gradient.h, state.sum.h,
          lr, lr_decay, weight_decay, eps, state.step)
        if status ~= 0 then
          error(ffi.string(lib.lt_last_error()), 3)
        end
      end
    end
  end
  self:sync_lr()
  return self
end

return Adagrad
