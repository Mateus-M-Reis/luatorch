--
-- torch.optim.rmsprop
--
local torch = require("torch")
local libmod = require("torch.lib")
local lib = libmod.lib
local ffi = libmod.ffi
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

      if not gradient:is_contiguous() then
        gradient = gradient:contiguous()
      end

      local status = lib.lt_rmsprop_step(
        parameter.h, gradient.h, state.square_avg.h,
        state.grad_avg and state.grad_avg.h or nil,
        state.momentum_buffer and state.momentum_buffer.h or nil,
        self.lr, self.alpha, self.eps, self.weight_decay,
        self.momentum, self.centered and 1 or 0)
      if status ~= 0 then
        error(ffi.string(lib.lt_last_error()), 3)
      end
    end
  end
  return self
end

return RMSprop
