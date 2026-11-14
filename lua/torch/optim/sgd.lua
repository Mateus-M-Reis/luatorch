--
-- torch.optim.sgd
--
local torch = require("torch")
local libmod = require("torch.lib")
local lib = libmod.lib
local ffi = libmod.ffi
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
  for _, parameter in ipairs(self.params) do
    local gradient = parameter:grad()
    if gradient then
      if not gradient:is_contiguous() then
        gradient = gradient:contiguous()
      end

      local buffer = nil
      local first_step = 0
      if self.momentum ~= 0 then
        local state = self:state_for(parameter)
        if state.momentum_buffer == nil then
          state.momentum_buffer = torch.zeros_like(parameter)
          state.first_step = true
        end
        buffer = state.momentum_buffer
        if state.first_step then
          first_step = 1
          state.first_step = false
        end
      end

      local status = lib.lt_sgd_step(
        parameter.h, gradient.h, buffer and buffer.h or nil,
        self.lr, self.momentum, self.dampening,
        self.weight_decay, self.nesterov and 1 or 0, first_step)
      if status ~= 0 then
        error(ffi.string(lib.lt_last_error()), 3)
      end
    end
  end
  return self
end

return SGD
