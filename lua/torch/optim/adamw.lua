--
-- torch.optim.adamw
--
-- Adam with decoupled weight decay (Loshchilov & Hutter).
--
local Adam = require("torch.optim.adam")

local AdamW = setmetatable({}, {
  __index = Adam,
  __call = function(c, ...)
    return c.new(...)
  end,
})
AdamW.__index = AdamW
AdamW.__name = "AdamW"
AdamW.__tostring = function(self)
  return self:format()
end

function AdamW.new(params, options)
  local self = Adam.new(params, options or {})
  setmetatable(self, AdamW)
  self.decoupled_weight_decay = true
  return self
end

return AdamW
