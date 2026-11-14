--
-- torch.optim.clip
--
local torch = require("torch")

local Clip = {}

local function collect_gradients(parameters)
  local gradients = {}
  for _, parameter in ipairs(parameters) do
    local gradient = parameter:grad()
    if gradient then
      gradients[#gradients + 1] = gradient
    end
  end
  return gradients
end

-- Clips gradients so that their global norm is at most max_norm.
-- Returns the (pre-clipping) total norm.
function Clip.clip_grad_norm_(parameters, max_norm, norm_type)
  norm_type = norm_type or 2
  max_norm = max_norm or 1
  if max_norm <= 0 then
    error("clip_grad_norm_: max_norm must be positive", 3)
  end

  local gradients = collect_gradients(parameters)
  local total
  if norm_type == math.huge then
    total = 0
    for _, gradient in ipairs(gradients) do
      total = math.max(total, gradient:abs():max():item())
    end
  else
    total = 0
    for _, gradient in ipairs(gradients) do
      total = total + gradient:abs():pow(norm_type):sum():item()
    end
    total = total ^ (1 / norm_type)
  end

  local coefficient = max_norm / (total + 1e-6)
  if coefficient < 1 and #gradients > 0 then
    torch.no_grad(function()
      for _, gradient in ipairs(gradients) do
        gradient:mul_(coefficient)
      end
    end)
  end
  return total
end

function Clip.clip_grad_value_(parameters, clip_value)
  if clip_value == nil or clip_value < 0 then
    error("clip_grad_value_: clip_value must be non-negative", 3)
  end
  local gradients = collect_gradients(parameters)
  torch.no_grad(function()
    for _, gradient in ipairs(gradients) do
      gradient:clamp_(-clip_value, clip_value)
    end
  end)
  return parameters
end

return Clip
