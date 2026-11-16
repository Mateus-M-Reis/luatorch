--
-- torch.nn.functional
--
-- Stateless neural-network operations.  Modules are thin wrappers around
-- these functions so custom layers can reuse the same code.
--
local torch = require("torch")

local F = {}

local reduction_codes = { none = 0, mean = 1, sum = 2 }

local function normalize_reduction(reduction)
  reduction = reduction or "mean"
  if type(reduction) == "number" then
    if reduction == 0 then return "none" end
    if reduction == 1 then return "mean" end
    if reduction == 2 then return "sum" end
    error("invalid reduction code: " .. tostring(reduction), 3)
  end
  if reduction_codes[reduction] == nil then
    error("invalid reduction: " .. tostring(reduction), 3)
  end
  return reduction
end

local function apply_reduction(loss, reduction)
  if reduction == "mean" then
    return loss:mean()
  elseif reduction == "sum" then
    return loss:sum()
  end
  return loss
end

F.normalize_reduction = normalize_reduction
F.apply_reduction = apply_reduction

-- Elementwise ----------------------------------------------------------------

function F.relu(input, inplace)
  if inplace then
    return input:relu_()
  end
  return input:relu()
end

function F.leaky_relu(input, negative_slope, inplace)
  negative_slope = negative_slope or 0.01
  local output = input:maximum(input:mul(negative_slope))
  if inplace then
    input:copy_(output)
    return input
  end
  return output
end

function F.sigmoid(input)
  return input:sigmoid()
end

function F.tanh(input)
  return input:tanh()
end

function F.softmax(input, dim)
  return input:softmax(dim)
end

function F.log_softmax(input, dim)
  return input:log_softmax(dim)
end

-- Linear / convolution / pooling --------------------------------------------

function F.linear(input, weight, bias)
  local output = input:matmul(weight:t())
  if bias ~= nil then
    output = output:add(bias)
  end
  return output
end

function F.conv2d(input, weight, bias, stride, padding, dilation, groups)
  return torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
end

function F.conv_transpose2d(input, weight, bias, stride, padding,
    output_padding, groups, dilation)
  return torch.conv_transpose2d(input, weight, bias, stride, padding,
    output_padding, groups, dilation)
end

function F.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
end

function F.adaptive_avg_pool2d(input, output_size)
  -- Average-pool to a single value per channel: global average pooling.
  if type(output_size) == "table" then
    if output_size[1] == 1 and output_size[2] == 1 then
      output_size = 1
    else
      error("adaptive_avg_pool2d currently supports output size 1x1", 3)
    end
  end
  if output_size ~= 1 then
    error("adaptive_avg_pool2d currently supports output size 1x1", 3)
  end
  return input:mean(3, true):mean(4, true)
end

-- Losses ---------------------------------------------------------------------

function F.mse_loss(input, target, reduction)
  reduction = normalize_reduction(reduction)
  local squared = input:sub(target):pow(2)
  return apply_reduction(squared, reduction)
end

function F.l1_loss(input, target, reduction)
  reduction = normalize_reduction(reduction)
  local absolute = input:sub(target):abs()
  return apply_reduction(absolute, reduction)
end

function F.cross_entropy(input, target, reduction)
  reduction = normalize_reduction(reduction)
  return torch.cross_entropy(input, target, reduction)
end

function F.nll_loss(input, target, reduction)
  reduction = normalize_reduction(reduction)
  return torch.nll_loss(input, target, reduction)
end

function F.binary_cross_entropy_with_logits(input, target, reduction)
  reduction = normalize_reduction(reduction)
  return torch.binary_cross_entropy_with_logits(input, target, reduction)
end

function F.binary_cross_entropy(input, target, reduction)
  reduction = normalize_reduction(reduction)
  local epsilon = 1e-12
  local probabilities = input:clamp(epsilon, 1 - epsilon)
  local one_minus_target = target:neg():add(1)
  local one_minus_probability = probabilities:neg():add(1)
  local loss = target:mul(probabilities:log())
    :add(one_minus_target:mul(one_minus_probability:log()))
    :neg()
  return apply_reduction(loss, reduction)
end

-- Dropout --------------------------------------------------------------------

function F.dropout(input, p, training, inplace)
  p = p or 0.5
  if p < 0 or p > 1 then
    error("dropout probability has to be between 0 and 1", 3)
  end
  if not training or p == 0 then
    return input
  end
  if p == 1 then
    if inplace then
      return input:zero_()
    end
    return torch.zeros_like(input)
  end

  local sizes = input:sizes()
  local mask
  if #sizes == 0 then
    mask = torch.rand(1, { dtype = input:dtype() }):ge(1 - p):reshape({})
  else
    mask = torch.rand(sizes, { dtype = input:dtype() }):ge(1 - p)
  end
  local scaled_mask = mask:to(input:dtype()):mul(1 / (1 - p))

  if inplace then
    input:mul_(scaled_mask)
    return input
  end
  return input:mul(scaled_mask)
end

return F
