--
-- torch.nn.loss
--
local torch = require("torch")
local Module = require("torch.nn.module")
local F = require("torch.nn.functional")

local function class(name)
  return Module.class(name)
end

local function can_fuse_elementwise_loss(input, target, reduction)
  if reduction == "none" then
    return false
  end
  if not input or not target or not input.sizes or not target.sizes then
    return false
  end
  local input_dtype = input:dtype_id()
  local target_dtype = target:dtype_id()
  if input_dtype ~= target_dtype or
    (input_dtype ~= 0 and input_dtype ~= 1) then
    return false
  end
  local input_sizes = input:sizes()
  local target_sizes = target:sizes()
  if #input_sizes ~= #target_sizes then
    return false
  end
  for index = 1, #input_sizes do
    if input_sizes[index] ~= target_sizes[index] then
      return false
    end
  end
  return input:numel() > 0
end

local MSELoss = class("MSELoss")

function MSELoss.new(reduction)
  local self = setmetatable(Module.new(), MSELoss)
  self.reduction = F.normalize_reduction(reduction)
  return self
end

function MSELoss:forward(input, target)
  if can_fuse_elementwise_loss(input, target, self.reduction) then
    self._fused = self._fused or torch.fuse_reduce(function(a, b)
      return (a - b) ^ 2
    end, self.reduction)
    return self._fused(input, target)
  end
  return F.mse_loss(input, target, self.reduction)
end

local L1Loss = class("L1Loss")

function L1Loss.new(reduction)
  local self = setmetatable(Module.new(), L1Loss)
  self.reduction = F.normalize_reduction(reduction)
  return self
end

function L1Loss:forward(input, target)
  if can_fuse_elementwise_loss(input, target, self.reduction) then
    self._fused = self._fused or torch.fuse_reduce(function(a, b)
      return (a - b):abs()
    end, self.reduction)
    return self._fused(input, target)
  end
  return F.l1_loss(input, target, self.reduction)
end

local CrossEntropyLoss = class("CrossEntropyLoss")

function CrossEntropyLoss.new(options)
  local self = setmetatable(Module.new(), CrossEntropyLoss)
  if type(options) == "string" then
    options = { reduction = options }
  end
  options = options or {}
  self.reduction = F.normalize_reduction(options.reduction)
  self.weight = options.weight
  self.ignore_index = options.ignore_index
  self.label_smoothing = options.label_smoothing
  return self
end

function CrossEntropyLoss:forward(input, target)
  return F.cross_entropy(input, target, {
    reduction = self.reduction,
    weight = self.weight,
    ignore_index = self.ignore_index,
    label_smoothing = self.label_smoothing,
  })
end

local NLLLoss = class("NLLLoss")

function NLLLoss.new(options)
  local self = setmetatable(Module.new(), NLLLoss)
  if type(options) == "string" then
    options = { reduction = options }
  end
  options = options or {}
  self.reduction = F.normalize_reduction(options.reduction)
  self.weight = options.weight
  self.ignore_index = options.ignore_index
  return self
end

function NLLLoss:forward(input, target)
  return F.nll_loss(input, target, {
    reduction = self.reduction,
    weight = self.weight,
    ignore_index = self.ignore_index,
  })
end

local BCEWithLogitsLoss = class("BCEWithLogitsLoss")

function BCEWithLogitsLoss.new(reduction)
  local self = setmetatable(Module.new(), BCEWithLogitsLoss)
  self.reduction = F.normalize_reduction(reduction)
  return self
end

function BCEWithLogitsLoss:forward(input, target)
  return F.binary_cross_entropy_with_logits(input, target, self.reduction)
end

local BCELoss = class("BCELoss")

function BCELoss.new(reduction)
  local self = setmetatable(Module.new(), BCELoss)
  self.reduction = F.normalize_reduction(reduction)
  return self
end

function BCELoss:forward(input, target)
  return F.binary_cross_entropy(input, target, self.reduction)
end

return {
  MSELoss = MSELoss,
  L1Loss = L1Loss,
  CrossEntropyLoss = CrossEntropyLoss,
  NLLLoss = NLLLoss,
  BCEWithLogitsLoss = BCEWithLogitsLoss,
  BCELoss = BCELoss,
}
