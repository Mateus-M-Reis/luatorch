--
-- torch.nn.loss
--
local torch = require("torch")
local Module = require("torch.nn.module")
local F = require("torch.nn.functional")

local function class(name)
  return Module.class(name)
end

local MSELoss = class("MSELoss")

function MSELoss.new(reduction)
  local self = setmetatable(Module.new(), MSELoss)
  self.reduction = F.normalize_reduction(reduction)
  return self
end

function MSELoss:forward(input, target)
  return F.mse_loss(input, target, self.reduction)
end

local L1Loss = class("L1Loss")

function L1Loss.new(reduction)
  local self = setmetatable(Module.new(), L1Loss)
  self.reduction = F.normalize_reduction(reduction)
  return self
end

function L1Loss:forward(input, target)
  return F.l1_loss(input, target, self.reduction)
end

local CrossEntropyLoss = class("CrossEntropyLoss")

function CrossEntropyLoss.new(options)
  local self = setmetatable(Module.new(), CrossEntropyLoss)
  if type(options) == "string" then
    options = { reduction = options }
  end
  options = options or {}
  if options.weight ~= nil or options.ignore_index ~= nil or
    options.label_smoothing ~= nil then
    error("CrossEntropyLoss: weight/ignore_index/label_smoothing are not supported yet", 3)
  end
  self.reduction = F.normalize_reduction(options.reduction)
  return self
end

function CrossEntropyLoss:forward(input, target)
  return F.cross_entropy(input, target, self.reduction)
end

local NLLLoss = class("NLLLoss")

function NLLLoss.new(options)
  local self = setmetatable(Module.new(), NLLLoss)
  if type(options) == "string" then
    options = { reduction = options }
  end
  options = options or {}
  if options.weight ~= nil or options.ignore_index ~= nil then
    error("NLLLoss: weight/ignore_index are not supported yet", 3)
  end
  self.reduction = F.normalize_reduction(options.reduction)
  return self
end

function NLLLoss:forward(input, target)
  return F.nll_loss(input, target, self.reduction)
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
