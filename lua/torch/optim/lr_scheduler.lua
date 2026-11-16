--
-- torch.optim.lr_scheduler
--
-- Schedulers mutate optimizer.lr.  Call scheduler:step() once per epoch,
-- after optimizer:step().
--
local Scheduler = {}
Scheduler.__index = Scheduler

local function class(name)
  local cls = {}
  cls.__index = cls
  cls.__name = name
  cls.__tostring = function(self)
    return self:format()
  end
  setmetatable(cls, {
    __index = Scheduler,
    __call = function(c, ...)
      return c.new(...)
    end,
  })
  return cls
end

function Scheduler.new(optimizer)
  if optimizer == nil or optimizer.lr == nil then
    error("a scheduler expects an optimizer with a learning rate", 3)
  end
  local self = setmetatable({
    optimizer = optimizer,
    last_epoch = 0,
    base_lr = optimizer.lr,
  }, Scheduler)
  return self
end

function Scheduler:step()
  self.last_epoch = self.last_epoch + 1
  return self
end

function Scheduler:get_last_lr()
  return self.optimizer.lr
end

function Scheduler:format()
  local name = getmetatable(self).__name or "Scheduler"
  return string.format("%s(lr=%.6g, epoch=%d)",
    name, self.optimizer.lr, self.last_epoch)
end

-- StepLR ---------------------------------------------------------------------

local StepLR = class("StepLR")

function StepLR.new(optimizer, step_size, gamma)
  local self = setmetatable(Scheduler.new(optimizer), StepLR)
  self.step_size = step_size
  self.gamma = gamma or 0.1
  if type(step_size) ~= "number" or step_size < 1 then
    error("StepLR: step_size must be a positive integer", 3)
  end
  return self
end

function StepLR:step()
  Scheduler.step(self)
  if self.last_epoch % self.step_size == 0 then
    self.optimizer.lr = self.optimizer.lr * self.gamma
  end
  return self
end

-- MultiStepLR ----------------------------------------------------------------

local MultiStepLR = class("MultiStepLR")

function MultiStepLR.new(optimizer, milestones, gamma)
  local self = setmetatable(Scheduler.new(optimizer), MultiStepLR)
  self.milestones = milestones or {}
  self.gamma = gamma or 0.1
  return self
end

function MultiStepLR:step()
  Scheduler.step(self)
  for _, milestone in ipairs(self.milestones) do
    if self.last_epoch == milestone then
      self.optimizer.lr = self.optimizer.lr * self.gamma
      break
    end
  end
  return self
end

-- ExponentialLR --------------------------------------------------------------

local ExponentialLR = class("ExponentialLR")

function ExponentialLR.new(optimizer, gamma)
  local self = setmetatable(Scheduler.new(optimizer), ExponentialLR)
  self.gamma = gamma or 0.1
  return self
end

function ExponentialLR:step()
  Scheduler.step(self)
  self.optimizer.lr = self.optimizer.lr * self.gamma
  return self
end

-- CosineAnnealingLR ----------------------------------------------------------

local CosineAnnealingLR = class("CosineAnnealingLR")

function CosineAnnealingLR.new(optimizer, T_max, eta_min)
  local self = setmetatable(Scheduler.new(optimizer), CosineAnnealingLR)
  self.T_max = T_max
  self.eta_min = eta_min or 0
  if type(T_max) ~= "number" or T_max <= 0 then
    error("CosineAnnealingLR: T_max must be positive", 3)
  end
  return self
end

function CosineAnnealingLR:step()
  Scheduler.step(self)
  local epoch = math.min(self.last_epoch, self.T_max)
  self.optimizer.lr = self.eta_min +
    (self.base_lr - self.eta_min) *
    (1 + math.cos(math.pi * epoch / self.T_max)) / 2
  return self
end

-- ReduceLROnPlateau ----------------------------------------------------------

local ReduceLROnPlateau = class("ReduceLROnPlateau")

function ReduceLROnPlateau.new(optimizer, options)
  options = options or {}
  local self = setmetatable(Scheduler.new(optimizer), ReduceLROnPlateau)
  self.mode = options.mode or "min"
  self.factor = options.factor or 0.1
  self.patience = options.patience or 10
  self.threshold = options.threshold or 1e-4
  self.threshold_mode = options.threshold_mode or "rel"
  self.cooldown = options.cooldown or 0
  self.min_lr = options.min_lr or 0
  self.eps = options.eps or 1e-8
  self.best = nil
  self.num_bad_epochs = 0
  self.cooldown_counter = 0
  if self.mode ~= "min" and self.mode ~= "max" then
    error("ReduceLROnPlateau: mode must be 'min' or 'max'", 3)
  end
  if type(self.factor) ~= "number" or self.factor <= 0 or self.factor >= 1 then
    error("ReduceLROnPlateau: factor must be in (0, 1)", 3)
  end
  if type(self.patience) ~= "number" or self.patience < 0 then
    error("ReduceLROnPlateau: patience must be non-negative", 3)
  end
  return self
end

function ReduceLROnPlateau:_is_better(value)
  if self.best == nil then return true end
  if self.threshold_mode == "abs" then
    if self.mode == "min" then
      return value < self.best - self.threshold
    end
    return value > self.best + self.threshold
  end
  if self.mode == "min" then
    return value < self.best * (1 - self.threshold)
  end
  return value > self.best * (1 + self.threshold)
end

function ReduceLROnPlateau:step(metric)
  if type(metric) ~= "number" or metric ~= metric then
    error("ReduceLROnPlateau:step expects a numeric metric", 3)
  end
  Scheduler.step(self)
  if self.cooldown_counter > 0 then
    self.cooldown_counter = self.cooldown_counter - 1
    self.num_bad_epochs = 0
    return self
  end
  if self:_is_better(metric) then
    self.best = metric
    self.num_bad_epochs = 0
  else
    self.num_bad_epochs = self.num_bad_epochs + 1
  end
  if self.num_bad_epochs > self.patience then
    local reduced = self.optimizer.lr * self.factor
    self.optimizer.lr = math.max(self.min_lr, reduced)
    self.num_bad_epochs = 0
    self.cooldown_counter = self.cooldown
  end
  return self
end

-- LambdaLR -------------------------------------------------------------------

local LambdaLR = class("LambdaLR")

function LambdaLR.new(optimizer, lr_lambda)
  if type(lr_lambda) ~= "function" then
    error("LambdaLR: lr_lambda must be a function", 3)
  end
  local self = setmetatable(Scheduler.new(optimizer), LambdaLR)
  self.lr_lambda = lr_lambda
  return self
end

function LambdaLR:step()
  Scheduler.step(self)
  self.optimizer.lr = self.base_lr * self.lr_lambda(self.last_epoch)
  return self
end

return {
  Scheduler = Scheduler,
  StepLR = StepLR,
  MultiStepLR = MultiStepLR,
  ExponentialLR = ExponentialLR,
  CosineAnnealingLR = CosineAnnealingLR,
  ReduceLROnPlateau = ReduceLROnPlateau,
  LambdaLR = LambdaLR,
}
