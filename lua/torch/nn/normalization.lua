--
-- torch.nn.normalization
--
local torch = require("torch")
local Module = require("torch.nn.module")

local BatchNorm2d = Module.class("BatchNorm2d")

function BatchNorm2d.new(num_features, options)
  options = options or {}
  local self = setmetatable(Module.new(), BatchNorm2d)
  self.num_features = num_features
  self.eps = options.eps or 1e-5
  self.momentum = options.momentum or 0.1
  self.affine = options.affine ~= false
  self.track_running_stats = options.track_running_stats ~= false

  if self.affine then
    self:register_parameter("weight", torch.ones(num_features))
    self:register_parameter("bias", torch.zeros(num_features))
  end
  if self.track_running_stats then
    self:register_buffer("running_mean", torch.zeros(num_features))
    self:register_buffer("running_var", torch.ones(num_features))
  end
  self:_refresh_aliases()
  return self
end

function BatchNorm2d:_refresh_aliases()
  self.weight = self._parameters.weight
  self.bias = self._parameters.bias
  self.running_mean = self._buffers.running_mean
  self.running_var = self._buffers.running_var
end

function BatchNorm2d:forward(input)
  local shape = input:sizes()
  if #shape ~= 4 then
    error("BatchNorm2d expects an NCHW tensor", 2)
  end
  local batch, channels, height, width = shape[1], shape[2], shape[3], shape[4]
  if channels ~= self.num_features then
    error(string.format("BatchNorm2d: expected %d channels, got %d",
      self.num_features, channels), 2)
  end

  local use_batch_statistics = self.training or not self.track_running_stats
  local mean, variance
  if use_batch_statistics then
    -- Mean and E[x^2] over N, H, W (dimensions 1, 3 and 4).
    mean = input:mean(4, true):mean(3, true):mean(1, true)
    local second_moment = input:pow(2):mean(4, true):mean(3, true):mean(1, true)
    -- clamp(0) guards against tiny negative values from float cancellation
    variance = second_moment:sub(mean:pow(2)):clamp(0, math.huge)
  else
    mean = self.running_mean:reshape(1, channels, 1, 1)
    variance = self.running_var:reshape(1, channels, 1, 1)
  end

  local normalized = input:sub(mean):div(variance:add(self.eps):sqrt())
  if self.weight then
    normalized = normalized:mul(self.weight:reshape(1, channels, 1, 1))
  end
  if self.bias then
    normalized = normalized:add(self.bias:reshape(1, channels, 1, 1))
  end

  if self.training and self.track_running_stats then
    local count = batch * height * width
    if count <= 1 then
      error("BatchNorm2d: expected more than one value per channel while training", 2)
    end
    -- Running variance uses the unbiased estimate, as PyTorch does.
    local unbiased = variance:mul(count / (count - 1)):reshape(channels)
    local batch_mean = mean:reshape(channels)
    torch.no_grad(function()
      self.running_mean:mul_(1 - self.momentum):add_(batch_mean:mul(self.momentum))
      self.running_var:mul_(1 - self.momentum):add_(unbiased:mul(self.momentum))
    end)
  end

  return normalized
end

function BatchNorm2d:format()
  return string.format("BatchNorm2d(%d, eps=%g, momentum=%g)",
    self.num_features, self.eps, self.momentum)
end

local GroupNorm = Module.class("GroupNorm")

function GroupNorm.new(num_groups, num_channels, options)
  options = options or {}
  if type(num_groups) ~= "number" or num_groups < 1 or num_groups % 1 ~= 0 then
    error("GroupNorm: num_groups must be a positive integer", 3)
  end
  if type(num_channels) ~= "number" or num_channels < 1 or
     num_channels % 1 ~= 0 then
    error("GroupNorm: num_channels must be a positive integer", 3)
  end
  if num_channels % num_groups ~= 0 then
    error("GroupNorm: num_channels must be divisible by num_groups", 3)
  end
  local self = setmetatable(Module.new(), GroupNorm)
  self.num_groups = num_groups
  self.num_channels = num_channels
  self.eps = options.eps or 1e-5
  self.affine = options.affine ~= false
  if self.affine then
    self:register_parameter("weight", torch.ones(num_channels))
    self:register_parameter("bias", torch.zeros(num_channels))
  end
  self:_refresh_aliases()
  return self
end

function GroupNorm:_refresh_aliases()
  self.weight = self._parameters.weight
  self.bias = self._parameters.bias
end

function GroupNorm:forward(input)
  local shape = input:sizes()
  if #shape ~= 4 then
    error("GroupNorm expects an NCHW tensor", 2)
  end
  local batch, channels, height, width = shape[1], shape[2], shape[3], shape[4]
  if channels ~= self.num_channels then
    error(string.format("GroupNorm: expected %d channels, got %d",
      self.num_channels, channels), 2)
  end
  local per_group = math.floor(channels / self.num_groups)
  local group_size = per_group * height * width
  local grouped = input:contiguous():view(batch, self.num_groups, group_size)
  local mean = grouped:mean(3, true)
  local second = grouped:pow(2):mean(3, true)
  local variance = second:sub(mean:pow(2)):clamp(0, math.huge)
  local normalized = grouped:sub(mean):div(variance:add(self.eps):sqrt())
  normalized = normalized:view(batch, channels, height, width)
  if self.weight then
    normalized = normalized:mul(self.weight:reshape(1, channels, 1, 1))
  end
  if self.bias then
    normalized = normalized:add(self.bias:reshape(1, channels, 1, 1))
  end
  return normalized
end

function GroupNorm:format()
  return string.format("GroupNorm(%d, %d, eps=%g, affine=%s)",
    self.num_groups, self.num_channels, self.eps, tostring(self.affine))
end

local LayerNorm = Module.class("LayerNorm")

function LayerNorm.new(normalized_shape, options)
  options = options or {}
  local shape
  if type(normalized_shape) == "number" then
    shape = { normalized_shape }
  elseif type(normalized_shape) == "table" then
    shape = normalized_shape
  else
    error("LayerNorm expects a number or a shape table", 3)
  end
  for i = 1, #shape do
    if type(shape[i]) ~= "number" or shape[i] % 1 ~= 0 or shape[i] <= 0 then
      error("LayerNorm: invalid normalized shape", 3)
    end
  end

  local self = setmetatable(Module.new(), LayerNorm)
  self.normalized_shape = shape
  self.eps = options.eps or 1e-5
  self.elementwise_affine = options.elementwise_affine ~= false
  if self.elementwise_affine then
    self:register_parameter("weight", torch.ones(shape))
    self:register_parameter("bias", torch.zeros(shape))
  end
  self:_refresh_aliases()
  return self
end

function LayerNorm:_refresh_aliases()
  self.weight = self._parameters.weight
  self.bias = self._parameters.bias
end

function LayerNorm:forward(input)
  return torch.layer_norm(
    input, self.normalized_shape, self.weight, self.bias, self.eps)
end

function LayerNorm:format()
  local parts = {}
  for i = 1, #self.normalized_shape do
    parts[i] = tostring(self.normalized_shape[i])
  end
  return "LayerNorm(" .. table.concat(parts, ", ") .. ")"
end

return {
  BatchNorm2d = BatchNorm2d,
  GroupNorm = GroupNorm,
  LayerNorm = LayerNorm,
}
