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

return {
  BatchNorm2d = BatchNorm2d,
}
