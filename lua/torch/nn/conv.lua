--
-- torch.nn.conv
--
local torch = require("torch")
local Module = require("torch.nn.module")

local Conv2d = Module.class("Conv2d")

local function pair(value, default, name)
  if value == nil then
    return default, default
  end
  if type(value) == "number" then
    return value, value
  end
  if type(value) == "table" and #value == 2 then
    return value[1], value[2]
  end
  error(name .. " must be a number or a {height, width} table", 3)
end

function Conv2d.new(in_channels, out_channels, kernel_size,
    stride, padding, dilation, groups, bias)
  if type(stride) == "table" and
    (stride.stride or stride.padding or stride.dilation or
      stride.groups or stride.bias ~= nil) then
    local options = stride
    stride = options.stride
    padding = options.padding
    dilation = options.dilation
    groups = options.groups
    if options.bias ~= nil then
      bias = options.bias
    end
  end
  if bias == nil then
    bias = true
  end
  if groups == nil then
    groups = 1
  end

  local kernel_h, kernel_w = pair(kernel_size, nil, "kernel_size")
  local stride_h, stride_w = pair(stride, 1, "stride")
  local pad_h, pad_w = pair(padding, 0, "padding")
  local dilation_h, dilation_w = pair(dilation, 1, "dilation")

  if in_channels % groups ~= 0 or out_channels % groups ~= 0 then
    error("Conv2d: channels must be divisible by groups", 3)
  end

  local self = setmetatable(Module.new(), Conv2d)
  self.in_channels = in_channels
  self.out_channels = out_channels
  self.kernel_size = { kernel_h, kernel_w }
  self.stride = { stride_h, stride_w }
  self.padding = { pad_h, pad_w }
  self.dilation = { dilation_h, dilation_w }
  self.groups = groups

  local fan_in = (in_channels / groups) * kernel_h * kernel_w
  local bound = 1 / math.sqrt(fan_in)
  local weight = torch.rand(
    out_channels, in_channels / groups, kernel_h, kernel_w)
  weight:mul_(2 * bound):sub_(bound)
  self:register_parameter("weight", weight)
  self.weight = self._parameters.weight

  if bias then
    local bias_tensor = torch.rand(out_channels):mul_(2 * bound):sub_(bound)
    self:register_parameter("bias", bias_tensor)
    self.bias = self._parameters.bias
  else
    self.bias = nil
  end
  return self
end

function Conv2d:forward(input)
  return torch.conv2d(
    input, self.weight, self.bias,
    self.stride, self.padding, self.dilation, self.groups)
end

function Conv2d:_refresh_aliases()
  self.weight = self._parameters.weight
  self.bias = self._parameters.bias
end

function Conv2d:format()
  return string.format(
    "Conv2d(%d -> %d, kernel=%dx%d, stride=%dx%d, padding=%dx%d, groups=%d)",
    self.in_channels, self.out_channels,
    self.kernel_size[1], self.kernel_size[2],
    self.stride[1], self.stride[2],
    self.padding[1], self.padding[2], self.groups)
end

local ConvTranspose2d = Module.class("ConvTranspose2d")

function ConvTranspose2d.new(in_channels, out_channels, kernel_size,
    stride, padding, output_padding, groups, bias, dilation)
  if type(stride) == "table" and
    (stride.stride or stride.padding or stride.output_padding or
      stride.dilation or stride.groups or stride.bias ~= nil) then
    local options = stride
    stride = options.stride
    padding = options.padding
    output_padding = options.output_padding
    dilation = options.dilation
    groups = options.groups
    if options.bias ~= nil then bias = options.bias end
  end
  if bias == nil then bias = true end
  if groups == nil then groups = 1 end
  local kernel_h, kernel_w = pair(kernel_size, nil, "kernel_size")
  local stride_h, stride_w = pair(stride, 1, "stride")
  local pad_h, pad_w = pair(padding, 0, "padding")
  local output_pad_h, output_pad_w = pair(output_padding, 0, "output_padding")
  local dilation_h, dilation_w = pair(dilation, 1, "dilation")
  if in_channels % groups ~= 0 or out_channels % groups ~= 0 then
    error("ConvTranspose2d: channels must be divisible by groups", 3)
  end
  if output_pad_h >= stride_h or output_pad_w >= stride_w then
    error("ConvTranspose2d: output_padding must be smaller than stride", 3)
  end
  local self = setmetatable(Module.new(), ConvTranspose2d)
  self.in_channels = in_channels
  self.out_channels = out_channels
  self.kernel_size = { kernel_h, kernel_w }
  self.stride = { stride_h, stride_w }
  self.padding = { pad_h, pad_w }
  self.output_padding = { output_pad_h, output_pad_w }
  self.dilation = { dilation_h, dilation_w }
  self.groups = groups
  local fan_in = (in_channels / groups) * kernel_h * kernel_w
  local bound = 1 / math.sqrt(fan_in)
  local weight = torch.rand(in_channels, out_channels / groups,
    kernel_h, kernel_w)
  weight:mul_(2 * bound):sub_(bound)
  self:register_parameter("weight", weight)
  self.weight = self._parameters.weight
  if bias then
    local bias_tensor = torch.rand(out_channels):mul_(2 * bound):sub_(bound)
    self:register_parameter("bias", bias_tensor)
    self.bias = self._parameters.bias
  else
    self.bias = nil
  end
  return self
end

function ConvTranspose2d:forward(input)
  return torch.conv_transpose2d(
    input, self.weight, self.bias, self.stride, self.padding,
    self.output_padding, self.groups, self.dilation)
end

function ConvTranspose2d:_refresh_aliases()
  self.weight = self._parameters.weight
  self.bias = self._parameters.bias
end

function ConvTranspose2d:format()
  return string.format(
    "ConvTranspose2d(%d -> %d, kernel=%dx%d, stride=%dx%d, " ..
    "padding=%dx%d, output_padding=%dx%d, groups=%d)",
    self.in_channels, self.out_channels, self.kernel_size[1],
    self.kernel_size[2], self.stride[1], self.stride[2], self.padding[1],
    self.padding[2], self.output_padding[1], self.output_padding[2],
    self.groups)
end

return {
  Conv2d = Conv2d,
  ConvTranspose2d = ConvTranspose2d,
}
