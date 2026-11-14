--
-- torch.nn.pooling
--
local torch = require("torch")
local Module = require("torch.nn.module")

local MaxPool2d = Module.class("MaxPool2d")

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

function MaxPool2d.new(kernel_size, stride, padding, dilation, ceil_mode)
  if type(stride) == "table" and
    (stride.stride or stride.padding or stride.dilation or stride.ceil_mode ~= nil) then
    local options = stride
    stride = options.stride
    padding = options.padding
    dilation = options.dilation
    ceil_mode = options.ceil_mode
  end

  local kernel_h, kernel_w = pair(kernel_size, nil, "kernel_size")
  local stride_h, stride_w = pair(stride, nil, "stride")
  if stride_h == nil then
    stride_h, stride_w = kernel_h, kernel_w
  end
  local pad_h, pad_w = pair(padding, 0, "padding")
  local dilation_h, dilation_w = pair(dilation, 1, "dilation")

  local self = setmetatable(Module.new(), MaxPool2d)
  self.kernel_size = { kernel_h, kernel_w }
  self.stride = { stride_h, stride_w }
  self.padding = { pad_h, pad_w }
  self.dilation = { dilation_h, dilation_w }
  self.ceil_mode = ceil_mode and true or false
  return self
end

function MaxPool2d:forward(input)
  return torch.max_pool2d(
    input, self.kernel_size, self.stride,
    self.padding, self.dilation, self.ceil_mode)
end

function MaxPool2d:format()
  return string.format("MaxPool2d(kernel=%dx%d, stride=%dx%d)",
    self.kernel_size[1], self.kernel_size[2],
    self.stride[1], self.stride[2])
end

-- Global average pooling over the spatial dimensions.
local AdaptiveAvgPool2d = Module.class("AdaptiveAvgPool2d")

function AdaptiveAvgPool2d.new(output_size)
  local self = setmetatable(Module.new(), AdaptiveAvgPool2d)
  self.output_size = output_size
  return self
end

function AdaptiveAvgPool2d:forward(input)
  return input:mean(3, true):mean(4, true)
end

return {
  MaxPool2d = MaxPool2d,
  AdaptiveAvgPool2d = AdaptiveAvgPool2d,
}
