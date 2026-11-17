--
-- Recurrent layers built from ordinary tensor operations.  Single layer,
-- unidirectional, batch-first or sequence-first; gradients come from the
-- regular autograd graph.
--
local torch = require("torch")
local Module = require("torch.nn.module")

local function class(name)
  return Module.class(name)
end

local function uniform_parameter(rows, columns)
  local bound = 1 / math.sqrt(columns)
  return torch.rand(rows, columns):mul(2 * bound):sub(bound)
end

local function validate_common(input_size, hidden_size, options)
  if type(input_size) ~= "number" or type(hidden_size) ~= "number" or
      input_size < 1 or hidden_size < 1 then
    error("recurrent layer requires positive input_size and hidden_size", 3)
  end
  options = options or {}
  if options.num_layers ~= nil and options.num_layers ~= 1 then
    error("recurrent layers currently support exactly one layer", 3)
  end
  if options.bidirectional then
    error("bidirectional recurrent layers are not implemented", 3)
  end
  return options
end

local function prepare_input(input, batch_first)
  local unbatched = input:dim() == 2
  if unbatched then input = input:reshape(input:size(1), 1, input:size(2)) end
  if input:dim() ~= 3 then
    error("recurrent layer expects [seq, batch, input] or [batch, seq, input]", 3)
  end
  if batch_first then input = input:transpose(1, 2) end
  return input, unbatched
end

local function prepare_hidden(hidden, batch, size, dtype, name)
  if hidden == nil then
    return torch.zeros(1, batch, size, { dtype = dtype })
  end
  if hidden:dim() == 2 then hidden = hidden:reshape(1, batch, size) end
  if hidden:dim() ~= 3 or hidden:size(1) ~= 1 or hidden:size(2) ~= batch or
      hidden:size(3) ~= size then
    error(name .. " must have shape [1, batch, hidden]", 3)
  end
  return hidden
end

local function finish_output(outputs, batch_first, unbatched)
  local output = torch.stack(outputs, 1)
  if batch_first and not unbatched then output = output:transpose(1, 2) end
  if unbatched then output = output:select(2, 1) end
  return output
end

-- RNN ------------------------------------------------------------------------

local RNN = class("RNN")

function RNN.new(input_size, hidden_size, options)
  options = validate_common(input_size, hidden_size, options)
  local nonlinearity = options.nonlinearity or "tanh"
  if nonlinearity ~= "tanh" and nonlinearity ~= "relu" then
    error("RNN nonlinearity must be 'tanh' or 'relu'", 3)
  end
  local self = setmetatable(Module.new(), RNN)
  self.input_size = input_size
  self.hidden_size = hidden_size
  self.batch_first = options.batch_first == true
  self.nonlinearity = nonlinearity
  self.has_bias = options.bias ~= false
  self:register_parameter("weight_ih_l0",
    uniform_parameter(hidden_size, input_size))
  self:register_parameter("weight_hh_l0",
    uniform_parameter(hidden_size, hidden_size))
  if self.has_bias then
    self:register_parameter("bias_ih_l0",
      uniform_parameter(1, hidden_size):reshape(hidden_size))
    self:register_parameter("bias_hh_l0",
      uniform_parameter(1, hidden_size):reshape(hidden_size))
  end
  self:_refresh_aliases()
  return self
end

function RNN:_refresh_aliases()
  self.weight_ih_l0 = self._parameters.weight_ih_l0
  self.weight_hh_l0 = self._parameters.weight_hh_l0
  self.bias_ih_l0 = self._parameters.bias_ih_l0
  self.bias_hh_l0 = self._parameters.bias_hh_l0
end

function RNN:forward(input, hidden)
  local sequence, unbatched = prepare_input(input, self.batch_first)
  local steps, batch = sequence:size(1), sequence:size(2)
  local h = prepare_hidden(hidden, batch, self.hidden_size,
    input:dtype(), "RNN hidden state"):select(1, 1)
  local outputs = {}
  for step = 1, steps do
    local x = sequence:select(1, step)
    local next_h = x:matmul(self.weight_ih_l0:t()):add(
      h:matmul(self.weight_hh_l0:t()))
    if self.has_bias then
      next_h = next_h:add(self.bias_ih_l0):add(self.bias_hh_l0)
    end
    h = self.nonlinearity == "relu" and next_h:relu() or next_h:tanh()
    outputs[#outputs + 1] = h
  end
  local output = finish_output(outputs, self.batch_first, unbatched)
  local final = h:reshape(1, batch, self.hidden_size)
  if unbatched then final = final:reshape(1, self.hidden_size) end
  return output, final
end

function RNN:format()
  return string.format("RNN(%d -> %d, %s, batch_first=%s)",
    self.input_size, self.hidden_size, self.nonlinearity,
    tostring(self.batch_first))
end

-- LSTM -----------------------------------------------------------------------

local LSTM = class("LSTM")

function LSTM.new(input_size, hidden_size, options)
  options = validate_common(input_size, hidden_size, options)
  local self = setmetatable(Module.new(), LSTM)
  self.input_size = input_size
  self.hidden_size = hidden_size
  self.batch_first = options.batch_first == true
  self.has_bias = options.bias ~= false
  self:register_parameter("weight_ih_l0",
    uniform_parameter(4 * hidden_size, input_size))
  self:register_parameter("weight_hh_l0",
    uniform_parameter(4 * hidden_size, hidden_size))
  if self.has_bias then
    self:register_parameter("bias_ih_l0",
      uniform_parameter(1, 4 * hidden_size):reshape(4 * hidden_size))
    self:register_parameter("bias_hh_l0",
      uniform_parameter(1, 4 * hidden_size):reshape(4 * hidden_size))
  end
  self:_refresh_aliases()
  return self
end

function LSTM:_refresh_aliases()
  self.weight_ih_l0 = self._parameters.weight_ih_l0
  self.weight_hh_l0 = self._parameters.weight_hh_l0
  self.bias_ih_l0 = self._parameters.bias_ih_l0
  self.bias_hh_l0 = self._parameters.bias_hh_l0
end

function LSTM:forward(input, hidden, cell)
  local sequence, unbatched = prepare_input(input, self.batch_first)
  local steps, batch = sequence:size(1), sequence:size(2)
  local h = prepare_hidden(hidden, batch, self.hidden_size,
    input:dtype(), "LSTM hidden state"):select(1, 1)
  local c = prepare_hidden(cell, batch, self.hidden_size,
    input:dtype(), "LSTM cell state"):select(1, 1)
  local outputs = {}
  local hidden_size = self.hidden_size
  for step = 1, steps do
    local x = sequence:select(1, step)
    local gates = x:matmul(self.weight_ih_l0:t()):add(
      h:matmul(self.weight_hh_l0:t()))
    if self.has_bias then
      gates = gates:add(self.bias_ih_l0):add(self.bias_hh_l0)
    end
    local input_gate = gates:narrow(2, 1, hidden_size):sigmoid()
    local forget_gate = gates:narrow(2, hidden_size + 1, hidden_size):sigmoid()
    local cell_gate = gates:narrow(2, 2 * hidden_size + 1, hidden_size):tanh()
    local output_gate = gates:narrow(2, 3 * hidden_size + 1, hidden_size):sigmoid()
    c = forget_gate:mul(c):add(input_gate:mul(cell_gate))
    h = output_gate:mul(c:tanh())
    outputs[#outputs + 1] = h
  end
  local output = finish_output(outputs, self.batch_first, unbatched)
  local final_h = h:reshape(1, batch, hidden_size)
  local final_c = c:reshape(1, batch, hidden_size)
  if unbatched then
    final_h = final_h:reshape(1, hidden_size)
    final_c = final_c:reshape(1, hidden_size)
  end
  return output, final_h, final_c
end

function LSTM:format()
  return string.format("LSTM(%d -> %d, batch_first=%s)",
    self.input_size, self.hidden_size, tostring(self.batch_first))
end

return {
  RNN = RNN,
  LSTM = LSTM,
}
