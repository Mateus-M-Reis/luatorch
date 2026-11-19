--
-- Recurrent layers built from ordinary tensor operations.  Single- and
-- multi-layer, optional bidirectional traversal, and inter-layer dropout.
-- Gradients come from the regular autograd graph; parameter names follow
-- PyTorch's `_lN` / `_lN_reverse` convention.
--
local torch = require("torch")
local Module = require("torch.nn.module")
local F = require("torch.nn.functional")

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
  local layers = options.num_layers or 1
  if type(layers) ~= "number" or layers % 1 ~= 0 or layers < 1 then
    error("recurrent num_layers must be a positive integer", 3)
  end
  local dropout = options.dropout or 0
  if type(dropout) ~= "number" or dropout < 0 or dropout > 1 then
    error("recurrent dropout must be in [0, 1]", 3)
  end
  return options, layers, dropout
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

local function finish_output(outputs, batch_first, unbatched)
  local output = torch.stack(outputs, 1)
  if batch_first and not unbatched then output = output:transpose(1, 2) end
  if unbatched then output = output:select(2, 1) end
  return output
end

local function initial_state(value, layers, directions, batch, hidden, dtype, name)
  local states = layers * directions
  if value == nil then
    return torch.zeros(states, batch, hidden, { dtype = dtype }), states
  end
  if value:dim() == 2 and states == 1 then
    value = value:reshape(1, batch, hidden)
  end
  if value:dim() ~= 3 or value:size(1) ~= states or value:size(2) ~= batch or
      value:size(3) ~= hidden then
    error(string.format("%s must have shape [%d, %d, %d]", name, states, batch,
      hidden), 3)
  end
  return value, states
end

local function suffix_for(layer, reverse)
  return "l" .. (layer - 1) .. (reverse and "_reverse" or "")
end

local function register_cell(module, layer, reverse, input_size, hidden_size,
    gates, has_bias)
  local suffix = suffix_for(layer, reverse)
  module:register_parameter("weight_ih_" .. suffix,
    uniform_parameter(gates * hidden_size, input_size))
  module:register_parameter("weight_hh_" .. suffix,
    uniform_parameter(gates * hidden_size, hidden_size))
  if has_bias then
    module:register_parameter("bias_ih_" .. suffix,
      uniform_parameter(1, gates * hidden_size):reshape(gates * hidden_size))
    module:register_parameter("bias_hh_" .. suffix,
      uniform_parameter(1, gates * hidden_size):reshape(gates * hidden_size))
  end
end

local function cell_forward(module, layer, reverse, sequence, initial, cell,
    hidden_size, gates, nonlinearity)
  local suffix = suffix_for(layer, reverse)
  local weight_ih = module._parameters["weight_ih_" .. suffix]
  local weight_hh = module._parameters["weight_hh_" .. suffix]
  local bias_ih = module._parameters["bias_ih_" .. suffix]
  local bias_hh = module._parameters["bias_hh_" .. suffix]
  local steps, batch = sequence:size(1), sequence:size(2)
  local outputs = {}
  local h = initial
  local c = cell
  local first, last, step = 1, steps, 1
  if reverse then first, last, step = steps, 1, -1 end
  local t = first
  while (step > 0 and t <= last) or (step < 0 and t >= last) do
    local x = sequence:select(1, t)
    local combined = x:matmul(weight_ih:t()):add(h:matmul(weight_hh:t()))
    if bias_ih ~= nil then
      combined = combined:add(bias_ih):add(bias_hh)
    end
    if gates == 1 then
      h = nonlinearity == "relu" and combined:relu() or combined:tanh()
    else
      local input_gate = combined:narrow(2, 1, hidden_size):sigmoid()
      local forget_gate = combined:narrow(2, hidden_size + 1, hidden_size):sigmoid()
      local cell_gate = combined:narrow(2, 2 * hidden_size + 1, hidden_size):tanh()
      local output_gate = combined:narrow(2, 3 * hidden_size + 1, hidden_size):sigmoid()
      c = forget_gate:mul(c):add(input_gate:mul(cell_gate))
      h = output_gate:mul(c:tanh())
    end
    outputs[t] = h
    t = t + step
  end
  return outputs, h, c
end

local function run_layers(module, input, hidden, cell)
  local sequence, unbatched = prepare_input(input, module.batch_first)
  local steps, batch = sequence:size(1), sequence:size(2)
  local hidden_size = module.hidden_size
  local directions = module.bidirectional and 2 or 1
  local gates = module.gates
  local h0, states = initial_state(hidden, module.num_layers, directions, batch,
    hidden_size, input:dtype(), "hidden state")
  local c0
  if gates > 1 then
    c0 = initial_state(cell, module.num_layers, directions, batch, hidden_size,
      input:dtype(), "cell state")
  end
  local layer_input = sequence
  local h_states, c_states = {}, {}
  for layer = 1, module.num_layers do
    local direction_outputs = {}
    for direction = 1, directions do
      local reverse = direction == 2
      local index = (layer - 1) * directions + direction
      local h = h0:select(1, index)
      local c = c0 and c0:select(1, index) or nil
      local outputs, final_h, final_c = cell_forward(module, layer, reverse,
        layer_input, h, c, hidden_size, gates, module.nonlinearity)
      direction_outputs[direction] = torch.stack(outputs, 1)
      h_states[#h_states + 1] = final_h
      if final_c then c_states[#c_states + 1] = final_c end
    end
    local layer_output = directions == 2 and torch.cat(direction_outputs, 3) or
      direction_outputs[1]
    if layer < module.num_layers and module.dropout > 0 and module.training then
      layer_output = F.dropout(layer_output, module.dropout, true)
    end
    layer_input = layer_output
  end
  local output = layer_input
  if module.batch_first and not unbatched then
    output = output:transpose(1, 2)
  end
  if unbatched then output = output:select(2, 1) end
  return output, unbatched, h_states, c_states
end

-- RNN ------------------------------------------------------------------------

local RNN = class("RNN")

function RNN.new(input_size, hidden_size, options)
  options, layers, dropout = validate_common(input_size, hidden_size, options)
  local nonlinearity = options.nonlinearity or "tanh"
  if nonlinearity ~= "tanh" and nonlinearity ~= "relu" then
    error("RNN nonlinearity must be 'tanh' or 'relu'", 3)
  end
  local self = setmetatable(Module.new(), RNN)
  self.input_size = input_size
  self.hidden_size = hidden_size
  self.num_layers = layers
  self.gates = 1
  self.batch_first = options.batch_first == true
  self.bidirectional = options.bidirectional == true
  self.dropout = dropout
  self.nonlinearity = nonlinearity
  self.has_bias = options.bias ~= false
  local directions = self.bidirectional and 2 or 1
  local layer_input = input_size
  for layer = 1, layers do
    for direction = 1, directions do
      register_cell(self, layer, direction == 2, layer_input, hidden_size,
        1, self.has_bias)
    end
    layer_input = hidden_size * directions
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
  local full, unbatched, h_states = run_layers(self, input, hidden, nil)
  local hidden_out = torch.stack(h_states, 1)
  if unbatched then
    hidden_out = hidden_out:select(2, 1)
  end
  return full, hidden_out
end

function RNN:format()
  return string.format("RNN(%d -> %d, layers=%d, bidirectional=%s, %s)",
    self.input_size, self.hidden_size, self.num_layers,
    tostring(self.bidirectional), self.nonlinearity)
end

-- LSTM -----------------------------------------------------------------------

local LSTM = class("LSTM")

function LSTM.new(input_size, hidden_size, options)
  options, layers, dropout = validate_common(input_size, hidden_size, options)
  local self = setmetatable(Module.new(), LSTM)
  self.input_size = input_size
  self.hidden_size = hidden_size
  self.num_layers = layers
  self.gates = 4
  self.batch_first = options.batch_first == true
  self.bidirectional = options.bidirectional == true
  self.dropout = dropout
  self.nonlinearity = "tanh"
  self.has_bias = options.bias ~= false
  local directions = self.bidirectional and 2 or 1
  local layer_input = input_size
  for layer = 1, layers do
    for direction = 1, directions do
      register_cell(self, layer, direction == 2, layer_input, hidden_size,
        4, self.has_bias)
    end
    layer_input = hidden_size * directions
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
  local full, unbatched, h_states, c_states = run_layers(self, input, hidden,
    cell)
  local final_h = torch.stack(h_states, 1)
  local final_c = torch.stack(c_states, 1)
  if unbatched then
    final_h = final_h:select(2, 1)
    final_c = final_c:select(2, 1)
  end
  return full, final_h, final_c
end

function LSTM:format()
  return string.format("LSTM(%d -> %d, layers=%d, bidirectional=%s)",
    self.input_size, self.hidden_size, self.num_layers,
    tostring(self.bidirectional))
end

return {
  RNN = RNN,
  LSTM = LSTM,
}
