--
-- torch.optim
--
--   local optim = require("torch.optim")
--   local optimizer = optim.Adam(model:parameters(), {lr = 1e-3})
--   optimizer:zero_grad()
--   loss:backward()
--   optimizer:step()
--
local Optimizer = require("torch.optim.optimizer")

local optim = {
  Optimizer = Optimizer,
  SGD = require("torch.optim.sgd"),
  Adam = require("torch.optim.adam"),
  RMSprop = require("torch.optim.rmsprop"),
  Adagrad = require("torch.optim.adagrad"),
}

return optim
