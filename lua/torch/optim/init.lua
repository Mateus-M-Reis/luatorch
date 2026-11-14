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
local clip = require("torch.optim.clip")

local optim = {
  Optimizer = Optimizer,
  SGD = require("torch.optim.sgd"),
  Adam = require("torch.optim.adam"),
  AdamW = require("torch.optim.adamw"),
  RMSprop = require("torch.optim.rmsprop"),
  Adagrad = require("torch.optim.adagrad"),
  lr_scheduler = require("torch.optim.lr_scheduler"),
  clip_grad_norm_ = clip.clip_grad_norm_,
  clip_grad_value_ = clip.clip_grad_value_,
}

return optim
