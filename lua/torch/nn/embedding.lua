--
-- torch.nn.embedding
--
local torch = require("torch")
local Module = require("torch.nn.module")

local Embedding = Module.class("Embedding")

function Embedding.new(num_embeddings, embedding_dim, options)
  options = options or {}
  if options.padding_idx ~= nil then
    error("Embedding: padding_idx is not supported yet", 3)
  end
  if type(num_embeddings) ~= "number" or type(embedding_dim) ~= "number" then
    error("Embedding(num_embeddings, embedding_dim)", 3)
  end

  local self = setmetatable(Module.new(), Embedding)
  self.num_embeddings = num_embeddings
  self.embedding_dim = embedding_dim

  -- PyTorch initialises embeddings from N(0, 1).
  self:register_parameter("weight", torch.randn(num_embeddings, embedding_dim))
  self.weight = self._parameters.weight
  return self
end

function Embedding:forward(indices)
  return torch.embedding(self.weight, indices)
end

function Embedding:_refresh_aliases()
  self.weight = self._parameters.weight
end

function Embedding:format()
  return string.format("Embedding(%d -> %d)", self.num_embeddings, self.embedding_dim)
end

return {
  Embedding = Embedding,
}
