# luatorch 0.17.0

- `torch.einsum` with validated equations, scalar operands, ellipsis batching,
  and autograd;
- direct `float16` construction from Lua tables and scalars, plus printing
  parity for values and `nan`/`inf`;
- `torch.isnan` / `torch.isinf` and `Tensor:isnan` / `Tensor:isinf` bool masks;
- 359 tests.
