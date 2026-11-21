# luatorch 0.18.0

- `torch.isfinite` / `Tensor:isfinite` bool masks, alongside `isnan`/`isinf`;
- `torch.from_blob` accepts `float16` buffers with raw binary16 elements,
  zero-copy, including strings, typed pointers, and shared mutation;
- einsum hardening: clear mixed-dtype and bool-operand errors, plus coverage
  for ellipsis, repeated labels, empty/scalar operands, and rank error paths;
- documentation, rockspec, and version consistency guarded by a new test;
- 382 tests.
