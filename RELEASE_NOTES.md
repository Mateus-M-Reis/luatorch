# luatorch 0.20.0

- `torch.linalg.lstsq` / `Tensor:lstsq(b, opts?)` returns `solution`,
  `residuals`, `rank`, and `singular_values` from `at::linalg_lstsq`;
- supports the `gels`, `gelsy`, `gelsd`, and `gelss` drivers, `rcond` rank
  truncation, batched and multiple right-hand sides, and autograd through the
  solution;
- six linalg tests cover the normal equations, driver outputs, batched and
  underdetermined systems, `rcond` truncation, gradients, and error paths;
- docs describe option, shape, residual, dtype, and driver behavior, with a
  README example and an API alias guard;
- benchmark entries for `lstsq` forward and forward+backward (no timing gate);
- 407 tests.

## luatorch 0.19.0

- `torch.linalg.slogdet` / `Tensor:slogdet()` returns `sign` and `logabsdet`,
  supports batched square matrices, and differentiates through `logabsdet`;
- autograd coverage for einsum traces, batched ellipses, implicit outputs,
  scalar operands, and explicit gradients;
- float16 autograd coverage for elementwise graphs, casts, an `nn.Linear`
  layer, and manual parameter updates;
- predicate masks (`isnan`/`isinf`/`isfinite`) are documented as detached and
  tested with `where` and boolean indexing;
- documentation corrections and checks for linalg aliases and integer/bool
  predicate semantics;
- benchmark entries for einsum, float16 matmul/add, `isfinite`, and `slogdet`;
- 401 tests.
