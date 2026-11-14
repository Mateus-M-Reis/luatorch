/*
 * luatorch C shim implementation.
 *
 * Everything in this file is plain C-compatible at the ABI level; the
 * at::Tensor handles are heap-allocated and owned by the caller (Lua's
 * GC, via ffi.gc).  Each entry point catches C++ exceptions and turns
 * them into messages retrievable with lt_last_error().
 */
#include "luatorch.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/Logging.h>
#include <torch/serialize.h>
#include <torch/version.h>

#include <cmath>
#include <cstring>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct lt_tensor {
  at::Tensor t;
};

namespace {

thread_local std::string g_last_error;

// c10::Error normally unwinds the C++ stack to build an error backtrace.  When
// the throw happens under LuaJIT's JIT-compiled frames, _Unwind_Backtrace can
// abort because those frames are not unwindable.  We do not expose the C++
// backtrace to users anyway (the message and source location are kept), so
// install an empty stack-trace fetcher before any ATen call can throw.
struct StackTraceSilencer {
  StackTraceSilencer() {
    // There are two fetcher overloads; disable both.  The lazy Backtrace
    // overload is the one used while constructing c10::Error.
    c10::SetStackTraceFetcher([]() -> c10::Backtrace { return nullptr; });
    c10::SetStackTraceFetcher([]() { return std::string(); });
  }
};

const StackTraceSilencer g_stacktrace_silencer;

void clear_error() {
  g_last_error.clear();
}

void set_error(const std::string& message) {
  g_last_error = message;
}

void set_error(const std::exception& error) {
  g_last_error = error.what();
}

at::ScalarType to_scalar_type(int dtype) {
  switch (dtype) {
    case LT_DTYPE_F32:  return at::kFloat;
    case LT_DTYPE_F64:  return at::kDouble;
    case LT_DTYPE_I64:  return at::kLong;
    case LT_DTYPE_I32:  return at::kInt;
    case LT_DTYPE_I16:  return at::kShort;
    case LT_DTYPE_I8:   return at::kChar;
    case LT_DTYPE_U8:   return at::kByte;
    case LT_DTYPE_BOOL: return at::kBool;
    case LT_DTYPE_F16:  return at::kHalf;
    default:
      throw std::runtime_error("unknown dtype id " + std::to_string(dtype));
  }
}

int from_scalar_type(at::ScalarType type) {
  switch (type) {
    case at::kFloat:  return LT_DTYPE_F32;
    case at::kDouble: return LT_DTYPE_F64;
    case at::kLong:   return LT_DTYPE_I64;
    case at::kInt:    return LT_DTYPE_I32;
    case at::kShort:  return LT_DTYPE_I16;
    case at::kChar:   return LT_DTYPE_I8;
    case at::kByte:   return LT_DTYPE_U8;
    case at::kBool:   return LT_DTYPE_BOOL;
    case at::kHalf:   return LT_DTYPE_F16;
    default:
      throw std::runtime_error(
          std::string("unsupported scalar type: ") +
          c10::toString(type));
  }
}

at::TensorOptions make_options(int dtype) {
  return at::TensorOptions().dtype(to_scalar_type(dtype));
}

std::vector<int64_t> make_shape(int64_t ndim, const int64_t* sizes) {
  if (ndim < 0) {
    throw std::runtime_error("rank must be non-negative");
  }
  if (ndim > 0 && sizes == nullptr) {
    throw std::runtime_error("null shape pointer");
  }
  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(ndim));
  for (int64_t i = 0; i < ndim; ++i) {
    shape.push_back(sizes[i]);
  }
  return shape;
}

int64_t resolve_dim(int64_t dim, int64_t ndim, const char* what) {
  int64_t d = dim;
  if (d < 0) {
    d += ndim;
  }
  if (d < 0 || d >= ndim) {
    throw std::runtime_error(
        std::string(what) + ": dimension " + std::to_string(dim) +
        " out of range for tensor with " + std::to_string(ndim) + " dims");
  }
  return d;
}

int64_t resolve_index(int64_t index, int64_t size, const char* what) {
  int64_t i = index;
  if (i < 0) {
    i += size;
  }
  if (i < 0 || i >= size) {
    throw std::runtime_error(
        std::string(what) + ": index " + std::to_string(index) +
        " out of range for size " + std::to_string(size));
  }
  return i;
}

std::vector<int64_t> all_dims(int64_t ndim) {
  std::vector<int64_t> dims(static_cast<size_t>(ndim));
  for (int64_t i = 0; i < ndim; ++i) {
    dims[static_cast<size_t>(i)] = i;
  }
  return dims;
}

lt_tensor* wrap(at::Tensor tensor) {
  auto* handle = new lt_tensor();
  handle->t = std::move(tensor);
  return handle;
}

// Lua only has doubles, but users expect `long_tensor + 1` to stay integral.
// When the tensor is integral and the value is a whole number, build an
// integer Scalar so ATen does not promote the result to float.
at::Scalar scalar_for(const at::Tensor& tensor, double value) {
  switch (tensor.scalar_type()) {
    case at::kLong:
    case at::kInt:
    case at::kShort:
    case at::kChar:
    case at::kByte:
    case at::kBool:
      if (std::isfinite(value) && value == std::floor(value)) {
        return at::Scalar(static_cast<int64_t>(value));
      }
      break;
    default:
      break;
  }
  return at::Scalar(value);
}

at::Tensor scalar_like(const at::Tensor& like, double value) {
  switch (like.scalar_type()) {
    case at::kBool:
      return at::scalar_tensor(at::Scalar(value != 0.0), like.options());
    case at::kLong:
      return at::scalar_tensor(at::Scalar(static_cast<int64_t>(value)), like.options());
    case at::kInt:
      return at::scalar_tensor(at::Scalar(static_cast<int32_t>(value)), like.options());
    case at::kShort:
      return at::scalar_tensor(at::Scalar(static_cast<int16_t>(value)), like.options());
    case at::kChar:
      return at::scalar_tensor(at::Scalar(static_cast<int8_t>(value)), like.options());
    case at::kByte:
      return at::scalar_tensor(at::Scalar(static_cast<uint8_t>(value)), like.options());
    default:
      return at::scalar_tensor(at::Scalar(value), like.options());
  }
}

at::Tensor item_prefix(const at::Tensor& tensor, int64_t ndim, const int64_t* index) {
  if (ndim != tensor.dim()) {
    throw std::runtime_error(
        "expected " + std::to_string(tensor.dim()) + " indices, got " +
        std::to_string(ndim));
  }
  at::Tensor view = tensor;
  for (int64_t i = 0; i < ndim; ++i) {
    int64_t idx = resolve_index(index[i], view.size(0), "index");
    view = view.select(0, idx);
  }
  return view;
}

at::Tensor variance_impl(
    const at::Tensor& tensor,
    int64_t dim,
    bool keepdim,
    bool unbiased) {
  if (dim == LT_ALL_DIMS) {
    at::Tensor flat = tensor.flatten();
    at::Tensor mean = flat.mean();
    at::Tensor diff = flat - mean;
    at::Tensor value = (diff * diff).sum();
    int64_t n = flat.numel();
    value = unbiased ? value / (n - 1) : value / n;
    if (keepdim) {
      value = value.reshape(std::vector<int64_t>(static_cast<size_t>(tensor.dim()), 1));
    }
    return value;
  }
  int64_t d = resolve_dim(dim, tensor.dim(), "var");
  at::Tensor mean = tensor.mean(at::IntArrayRef(&d, 1), true);
  at::Tensor diff = tensor - mean;
  at::Tensor value = (diff * diff).sum(at::IntArrayRef(&d, 1), keepdim);
  int64_t n = tensor.size(d);
  value = unbiased ? value / (n - 1) : value / n;
  return value;
}

at::Tensor norm_impl(
    const at::Tensor& tensor,
    double p,
    int64_t dim,
    bool keepdim) {
  at::Tensor values = tensor.abs();
  if (std::isinf(p)) {
    if (p < 0) {
      throw std::runtime_error("norm: infinite p with negative sign is not supported");
    }
    if (dim == LT_ALL_DIMS) {
      return values.max();
    }
    int64_t d = resolve_dim(dim, tensor.dim(), "norm");
    return std::get<0>(values.max(d, keepdim));
  }
  if (p == 0.0) {
    at::Tensor nz = values.ne(0).to(tensor.scalar_type());
    if (dim == LT_ALL_DIMS) {
      return nz.sum();
    }
    int64_t d = resolve_dim(dim, tensor.dim(), "norm");
    return nz.sum(at::IntArrayRef(&d, 1), keepdim);
  }
  if (p < 0.0) {
    throw std::runtime_error("norm: negative p is not supported");
  }
  at::Tensor powered = values.pow(p);
  if (dim == LT_ALL_DIMS) {
    return powered.sum().pow(1.0 / p);
  }
  int64_t d = resolve_dim(dim, tensor.dim(), "norm");
  return powered.sum(at::IntArrayRef(&d, 1), keepdim).pow(1.0 / p);
}

#define LT_CATCH_NULL                                  \
  catch (const std::exception& error) {                \
    set_error(error);                                  \
    return nullptr;                                    \
  }                                                    \
  catch (...) {                                        \
    set_error("unknown native error");                 \
    return nullptr;                                    \
  }

#define LT_CATCH_INT                                   \
  catch (const std::exception& error) {                \
    set_error(error);                                  \
    return 1;                                          \
  }                                                    \
  catch (...) {                                        \
    set_error("unknown native error");                 \
    return 1;                                          \
  }

}  // namespace

extern "C" {

/* ------------------------------------------------------------------ */
/* lifecycle / diagnostics                                            */
/* ------------------------------------------------------------------ */

const char* lt_version(void) {
  return TORCH_VERSION;
}

const char* lt_last_error(void) {
  return g_last_error.c_str();
}

void lt_free(lt_tensor* t) {
  delete t;
}

int64_t lt_free_counted(lt_tensor* t) {
  if (t == nullptr) {
    return 0;
  }
  int64_t bytes = t->t.numel() * t->t.element_size();
  delete t;
  return bytes;
}

/* ------------------------------------------------------------------ */
/* creation                                                           */
/* ------------------------------------------------------------------ */

#define LT_CREATE(name, call)                                            \
  lt_tensor* lt_##name(int64_t ndim, const int64_t* sizes, int dtype) { \
    try {                                                                \
      clear_error();                                                     \
      return wrap(call);                                                 \
    }                                                                    \
    LT_CATCH_NULL                                                        \
  }

LT_CREATE(zeros, at::zeros(make_shape(ndim, sizes), make_options(dtype)))
LT_CREATE(ones, at::ones(make_shape(ndim, sizes), make_options(dtype)))
LT_CREATE(empty, at::empty(make_shape(ndim, sizes), make_options(dtype)))
LT_CREATE(rand, at::rand(make_shape(ndim, sizes), make_options(dtype)))
LT_CREATE(randn, at::randn(make_shape(ndim, sizes), make_options(dtype)))

lt_tensor* lt_full(int64_t ndim, const int64_t* sizes, double value, int dtype) {
  try {
    clear_error();
    return wrap(at::full(make_shape(ndim, sizes), value, make_options(dtype)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_eye(int64_t n, int dtype) {
  try {
    clear_error();
    if (n < 0) {
      throw std::runtime_error("eye: negative size");
    }
    return wrap(at::eye(n, make_options(dtype)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_arange(double start, double stop, double step, int dtype) {
  try {
    clear_error();
    if (step == 0.0) {
      throw std::runtime_error("arange: step must be non-zero");
    }
    return wrap(at::arange(start, stop, step, make_options(dtype)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_linspace(double start, double stop, int64_t steps, int dtype) {
  try {
    clear_error();
    if (steps < 0) {
      throw std::runtime_error("linspace: negative steps");
    }
    return wrap(at::linspace(start, stop, steps, make_options(dtype)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_randperm(int64_t n, int dtype) {
  try {
    clear_error();
    if (n < 0) {
      throw std::runtime_error("randperm: negative size");
    }
    return wrap(at::randperm(n, make_options(dtype)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_from_data(const void* data, int64_t ndim, const int64_t* sizes, int dtype) {
  try {
    clear_error();
    std::vector<int64_t> shape = make_shape(ndim, sizes);
    int64_t numel = 1;
    for (int64_t size : shape) {
      numel *= size;
    }
    if (numel == 0) {
      return wrap(at::empty(shape, make_options(dtype)));
    }
    if (data == nullptr) {
      throw std::runtime_error("from_data: null data pointer");
    }
    at::Tensor tensor = at::from_blob(
        const_cast<void*>(data), shape, make_options(dtype));
    return wrap(tensor.clone());
  }
  LT_CATCH_NULL
}

void lt_manual_seed(uint64_t seed) {
  at::manual_seed(seed);
}

/* ------------------------------------------------------------------ */
/* metadata                                                           */
/* ------------------------------------------------------------------ */

#define LT_META_INT64(name, expr)                    \
  int64_t lt_##name(const lt_tensor* t) {            \
    try {                                            \
      clear_error();                                 \
      return (expr);                                 \
    }                                                \
    catch (const std::exception& error) {            \
      set_error(error);                              \
      return -1;                                     \
    }                                                \
    catch (...) {                                    \
      set_error("unknown native error");             \
      return -1;                                     \
    }                                                \
  }

LT_META_INT64(dim, t->t.dim())
LT_META_INT64(numel, t->t.numel())
LT_META_INT64(nbytes, t->t.numel() * t->t.element_size())
LT_META_INT64(element_size, t->t.element_size())

int lt_dtype(const lt_tensor* t) {
  try {
    clear_error();
    return from_scalar_type(t->t.scalar_type());
  } catch (...) {
    set_error("cannot query dtype");
    return -1;
  }
}

int lt_device(const lt_tensor* t) {
  try {
    clear_error();
    return t->t.device().is_cuda() ? 1 : 0;
  } catch (...) {
    set_error("cannot query device");
    return -1;
  }
}

int64_t lt_size(const lt_tensor* t, int64_t dim) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, t->t.dim(), "size");
    return t->t.size(d);
  } catch (const std::exception& error) {
    set_error(error);
    return -1;
  }
}

int lt_sizes(const lt_tensor* t, int64_t* out) {
  try {
    clear_error();
    if (out == nullptr && t->t.dim() > 0) {
      throw std::runtime_error("sizes: null output pointer");
    }
    for (int64_t i = 0; i < t->t.dim(); ++i) {
      out[i] = t->t.size(i);
    }
    return 0;
  }
  LT_CATCH_INT
}

#define LT_META_BOOL(name, expr)                     \
  int lt_##name(const lt_tensor* t) {                \
    try {                                            \
      clear_error();                                 \
      return (expr) ? 1 : 0;                         \
    }                                                \
    catch (const std::exception& error) {            \
      set_error(error);                              \
      return -1;                                     \
    }                                                \
    catch (...) {                                    \
      set_error("unknown native error");             \
      return -1;                                     \
    }                                                \
  }

LT_META_BOOL(is_contiguous, t->t.is_contiguous())
LT_META_BOOL(is_floating_point, t->t.is_floating_point())
LT_META_BOOL(requires_grad, t->t.requires_grad())

const void* lt_data_ptr(const lt_tensor* t) {
  try {
    clear_error();
    if (!t->t.is_contiguous()) {
      throw std::runtime_error("data_ptr: tensor is not contiguous (call contiguous())");
    }
    if (!t->t.device().is_cpu()) {
      throw std::runtime_error("data_ptr: tensor is not on CPU");
    }
    return t->t.const_data_ptr();
  } catch (const std::exception& error) {
    set_error(error);
    return nullptr;
  }
}

/* ------------------------------------------------------------------ */
/* copies / dtypes / autograd                                         */
/* ------------------------------------------------------------------ */

lt_tensor* lt_contiguous(const lt_tensor* t) {
  try {
    clear_error();
    return wrap(t->t.contiguous());
  }
  LT_CATCH_NULL
}

lt_tensor* lt_clone(const lt_tensor* t) {
  try {
    clear_error();
    return wrap(t->t.clone());
  }
  LT_CATCH_NULL
}

lt_tensor* lt_to_dtype(const lt_tensor* t, int dtype) {
  try {
    clear_error();
    return wrap(t->t.to(to_scalar_type(dtype)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_detach(const lt_tensor* t) {
  try {
    clear_error();
    return wrap(t->t.detach());
  }
  LT_CATCH_NULL
}

int lt_copy_(lt_tensor* dst, const lt_tensor* src) {
  try {
    clear_error();
    dst->t.copy_(src->t);
    return 0;
  }
  LT_CATCH_INT
}

int lt_set_requires_grad(lt_tensor* t, int requires_grad) {
  try {
    clear_error();
    t->t.set_requires_grad(requires_grad != 0);
    return 0;
  }
  LT_CATCH_INT
}

int lt_has_grad(const lt_tensor* t) {
  try {
    clear_error();
    return t->t.grad().defined() ? 1 : 0;
  } catch (...) {
    set_error("has_grad failed");
    return -1;
  }
}

lt_tensor* lt_grad(const lt_tensor* t) {
  try {
    clear_error();
    at::Tensor grad = t->t.grad();
    if (!grad.defined()) {
      return nullptr;
    }
    return wrap(grad);
  }
  LT_CATCH_NULL
}

int lt_backward(lt_tensor* t, const lt_tensor* gradient) {
  try {
    clear_error();
    if (gradient != nullptr) {
      t->t.backward(gradient->t);
    } else {
      t->t.backward();
    }
    return 0;
  }
  LT_CATCH_INT
}

/* ------------------------------------------------------------------ */
/* scalar element access                                              */
/* ------------------------------------------------------------------ */

int lt_item_f64(const lt_tensor* t, int64_t ndim, const int64_t* index, double* out) {
  try {
    clear_error();
    if (out == nullptr) {
      throw std::runtime_error("item: null output pointer");
    }
    *out = item_prefix(t->t, ndim, index).item<double>();
    return 0;
  }
  LT_CATCH_INT
}

int lt_item_i64(const lt_tensor* t, int64_t ndim, const int64_t* index, int64_t* out) {
  try {
    clear_error();
    if (out == nullptr) {
      throw std::runtime_error("item: null output pointer");
    }
    *out = item_prefix(t->t, ndim, index).item<int64_t>();
    return 0;
  }
  LT_CATCH_INT
}

int lt_item_bool(const lt_tensor* t, int64_t ndim, const int64_t* index, int* out) {
  try {
    clear_error();
    if (out == nullptr) {
      throw std::runtime_error("item: null output pointer");
    }
    *out = item_prefix(t->t, ndim, index).item<bool>() ? 1 : 0;
    return 0;
  }
  LT_CATCH_INT
}

int lt_set_item_f64(lt_tensor* t, int64_t ndim, const int64_t* index, double value) {
  try {
    clear_error();
    if (ndim == 0) {
      t->t.fill_(scalar_like(t->t, value));
      return 0;
    }
    if (ndim != t->t.dim()) {
      throw std::runtime_error(
          "set: expected " + std::to_string(t->t.dim()) + " indices, got " +
          std::to_string(ndim));
    }
    at::Tensor view = t->t;
    for (int64_t i = 0; i < ndim; ++i) {
      int64_t idx = resolve_index(index[i], view.size(0), "set");
      view = view.select(0, idx);
    }
    view.fill_(scalar_like(view, value));
    return 0;
  }
  LT_CATCH_INT
}

int lt_set_item_i64(lt_tensor* t, int64_t ndim, const int64_t* index, int64_t value) {
  try {
    clear_error();
    if (ndim == 0) {
      t->t.fill_(scalar_like(t->t, static_cast<double>(value)));
      return 0;
    }
    if (ndim != t->t.dim()) {
      throw std::runtime_error(
          "set: expected " + std::to_string(t->t.dim()) + " indices, got " +
          std::to_string(ndim));
    }
    at::Tensor view = t->t;
    for (int64_t i = 0; i < ndim; ++i) {
      int64_t idx = resolve_index(index[i], view.size(0), "set");
      view = view.select(0, idx);
    }
    view.fill_(scalar_like(view, static_cast<double>(value)));
    return 0;
  }
  LT_CATCH_INT
}

/* ------------------------------------------------------------------ */
/* binary ops                                                         */
/* ------------------------------------------------------------------ */

#define LT_BINOP(name, method)                                  \
  lt_tensor* lt_##name(lt_tensor* a, lt_tensor* b, int inplace) { \
    try {                                                       \
      clear_error();                                            \
      if (inplace) {                                            \
        a->t.method##_(b->t);                                   \
        return a;                                               \
      }                                                         \
      return wrap(a->t.method(b->t));                           \
    }                                                           \
    LT_CATCH_NULL                                               \
  }

LT_BINOP(add, add)
LT_BINOP(sub, sub)
LT_BINOP(mul, mul)
LT_BINOP(div, div)
LT_BINOP(pow, pow)

#undef LT_BINOP

lt_tensor* lt_remainder(lt_tensor* a, lt_tensor* b, int inplace) {
  try {
    clear_error();
    if (inplace) {
      a->t.copy_(a->t.remainder(b->t));
      return a;
    }
    return wrap(a->t.remainder(b->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_maximum(lt_tensor* a, lt_tensor* b, int inplace) {
  try {
    clear_error();
    if (inplace) {
      a->t.copy_(at::maximum(a->t, b->t));
      return a;
    }
    return wrap(at::maximum(a->t, b->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_minimum(lt_tensor* a, lt_tensor* b, int inplace) {
  try {
    clear_error();
    if (inplace) {
      a->t.copy_(at::minimum(a->t, b->t));
      return a;
    }
    return wrap(at::minimum(a->t, b->t));
  }
  LT_CATCH_NULL
}

#define LT_BINOP_SCALAR(name, method)                              \
  lt_tensor* lt_##name##_scalar(lt_tensor* a, double v, int inplace) { \
    try {                                                          \
      clear_error();                                               \
      if (inplace) {                                               \
        a->t.method##_(scalar_for(a->t, v));                                         \
        return a;                                                  \
      }                                                            \
      return wrap(a->t.method(scalar_for(a->t, v)));                                 \
    }                                                              \
    LT_CATCH_NULL                                                  \
  }

LT_BINOP_SCALAR(add, add)
LT_BINOP_SCALAR(sub, sub)
LT_BINOP_SCALAR(mul, mul)
LT_BINOP_SCALAR(div, div)
LT_BINOP_SCALAR(pow, pow)

#undef LT_BINOP_SCALAR

lt_tensor* lt_remainder_scalar(lt_tensor* a, double v, int inplace) {
  try {
    clear_error();
    if (inplace) {
      a->t.copy_(a->t.remainder(scalar_for(a->t, v)));
      return a;
    }
    return wrap(a->t.remainder(scalar_for(a->t, v)));
  }
  LT_CATCH_NULL
}

#define LT_CMP(name, method)                       \
  lt_tensor* lt_##name(lt_tensor* a, lt_tensor* b) { \
    try {                                          \
      clear_error();                               \
      return wrap(a->t.method(b->t));              \
    }                                              \
    LT_CATCH_NULL                                  \
  }

LT_CMP(eq, eq)
LT_CMP(ne, ne)
LT_CMP(lt, lt)
LT_CMP(le, le)
LT_CMP(gt, gt)
LT_CMP(ge, ge)

#undef LT_CMP

#define LT_CMP_SCALAR(name, method)                   \
  lt_tensor* lt_##name##_scalar(lt_tensor* a, double v) { \
    try {                                             \
      clear_error();                                  \
      return wrap(a->t.method(scalar_for(a->t, v)));                    \
    }                                                 \
    LT_CATCH_NULL                                     \
  }

LT_CMP_SCALAR(eq, eq)
LT_CMP_SCALAR(ne, ne)
LT_CMP_SCALAR(lt, lt)
LT_CMP_SCALAR(le, le)
LT_CMP_SCALAR(gt, gt)
LT_CMP_SCALAR(ge, ge)

#undef LT_CMP_SCALAR

lt_tensor* lt_logical_and(lt_tensor* a, lt_tensor* b) {
  try {
    clear_error();
    return wrap(a->t.logical_and(b->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_logical_or(lt_tensor* a, lt_tensor* b) {
  try {
    clear_error();
    return wrap(a->t.logical_or(b->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_logical_not(lt_tensor* a) {
  try {
    clear_error();
    return wrap(a->t.logical_not());
  }
  LT_CATCH_NULL
}

/* ------------------------------------------------------------------ */
/* unary ops                                                          */
/* ------------------------------------------------------------------ */

#define LT_UNARY(name, method)                          \
  lt_tensor* lt_##name(lt_tensor* a, int inplace) {     \
    try {                                               \
      clear_error();                                    \
      if (inplace) {                                    \
        a->t.copy_(a->t.method());                      \
        return a;                                       \
      }                                                 \
      return wrap(a->t.method());                       \
    }                                                   \
    LT_CATCH_NULL                                       \
  }

LT_UNARY(neg, neg)
LT_UNARY(abs, abs)
LT_UNARY(exp, exp)
LT_UNARY(log, log)
LT_UNARY(log1p, log1p)
LT_UNARY(sqrt, sqrt)
LT_UNARY(rsqrt, rsqrt)
LT_UNARY(sin, sin)
LT_UNARY(cos, cos)
LT_UNARY(tan, tan)
LT_UNARY(asin, asin)
LT_UNARY(acos, acos)
LT_UNARY(atan, atan)
LT_UNARY(sinh, sinh)
LT_UNARY(cosh, cosh)
LT_UNARY(tanh, tanh)
LT_UNARY(sigmoid, sigmoid)
LT_UNARY(relu, relu)
LT_UNARY(floor, floor)
LT_UNARY(ceil, ceil)
LT_UNARY(round, round)
LT_UNARY(trunc, trunc)
LT_UNARY(sign, sign)
LT_UNARY(square, square)
LT_UNARY(reciprocal, reciprocal)

#undef LT_UNARY

lt_tensor* lt_clamp(lt_tensor* a, double lo, double hi, int inplace) {
  try {
    clear_error();
    if (inplace) {
      a->t.copy_(a->t.clamp(lo, hi));
      return a;
    }
    return wrap(a->t.clamp(lo, hi));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_fill(lt_tensor* a, double value, int inplace) {
  try {
    clear_error();
    if (inplace) {
      a->t.fill_(value);
      return a;
    }
    return wrap(at::full_like(a->t, value));
  }
  LT_CATCH_NULL
}

/* ------------------------------------------------------------------ */
/* linear algebra                                                     */
/* ------------------------------------------------------------------ */

lt_tensor* lt_matmul(lt_tensor* a, lt_tensor* b) {
  try {
    clear_error();
    return wrap(a->t.matmul(b->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_mm(lt_tensor* a, lt_tensor* b) {
  try {
    clear_error();
    return wrap(at::mm(a->t, b->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_bmm(lt_tensor* a, lt_tensor* b) {
  try {
    clear_error();
    return wrap(at::bmm(a->t, b->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_dot(lt_tensor* a, lt_tensor* b) {
  try {
    clear_error();
    return wrap(at::dot(a->t, b->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_linalg_inv(lt_tensor* a) {
  try {
    clear_error();
    return wrap(at::linalg_inv(a->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_linalg_det(lt_tensor* a) {
  try {
    clear_error();
    return wrap(at::linalg_det(a->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_linalg_solve(lt_tensor* a, lt_tensor* b) {
  try {
    clear_error();
    return wrap(at::linalg_solve(a->t, b->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_linalg_cholesky(lt_tensor* a, int upper) {
  try {
    clear_error();
    return wrap(at::linalg_cholesky(a->t, upper != 0));
  }
  LT_CATCH_NULL
}

int lt_linalg_svd(lt_tensor* a, int full_matrices,
                  lt_tensor** U, lt_tensor** S, lt_tensor** Vh) {
  try {
    clear_error();
    if (U == nullptr || S == nullptr || Vh == nullptr) {
      throw std::runtime_error("svd: null output pointer");
    }
    auto result = at::linalg_svd(a->t, full_matrices != 0, std::nullopt);
    *U = wrap(std::get<0>(result));
    *S = wrap(std::get<1>(result));
    *Vh = wrap(std::get<2>(result));
    return 0;
  }
  LT_CATCH_INT
}

int lt_linalg_qr(lt_tensor* a, const char* mode, lt_tensor** Q, lt_tensor** R) {
  try {
    clear_error();
    if (Q == nullptr || R == nullptr) {
      throw std::runtime_error("qr: null output pointer");
    }
    std::string mode_string = mode == nullptr ? "reduced" : std::string(mode);
    auto result = at::linalg_qr(a->t, mode_string);
    *Q = wrap(std::get<0>(result));
    *R = wrap(std::get<1>(result));
    return 0;
  }
  LT_CATCH_INT
}

lt_tensor* lt_mv(lt_tensor* a, lt_tensor* v) {
  try {
    clear_error();
    return wrap(at::mv(a->t, v->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_outer(lt_tensor* a, lt_tensor* b) {
  try {
    clear_error();
    return wrap(at::outer(a->t, b->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_addmm(lt_tensor* self, lt_tensor* a, lt_tensor* b,
                    double beta, double alpha) {
  try {
    clear_error();
    return wrap(at::addmm(self->t, a->t, b->t, beta, alpha));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_baddbmm(lt_tensor* self, lt_tensor* a, lt_tensor* b,
                      double beta, double alpha) {
  try {
    clear_error();
    return wrap(at::baddbmm(self->t, a->t, b->t, beta, alpha));
  }
  LT_CATCH_NULL
}

/* ------------------------------------------------------------------ */
/* neural-network primitives                                          */
/* ------------------------------------------------------------------ */

lt_tensor* lt_softmax(lt_tensor* a, int64_t dim) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "softmax");
    return wrap(at::softmax(a->t, d));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_log_softmax(lt_tensor* a, int64_t dim) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "log_softmax");
    return wrap(at::log_softmax(a->t, d));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_conv2d(
    lt_tensor* input, lt_tensor* weight, const lt_tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t groups) {
  try {
    clear_error();
    std::optional<at::Tensor> bias_tensor;
    if (bias != nullptr) {
      bias_tensor = bias->t;
    }
    return wrap(at::conv2d(
        input->t, weight->t, bias_tensor,
        at::IntArrayRef({stride_h, stride_w}),
        at::IntArrayRef({pad_h, pad_w}),
        at::IntArrayRef({dilation_h, dilation_w}),
        groups));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_max_pool2d(
    lt_tensor* input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int ceil_mode) {
  try {
    clear_error();
    return wrap(at::max_pool2d(
        input->t,
        at::IntArrayRef({kernel_h, kernel_w}),
        at::IntArrayRef({stride_h, stride_w}),
        at::IntArrayRef({pad_h, pad_w}),
        at::IntArrayRef({dilation_h, dilation_w}),
        ceil_mode != 0));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_avg_pool2d(
    lt_tensor* input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int ceil_mode, int count_include_pad) {
  try {
    clear_error();
    return wrap(at::avg_pool2d(
        input->t,
        at::IntArrayRef({kernel_h, kernel_w}),
        at::IntArrayRef({stride_h, stride_w}),
        at::IntArrayRef({pad_h, pad_w}),
        ceil_mode != 0,
        count_include_pad != 0));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_embedding(lt_tensor* weight, lt_tensor* indices) {
  try {
    clear_error();
    if (indices->t.scalar_type() != at::kLong) {
      throw std::runtime_error("embedding: indices must be int64");
    }
    return wrap(at::embedding(weight->t, indices->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_layer_norm(
    lt_tensor* input, const lt_tensor* weight, const lt_tensor* bias,
    int64_t ndim, const int64_t* normalized_shape, double eps) {
  try {
    clear_error();
    std::optional<at::Tensor> weight_tensor;
    std::optional<at::Tensor> bias_tensor;
    if (weight != nullptr) {
      weight_tensor = weight->t;
    }
    if (bias != nullptr) {
      bias_tensor = bias->t;
    }
    return wrap(at::layer_norm(
        input->t,
        make_shape(ndim, normalized_shape),
        weight_tensor,
        bias_tensor,
        eps));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_cross_entropy(lt_tensor* input, lt_tensor* target, int64_t reduction) {
  try {
    clear_error();
    at::Tensor log_probabilities = at::log_softmax(input->t, 1);
    return wrap(at::nll_loss(
        log_probabilities, target->t, std::nullopt, reduction, -100));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_nll_loss(lt_tensor* input, lt_tensor* target, int64_t reduction) {
  try {
    clear_error();
    return wrap(at::nll_loss(input->t, target->t, std::nullopt, reduction, -100));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_binary_cross_entropy_with_logits(
    lt_tensor* input, lt_tensor* target, int64_t reduction) {
  try {
    clear_error();
    return wrap(at::binary_cross_entropy_with_logits(
        input->t, target->t, std::nullopt, std::nullopt, reduction));
  }
  LT_CATCH_NULL
}

int lt_uniform_(lt_tensor* t, double low, double high) {
  try {
    clear_error();
    t->t.uniform_(low, high);
    return 0;
  }
  LT_CATCH_INT
}

/* ------------------------------------------------------------------ */
/* shape manipulation                                                 */
/* ------------------------------------------------------------------ */

lt_tensor* lt_reshape(lt_tensor* a, int64_t ndim, const int64_t* sizes) {
  try {
    clear_error();
    return wrap(a->t.reshape(make_shape(ndim, sizes)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_view(lt_tensor* a, int64_t ndim, const int64_t* sizes) {
  try {
    clear_error();
    return wrap(a->t.view(make_shape(ndim, sizes)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_transpose(lt_tensor* a, int64_t d0, int64_t d1) {
  try {
    clear_error();
    int64_t ndim = a->t.dim();
    int64_t dim0 = resolve_dim(d0, ndim, "transpose");
    int64_t dim1 = resolve_dim(d1, ndim, "transpose");
    return wrap(a->t.transpose(dim0, dim1));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_t(lt_tensor* a) {
  try {
    clear_error();
    return wrap(a->t.t());
  }
  LT_CATCH_NULL
}

lt_tensor* lt_permute(lt_tensor* a, int64_t ndim, const int64_t* dims) {
  try {
    clear_error();
    if (ndim != a->t.dim()) {
      throw std::runtime_error(
          "permute: expected " + std::to_string(a->t.dim()) +
          " dimensions, got " + std::to_string(ndim));
    }
    std::vector<int64_t> resolved;
    resolved.reserve(static_cast<size_t>(ndim));
    for (int64_t i = 0; i < ndim; ++i) {
      resolved.push_back(resolve_dim(dims[i], ndim, "permute"));
    }
    return wrap(a->t.permute(resolved));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_squeeze(lt_tensor* a, int64_t dim) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "squeeze");
    return wrap(a->t.squeeze(d));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_squeeze_all(lt_tensor* a) {
  try {
    clear_error();
    return wrap(a->t.squeeze());
  }
  LT_CATCH_NULL
}

lt_tensor* lt_unsqueeze(lt_tensor* a, int64_t dim) {
  try {
    clear_error();
    int64_t ndim = a->t.dim();
    int64_t d = dim;
    if (d < 0) {
      d += ndim + 1;
    }
    if (d < 0 || d > ndim) {
      throw std::runtime_error(
          "unsqueeze: dimension " + std::to_string(dim) + " out of range");
    }
    return wrap(a->t.unsqueeze(d));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_flatten(lt_tensor* a, int64_t start, int64_t end) {
  try {
    clear_error();
    int64_t ndim = a->t.dim();
    int64_t s = resolve_dim(start, ndim, "flatten");
    int64_t e = end;
    if (e < 0) {
      e += ndim;
    }
    if (e < s || e >= ndim) {
      throw std::runtime_error("flatten: invalid end dimension");
    }
    return wrap(a->t.flatten(s, e));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_expand(lt_tensor* a, int64_t ndim, const int64_t* sizes) {
  try {
    clear_error();
    return wrap(a->t.expand(make_shape(ndim, sizes)));
  }
  LT_CATCH_NULL
}

/* ------------------------------------------------------------------ */
/* indexing                                                           */
/* ------------------------------------------------------------------ */

lt_tensor* lt_select(lt_tensor* a, int64_t dim, int64_t index) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "select");
    int64_t i = resolve_index(index, a->t.size(d), "select");
    return wrap(a->t.select(d, i));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_narrow(lt_tensor* a, int64_t dim, int64_t start, int64_t length) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "narrow");
    int64_t size = a->t.size(d);
    int64_t s = start;
    if (s < 0) {
      s += size;
    }
    if (s < 0 || s > size) {
      throw std::runtime_error("narrow: start out of range");
    }
    if (length < 0) {
      throw std::runtime_error("narrow: negative length");
    }
    return wrap(a->t.narrow(d, s, std::min(length, size - s)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_slice(lt_tensor* a, int64_t dim, int64_t start, int64_t stop, int64_t step) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "slice");
    int64_t size = a->t.size(d);
    int64_t s = start < 0 ? start + size : start;
    int64_t e = stop < 0 ? stop + size : stop;
    if (s < 0) {
      s = 0;
    }
    if (e > size) {
      e = size;
    }
    if (s > size || e < 0 || e < s) {
      throw std::runtime_error("slice: invalid range");
    }
    if (step <= 0) {
      throw std::runtime_error("slice: step must be positive");
    }
    return wrap(a->t.slice(d, s, e, step));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_flip(lt_tensor* a, int64_t dim) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "flip");
    return wrap(a->t.flip(at::IntArrayRef(&d, 1)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_cat(lt_tensor* const* tensors, int64_t n, int64_t dim) {
  try {
    clear_error();
    if (n <= 0 || tensors == nullptr) {
      throw std::runtime_error("cat: empty tensor list");
    }
    std::vector<at::Tensor> values;
    values.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      values.push_back(tensors[i]->t);
    }
    return wrap(at::cat(values, dim));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_stack(lt_tensor* const* tensors, int64_t n, int64_t dim) {
  try {
    clear_error();
    if (n <= 0 || tensors == nullptr) {
      throw std::runtime_error("stack: empty tensor list");
    }
    std::vector<at::Tensor> values;
    values.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      values.push_back(tensors[i]->t);
    }
    return wrap(at::stack(values, dim));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_where(lt_tensor* cond, lt_tensor* a, lt_tensor* b) {
  try {
    clear_error();
    return wrap(at::where(cond->t, a->t, b->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_masked_fill(lt_tensor* a, lt_tensor* mask, double value) {
  try {
    clear_error();
    return wrap(a->t.masked_fill(mask->t, value));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_masked_select(lt_tensor* a, lt_tensor* mask) {
  try {
    clear_error();
    if (mask->t.scalar_type() != at::kBool) {
      throw std::runtime_error("masked_select: mask must be a bool tensor");
    }
    return wrap(at::masked_select(a->t, mask->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_index_select(lt_tensor* a, int64_t dim, lt_tensor* indices) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "index_select");
    if (indices->t.scalar_type() != at::kLong) {
      throw std::runtime_error("index_select: indices must be int64");
    }
    return wrap(a->t.index_select(d, indices->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_nonzero(lt_tensor* a) {
  try {
    clear_error();
    return wrap(a->t.nonzero());
  }
  LT_CATCH_NULL
}

lt_tensor* lt_gather(lt_tensor* a, int64_t dim, lt_tensor* index) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "gather");
    if (index->t.scalar_type() != at::kLong) {
      throw std::runtime_error("gather: index must be int64");
    }
    return wrap(at::gather(a->t, d, index->t));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_scatter(lt_tensor* a, int64_t dim, lt_tensor* index, lt_tensor* src) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "scatter");
    if (index->t.scalar_type() != at::kLong) {
      throw std::runtime_error("scatter: index must be int64");
    }
    return wrap(at::scatter(a->t, d, index->t, src->t));
  }
  LT_CATCH_NULL
}

int lt_sort(lt_tensor* a, int64_t dim, int descending,
            lt_tensor** values, lt_tensor** indices) {
  try {
    clear_error();
    if (values == nullptr || indices == nullptr) {
      throw std::runtime_error("sort: null output pointer");
    }
    int64_t d = resolve_dim(dim, a->t.dim(), "sort");
    auto result = a->t.sort(d, descending != 0);
    *values = wrap(std::get<0>(result));
    *indices = wrap(std::get<1>(result));
    return 0;
  }
  LT_CATCH_INT
}

int lt_topk(lt_tensor* a, int64_t k, int64_t dim, int largest, int sorted,
            lt_tensor** values, lt_tensor** indices) {
  try {
    clear_error();
    if (values == nullptr || indices == nullptr) {
      throw std::runtime_error("topk: null output pointer");
    }
    int64_t d = resolve_dim(dim, a->t.dim(), "topk");
    auto result = a->t.topk(k, d, largest != 0, sorted != 0);
    *values = wrap(std::get<0>(result));
    *indices = wrap(std::get<1>(result));
    return 0;
  }
  LT_CATCH_INT
}

int lt_unique(lt_tensor* a, int sorted, int return_inverse, int return_counts,
              lt_tensor** values, lt_tensor** inverse, lt_tensor** counts) {
  try {
    clear_error();
    if (values == nullptr) {
      throw std::runtime_error("unique: null output pointer");
    }
    auto result = at::_unique2(
        a->t, sorted != 0, return_inverse != 0, return_counts != 0);
    *values = wrap(std::get<0>(result));
    if (inverse != nullptr && return_inverse) {
      *inverse = wrap(std::get<1>(result));
    } else if (inverse != nullptr) {
      *inverse = nullptr;
    }
    if (counts != nullptr && return_counts) {
      *counts = wrap(std::get<2>(result));
    } else if (counts != nullptr) {
      *counts = nullptr;
    }
    return 0;
  }
  LT_CATCH_INT
}

/* ------------------------------------------------------------------ */
/* reductions                                                         */
/* ------------------------------------------------------------------ */

#define LT_REDUCE(name, method)                                          \
  lt_tensor* lt_##name(lt_tensor* a, int64_t dim, int keepdim) {         \
    try {                                                                \
      clear_error();                                                     \
      int64_t ndim = a->t.dim();                                         \
      bool keep = keepdim != 0;                                          \
      if (dim == LT_ALL_DIMS) {                                          \
        if (keep) {                                                      \
          std::vector<int64_t> dims = all_dims(ndim);                    \
          return wrap(a->t.method(dims, true));                          \
        }                                                                \
        return wrap(a->t.method());                                      \
      }                                                                  \
      int64_t d = resolve_dim(dim, ndim, #name);                         \
      return wrap(a->t.method(at::IntArrayRef(&d, 1), keep));            \
    }                                                                    \
    LT_CATCH_NULL                                                        \
  }

LT_REDUCE(sum, sum)
LT_REDUCE(mean, mean)

#undef LT_REDUCE

lt_tensor* lt_prod(lt_tensor* a, int64_t dim, int keepdim) {
  try {
    clear_error();
    int64_t ndim = a->t.dim();
    bool keep = keepdim != 0;
    if (dim == LT_ALL_DIMS) {
      at::Tensor value = a->t.prod();
      if (keep) {
        std::vector<int64_t> shape(static_cast<size_t>(ndim), 1);
        value = value.reshape(shape);
      }
      return wrap(value);
    }
    int64_t d = resolve_dim(dim, ndim, "prod");
    return wrap(a->t.prod(d, keep));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_max(lt_tensor* a, int64_t dim, int keepdim) {
  try {
    clear_error();
    int64_t ndim = a->t.dim();
    bool keep = keepdim != 0;
    if (dim == LT_ALL_DIMS) {
      if (keep) {
        std::vector<int64_t> dims = all_dims(ndim);
        return wrap(at::amax(a->t, dims, true));
      }
      return wrap(a->t.max());
    }
    int64_t d = resolve_dim(dim, ndim, "max");
    return wrap(std::get<0>(a->t.max(d, keep)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_min(lt_tensor* a, int64_t dim, int keepdim) {
  try {
    clear_error();
    int64_t ndim = a->t.dim();
    bool keep = keepdim != 0;
    if (dim == LT_ALL_DIMS) {
      if (keep) {
        std::vector<int64_t> dims = all_dims(ndim);
        return wrap(at::amin(a->t, dims, true));
      }
      return wrap(a->t.min());
    }
    int64_t d = resolve_dim(dim, ndim, "min");
    return wrap(std::get<0>(a->t.min(d, keep)));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_argmax(lt_tensor* a, int64_t dim, int keepdim) {
  try {
    clear_error();
    if (dim == LT_ALL_DIMS) {
      return wrap(a->t.argmax());
    }
    int64_t d = resolve_dim(dim, a->t.dim(), "argmax");
    return wrap(a->t.argmax(d, keepdim != 0));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_argmin(lt_tensor* a, int64_t dim, int keepdim) {
  try {
    clear_error();
    if (dim == LT_ALL_DIMS) {
      return wrap(a->t.argmin());
    }
    int64_t d = resolve_dim(dim, a->t.dim(), "argmin");
    return wrap(a->t.argmin(d, keepdim != 0));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_var(lt_tensor* a, int64_t dim, int keepdim, int unbiased) {
  try {
    clear_error();
    return wrap(variance_impl(a->t, dim, keepdim != 0, unbiased != 0));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_std(lt_tensor* a, int64_t dim, int keepdim, int unbiased) {
  try {
    clear_error();
    return wrap(variance_impl(a->t, dim, keepdim != 0, unbiased != 0).sqrt());
  }
  LT_CATCH_NULL
}

lt_tensor* lt_norm(lt_tensor* a, double p, int64_t dim, int keepdim) {
  try {
    clear_error();
    return wrap(norm_impl(a->t, p, dim, keepdim != 0));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_cumsum(lt_tensor* a, int64_t dim) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "cumsum");
    return wrap(a->t.cumsum(d));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_cumprod(lt_tensor* a, int64_t dim) {
  try {
    clear_error();
    int64_t d = resolve_dim(dim, a->t.dim(), "cumprod");
    return wrap(a->t.cumprod(d));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_all(lt_tensor* a, int64_t dim, int keepdim) {
  try {
    clear_error();
    if (dim == LT_ALL_DIMS) {
      return wrap(a->t.all());
    }
    int64_t d = resolve_dim(dim, a->t.dim(), "all");
    return wrap(a->t.all(d, keepdim != 0));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_any(lt_tensor* a, int64_t dim, int keepdim) {
  try {
    clear_error();
    if (dim == LT_ALL_DIMS) {
      return wrap(a->t.any());
    }
    int64_t d = resolve_dim(dim, a->t.dim(), "any");
    return wrap(a->t.any(d, keepdim != 0));
  }
  LT_CATCH_NULL
}

lt_tensor* lt_logsumexp(lt_tensor* a, int64_t dim, int keepdim) {
  try {
    clear_error();
    if (dim == LT_ALL_DIMS) {
      std::vector<int64_t> dims = all_dims(a->t.dim());
      return wrap(at::logsumexp(a->t, dims, keepdim != 0));
    }
    int64_t d = resolve_dim(dim, a->t.dim(), "logsumexp");
    return wrap(at::logsumexp(a->t, at::IntArrayRef(&d, 1), keepdim != 0));
  }
  LT_CATCH_NULL
}

int lt_normal_(lt_tensor* a, double mean, double std) {
  try {
    clear_error();
    a->t.normal_(mean, std);
    return 0;
  }
  LT_CATCH_INT
}

int lt_exponential_(lt_tensor* a, double lambd) {
  try {
    clear_error();
    a->t.exponential_(lambd);
    return 0;
  }
  LT_CATCH_INT
}

int lt_bernoulli_(lt_tensor* a, double p) {
  try {
    clear_error();
    a->t.bernoulli_(p);
    return 0;
  }
  LT_CATCH_INT
}

lt_tensor* lt_multinomial(lt_tensor* a, int64_t num_samples, int replacement) {
  try {
    clear_error();
    return wrap(at::multinomial(a->t, num_samples, replacement != 0));
  }
  LT_CATCH_NULL
}

void lt_set_num_threads(int num_threads) {
  at::set_num_threads(num_threads);
}

int lt_get_num_threads(void) {
  return at::get_num_threads();
}

/* ------------------------------------------------------------------ */
/* grad mode                                                          */
/* ------------------------------------------------------------------ */

void lt_set_grad_enabled(int enabled) {
  at::GradMode::set_enabled(enabled != 0);
}

int lt_is_grad_enabled(void) {
  return at::GradMode::is_enabled() ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* serialization                                                      */
/* ------------------------------------------------------------------ */

int lt_save(const lt_tensor* t, const char* path) {
  try {
    clear_error();
    if (path == nullptr) {
      throw std::runtime_error("save: null path");
    }
    torch::save(t->t, std::string(path));
    return 0;
  }
  LT_CATCH_INT
}

lt_tensor* lt_load(const char* path) {
  try {
    clear_error();
    if (path == nullptr) {
      throw std::runtime_error("load: null path");
    }
    at::Tensor tensor;
    torch::load(tensor, std::string(path));
    return wrap(tensor);
  }
  LT_CATCH_NULL
}

}  // extern "C"
