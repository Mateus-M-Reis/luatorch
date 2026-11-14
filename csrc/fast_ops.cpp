// Fast-math kernels that are small and self-contained.  Keeping them in a
// separate translation unit lets CMake compile only this file with
// -ffast-math; the main ATen shim stays precise.

#include <cmath>
#include <cstdint>

extern "C" {

void lt_log_softmax_f32(
    const float* input, float* output, int64_t rows, int64_t columns) {
  for (int64_t row = 0; row < rows; ++row) {
    const float* source = input + row * columns;
    float* destination = output + row * columns;
    float maximum = -INFINITY;
    for (int64_t column = 0; column < columns; ++column) {
      if (source[column] > maximum) maximum = source[column];
    }
    float sum = 0.0f;
    for (int64_t column = 0; column < columns; ++column) {
      sum += expf(source[column] - maximum);
    }
    const float offset = maximum + logf(sum);
    for (int64_t column = 0; column < columns; ++column) {
      destination[column] = source[column] - offset;
    }
  }
}

void lt_log_softmax_f64(
    const double* input, double* output, int64_t rows, int64_t columns) {
  for (int64_t row = 0; row < rows; ++row) {
    const double* source = input + row * columns;
    double* destination = output + row * columns;
    double maximum = -INFINITY;
    for (int64_t column = 0; column < columns; ++column) {
      if (source[column] > maximum) maximum = source[column];
    }
    double sum = 0.0;
    for (int64_t column = 0; column < columns; ++column) {
      sum += exp(source[column] - maximum);
    }
    const double offset = maximum + log(sum);
    for (int64_t column = 0; column < columns; ++column) {
      destination[column] = source[column] - offset;
    }
  }
}

void lt_log_softmax_backward_f32(
    const float* output, const float* grad_output, float* grad_input,
    int64_t rows, int64_t columns) {
  for (int64_t row = 0; row < rows; ++row) {
    const float* source = output + row * columns;
    const float* upstream = grad_output + row * columns;
    float* destination = grad_input + row * columns;
    float sum = 0.0f;
    for (int64_t column = 0; column < columns; ++column) {
      sum += upstream[column];
    }
    for (int64_t column = 0; column < columns; ++column) {
      destination[column] = upstream[column] - expf(source[column]) * sum;
    }
  }
}

void lt_log_softmax_backward_f64(
    const double* output, const double* grad_output, double* grad_input,
    int64_t rows, int64_t columns) {
  for (int64_t row = 0; row < rows; ++row) {
    const double* source = output + row * columns;
    const double* upstream = grad_output + row * columns;
    double* destination = grad_input + row * columns;
    double sum = 0.0;
    for (int64_t column = 0; column < columns; ++column) {
      sum += upstream[column];
    }
    for (int64_t column = 0; column < columns; ++column) {
      destination[column] = upstream[column] - exp(source[column]) * sum;
    }
  }
}

}  // extern "C"
