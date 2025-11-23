/**
 * @file utils.h
 * @brief Utility primitives exposed from the ΛSNARK-R core.
 *
 * Provides FFI-safe declarations for helper routines implemented in
 * `src/utils.cpp`, including the constant-time discrete Gaussian sampler.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sample coefficients from a discrete Gaussian distribution.
 *
 * @param output Pointer to the destination buffer (must have `len` elements).
 * @param len    Number of samples to generate.
 * @param sigma  Gaussian width parameter (standard deviation).
 *
 * @return 0 on success, -1 on invalid parameters.
 */
int sample_gaussian(uint64_t* output, size_t len, double sigma) noexcept;

#ifdef __cplusplus
}  // extern "C"
#endif
