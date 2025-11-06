/**
 * @file utils.cpp
 * @brief Utility functions (sampling, serialization, etc.).
 * 
 * @copyright Copyright (c) 2025 URPKS Contributors
 * @license Apache-2.0 OR MIT
 */

#include "lambda_snark/types.h"
#include <cstdlib>

extern "C" {

/**
 * @brief Sample from discrete Gaussian distribution (stub).
 * 
 * TODO: Implement constant-time Gaussian sampling.
 */
int sample_gaussian(uint64_t* output, size_t len, double sigma) noexcept {
    // Stub: uniform random for now
    for (size_t i = 0; i < len; ++i) {
        output[i] = rand() % 1024;  // INSECURE: replace with proper sampling
    }
    return 0;
}

}  // extern "C"
