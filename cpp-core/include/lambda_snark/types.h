/**
 * @file types.h
 * @brief Common types for ΛSNARK-R C++ core.
 * 
 * This header defines the fundamental data structures and type aliases
 * used throughout the ΛSNARK-R C++ performance kernel.
 * 
 * @copyright Copyright (c) 2025 URPKS Contributors
 * @license Apache-2.0 OR MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to LWE context (contains public parameters).
 * 
 * This structure is intentionally opaque to Rust FFI; internal layout
 * defined in commitment.cpp.
 */
typedef struct LweContext LweContext;

/**
 * @brief LWE commitment structure (FFI-safe).
 * 
 * Layout: flat array of uint64_t elements representing polynomial coefficients
 * in the cyclotomic ring R_q = Z_q[X]/(X^n + 1).
 */
typedef struct {
    uint64_t* data;  ///< Pointer to commitment data (caller must free)
    size_t    len;   ///< Length of data array
} LweCommitment;

/**
 * @brief Opening information for commitment verification.
 */
typedef struct {
    uint64_t* randomness;  ///< Random coins used in commitment
    size_t    rand_len;    ///< Length of randomness
} LweOpening;

/**
 * @brief Parameter profile identifier.
 */
typedef enum {
    PROFILE_SCALAR_A = 0,  ///< Scalar profile (Z_q with q > 2^24)
    PROFILE_RING_B   = 1,  ///< Ring profile (R_q with n=256, k=2)
} ProfileType;

/**
 * @brief Public parameters for ΛSNARK-R.
 */
typedef struct {
    ProfileType profile;
    uint32_t    security_level;  ///< λ ∈ {128, 192, 256}
    uint64_t    modulus;          ///< q (prime modulus)
    uint32_t    ring_degree;      ///< n (for ring profile)
    uint32_t    module_rank;      ///< k (for module-LWE)
    double      sigma;            ///< Gaussian parameter
} PublicParams;

/**
 * @brief Error codes (FFI-safe).
 */
typedef enum {
    LAMBDA_SNARK_OK              = 0,
    LAMBDA_SNARK_ERR_NULL_PTR    = 1,
    LAMBDA_SNARK_ERR_INVALID_PARAMS = 2,
    LAMBDA_SNARK_ERR_ALLOC_FAILED   = 3,
    LAMBDA_SNARK_ERR_CRYPTO_FAILED  = 4,
} LambdaSnarkError;

#ifdef __cplusplus
}  // extern "C"
#endif
