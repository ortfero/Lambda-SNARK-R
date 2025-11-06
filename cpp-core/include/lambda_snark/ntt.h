/**
 * @file ntt.h
 * @brief Number-Theoretic Transform for cyclotomic rings.
 * 
 * Optimized NTT implementation for R_q = Z_q[X]/(X^n + 1) using:
 * - Cooley-Tukey butterfly operations
 * - Bit-reversal permutation
 * - AVX-512 SIMD (if available)
 * 
 * @copyright Copyright (c) 2025 URPKS Contributors
 * @license Apache-2.0 OR MIT
 */

#pragma once

#include "lambda_snark/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NTT context (precomputed twiddle factors).
 */
typedef struct NttContext NttContext;

/**
 * @brief Initialize NTT context for given ring parameters.
 * 
 * @param q Modulus (must be prime, q ≡ 1 mod 2n).
 * @param n Ring degree (must be power of 2).
 * @return Pointer to NttContext, or NULL on failure.
 */
NttContext* ntt_context_create(uint64_t q, uint32_t n) noexcept;

/**
 * @brief Free NTT context.
 * 
 * @param ctx Context to free (NULL-safe).
 */
void ntt_context_free(NttContext* ctx) noexcept;

/**
 * @brief Forward NTT (polynomial to evaluation representation).
 * 
 * Transforms coeffs[0..n-1] in-place.
 * 
 * @param ctx NTT context.
 * @param coeffs Coefficient array (length n, modified in-place).
 * @param n Length of array (must match ctx).
 * @return 0 on success, error code on failure.
 * 
 * @note This function is NOT constant-time (optimization over security).
 */
int ntt_forward(
    const NttContext* ctx,
    uint64_t* coeffs,
    uint32_t n
) noexcept;

/**
 * @brief Inverse NTT (evaluation to coefficient representation).
 * 
 * @param ctx NTT context.
 * @param evals Evaluation array (length n, modified in-place).
 * @param n Length of array.
 * @return 0 on success, error code on failure.
 */
int ntt_inverse(
    const NttContext* ctx,
    uint64_t* evals,
    uint32_t n
) noexcept;

/**
 * @brief Pointwise multiplication in NTT domain.
 * 
 * Computes result[i] = a[i] * b[i] mod q for all i.
 * 
 * @param ctx NTT context.
 * @param result Output array (can alias a or b).
 * @param a First operand (NTT form).
 * @param b Second operand (NTT form).
 * @param n Length of arrays.
 */
void ntt_mul_pointwise(
    const NttContext* ctx,
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint32_t n
) noexcept;

#ifdef __cplusplus
}  // extern "C"
#endif
