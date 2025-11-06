/**
 * @file commitment.h
 * @brief LWE-based vector commitment interface.
 * 
 * Provides commitment scheme based on Module-LWE with:
 * - Statistical hiding (under appropriate σ)
 * - Binding under Module-SIS hardness
 * - Homomorphic properties for linear checks
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
 * @brief Initialize LWE context from public parameters.
 * 
 * @param params Public parameters (must be validated).
 * @return Pointer to LweContext, or NULL on failure.
 * 
 * @note Caller must call lwe_context_free() to release resources.
 * @note This function may allocate large matrices (A, M) for commitment.
 */
LweContext* lwe_context_create(const PublicParams* params) noexcept;

/**
 * @brief Free LWE context and zeroize sensitive data.
 * 
 * @param ctx Context to free (NULL-safe).
 */
void lwe_context_free(LweContext* ctx) noexcept;

/**
 * @brief Commit to a message vector.
 * 
 * Computes c = A·s + M·message + e, where:
 * - A: public matrix (from ctx)
 * - s: secret sampled from χ
 * - e: error sampled from χ
 * - M: encoding matrix
 * 
 * @param ctx LWE context (must be non-NULL).
 * @param message Message vector (length determined by ctx).
 * @param msg_len Length of message.
 * @param seed Random seed for deterministic commitment (0 = random).
 * @return Pointer to LweCommitment, or NULL on failure.
 * 
 * @note Caller must call lwe_commitment_free() to release.
 * @note This function is NOT constant-time w.r.t. message (only randomness).
 */
LweCommitment* lwe_commit(
    LweContext* ctx,
    const uint64_t* message,
    size_t msg_len,
    uint64_t seed
) noexcept;

/**
 * @brief Free commitment and zeroize data.
 * 
 * @param comm Commitment to free (NULL-safe).
 */
void lwe_commitment_free(LweCommitment* comm) noexcept;

/**
 * @brief Verify opening of commitment (constant-time).
 * 
 * Recomputes commitment and checks equality using constant-time comparison.
 * 
 * @param ctx LWE context.
 * @param commitment Claimed commitment.
 * @param message Claimed message.
 * @param msg_len Length of message.
 * @param opening Opening information (randomness).
 * @return 1 if valid, 0 if invalid, -1 on error.
 * 
 * @note MUST be constant-time in message and randomness.
 */
int lwe_verify_opening(
    const LweContext* ctx,
    const LweCommitment* commitment,
    const uint64_t* message,
    size_t msg_len,
    const LweOpening* opening
) noexcept;

/**
 * @brief Compute linear combination of commitments (homomorphic).
 * 
 * Computes result = Σ coeffs[i] * commitments[i].
 * 
 * @param ctx LWE context.
 * @param commitments Array of commitments.
 * @param coeffs Coefficients for linear combination.
 * @param count Number of commitments.
 * @return New commitment, or NULL on failure.
 */
LweCommitment* lwe_linear_combine(
    const LweContext* ctx,
    const LweCommitment** commitments,
    const uint64_t* coeffs,
    size_t count
) noexcept;

#ifdef __cplusplus
}  // extern "C"
#endif
