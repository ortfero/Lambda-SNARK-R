/**
 * @file r1cs.h
 * @brief Rank-1 Constraint System (R1CS) implementation for ΛSNARK-R.
 * 
 * This header defines the R1CS constraint system structure and operations.
 * R1CS represents arithmetic constraints as:
 *     (A·z) ∘ (B·z) = (C·z)
 * where:
 * - A, B, C are m×n matrices (m constraints, n variables)
 * - z is the witness vector [1, x₁, x₂, ..., xₙ₋₁]
 * - ∘ is element-wise (Hadamard) product
 * 
 * Matrices are stored in sparse Coordinate (COO) format:
 *     (row, col, value) triples
 * 
 * @copyright Copyright (c) 2025 URPKS Contributors
 * @license Apache-2.0 OR MIT
 */

#pragma once

#include "types.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include <NTL/ZZ_p.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sparse matrix entry (row, col, value).
 * 
 * Represents a single non-zero element in a matrix.
 * Value is stored as uint64_t for FFI compatibility.
 */
typedef struct {
    uint32_t row;     ///< Row index (0-based)
    uint32_t col;     ///< Column index (0-based)
    uint64_t value;   ///< Field element (mod q)
} SparseEntry;

/**
 * @brief Sparse matrix in COO format.
 * 
 * FFI-safe representation of a sparse matrix.
 */
typedef struct {
    SparseEntry* entries;  ///< Array of non-zero entries
    size_t       n_entries; ///< Number of entries
    uint32_t     n_rows;    ///< Number of rows
    uint32_t     n_cols;    ///< Number of columns
} SparseMatrix;

/**
 * @brief R1CS constraint system.
 * 
 * Represents (A·z) ∘ (B·z) = (C·z) for witness z.
 * FFI-safe structure.
 */
typedef struct {
    SparseMatrix A;  ///< Left multiplication matrix
    SparseMatrix B;  ///< Right multiplication matrix
    SparseMatrix C;  ///< Output constraint matrix
    uint32_t n_vars; ///< Number of variables (n)
    uint32_t n_public_inputs; ///< Number of public inputs (l)
    uint32_t n_constraints; ///< Number of constraints (m)
} R1CSConstraintSystem;

/**
 * @brief Witness vector for R1CS.
 * 
 * FFI-safe representation of witness z = [1, x₁, ..., xₙ₋₁].
 */
typedef struct {
    uint64_t* values;  ///< Witness values (mod q)
    size_t    len;     ///< Length (must equal n_vars)
} R1CSWitness;

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus

namespace lambda_snark {

/**
 * @brief C++ wrapper for R1CS constraint system.
 * 
 * Provides safe RAII semantics and NTL integration.
 */
class R1CS {
public:
    /**
     * @brief Construct R1CS from matrices.
     * 
     * @param A Left matrix (sparse)
     * @param B Right matrix (sparse)
     * @param C Output matrix (sparse)
     * @param modulus Field modulus q
     * @throws std::invalid_argument if dimensions mismatch
     */
    R1CS(const SparseMatrix& A, 
         const SparseMatrix& B,
         const SparseMatrix& C,
         uint64_t modulus);

    /**
     * @brief Destructor (cleans up NTL context).
     */
    ~R1CS();

    // Delete copy (sparse matrices are expensive)
    R1CS(const R1CS&) = delete;
    R1CS& operator=(const R1CS&) = delete;

    // Allow move
    R1CS(R1CS&&) noexcept;
    R1CS& operator=(R1CS&&) noexcept;

    /**
     * @brief Validate witness satisfies all constraints.
     * 
     * Checks: (A·z) ∘ (B·z) = (C·z) for all rows.
     * 
     * @param witness Witness vector z
     * @return true if all constraints satisfied
     * @throws std::invalid_argument if witness length mismatch
     */
    bool validate_witness(const std::vector<uint64_t>& witness) const;

    /**
     * @brief Compute A·z for given witness.
     * 
     * @param witness Witness vector
     * @return Result vector (length = n_constraints)
     */
    std::vector<uint64_t> compute_Az(const std::vector<uint64_t>& witness) const;

    /**
     * @brief Compute B·z for given witness.
     * 
     * @param witness Witness vector
     * @return Result vector (length = n_constraints)
     */
    std::vector<uint64_t> compute_Bz(const std::vector<uint64_t>& witness) const;

    /**
     * @brief Compute C·z for given witness.
     * 
     * @param witness Witness vector
     * @return Result vector (length = n_constraints)
     */
    std::vector<uint64_t> compute_Cz(const std::vector<uint64_t>& witness) const;

    /**
     * @brief Get number of constraints.
     */
    uint32_t num_constraints() const { return n_constraints_; }

    /**
     * @brief Get number of variables.
     */
    uint32_t num_variables() const { return n_vars_; }

    /**
     * @brief Get field modulus.
     */
    uint64_t modulus() const { return modulus_; }

private:
    /**
     * @brief Sparse matrix-vector multiplication: M·v.
     * 
     * @param matrix Sparse matrix
     * @param vec Input vector
     * @return Result vector (length = matrix.n_rows)
     */
    std::vector<uint64_t> sparse_mv(
        const SparseMatrix& matrix,
        const std::vector<uint64_t>& vec) const;

    SparseMatrix A_;
    SparseMatrix B_;
    SparseMatrix C_;
    uint64_t modulus_;
    uint32_t n_vars_;
    uint32_t n_constraints_;
};

}  // namespace lambda_snark

#endif  // __cplusplus
