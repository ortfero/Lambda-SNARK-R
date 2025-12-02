/**
 * @file r1cs.cpp
 * @brief R1CS constraint system implementation.
 * 
 * @copyright Copyright (c) 2025 URPKS Contributors
 * @license Apache-2.0 OR MIT
 */

#include "lambda_snark/r1cs.h"
#include <stdexcept>
#include <cstring>
#include <sstream>

namespace lambda_snark {

R1CS::R1CS(const SparseMatrix& A, 
           const SparseMatrix& B,
           const SparseMatrix& C,
           uint64_t modulus)
    : modulus_(modulus)
{
    // Validate dimensions
    if (A.n_rows != B.n_rows || B.n_rows != C.n_rows) {
        throw std::invalid_argument("R1CS: Matrix row counts must match");
    }
    if (A.n_cols != B.n_cols || B.n_cols != C.n_cols) {
        throw std::invalid_argument("R1CS: Matrix column counts must match");
    }
    
    n_constraints_ = A.n_rows;
    n_vars_ = A.n_cols;

    // Deep copy matrices
    auto copy_matrix = [](const SparseMatrix& src) -> SparseMatrix {
        SparseMatrix dst;
        dst.n_rows = src.n_rows;
        dst.n_cols = src.n_cols;
        dst.n_entries = src.n_entries;
        dst.entries = new SparseEntry[src.n_entries];
        std::memcpy(dst.entries, src.entries, src.n_entries * sizeof(SparseEntry));
        return dst;
    };

    A_ = copy_matrix(A);
    B_ = copy_matrix(B);
    C_ = copy_matrix(C);

    // Initialize NTL modulus context
    NTL::ZZ q;
    // Disambiguate conv overload for uint64_t by casting to unsigned long
    NTL::conv(q, static_cast<unsigned long>(modulus_));
    NTL::ZZ_p::init(q);
}

R1CS::~R1CS() {
    delete[] A_.entries;
    delete[] B_.entries;
    delete[] C_.entries;
}

R1CS::R1CS(R1CS&& other) noexcept
    : A_(other.A_)
    , B_(other.B_)
    , C_(other.C_)
    , modulus_(other.modulus_)
    , n_vars_(other.n_vars_)
    , n_constraints_(other.n_constraints_)
{
    // Null out moved-from object
    other.A_.entries = nullptr;
    other.B_.entries = nullptr;
    other.C_.entries = nullptr;
}

R1CS& R1CS::operator=(R1CS&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        delete[] A_.entries;
        delete[] B_.entries;
        delete[] C_.entries;

        // Move from other
        A_ = other.A_;
        B_ = other.B_;
        C_ = other.C_;
        modulus_ = other.modulus_;
        n_vars_ = other.n_vars_;
        n_constraints_ = other.n_constraints_;

        // Null out moved-from object
        other.A_.entries = nullptr;
        other.B_.entries = nullptr;
        other.C_.entries = nullptr;
    }
    return *this;
}

bool R1CS::validate_witness(const std::vector<uint64_t>& witness) const {
    // Check witness length
    if (witness.size() != n_vars_) {
        std::ostringstream oss;
        oss << "R1CS: Witness length mismatch (expected " << n_vars_ 
            << ", got " << witness.size() << ")";
        throw std::invalid_argument(oss.str());
    }

    // Check first element is 1
    if (witness[0] != 1) {
        throw std::invalid_argument("R1CS: Witness must start with 1");
    }

    // Compute A·z, B·z, C·z
    auto Az = compute_Az(witness);
    auto Bz = compute_Bz(witness);
    auto Cz = compute_Cz(witness);

    // Check (A·z) ∘ (B·z) = C·z for each constraint
    for (uint32_t i = 0; i < n_constraints_; ++i) {
        // Use NTL for modular arithmetic
        NTL::ZZ_p az_i, bz_i, cz_i, product;
        
        NTL::conv(az_i, static_cast<long>(Az[i]));
        NTL::conv(bz_i, static_cast<long>(Bz[i]));
        NTL::conv(cz_i, static_cast<long>(Cz[i]));
        
        product = az_i * bz_i;

        if (product != cz_i) {
            return false;
        }
    }

    return true;
}

std::vector<uint64_t> R1CS::compute_Az(const std::vector<uint64_t>& witness) const {
    return sparse_mv(A_, witness);
}

std::vector<uint64_t> R1CS::compute_Bz(const std::vector<uint64_t>& witness) const {
    return sparse_mv(B_, witness);
}

std::vector<uint64_t> R1CS::compute_Cz(const std::vector<uint64_t>& witness) const {
    return sparse_mv(C_, witness);
}

std::vector<uint64_t> R1CS::sparse_mv(
    const SparseMatrix& matrix,
    const std::vector<uint64_t>& vec) const 
{
    // Initialize result vector with zeros
    std::vector<uint64_t> result(matrix.n_rows, 0);

    // Accumulate each non-zero entry: result[row] += matrix[row,col] * vec[col]
    for (size_t i = 0; i < matrix.n_entries; ++i) {
        const auto& entry = matrix.entries[i];
        
        if (entry.col >= vec.size()) {
            throw std::out_of_range("R1CS: Column index exceeds witness length");
        }

        // Modular arithmetic: result[row] += value * vec[col] (mod q)
        NTL::ZZ_p acc, val, v;
        NTL::conv(acc, static_cast<long>(result[entry.row]));
        NTL::conv(val, static_cast<long>(entry.value));
        NTL::conv(v, static_cast<long>(vec[entry.col]));
        
        acc += val * v;

        // Convert ZZ_p back to host integer
        unsigned long acc_ulong = 0;
        NTL::ZZ acc_zz;
        NTL::conv(acc_zz, acc);
        NTL::conv(acc_ulong, acc_zz);
        result[entry.row] = static_cast<uint64_t>(acc_ulong);
    }

    return result;
}

}  // namespace lambda_snark
