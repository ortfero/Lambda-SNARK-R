/**
 * @file test_r1cs.cpp
 * @brief Unit tests for R1CS constraint system.
 * 
 * @copyright Copyright (c) 2025 URPKS Contributors
 * @license Apache-2.0 OR MIT
 */

#include "lambda_snark/r1cs.h"
#include <gtest/gtest.h>
#include <vector>

using namespace lambda_snark;

// Test modulus: q = 2^44 + 1 (from Phase 1)
constexpr uint64_t TEST_MODULUS = 17592186044417ULL;

/**
 * @brief Test fixture for R1CS tests.
 */
class R1CSTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize NTL context
        NTL::ZZ q;
        NTL::conv(q, static_cast<unsigned long>(TEST_MODULUS));
        NTL::ZZ_p::init(q);
    }
};

/**
 * @brief Test: Simple multiplication constraint a * b = c.
 * 
 * Witness: z = [1, a, b, c]
 * Constraint: z[1] * z[2] = z[3]
 * 
 * R1CS encoding:
 * A = [[0, 1, 0, 0]]  (selects a)
 * B = [[0, 0, 1, 0]]  (selects b)
 * C = [[0, 0, 0, 1]]  (selects c)
 */
TEST_F(R1CSTest, SimpleMultiplication) {
    // Define sparse matrices (1 constraint, 4 variables)
    SparseEntry a_entries[] = {{0, 1, 1}};  // A[0,1] = 1
    SparseEntry b_entries[] = {{0, 2, 1}};  // B[0,2] = 1
    SparseEntry c_entries[] = {{0, 3, 1}};  // C[0,3] = 1

    SparseMatrix A = {a_entries, 1, 1, 4};
    SparseMatrix B = {b_entries, 1, 1, 4};
    SparseMatrix C = {c_entries, 1, 1, 4};

    R1CS r1cs(A, B, C, TEST_MODULUS);

    EXPECT_EQ(r1cs.num_constraints(), 1);
    EXPECT_EQ(r1cs.num_variables(), 4);

    // Valid witness: [1, 7, 13, 91] (7 * 13 = 91)
    std::vector<uint64_t> valid_witness = {1, 7, 13, 91};
    EXPECT_TRUE(r1cs.validate_witness(valid_witness));

    // Invalid witness: [1, 7, 13, 92] (7 * 13 ≠ 92)
    std::vector<uint64_t> invalid_witness = {1, 7, 13, 92};
    EXPECT_FALSE(r1cs.validate_witness(invalid_witness));
}

/**
 * @brief Test: Two multiplication constraints (TV-1 style).
 * 
 * Constraints:
 * 1. a * b = c
 * 2. c * 1 = c  (identity check)
 */
TEST_F(R1CSTest, TwoConstraints) {
    // 2 constraints, 4 variables
    SparseEntry a_entries[] = {
        {0, 1, 1},  // A[0,1] = 1 (constraint 0: select a)
        {1, 3, 1}   // A[1,3] = 1 (constraint 1: select c)
    };
    SparseEntry b_entries[] = {
        {0, 2, 1},  // B[0,2] = 1 (constraint 0: select b)
        {1, 0, 1}   // B[1,0] = 1 (constraint 1: select constant 1)
    };
    SparseEntry c_entries[] = {
        {0, 3, 1},  // C[0,3] = 1 (constraint 0: select c)
        {1, 3, 1}   // C[1,3] = 1 (constraint 1: select c)
    };

    SparseMatrix A = {a_entries, 2, 2, 4};
    SparseMatrix B = {b_entries, 2, 2, 4};
    SparseMatrix C = {c_entries, 2, 2, 4};

    R1CS r1cs(A, B, C, TEST_MODULUS);

    EXPECT_EQ(r1cs.num_constraints(), 2);

    std::vector<uint64_t> witness = {1, 7, 13, 91};
    EXPECT_TRUE(r1cs.validate_witness(witness));
}

/**
 * @brief Test: Sparse matrix-vector multiplication.
 * 
 * Tests compute_Az, compute_Bz, compute_Cz.
 */
TEST_F(R1CSTest, SparseMatrixVector) {
    // 2x3 matrix with 3 non-zero entries
    // A = [[1, 0, 2],
    //      [0, 3, 0]]
    SparseEntry a_entries[] = {
        {0, 0, 1},  // A[0,0] = 1
        {0, 2, 2},  // A[0,2] = 2
        {1, 1, 3}   // A[1,1] = 3
    };
    
    SparseMatrix A = {a_entries, 3, 2, 3};
    SparseMatrix B = {nullptr, 0, 2, 3};  // Not used
    SparseMatrix C = {nullptr, 0, 2, 3};  // Not used

    R1CS r1cs(A, B, C, TEST_MODULUS);

    // Witness: z = [1, 5, 7]
    std::vector<uint64_t> witness = {1, 5, 7};
    
    auto Az = r1cs.compute_Az(witness);
    
    // A·z = [1*1 + 0*5 + 2*7, 0*1 + 3*5 + 0*7] = [15, 15]
    ASSERT_EQ(Az.size(), 2);
    EXPECT_EQ(Az[0], 15);
    EXPECT_EQ(Az[1], 15);
}

/**
 * @brief Test: Empty R1CS (0 constraints).
 */
TEST_F(R1CSTest, EmptyConstraints) {
    SparseMatrix A = {nullptr, 0, 0, 3};
    SparseMatrix B = {nullptr, 0, 0, 3};
    SparseMatrix C = {nullptr, 0, 0, 3};

    R1CS r1cs(A, B, C, TEST_MODULUS);

    EXPECT_EQ(r1cs.num_constraints(), 0);
    
    // Any witness should be valid (no constraints to violate)
    std::vector<uint64_t> witness = {1, 42, 100};
    EXPECT_TRUE(r1cs.validate_witness(witness));
}

/**
 * @brief Test: Witness length mismatch.
 */
TEST_F(R1CSTest, WitnessLengthMismatch) {
    SparseEntry a_entries[] = {{0, 1, 1}};
    SparseMatrix A = {a_entries, 1, 1, 4};
    SparseMatrix B = {nullptr, 0, 1, 4};
    SparseMatrix C = {nullptr, 0, 1, 4};

    R1CS r1cs(A, B, C, TEST_MODULUS);

    // Witness too short
    std::vector<uint64_t> short_witness = {1, 2, 3};
    EXPECT_THROW(r1cs.validate_witness(short_witness), std::invalid_argument);

    // Witness too long
    std::vector<uint64_t> long_witness = {1, 2, 3, 4, 5};
    EXPECT_THROW(r1cs.validate_witness(long_witness), std::invalid_argument);
}

/**
 * @brief Test: First witness element must be 1.
 */
TEST_F(R1CSTest, WitnessFirstElementCheck) {
    SparseMatrix A = {nullptr, 0, 0, 3};
    SparseMatrix B = {nullptr, 0, 0, 3};
    SparseMatrix C = {nullptr, 0, 0, 3};

    R1CS r1cs(A, B, C, TEST_MODULUS);

    // First element is not 1
    std::vector<uint64_t> invalid_witness = {2, 42, 100};
    EXPECT_THROW(r1cs.validate_witness(invalid_witness), std::invalid_argument);
}

/**
 * @brief Test: Dimension mismatch detection.
 */
TEST_F(R1CSTest, DimensionMismatch) {
    SparseEntry a_entries[] = {{0, 0, 1}};
    SparseEntry b_entries[] = {{0, 0, 1}};
    
    SparseMatrix A = {a_entries, 1, 1, 3};
    SparseMatrix B = {b_entries, 1, 2, 3};  // Different row count
    SparseMatrix C = {nullptr, 0, 1, 3};

    EXPECT_THROW(R1CS(A, B, C, TEST_MODULUS), std::invalid_argument);
}

/**
 * @brief Test: Modular arithmetic overflow.
 * 
 * Tests that large values are correctly reduced modulo q.
 */
TEST_F(R1CSTest, ModularArithmetic) {
    // Single constraint: a * b = c
    SparseEntry a_entries[] = {{0, 1, 1}};
    SparseEntry b_entries[] = {{0, 2, 1}};
    SparseEntry c_entries[] = {{0, 3, 1}};

    SparseMatrix A = {a_entries, 1, 1, 4};
    SparseMatrix B = {b_entries, 1, 1, 4};
    SparseMatrix C = {c_entries, 1, 1, 4};

    R1CS r1cs(A, B, C, TEST_MODULUS);

    // Test: (q-1) * (q-1) mod q = (q^2 - 2q + 1) mod q = 1
    // But we need to compute this correctly using NTL
    uint64_t a = TEST_MODULUS - 1;
    uint64_t b = TEST_MODULUS - 1;
    
    // Compute c = (a * b) % q using NTL
    NTL::ZZ_p a_zp, b_zp, c_zp;
    NTL::conv(a_zp, static_cast<long>(a));
    NTL::conv(b_zp, static_cast<long>(b));
    c_zp = a_zp * b_zp;
    unsigned long c_ul = 0;
    NTL::ZZ c_zz;
    NTL::conv(c_zz, c_zp);
    NTL::conv(c_ul, c_zz);
    uint64_t c = c_ul;

    std::vector<uint64_t> witness = {1, a, b, c};
    EXPECT_TRUE(r1cs.validate_witness(witness));
    
    // Also test that c should equal 1
    EXPECT_EQ(c, 1);
}

/**
 * @brief Test: Linear combination constraint.
 * 
 * Constraint: (a + 2b) * c = d
 */
TEST_F(R1CSTest, LinearCombination) {
    // (a + 2b) * c = d
    // Witness: z = [1, a, b, c, d]
    SparseEntry a_entries[] = {
        {0, 1, 1},  // A[0,1] = 1 (select a)
        {0, 2, 2}   // A[0,2] = 2 (select 2*b)
    };
    SparseEntry b_entries[] = {{0, 3, 1}};  // B[0,3] = 1 (select c)
    SparseEntry c_entries[] = {{0, 4, 1}};  // C[0,4] = 1 (select d)

    SparseMatrix A = {a_entries, 2, 1, 5};
    SparseMatrix B = {b_entries, 1, 1, 5};
    SparseMatrix C = {c_entries, 1, 1, 5};

    R1CS r1cs(A, B, C, TEST_MODULUS);

    // Test: (3 + 2*5) * 7 = 91
    // a=3, b=5, c=7, d=91
    std::vector<uint64_t> witness = {1, 3, 5, 7, 91};
    EXPECT_TRUE(r1cs.validate_witness(witness));

    // Test invalid: (3 + 2*5) * 7 ≠ 90
    std::vector<uint64_t> invalid = {1, 3, 5, 7, 90};
    EXPECT_FALSE(r1cs.validate_witness(invalid));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
