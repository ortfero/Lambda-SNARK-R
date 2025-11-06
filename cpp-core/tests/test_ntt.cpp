/**
 * @file test_ntt.cpp
 * @brief Unit tests for NTT.
 */

#include "lambda_snark/ntt.h"
#include <gtest/gtest.h>
#include <vector>

class NttTest : public ::testing::Test {
protected:
    void SetUp() override {
        q = 12289;  // Prime: q = 1 + 2^12 * 3 (NTT-friendly)
        n = 256;    // Power of 2
        
        ctx = ntt_context_create(q, n);
        ASSERT_NE(ctx, nullptr);
    }
    
    void TearDown() override {
        ntt_context_free(ctx);
    }
    
    uint64_t q;
    uint32_t n;
    NttContext* ctx = nullptr;
};

TEST_F(NttTest, CreateAndFree) {
    EXPECT_NE(ctx, nullptr);
}

TEST_F(NttTest, ForwardNttBasic) {
    std::vector<uint64_t> coeffs(n, 1);  // All ones
    
    int result = ntt_forward(ctx, coeffs.data(), n);
    EXPECT_EQ(result, 0);
}

TEST_F(NttTest, InverseNttBasic) {
    std::vector<uint64_t> evals(n, 1);
    
    int result = ntt_inverse(ctx, evals.data(), n);
    EXPECT_EQ(result, 0);
}

TEST_F(NttTest, ForwardInverseIdentity) {
    std::vector<uint64_t> original = {1, 2, 3, 4, 0, 0, 0, 0};
    original.resize(n, 0);
    
    std::vector<uint64_t> transformed = original;
    
    ntt_forward(ctx, transformed.data(), n);
    ntt_inverse(ctx, transformed.data(), n);
    
    // After forward + inverse, should recover original (up to scaling)
    // TODO: Implement proper test once NTT is functional
}

TEST_F(NttTest, PointwiseMultiplication) {
    std::vector<uint64_t> a(n, 2);
    std::vector<uint64_t> b(n, 3);
    std::vector<uint64_t> result(n);
    
    ntt_mul_pointwise(ctx, result.data(), a.data(), b.data(), n);
    
    // 2 * 3 = 6 for all elements
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(result[i], 6);
    }
}

TEST_F(NttTest, NullPointerHandling) {
    std::vector<uint64_t> dummy(n, 0);
    
    EXPECT_EQ(ntt_forward(nullptr, dummy.data(), n), -1);
    EXPECT_EQ(ntt_forward(ctx, nullptr, n), -1);
    
    ntt_context_free(nullptr);  // Should not crash
}
