/**
 * @file test_commitment.cpp
 * @brief UnitTEST_F(CommitmentTest, CommitDeterministic) {
    // Deterministic encryption: same seed → same commitment
    uint64_t message[] = {1, 2, 3};
    size_t msg_len = 3;
    uint64_t seed = 0xDEADBEEF;ts for LWE commitment.
 */

#include "lambda_snark/commitment.h"
#include <gtest/gtest.h>

class CommitmentTest : public ::testing::Test {
protected:
    void SetUp() override {
        params.profile = PROFILE_RING_B;
        params.security_level = 128;
        params.modulus = 12289;
        params.ring_degree = 4096;  // SEAL requires power-of-2 >= 1024
        params.module_rank = 2;
        params.sigma = 3.19;
        
        ctx = lwe_context_create(&params);
        ASSERT_NE(ctx, nullptr);
    }
    
    void TearDown() override {
        lwe_context_free(ctx);
    }
    
    PublicParams params;
    LweContext* ctx = nullptr;
};

TEST_F(CommitmentTest, CreateAndFree) {
    // Context already created in SetUp
    EXPECT_NE(ctx, nullptr);
}

TEST_F(CommitmentTest, CommitBasic) {
    uint64_t message[] = {1, 2, 3, 4};
    size_t msg_len = 4;
    
    auto comm = lwe_commit(ctx, message, msg_len, 0x1234);
    ASSERT_NE(comm, nullptr);
    EXPECT_GT(comm->len, 0);
    EXPECT_NE(comm->data, nullptr);
    
    lwe_commitment_free(comm);
}

TEST_F(CommitmentTest, CommitBinding) {
    // Binding property: different messages → different commitments (with high probability)
    uint64_t msg1[] = {1, 2, 3};
    uint64_t msg2[] = {4, 5, 6};
    size_t msg_len = 3;
    
    auto comm1 = lwe_commit(ctx, msg1, msg_len, 0);
    auto comm2 = lwe_commit(ctx, msg2, msg_len, 0);
    
    ASSERT_NE(comm1, nullptr);
    ASSERT_NE(comm2, nullptr);
    ASSERT_EQ(comm1->len, comm2->len);
    
    // Different messages → commitments should differ (with overwhelming probability)
    bool different = false;
    for (size_t i = 0; i < comm1->len; ++i) {
        if (comm1->data[i] != comm2->data[i]) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different) << "Commitments to different messages should differ";
    
    lwe_commitment_free(comm1);
    lwe_commitment_free(comm2);
}

TEST_F(CommitmentTest, CommitDifferentMessages) {
    uint64_t msg1[] = {1, 2, 3};
    uint64_t msg2[] = {4, 5, 6};
    uint64_t seed = 0x1234;
    
    auto comm1 = lwe_commit(ctx, msg1, 3, seed);
    auto comm2 = lwe_commit(ctx, msg2, 3, seed);
    
    ASSERT_NE(comm1, nullptr);
    ASSERT_NE(comm2, nullptr);
    
    // Different messages => different commitments (with high probability)
    bool different = false;
    for (size_t i = 0; i < std::min(comm1->len, comm2->len); ++i) {
        if (comm1->data[i] != comm2->data[i]) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different) << "Commitments should differ for different messages";
    
    lwe_commitment_free(comm1);
    lwe_commitment_free(comm2);
}

TEST_F(CommitmentTest, NullPointerHandling) {
    // Null context
    auto comm = lwe_commit(nullptr, nullptr, 0, 0);
    EXPECT_EQ(comm, nullptr);
    
    // Null message
    comm = lwe_commit(ctx, nullptr, 10, 0);
    EXPECT_EQ(comm, nullptr);
    
    // Free null commitment (should not crash)
    lwe_commitment_free(nullptr);
}

// TODO: Add tests for verify_opening once implemented
// TODO: Add tests for linear_combine once implemented
