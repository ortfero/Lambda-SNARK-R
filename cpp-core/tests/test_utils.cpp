/**
 * @file test_utils.cpp
 * @brief Tests for security-sensitive utility helpers.
 */

#include "lambda_snark/types.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" {
int sample_gaussian(uint64_t* output, size_t len, double sigma) noexcept;
}

namespace {

inline int64_t decode_sample(uint64_t word) {
    return static_cast<int64_t>(word);
}

}  // namespace

TEST(SampleGaussian, RejectsInvalidInputs) {
    std::vector<uint64_t> buffer(16, 0);

    EXPECT_EQ(sample_gaussian(nullptr, buffer.size(), 3.2), -1);
    EXPECT_EQ(sample_gaussian(buffer.data(), 0, 3.2), -1);
    EXPECT_EQ(sample_gaussian(buffer.data(), buffer.size(), 0.0), -1);
    EXPECT_EQ(sample_gaussian(buffer.data(), buffer.size(), std::numeric_limits<double>::infinity()), -1);
}

TEST(SampleGaussian, EmpiricalMomentsWithinBounds) {
    constexpr size_t kSamples = 4096;
    constexpr double kSigma = 3.2;

    std::vector<uint64_t> buffer(kSamples, 0);
    ASSERT_EQ(sample_gaussian(buffer.data(), buffer.size(), kSigma), 0);

    double mean = 0.0;
    double m2 = 0.0;
    size_t positives = 0;
    size_t negatives = 0;

    for (size_t i = 0; i < buffer.size(); ++i) {
        const int64_t value = decode_sample(buffer[i]);
        const double x = static_cast<double>(value);
        const double delta = x - mean;
        mean += delta / static_cast<double>(i + 1);
        m2 += delta * (x - mean);
        if (value > 0) {
            ++positives;
        } else if (value < 0) {
            ++negatives;
        }
    }

    const double variance = m2 / static_cast<double>(buffer.size() - 1);
    const double sigma_empirical = std::sqrt(variance);

    EXPECT_NEAR(mean, 0.0, 0.5);
    EXPECT_NEAR(sigma_empirical, kSigma, 0.8);

    EXPECT_GT(positives, buffer.size() / 4);
    EXPECT_GT(negatives, buffer.size() / 4);
    const auto imbalance = std::llabs(static_cast<long long>(positives) - static_cast<long long>(negatives));
    EXPECT_LT(imbalance, static_cast<long long>(buffer.size() / 5));
}
