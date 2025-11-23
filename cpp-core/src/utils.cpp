/**
 * @file utils.cpp
 * @brief Utility functions (sampling, serialization, etc.).
 * 
 * @copyright Copyright (c) 2025 URPKS Contributors
 * @license Apache-2.0 OR MIT
 */

#include "lambda_snark/types.h"
#include "lambda_snark/utils.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

struct GaussianTable {
    std::vector<uint64_t> cdf;
};

constexpr long double kTailCutoff = 12.0L;

GaussianTable build_cdf(double sigma) noexcept {
    GaussianTable table;

    const long double sigma_ld = static_cast<long double>(sigma);
    const long double sigma_sq = sigma_ld * sigma_ld;

    // Tail bound chosen so that exp(-B^2/(2*sigma^2)) ≈ 2^{-72}
    long double bound = std::ceil(kTailCutoff * sigma_ld);
    if (bound < 8.0L) {
        bound = 8.0L;  // ensure minimal support for stability
    }

    const std::size_t max_index = static_cast<std::size_t>(bound);
    std::vector<long double> weights(max_index + 1, 0.0L);

    long double sum = 0.0L;
    for (std::size_t k = 0; k <= max_index; ++k) {
        const long double kk = static_cast<long double>(k) * static_cast<long double>(k);
        const long double exponent = -kk / (2.0L * sigma_sq);
        long double weight = std::exp(exponent);
        if (k > 0) {
            weight *= 2.0L;  // account for ±k
        }
        weights[k] = weight;
        sum += weight;
    }

    table.cdf.resize(max_index + 1, 0);
    if (sum == 0.0L) {
        table.cdf[max_index] = std::numeric_limits<uint64_t>::max();
        return table;
    }

    const long double scale = static_cast<long double>(std::numeric_limits<uint64_t>::max()) / sum;
    long double cumulative = 0.0L;
    for (std::size_t k = 0; k <= max_index; ++k) {
        cumulative += weights[k];
        long double value = cumulative * scale;
        if (value >= static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
            table.cdf[k] = std::numeric_limits<uint64_t>::max();
        } else if (value <= 0.0L) {
            table.cdf[k] = 0;
        } else {
            table.cdf[k] = static_cast<uint64_t>(value);
        }
    }

    table.cdf.back() = std::numeric_limits<uint64_t>::max();
    return table;
}

uint64_t random_u64(std::random_device& rd) noexcept {
    uint64_t value = 0;
    constexpr unsigned int chunk_bits = std::numeric_limits<unsigned int>::digits;
    unsigned int produced = 0;

    while (produced < 64) {
        const unsigned int take = (64 - produced < chunk_bits) ? (64 - produced) : chunk_bits;
        value <<= take;
        const unsigned int sample = rd();
        const uint64_t mask = (take == 64) ? std::numeric_limits<uint64_t>::max()
                                           : ((1ULL << take) - 1ULL);
        value |= static_cast<uint64_t>(sample) & mask;
        produced += take;
    }

    return value;
}

int64_t sample_single(const GaussianTable& table, std::random_device& rd) noexcept {
    const uint64_t u = random_u64(rd);

    uint32_t chosen = static_cast<uint32_t>(table.cdf.size() - 1);
    uint64_t found = 0;

    for (std::size_t k = 0; k < table.cdf.size(); ++k) {
        const uint64_t ge_mask = static_cast<uint64_t>(table.cdf[k] >= u);
        const uint64_t select_mask = ge_mask & (1ULL ^ found);
        const uint32_t mask32 = static_cast<uint32_t>(-static_cast<int32_t>(select_mask));
        const uint32_t candidate = static_cast<uint32_t>(k);

        chosen = (chosen & ~mask32) | (candidate & mask32);
        found |= select_mask;
    }

    const uint64_t sign_bit = random_u64(rd) & 1ULL;
    const uint64_t nonzero = static_cast<uint64_t>(chosen != 0);
    const uint64_t sign_mask = sign_bit & nonzero;

    const int64_t magnitude = static_cast<int64_t>(chosen);
    const int64_t neg = -magnitude;
    const int64_t mask = -static_cast<int64_t>(sign_mask);
    const int64_t signed_val = (magnitude & ~mask) | (neg & mask);

    return signed_val;
}

}  // namespace

extern "C" {

/**
 * @brief Sample from discrete Gaussian distribution (stub).
 * 
 * TODO: Implement constant-time Gaussian sampling.
 */
int sample_gaussian(uint64_t* output, size_t len, double sigma) noexcept {
    if (!output || len == 0 || !(sigma > 0.0) || !std::isfinite(sigma)) {
        return -1;
    }

    const GaussianTable table = build_cdf(sigma);
    std::random_device rd;

    for (size_t i = 0; i < len; ++i) {
        const int64_t sample = sample_single(table, rd);
        output[i] = static_cast<uint64_t>(sample);  // two's-complement encoding for signed sample
    }

    return 0;
}

}  // extern "C"
