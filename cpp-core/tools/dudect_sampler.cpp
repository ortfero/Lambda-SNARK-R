#include "lambda_snark/utils.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

std::filesystem::path locate_repo_root() {
    auto current = std::filesystem::current_path();
    for (int depth = 0; depth < 5; ++depth) {
        const bool has_readme = std::filesystem::exists(current / "README.md");
        const bool has_security = std::filesystem::exists(current / "SECURITY.md");
        if (has_readme && has_security) {
            return current;
        }
        if (!current.has_parent_path()) {
            break;
        }
        current = current.parent_path();
    }
    return std::filesystem::current_path();
}

struct Moments {
    double mean;
    double variance;
    std::size_t count;
};

Moments compute_moments(const std::vector<double>& samples) {
    Moments stats{0.0, 0.0, samples.size()};
    if (samples.empty()) {
        return stats;
    }

    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean = sum / static_cast<double>(samples.size());

    double variance_acc = 0.0;
    for (const double value : samples) {
        const double diff = value - stats.mean;
        variance_acc += diff * diff;
    }

    if (samples.size() > 1) {
        stats.variance = variance_acc / static_cast<double>(samples.size() - 1);
    } else {
        stats.variance = 0.0;
    }
    return stats;
}

double welch_t_stat(const Moments& a, const Moments& b) {
    if (a.count == 0 || b.count == 0) {
        return 0.0;
    }

    const double denom = std::sqrt((a.variance / static_cast<double>(a.count)) +
                                   (b.variance / static_cast<double>(b.count)));
    if (denom == 0.0) {
        return 0.0;
    }

    return (a.mean - b.mean) / denom;
}

void write_report(const Moments& zero, const Moments& one, double t_stat, std::size_t total_samples,
                  std::size_t trace_len, const std::filesystem::path& output_path) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path);

    out << "# dudect sampler report\n\n";
    out << "- samples_per_class_zero: " << zero.count << "\n";
    out << "- samples_per_class_one: " << one.count << "\n";
    out << "- total_samples: " << total_samples << "\n";
    out << "- trace_length: " << trace_len << " coefficients\n";
    out << "- welch_t_statistic: " << std::fixed << std::setprecision(4) << t_stat << "\n\n";

    auto emit_table_row = [&out](const std::string& label, const Moments& stats) {
        const double stddev = std::sqrt(std::max(0.0, stats.variance));
        out << "| " << label << " | " << stats.count << " | " << std::setprecision(2) << std::fixed
            << stats.mean << " | " << stddev << " |\n";
    };

    out << "| class | samples | mean_ns | stddev_ns |\n";
    out << "|-------|---------|---------|-----------|\n";
    emit_table_row("0", zero);
    emit_table_row("1", one);

    out << "\n";
    out << "> Threshold guidance: |t| < 4.5 is typically treated as constant-time by dudect.\n";
}

}  // namespace

int main() {
    constexpr std::size_t kTraceLength = 64;
    constexpr std::size_t kSampleCount = 20000;
    constexpr double kSigma = 3.2;

    std::vector<uint64_t> buffer(kTraceLength, 0);
    std::vector<double> class_zero;
    std::vector<double> class_one;
    class_zero.reserve(kSampleCount / 2);
    class_one.reserve(kSampleCount / 2);

    for (std::size_t i = 0; i < kSampleCount; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const int status = sample_gaussian(buffer.data(), buffer.size(), kSigma);
        const auto end = std::chrono::steady_clock::now();

        if (status != 0) {
            std::cerr << "sample_gaussian returned error at iteration " << i << "\n";
            return 1;
        }

        const double nanos = std::chrono::duration<double, std::nano>(end - start).count();
        const bool classification = (buffer[0] & 1ULL) != 0ULL;
        if (classification) {
            class_one.push_back(nanos);
        } else {
            class_zero.push_back(nanos);
        }
    }

    const Moments zero_stats = compute_moments(class_zero);
    const Moments one_stats = compute_moments(class_one);
    const double t_stat = welch_t_stat(zero_stats, one_stats);

    const auto repo_root = locate_repo_root();
    const std::filesystem::path output_path = repo_root / "artifacts" / "dudect" /
                                              "gaussian_sampler_report.md";
    write_report(zero_stats, one_stats, t_stat, kSampleCount, kTraceLength, output_path);

    std::cout << "dudect sampler t-statistic: " << std::fixed << std::setprecision(4) << t_stat
              << "\nReport written to " << output_path << "\n";

    return 0;
}
