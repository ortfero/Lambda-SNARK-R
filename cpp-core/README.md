# C++ Performance Kernel

This directory contains the C++ performance-critical core of ΛSNARK-R, implementing:
- LWE/SIS-based commitments (via Microsoft SEAL)
- Number-Theoretic Transform (NTT) for cyclotomic rings
- Linear and multiplicative constraint checks

## Dependencies

Install via **vcpkg** (recommended):

```bash
# Clone vcpkg if not installed
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh

# Install dependencies (NTL is not packaged in vcpkg; install via system manager)
./vcpkg/vcpkg install seal gmp eigen3 libsodium gtest benchmark
```

Then install NTL separately:

```bash
# Linux (Ubuntu/Debian)
sudo apt-get install libntl-dev

# macOS (Homebrew)
brew install ntl

# From source (if needed)
curl -LO https://www.shoup.net/ntl/ntl-11.5.1.tar.gz
tar -xf ntl-11.5.1.tar.gz && cd ntl-11.5.1/src
./configure SHARED=on NTL_GMP_LIP=on && make -j$(nproc) && sudo make install
cd ../..
```

> **Note:** `vcpkg` builds `gmp` from source and requires system autotools (`autoconf`, `automake`, `libtool`). Install them before running `vcpkg install`.

Or install system packages:

```bash
# Ubuntu/Debian
sudo apt-get install libseal-dev libntl-dev libgmp-dev libeigen3-dev libsodium-dev autoconf automake libtool

# macOS
brew install seal ntl gmp eigen libsodium autoconf automake libtool
```

## Building

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build -j$(nproc)

# Run tests
cd build && ctest --output-on-failure

# Install (optional)
sudo cmake --install build
```

### Build Options

- `-DLAMBDA_SNARK_BUILD_TESTS=ON`: Build tests (default: ON)
- `-DLAMBDA_SNARK_BUILD_BENCHMARKS=ON`: Build benchmarks (default: ON)
- `-DLAMBDA_SNARK_USE_ASAN=ON`: Enable AddressSanitizer
- `-DLAMBDA_SNARK_USE_UBSAN=ON`: Enable UndefinedBehaviorSanitizer
- `-DLAMBDA_SNARK_NATIVE_ARCH=ON`: Compile for native CPU (default: ON)

## Testing

```bash
# All tests
make -C build test

# Specific test
./build/test_commitment
./build/test_ntt

# With sanitizers
cmake -B build-asan -DLAMBDA_SNARK_USE_ASAN=ON -DLAMBDA_SNARK_USE_UBSAN=ON
cmake --build build-asan
./build-asan/test_commitment
```

## Benchmarking

```bash
./build/bench_ntt --benchmark_min_time=5s
```

## API Overview

### LWE Commitment

```cpp
#include <lambda_snark/commitment.h>

// Setup
PublicParams params = { /* ... */ };
LweContext* ctx = lwe_context_create(&params);

// Commit
uint64_t message[] = {1, 2, 3};
LweCommitment* comm = lwe_commit(ctx, message, 3, 0x1234);

// Cleanup
lwe_commitment_free(comm);
lwe_context_free(ctx);
```

### NTT

```cpp
#include <lambda_snark/ntt.h>

NttContext* ctx = ntt_context_create(q, n);
uint64_t coeffs[256] = { /* ... */ };

ntt_forward(ctx, coeffs, 256);
// ... operations in NTT domain ...
ntt_inverse(ctx, coeffs, 256);

ntt_context_free(ctx);
```

## Directory Structure

```
cpp-core/
├── CMakeLists.txt         # Build configuration
├── vcpkg.json             # Dependency manifest
├── include/               # Public headers
│   └── lambda_snark/
│       ├── types.h
│       ├── commitment.h
│       └── ntt.h
├── src/                   # Implementation
│   ├── commitment.cpp
│   ├── ntt.cpp
│   ├── lincheck.cpp
│   ├── mulcheck.cpp
│   ├── ffi.cpp
│   └── utils.cpp
├── tests/                 # Unit tests (Google Test)
│   ├── test_commitment.cpp
│   └── test_ntt.cpp
└── benches/               # Benchmarks (Google Benchmark)
    └── bench_ntt.cpp
```

## Notes

- **FFI Safety**: All public functions are `extern "C"` and `noexcept`.
- **Memory Management**: Caller must free returned pointers (e.g., `lwe_commitment_free`).
- **Constant-Time**: Critical functions (verification) use constant-time operations.
- **Stub Implementation**: Some functions are stubs pending full implementation.

## Next Steps

1. Implement full NTT with NTL integration
2. Add rejection sampling for Gaussian distribution
3. Implement homomorphic linear combination
4. Add comprehensive benchmarks
5. Security audit (Q2 2026)

See main [README.md](../README.md) for full project documentation.
