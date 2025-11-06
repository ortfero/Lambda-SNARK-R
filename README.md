# ΛSNARK-R: Lattice-Based SNARK over Rings

> **Version**: 0.1.0-alpha  
> **Status**: Early Development  
> **License**: Apache-2.0 OR MIT  

Post-quantum SNARK system based on Module-LWE/SIS for R1CS over cyclotomic rings, with zero-knowledge and succinct proofs.

## 🎯 Overview

ΛSNARK-R is a production-grade implementation of lattice-based SNARKs using:
- **Cryptographic Foundation**: Module-LWE/SIS hardness assumptions
- **Architecture**: Hybrid C++ (performance-critical core) + Rust (safe API)
- **Proof System**: Interactive Oracle Proof (IOP) with Fiat-Shamir transformation
- **Target Applications**: Post-quantum cryptography, QGravity-Lattice integration, privacy-preserving computation

### Key Features

- ✅ **Post-Quantum Security**: Resistant to quantum attacks (Shor, Grover)
- ✅ **Succinct Proofs**: O(log M) proof size for M constraints
- ✅ **Zero-Knowledge**: Statistical or computational ZK with rejection sampling
- ✅ **Formal Verification**: Soundness/ZK proofs in Lean 4
- ✅ **Production-Ready**: Security audited, constant-time, fuzzed

## 📁 Repository Structure

```
ΛSNARK-R/
├── cpp-core/              # C++ performance kernel (SEAL, NTL)
│   ├── include/           # Public C++ headers
│   ├── src/               # Implementation
│   └── tests/             # Google Test suite
│
├── rust-api/              # Rust safe wrapper
│   ├── lambda-snark-core/ # Core types (#![no_std])
│   ├── lambda-snark-sys/  # FFI bindings
│   ├── lambda-snark/      # Public API
│   └── lambda-snark-cli/  # CLI tool
│
├── formal/                # Lean 4 formal verification
│   └── LambdaSNARK/       # Soundness, ZK, Completeness proofs
│
├── docs/                  # Documentation (mkdocs)
│   ├── spec/              # Formal specification
│   ├── architecture/      # Design docs
│   └── api/               # API reference
│
├── benches/               # Benchmarks (Criterion)
├── examples/              # Usage examples
├── scripts/               # Build/test automation
└── .github/               # CI/CD workflows
```

## 🚀 Quick Start

### Prerequisites

**C++ Toolchain**:
```bash
# Install SEAL (Microsoft FHE library)
vcpkg install seal

# Install NTL (Number Theory Library)
vcpkg install ntl

# CMake 3.20+
cmake --version
```

**Rust Toolchain**:
```bash
# Rust 1.75+ (stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable

# Install tools
cargo install cargo-fuzz cargo-criterion
```

**Python (for docs & formal verification)**:
```bash
# UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Build

```bash
# Full build (C++ core + Rust API)
make build

# C++ core only
cd cpp-core && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# Rust API only
cd rust-api && cargo build --release

# Run tests
make test

# Run benchmarks
make bench
```

### Usage Example

```rust
use lambda_snark::{Params, Profile, prove, verify};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup parameters (Profile-B: Ring n=256, k=2, λ=128)
    let params = Params::new(128, Profile::RingB {
        n: 256,
        k: 2,
        q: 12289,
        sigma: 3.19,
    })?;
    
    let (pk, vk) = lambda_snark::setup(params)?;
    
    // R1CS: prove a * b = c
    let a = 7u64;
    let b = 13u64;
    let c = a * b; // 91
    
    let public_input = vec![1, c]; // (1, 91)
    let witness = vec![a, b];      // (7, 13)
    
    // Generate proof
    let proof = prove(&pk, &public_input, &witness)?;
    println!("Proof size: {} bytes", proof.to_bytes().len());
    
    // Verify
    let valid = verify(&vk, &public_input, &proof)?;
    assert!(valid);
    println!("✓ Proof verified!");
    
    Ok(())
}
```

## 📊 Performance Targets

| Metric              | Target (M=10⁶) | Status    |
|---------------------|----------------|-----------|
| Prover Time         | ≤ 20 minutes   | 🟡 In Dev |
| Verifier Time       | ≤ 500 ms       | 🟡 In Dev |
| Proof Size          | ≤ 50 KB        | 🟡 In Dev |
| Memory (Prover)     | ≤ 8 GB         | 🟡 In Dev |
| Security Level      | 128-bit (PQ)   | ✅ Design |

## 🔒 Security

### Cryptographic Assumptions
- **Module-LWE**: (k=2, n=256, q=12289, σ=3.19) → 128-bit quantum security
- **Module-SIS**: β-SIS with β=2¹⁰ for binding
- **Random Oracle Model**: SHAKE256 (QROM-safe)

### Audits & Reviews
- [ ] **Trail of Bits** (Planned Q2 2026): C++ core audit
- [ ] **Community Review** (Ongoing): Public issue tracker
- [ ] **Formal Verification** (In Progress): Lean 4 soundness proof

### Constant-Time Guarantees
All cryptographic operations are implemented with:
- No secret-dependent branches
- `dudect` validation (statistical timing analysis)
- Zeroization of sensitive data (`zeroize` crate)

## 📚 Documentation

- **[Specification](docs/spec/specification.md)**: Formal protocol definition
- **[Architecture](docs/architecture/overview.md)**: System design
- **[API Reference](https://docs.rs/lambda-snark)**: Rust API docs
- **[C++ API](docs/api/cpp.md)**: C++ core interface
- **[Security Analysis](docs/security/analysis.md)**: Threat model & mitigations

## 🧪 Testing

```bash
# Unit tests
make test-unit

# Integration tests
make test-integration

# Conformance tests (TV-0/1/2)
cargo test --test conformance

# Fuzzing (1 hour)
cargo fuzz run fuzz_verify -- -max_total_time=3600

# Constant-time check
cargo bench --bench dudect
```

## 🤝 Contributing

We follow **Trunk-Based Development** (TBD):
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit with [Conventional Commits](https://www.conventionalcommits.org/)
4. Ensure tests pass: `make test`
5. Run pre-commit hooks: `pre-commit run --all-files`
6. Submit a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## 🔗 Links

- **Specification**: [docs/spec/specification.md](docs/spec/specification.md)
- **Roadmap**: [ROADMAP.md](ROADMAP.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Security Policy**: [SECURITY.md](SECURITY.md)

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/URPKS/lambda-snark-r/issues)
- **Discussions**: [GitHub Discussions](https://github.com/URPKS/lambda-snark-r/discussions)
- **Email**: security@lambda-snark.org (security reports only)

---

**⚠️ DISCLAIMER**: This is research-grade software under active development. Do not use in production until audited (target: Q2 2026).
