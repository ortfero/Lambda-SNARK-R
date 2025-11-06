# Welcome to ΛSNARK-R

ΛSNARK-R is a **post-quantum SNARK** system based on **Module-LWE/SIS** hardness assumptions, providing succinct zero-knowledge proofs for R1CS arithmetic circuits over cyclotomic rings.

## 🎯 Key Features

- **Post-Quantum Security**: Resistant to quantum attacks (Shor's algorithm)
- **Succinct Proofs**: O(log M) proof size for M constraints
- **Zero-Knowledge**: Statistical or computational hiding
- **Production-Ready**: Constant-time, memory-safe, formally verified

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/URPKS/lambda-snark-r.git
cd lambda-snark-r

# Setup environment
make setup

# Build
make build

# Run tests
make test
```

### Hello World

```rust
use lambda_snark::{Params, Profile, SecurityLevel, setup, prove, verify};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parameters: 128-bit security, Ring profile
    let params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 256,    // Ring degree
            k: 2,      // Module rank
            q: 12289,  // Prime modulus
            sigma: 3.19,  // Gaussian parameter
        },
    );
    
    let (pk, vk) = setup(params)?;
    
    // Prove: 7 * 13 = 91
    let public = vec![1, 91];
    let witness = vec![7, 13];
    
    let proof = prove(&pk, &public, &witness)?;
    let valid = verify(&vk, &public, &proof)?;
    
    assert!(valid);
    Ok(())
}
```

## 📚 Learn More

- **[Specification](spec/specification.md)**: Formal protocol definition
- **[Architecture](architecture/overview.md)**: System design
- **[User Guide](guide/quickstart.md)**: Tutorials and examples
- **[API Reference](api/rust.md)**: Detailed API documentation

## 🔒 Security

**Status**: Research-grade (⚠️ **DO NOT USE IN PRODUCTION** until audited)

- **Target Audit Date**: Q2 2026 (Trail of Bits)
- **Security Level**: 128-bit post-quantum
- **Threat Model**: Quantum adversary with QROM access

See [Security Policy](../SECURITY.md) for vulnerability reporting.

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

- 💬 [Discussions](https://github.com/URPKS/lambda-snark-r/discussions)
- 🐛 [Issues](https://github.com/URPKS/lambda-snark-r/issues)
- 📧 Email: dev@lambda-snark.org

## 📄 License

Licensed under **Apache-2.0 OR MIT** (dual-license).

---

**ΛSNARK-R** — Building the future of post-quantum zero-knowledge cryptography.
