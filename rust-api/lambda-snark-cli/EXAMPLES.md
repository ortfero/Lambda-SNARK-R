# ΛSNARK-R CLI Examples

This document demonstrates how to use the `lambda-snark` CLI tool for R1CS proof generation and verification.

## Table of Contents

- [Installation](#installation)
- [Basic Example: R1CS Multiplication](#basic-example-r1cs-multiplication)
- [Command Reference](#command-reference)
- [Advanced Examples (Coming Soon)](#advanced-examples-coming-soon)

## Installation

```bash
cd rust-api/lambda-snark-cli
cargo build --release
```

The binary will be available at `target/release/lambda-snark`.

## Basic Example: R1CS Multiplication

Demonstrates end-to-end proof generation for the statement **7 × 13 = 91**.

### Quick Start

```bash
# Run with default seed (42)
cargo run -- r1cs-example

# Run with verbose output
cargo run -- r1cs-example --verbose

# Run with custom seed
cargo run -- r1cs-example --seed 12345
```

### Expected Output (Concise Mode)

```
╔═══════════════════════════════════════════════════════════╗
║       ΛSNARK-R: R1CS Proof Example (TV-R1CS-1)           ║
╚═══════════════════════════════════════════════════════════╝

📋 Step 1: Building R1CS circuit for multiplication
   Statement: 7 × 13 = 91

   ✓ Circuit built: 1 constraints, 4 variables, modulus=17592186044417

🔐 Step 2: Preparing witness and public inputs
   Public:  constant=1, x=7
   Private: y=13, result=91
   ✓ Witness satisfies constraints

⚙️  Step 3: Initializing LWE commitment scheme
   LWE parameters: n=4096, q=17592186044417 (2^44+1), σ=3.19
   Security: 128-bit post-quantum (Module-LWE)

🔨 Step 4: Generating R1CS proof (seed=42)
   ✓ Proof generated successfully
   Proof size: ~216 bytes

✅ Step 5: Verifying R1CS proof
   ✓ Proof VALID ✓

╔═══════════════════════════════════════════════════════════╗
║  SUCCESS: Proof verified! 7 × 13 = 91 is proven correct  ║
╚═══════════════════════════════════════════════════════════╝

Summary:
  - Circuit:       1 constraints, 4 variables
  - Public inputs: 2 (constant=1, x=7)
  - Proof size:    ~216 bytes
  - Soundness:     ε ≤ 2^-48 (two Fiat-Shamir challenges)
  - Security:      128-bit quantum (Module-LWE)
```

### Verbose Mode Details

Using `--verbose` flag provides additional information:

- **Variable allocation**: Shows z_0, z_1, z_2, z_3 assignments
- **Constraint structure**: Displays the R1CS constraint equation
- **Full witness**: Complete witness vector `[1, 7, 13, 91]`
- **Challenge values**: Fiat-Shamir challenges α and β
- **Polynomial evaluations**: Q(α), Q(β), A_z(α), B_z(α), C_z(α), etc.

### Circuit Structure

The multiplication circuit consists of:

```
Variables:
  z_0 = 1        (constant, public)
  z_1 = x = 7    (public input)
  z_2 = y = 13   (private witness)
  z_3 = result   (private, but verifiable)

Constraint:
  z_1 × z_2 = z_3
  ⟺  x × y = result
  ⟺  7 × 13 = 91
```

### Security Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Ring degree (n)** | 4096 | Polynomial degree (SEAL requires power-of-2) |
| **Modulus (q)** | 2^44 + 1 | Prime modulus (17592186044417) |
| **Module rank (k)** | 2 | Number of polynomial components |
| **Gaussian width (σ)** | 3.19 | Noise distribution parameter |
| **Security level** | 128-bit | Post-quantum security (Module-LWE) |
| **Soundness error (ε)** | ≤ 2^-48 | Two independent Fiat-Shamir challenges |

## Command Reference

### `r1cs-example`

Run a complete R1CS proof example (multiplication circuit).

**Options:**
- `--seed <u64>` — Random seed for proof generation (default: 42)
- `--verbose` — Show detailed information about proof components

**Examples:**
```bash
# Default run
lambda-snark r1cs-example

# Verbose with custom seed
lambda-snark r1cs-example --seed 999 --verbose
```

### `info`

Display version and build information.

```bash
lambda-snark info
```

**Output:**
```
ΛSNARK-R v0.1.0

Architecture: Hybrid (C++ Core + Rust API)
Target: x86_64-unknown-linux-gnu

Features:
  - Post-quantum security (Module-LWE/SIS)
  - Succinct proofs (O(log M) size)
  - Zero-knowledge

Status: ⚠️  Pre-alpha (NOT FOR PRODUCTION)
License: Apache-2.0 OR MIT
```

## Advanced Examples (Coming Soon)

### M4.8: Complex Circuits

- **Range Proof**: Prove that a value is in [0, 2^k) without revealing the value
- **Multiple Gates**: Circuits with multiple multiplication constraints
- **Benchmarks**: Performance analysis for varying constraint sizes

### M5: Optimizations

- **FFT-based Polynomial Operations**: O(m log m) instead of O(m²)
- **Zero-Knowledge Extension**: Blinded polynomials with prove_r1cs_zk()

## Implementation Details

### Proof Structure

```rust
pub struct ProofR1CS {
    commitment_q: Commitment,       // LWE commitment to Q(X)
    challenge_alpha: Challenge,     // First Fiat-Shamir challenge α
    challenge_beta: Challenge,      // Second Fiat-Shamir challenge β
    q_alpha: u64,                   // Q(α) evaluation
    q_beta: u64,                    // Q(β) evaluation
    a_z_alpha: u64,                 // A_z(α) evaluation
    b_z_alpha: u64,                 // B_z(α) evaluation
    c_z_alpha: u64,                 // C_z(α) evaluation
    a_z_beta: u64,                  // A_z(β) evaluation
    b_z_beta: u64,                  // B_z(β) evaluation
    c_z_beta: u64,                  // C_z(β) evaluation
    opening_alpha: Opening,         // Opening proof at α
    opening_beta: Opening,          // Opening proof at β
}
```

### Verification Equations

The verifier checks two independent equations:

1. **At challenge α**:
   ```
   Q(α) · Z_H(α) = A_z(α) · B_z(α) - C_z(α)
   ```

2. **At challenge β**:
   ```
   Q(β) · Z_H(β) = A_z(β) · B_z(β) - C_z(β)
   ```

Where `Z_H(X) = ∏_{i=0}^{m-1} (X - i)` is the vanishing polynomial.

### Soundness Analysis

- **Single challenge**: ε ≤ 1/|F| (trivial for large fields)
- **Two challenges**: ε ≤ 2 · deg(Q) / |F| ≈ 2^-48 for deg(Q) < 1000, |F| ≈ 2^44

The dual-challenge construction prevents polynomial forgery attacks.

## Performance Notes

### Current Implementation (Naïve)

- **Polynomial operations**: O(m²) Lagrange interpolation
- **Suitable for**: m ≤ 1000 constraints
- **Typical runtime**: ~50ms for m=100 constraints

### Future Optimization (M5.1)

- **FFT-based operations**: O(m log m) via NTT
- **Target speedup**: 1000× for m = 2^20
- **Requires**: NTT-friendly modulus (e.g., q = 2^64 - 2^32 + 1)

## Troubleshooting

### "Invalid modulus" Error

**Problem**: Modulus too small for LWE security.

**Solution**: Ensure modulus ≥ 2^24. The default example uses q = 2^44 + 1.

### "non-standard poly_modulus_degree" Error

**Problem**: SEAL library requires power-of-2 ring degree.

**Solution**: Use n ∈ {2048, 4096, 8192, 16384, 32768}.

### Verification Fails

**Problem**: Public inputs mismatch.

**Solution**: Ensure `public_inputs` matches the first `l` elements of the witness vector. Use `r1cs.public_inputs(&witness)` to extract them correctly.

## References

- **[R1CS Specification](../../docs/vdad/r1cs.md)**: Detailed protocol description
- **[Security Analysis](../../docs/vdad/security.md)**: Post-quantum security proofs
- **[API Documentation](https://docs.rs/lambda-snark)**: Rust API reference

## License

Apache-2.0 OR MIT
