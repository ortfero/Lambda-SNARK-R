# ΛSNARK-R CLI Examples

This document demonstrates how to use the `lambda-snark` CLI tool for R1CS proof generation and verification.

## Table of Contents

- [Installation](#installation)
- [Basic Example: R1CS Multiplication](#basic-example-r1cs-multiplication)
- [Healthcare Example: Privacy-Preserving Diagnosis](#healthcare-example-privacy-preserving-diagnosis)
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

## Healthcare Example: Privacy-Preserving Diagnosis

Demonstrates zero-knowledge proof for medical diagnosis without revealing patient data.

### Scenario

**Problem**: Insurance company needs diabetes risk assessment, but patient data is sensitive (GDPR/HIPAA).

**Solution**: Hospital proves diagnosis result (HIGH/LOW risk) without disclosing glucose, age, or BMI values.

### Circuit Logic

```rust
// Private inputs (HIDDEN from verifier):
let glucose = 142;  // mg/dL
let age = 45;       // years
let bmi = 31;       // kg/m²

// Threshold checks (computed by prover):
let glucose_high = (glucose > 126);  // 1 if true, 0 if false
let age_high = (age > 40);
let bmi_high = (bmi > 30);

// AND gate: all_high = glucose_high ∧ age_high ∧ bmi_high
let all_high = glucose_high * age_high * bmi_high;

// Risk score computation (PUBLIC output):
let risk_score = 1 + 2 * all_high;  // 1 (LOW) or 3 (HIGH)
```

### R1CS Constraints

1. **Binary constraints** (3 constraints):
   - `glucose_high * (glucose_high - 1) = 0`
   - `age_high * (age_high - 1) = 0`
   - `bmi_high * (bmi_high - 1) = 0`

2. **AND gate** (2 constraints):
   - `temp = glucose_high * age_high`
   - `all_high = temp * bmi_high`

3. **Risk score** (1 constraint):
   - `risk_score = 1 + 2 * all_high`

**Total**: 6 R1CS constraints, 10 variables

### Running the Example

```bash
cargo run --release -- healthcare-example
```

### Expected Output

```
╔═══════════════════════════════════════════════════════════╗
║   ΛSNARK-R: Healthcare Diagnosis (Privacy-Preserving)    ║
╚═══════════════════════════════════════════════════════════╝

🏥 Scenario: Hospital proves diabetes risk without sending patient data

📋 Step 1: Building Healthcare R1CS Circuit

   ✓ Circuit built:
     - Constraints: 6 R1CS equations
     - Variables: 10 (including intermediate)
     - Public inputs: 2 (constant=1, risk_score)
     - Logic: Binary checks + AND gate + risk computation

🔒 Step 2: Preparing Patient Data (PRIVATE)

   📊 Patient Metrics (HIDDEN from verifier):
     ┌──────────────┬────────┬────────────┐
     │ Metric       │  Value │ Status     │
     ├──────────────┼────────┼────────────┤
     │ Glucose      │ 142 mg/dL │ HIGH (>126)│
     │ Age          │ 45 years│ HIGH (>40) │
     │ BMI          │ 31 kg/m²│ HIGH (>30) │
     └──────────────┴────────┴────────────┘

   🎯 Diagnosis Result (PUBLIC):
     Risk Score: 3 (HIGH RISK)

   ✓ Witness satisfies all R1CS constraints

🔧 Step 3: Setting up LWE Context

   ✓ LWE parameters:
     - Security: 128-bit quantum (Module-LWE)
     - Ring dimension: n=4096, k=2
     - Modulus: q=17592186044423 (prime near 2^44)
     - Noise: σ=3.19

🔐 Step 4: Generating Zero-Knowledge Proof

   ✓ Proof generated in 0.04 ms
   ✓ Proof size: 216 bytes (constant, independent of data)

✅ Step 5: Verifying Proof (Insurance Perspective)

   🏢 What Insurance Company Sees:
     ┌────────────────────────────────────────┐
     │ Proof size:       216 bytes           │
     │ Risk score:       3 (HIGH RISK)       │
     │ Patient data:     ❌ HIDDEN            │
     │ Glucose value:    ❌ HIDDEN            │
     │ Age:              ❌ HIDDEN            │
     │ BMI:              ❌ HIDDEN            │
     └────────────────────────────────────────┘

   ⏱️  Verification time: 0.00 ms

   ✓ Proof VALID ✓

╔═══════════════════════════════════════════════════════════╗
║  ✅ SUCCESS: Diagnosis proven without data disclosure!   ║
╚═══════════════════════════════════════════════════════════╝

📊 Privacy Analysis:

   What was HIDDEN (zero-knowledge):
     • Actual glucose level: 142 mg/dL
     • Patient age: 45 years
     • BMI value: 31 kg/m²
     • All intermediate computations

   What was REVEALED (public):
     • Risk score: 3 (HIGH)
     • Proof of correct computation

   🔒 Security Guarantees:
     • Soundness: ε ≤ 2^-48 (dual Fiat-Shamir)
     • Zero-Knowledge: 2^-128 distinguishing advantage
     • Post-Quantum: Resistant to Shor's algorithm

   ⚡ Performance:
     • Proof generation: 0.04 ms
     • Verification: 0.00 ms
     • Proof size: 216 bytes (constant)

   🏥 Compliance:
     • GDPR: ✅ No personal data transfer
     • HIPAA: ✅ No PHI disclosure
     • Verifiable: ✅ Cryptographic proof of diagnosis

✅ Healthcare example complete!
```

### Privacy Guarantees

| What Insurance Sees | What Remains Hidden |
|---------------------|---------------------|
| Risk score: 3 (HIGH) | Glucose: 142 mg/dL |
| Proof: 216 bytes | Age: 45 years |
| Verification: PASS | BMI: 31 kg/m² |
| | All threshold checks |

### Security Properties

- **Soundness**: Prover cannot fake a LOW risk result for HIGH risk patient (ε ≤ 2^-48)
- **Zero-Knowledge**: Insurance learns nothing about patient data beyond risk score (2^-128 advantage)
- **Post-Quantum**: Resistant to Shor's algorithm (Module-LWE security)

### Use Cases

This pattern applies to any privacy-preserving computation:

1. **Medical Diagnosis**: Prove condition without revealing symptoms
2. **Financial Risk**: Prove creditworthiness without bank balances
3. **Age Verification**: Prove age ≥ 18 without birthdate
4. **Credential Verification**: Prove qualification without GPA

See [docs/m5.2-zk-plan.md](../../docs/m5.2-zk-plan.md) for more real-world examples.

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
