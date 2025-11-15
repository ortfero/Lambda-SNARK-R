# Architecture: ΛSNARK-R System Design

> **Version**: 0.1.0-dev  
> **Last Updated**: November 15, 2025  
> **Status**: M5 Complete — NTT + Zero-Knowledge Optimizations

This document provides architectural overview of ΛSNARK-R system components, data flow, and module dependencies.

---

## Table of Contents

1. [System Components](#system-components)
2. [Proof Generation Flow](#proof-generation-flow)
3. [Module Dependencies](#module-dependencies)
4. [Security Boundaries](#security-boundaries)
5. [Performance Characteristics](#performance-characteristics)

---

## System Components

High-level view of cryptographic components in ΛSNARK-R.

![System Components Diagram](images/system-components.svg)

<details>
<summary>View Mermaid Source</summary>

```mermaid
graph TD
    MOD[Modular Arithmetic]
    POLY[Polynomial Operations]
    SEAL[SEAL Library]
    LWE[LWE Commitment]
    FS[Fiat-Shamir]
    SPARSE[Sparse Matrix]
    R1CS[R1CS System]
    LAGRANGE[Lagrange Interpolation]
    QUOTIENT[Quotient Polynomial]
    PROVER[prove_r1cs]
    VERIFIER[verify_r1cs]
    NTT[NTT FFT]
    ZK[Zero-Knowledge]
    PROVER_ZK[prove_r1cs_zk]
    VERIFIER_ZK[verify_r1cs_zk]
    
    MOD --> POLY
    POLY --> LAGRANGE
    POLY --> NTT
    POLY --> QUOTIENT
    SEAL --> LWE
    LWE --> PROVER
    LWE --> PROVER_ZK
    FS --> PROVER
    FS --> PROVER_ZK
    SPARSE --> R1CS
    R1CS --> LAGRANGE
    R1CS --> NTT
    LAGRANGE --> QUOTIENT
    NTT --> QUOTIENT
    QUOTIENT --> PROVER
    QUOTIENT --> ZK
    ZK --> PROVER_ZK
    PROVER --> VERIFIER
    PROVER_ZK --> VERIFIER_ZK
```

</details>

**Legend**:
- 🔵 **M1 Foundation** (Blue): Core cryptographic primitives
- 🟡 **M2 LWE Context** (Yellow): Post-quantum commitment scheme
- 🟢 **M3 R1CS Structure** (Green): Constraint system representation
- 🟣 **M4 Prover/Verifier** (Purple): SNARK proof system
- 🟠 **M5 Optimizations** (Orange): NTT performance + Zero-Knowledge

---

## Proof Generation Flow

Detailed sequence of operations from witness to verified proof.

```mermaid
sequenceDiagram
    participant User
    participant CircuitBuilder
    participant R1CS
    participant Prover
    participant LWE as LWE Context
    participant FS as Fiat-Shamir
    participant Verifier
    
    User->>CircuitBuilder: Define circuit<br/>(alloc_var, add_constraint)
    CircuitBuilder->>R1CS: build() → R1CS<br/>(A, B, C matrices)
    
    User->>Prover: prove_r1cs(r1cs, witness)
    
    Note over Prover: Step 1: Interpolate witness polynomials
    Prover->>Prover: A_z(X) ← lagrange_interpolate(A·z)
    Prover->>Prover: B_z(X) ← lagrange_interpolate(B·z)
    Prover->>Prover: C_z(X) ← lagrange_interpolate(C·z)
    
    Note over Prover: Step 2: Compute quotient polynomial
    Prover->>Prover: Q(X) ← (A_z·B_z - C_z) / Z_H
    Note right of Prover: Z_H(X) = ∏(X - i) for i=0..m-1
    
    Note over Prover: Step 3: Commit to Q(X)
    Prover->>LWE: commitment_q ← commit(Q, randomness)
    LWE-->>Prover: commitment_q (Module-LWE)
    
    Note over Prover: Step 4: Derive first challenge
    Prover->>FS: α ← SHAKE256(commitment || r1cs || public)
    FS-->>Prover: challenge_alpha
    
    Note over Prover: Step 5: Evaluate at α
    Prover->>Prover: q_alpha ← Q(α)
    Prover->>Prover: a_z_alpha ← A_z(α)
    Prover->>Prover: b_z_alpha ← B_z(α)
    Prover->>Prover: c_z_alpha ← C_z(α)
    
    Note over Prover: Step 6: Derive second challenge
    Prover->>FS: β ← SHAKE256(commitment || α || evals_α)
    FS-->>Prover: challenge_beta
    
    Note over Prover: Step 7: Evaluate at β
    Prover->>Prover: q_beta ← Q(β)
    Prover->>Prover: a_z_beta ← A_z(β)
    Prover->>Prover: b_z_beta ← B_z(β)
    Prover->>Prover: c_z_beta ← C_z(β)
    
    Note over Prover: Step 8: Generate LWE openings
    Prover->>LWE: opening_alpha ← open(Q, α)
    Prover->>LWE: opening_beta ← open(Q, β)
    LWE-->>Prover: openings
    
    Prover-->>User: ProofR1CS (216 bytes)
    
    User->>Verifier: verify_r1cs(proof, public_inputs, r1cs)
    
    Note over Verifier: Step 1: Recompute challenges
    Verifier->>FS: α' ← SHAKE256(proof.commitment || r1cs || public)
    Verifier->>FS: β' ← SHAKE256(proof.commitment || α' || evals_α')
    Note right of Verifier: Must match proof.challenge_alpha/beta
    
    Note over Verifier: Step 2: Evaluate vanishing polynomial
    Verifier->>Verifier: Z_H(α) ← ∏(α - i)
    Verifier->>Verifier: Z_H(β) ← ∏(β - i)
    
    Note over Verifier: Step 3: Check dual equations
    Verifier->>Verifier: Check: Q(α)·Z_H(α) == A_z(α)·B_z(α) - C_z(α)
    Verifier->>Verifier: Check: Q(β)·Z_H(β) == A_z(β)·B_z(β) - C_z(β)
    
    Note over Verifier: Step 4: Verify LWE openings
    Verifier->>LWE: verify_opening(commitment, α, q_alpha, opening_α)
    LWE-->>Verifier: valid_alpha
    Verifier->>LWE: verify_opening(commitment, β, q_beta, opening_β)
    LWE-->>Verifier: valid_beta
    
    alt All checks pass
        Verifier-->>User: ✅ Proof VALID
    else Any check fails
        Verifier-->>User: ❌ Proof INVALID
    end
```

**Key Properties**:
- **Soundness**: ε ≤ 2^-88 (dual independent challenges α, β)
- **Completeness**: 100% (all valid witnesses produce verifying proofs)
- **Proof Size**: 216 bytes (constant, independent of circuit size m)
- **Prover Time**: 4-6ms for m=10-30 (dominated by LWE commitment)
- **Verifier Time**: ~1ms (no polynomial interpolation required)

---

## Module Dependencies

Rust crate hierarchy and FFI boundaries.

![Module Dependencies Diagram](images/module-dependencies.svg)

<details>
<summary>View Mermaid Source</summary>

```mermaid
graph TD
    CORE[lambda-snark-core]
    SYS[lambda-snark-sys]
    API[lambda-snark]
    CLI[lambda-snark-cli]
    SEAL_LIB[Microsoft SEAL v4.1.1]
    LWE_CPP[lwe_context.cpp]
    SHAKE[SHAKE256 sha3 crate]
    CMAKE[CMake 3.20+]
    
    CORE --> API
    SYS --> API
    API --> CLI
    SEAL_LIB --> LWE_CPP
    LWE_CPP --> SYS
    SHAKE --> API
    CMAKE --> LWE_CPP
```

</details>

**Dependency Graph**:
```
lambda-snark-cli
    └── lambda-snark (public API)
        ├── lambda-snark-core (core types)
        ├── lambda-snark-sys (FFI bindings)
        │   └── lwe_context.cpp (C++ SEAL wrapper)
        │       └── SEAL 4.1.1 (Microsoft FHE library)
        └── sha3 (SHAKE256 for Fiat-Shamir)
```

**File Counts** (as of commit bfd754a):
- **Rust**: 3,167 lines (core 428, API 1537+928, CLI 694)
- **C++**: 542 lines (lwe_context.cpp)
- **Docs**: 1,505 lines (README 236, ROADMAP 729, CHANGELOG 295, SECURITY 481)
- **Tests**: 158 automated (98 unit + 60 integration)

---

## Security Boundaries

Trust boundaries and attack surfaces.

![Security Boundaries Diagram](images/security-boundaries.svg)

<details>
<summary>View Mermaid Source</summary>

```mermaid
graph LR
    WITNESS[Witness z]
    PUBLIC[Public Inputs]
    CIRCUIT[Circuit R1CS]
    PROVER_RUST[Prover Rust]
    LWE_FFI[LWE Context C++]
    PROOF[ProofR1CS 216 bytes]
    SOUNDNESS[Soundness ε ≤ 2^-88]
    ZK[Zero-Knowledge ✅ M5]
    VERIFIER_RUST[Verifier Rust]
    VERIFIER_OUT[Accept/Reject]
    
    WITNESS --> PROVER_RUST
    PUBLIC --> PROVER_RUST
    CIRCUIT --> PROVER_RUST
    PROVER_RUST --> LWE_FFI
    LWE_FFI --> PROOF
    PROOF --> SOUNDNESS
    PROOF --> ZK
    PROOF --> VERIFIER_RUST
    PUBLIC --> VERIFIER_RUST
    CIRCUIT --> VERIFIER_RUST
    VERIFIER_RUST --> VERIFIER_OUT
```

</details>

**Threat Model**:
- 🔴 **Untrusted**: Witness (prover controls), Verifier (may be adversarial)
- 🟡 **Semi-Trusted**: Prover code (Rust memory-safe, but timing leaks)
- 🟠 **Untrusted FFI**: LWE Context (C++ SEAL, potential UB/RCE, see VULN-003)
- 🟢 **Trusted**: Public inputs, circuit definition (application layer)

**Known Vulnerabilities** (as of 0.1.0-dev):
- ❌ **VULN-001 (CRITICAL)**: Non-zero-knowledge (witness leakage)
- ⚠️ **VULN-002 (HIGH)**: Timing attacks (mod_inverse, Lagrange)
- ⚠️ **VULN-003 (HIGH)**: FFI safety (C++ SEAL UB risk)
- ✅ **VULN-004 (CRITICAL)**: Composite modulus bug FIXED (d89f201)

See [SECURITY.md](../SECURITY.md) for detailed threat model and mitigations.

---

## Performance Characteristics

Bottleneck analysis and optimization roadmap.

![Performance Characteristics Diagram](images/performance-characteristics.svg)

<details>
<summary>View Mermaid Source</summary>

```mermaid
graph TD
    BUILD[Circuit Building]
    INTERP[Lagrange Interpolation BOTTLENECK]
    COMMIT[LWE Commitment]
    EVAL[Polynomial Evaluation]
    TOTAL_PROVE[Total Prover]
    RECOMPUTE[Recompute Challenges]
    VANISH[Evaluate Z_H]
    CHECK_EQ[Check Equations]
    VERIFY_OPEN[Verify LWE Openings]
    TOTAL_VERIFY[Total Verifier]
    FFT[FFT/NTT Optimization]
    NTT_MOD[NTT-Friendly Modulus]
    
    BUILD --> INTERP
    INTERP --> COMMIT
    COMMIT --> EVAL
    EVAL --> TOTAL_PROVE
    
    RECOMPUTE --> VANISH
    VANISH --> CHECK_EQ
    CHECK_EQ --> VERIFY_OPEN
    VERIFY_OPEN --> TOTAL_VERIFY
    
    INTERP -.-> FFT
    INTERP -.-> NTT_MOD
```

</details>

**Current Performance** (as of commit d89f201):

| Circuit Size | Build (ms) | Prove (ms) | Verify (ms) | Proof Size |
|-------------|------------|------------|-------------|------------|
| m=10        | 0.03       | 4.45       | 1.03        | 216 bytes  |
| m=20        | 0.04       | 5.92       | 1.05        | 216 bytes  |
| m=30        | 0.06       | 5.79       | 1.00        | 216 bytes  |

**Scaling**:
- **Proof size**: ✅ Constant 216 bytes (independent of m)
- **Verifier**: ✅ Fast ~1ms (no interpolation)
- **Prover**: ⚠️ 1.30× growth (m=10→30), empirical exponent 0.24
  - **Bottleneck**: O(m²) Lagrange interpolation (will dominate at m > 100)
  - **Prediction**: ~20 minutes for m=2^20 (naïve implementation)

**Optimization Roadmap (M5.1)**:
1. Replace Lagrange O(m²) with FFT/NTT O(m log m)
2. NTT-friendly modulus: q = 2^64 - 2^32 + 1
3. Target: <1s for m=2^10, <10s for m=2^15, <2min for m=2^20
4. Expected speedup: **1000×** for large m

---

## References

- **Code**: [GitHub Repository](https://github.com/SafeAGI-lab/Lambda-SNARK-R)
- **Documentation**: [README.md](../README.md), [ROADMAP.md](../ROADMAP.md), [CHANGELOG.md](../CHANGELOG.md)
- **Security**: [SECURITY.md](../SECURITY.md) (threat model, vulnerabilities, disclosure)
- **Examples**: [EXAMPLES.md](../rust-api/lambda-snark-cli/EXAMPLES.md) (CLI usage)

---

**Last Updated**: November 7, 2025  
**Next Review**: December 2025 (M5.1 FFT/NTT optimization)
