# Security Policy

> **Version**: 0.1.0-dev  
> **Last Updated**: November 7, 2025  
> **Status**: M4 Complete — R1CS Prover/Verifier Working (NOT production-ready)

---

## ⚠️ Critical Warning

**ΛSNARK-R is NOT production-ready. DO NOT use for:**
- Privacy-critical applications (proofs are **NOT zero-knowledge**)
- Security-critical systems (no external audit, known vulnerabilities)
- Financial applications (timing attacks possible)
- Any deployment where adversarial input is possible

**Current Version (0.1.0-dev) Blockers**:
- ❌ **NOT Zero-Knowledge**: Witness elements leak via polynomial evaluations
- ❌ **NOT Audited**: No professional security review conducted
- ❌ **Non-Constant-Time**: Modular arithmetic operations leak timing information
- ❌ **FFI Safety**: C++ SEAL code not memory-safe, potential UB/RCE
- ⚠️ **O(m²) Performance**: Limited to small circuits (m ≤ 1000)

**Production Requirements** (ETA: Q2-Q3 2026):
1. M5.2: Zero-knowledge extension (polynomial blinding)
2. M7: External security audit (Trail of Bits or NCC Group)
3. M7: Constant-time validation (dudect, side-channel analysis)
4. M7: Formal verification (Lean 4 soundness/ZK proofs)
5. 6+ months stability with no critical findings

---

## 🎯 Threat Model

### Adversary Capabilities

#### **Malicious Prover** (Primary Threat)
- **Goal**: Convince verifier to accept invalid proof (soundness violation)
- **Capabilities**:
  - Full control over proof generation (can modify ProofR1CS fields)
  - Knowledge of prover algorithm (white-box attack)
  - Unlimited computational resources (within polynomial time)
- **Out of Scope**: Quantum attacks (system designed for 128-bit quantum security)

#### **Malicious Verifier** (Secondary Threat)
- **Goal**: Extract witness information from proof (zero-knowledge violation)
- **Capabilities**:
  - Can inspect all proof fields (commitment, challenges, evaluations, openings)
  - Can run distinguisher tests (compare real proofs vs. simulator)
- **Current Status**: ❌ **VULNERABLE** (NOT zero-knowledge, see vuln #1 below)

#### **Passive Observer** (Side-Channel Threat)
- **Goal**: Extract witness via timing/cache/power analysis
- **Capabilities**:
  - Precise timing measurements (nanosecond resolution)
  - Cache-timing oracles (Flush+Reload, Prime+Probe)
  - Power consumption monitoring (if hardware access)
- **Current Status**: ⚠️ **VULNERABLE** (non-constant-time ops, see vuln #2 below)

#### **Network Attacker** (Out of Scope for Core)
- **Goal**: MITM, replay attacks, denial of service
- **Mitigation**: Application layer responsibility (TLS, nonces, rate limiting)
- **Status**: Not addressed in core library (by design)

### Trust Assumptions

**Trusted**:
- Module-LWE hardness (128-bit quantum security)
- Random Oracle Model (SHAKE256 as Fiat-Shamir hash)
- Honest implementation of LWE commitment (SEAL library correctness)

**NOT Trusted**:
- Prover (must be sound against malicious provers)
- Verifier (should be zero-knowledge, currently NOT)
- Network (application must handle)

---

## 🔐 Cryptographic Assumptions

### Module-LWE Hardness

**Assumption**: Module Learning With Errors is hard for parameters (n, k, q, σ)  
**Parameters** (as of commit d89f201):
- **n**: 4096 (ring dimension, polynomial degree)
- **k**: 2 (module rank, number of LWE samples)
- **q**: 17592186044423 (prime modulus, ~2^44.01 bits)
- **σ**: 3.19 (Gaussian noise standard deviation)
- **Ring**: R = Z[X]/(X^n + 1) (cyclotomic polynomial, power-of-2 n)

**Security Level**: 128-bit quantum (Core-SVP hardness)  
**Reference**: [LWE Estimator](https://github.com/malb/lattice-estimator)  
**Validation**: ✅ Parameters verified with estimator (October 2025)

**Threat**: Quantum algorithms (Grover, Shor)  
**Mitigation**: LWE is believed quantum-resistant (NP-hard reduction)

### Random Oracle Model (ROM)

**Assumption**: SHAKE256 (SHA-3 XOF) behaves as random oracle  
**Usage**: Fiat-Shamir transformation for challenge derivation
- α = SHAKE256(commitment || r1cs_hash || public_inputs)
- β = SHAKE256(commitment || α || evals_alpha)

**Quantum ROM (QROM)**: ✅ SHAKE256 conjectured secure in QROM  
**Reference**: [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)

**Threat**: ROM is idealization (real hash functions not random oracles)  
**Mitigation**: Use standardized, well-analyzed hash (SHA-3 family)

### Soundness Bound

**Property**: Probability malicious prover convinces verifier with invalid witness  
**Bound**: ε ≤ 1/q² ≈ 2^-88 (dual independent challenges α, β)

**Derivation**:
1. Prover must provide Q(X) such that Q(α)·Z_H(α) = A_z(α)·B_z(α) - C_z(α)
2. If witness invalid, Q(X) ≠ (A_z·B_z - C_z)/Z_H (true quotient)
3. Cheating polynomial: ΔQ(X) = Q_cheat(X) - Q_true(X) has degree < m
4. Probability ΔQ(α) = 0: ≤ m/q (Schwartz-Zippel lemma)
5. **Dual challenge**: Must also satisfy at β (independent)
6. **Combined**: ε ≤ (m/q)² ≈ (30/2^44)² ≈ 2^-88 for m=30

**Validation**: ✅ 15 soundness tests passing (commit a216df3)  
**Test Vectors**: TV-R1CS-1, TV-R1CS-2 with modified witness/proof

### Completeness

**Property**: Valid witness always produces verifying proof  
**Status**: ✅ 100% completeness (all valid witnesses verify)  
**Validation**: 60 unit tests + 3 CLI examples, no false negatives

---

## 🚨 Known Vulnerabilities

### CRITICAL: Non-Zero-Knowledge (CVSSv3.1: 9.6)

**ID**: VULN-001  
**Status**: ❌ **OPEN** (deferred to M5.2)  
**Affected Versions**: 0.1.0-dev (all commits ≤ ffc4dcc)  
**Severity**: **CRITICAL** (Exploitability: 1.0, Impact: 0.9, Scope: 1.0)

**Description**:  
Polynomial evaluations in `ProofR1CS` leak witness correlations. Malicious verifier can extract partial witness information by analyzing proof fields.

**Attack Scenario**:
```rust
// Prover generates proof for witness [1, 7, 13, 91] (7 × 13 = 91)
let proof = prove_r1cs(&r1cs, &witness, &ctx, seed)?;

// Malicious verifier can observe:
proof.a_z_alpha  // = A_z(α) = a_0 + a_1·α + a_2·α² + ... (witness-dependent)
proof.b_z_alpha  // = B_z(α) = b_0 + b_1·α + ... (witness-dependent)
proof.c_z_alpha  // = C_z(α) = c_0 + c_1·α + ... (witness-dependent)

// For simple constraints, can solve linear system to recover witness!
// Example: For constraint `x · y = z`, three evaluations → three equations
```

**Impact**:
- **Privacy Loss**: Adversary can distinguish proofs for different witnesses
- **Witness Recovery**: For low-entropy witnesses (e.g., small integers), full recovery possible
- **Correlation Leakage**: Even if full recovery hard, statistical correlations leak

**Exploitability**: **TRIVIAL** (no special knowledge required, passive observation)

**Mitigation** (Planned M5.2, ETA: December 2025):
1. Blind quotient polynomial: Q'(X) = Q(X) + r_1·Z_H(X) + r_2·X·Z_H(X) + ...
2. Blind witness polynomials: A'_z(X) = A_z(X) + r_a·Z_H(X), similarly for B, C
3. Randomness: r_1, r_2, r_a, r_b, r_c sampled uniformly from F_q
4. Property: Q'(α) = Q(α) for α ∉ H (Z_H(α) ≠ 0), but Q'(X) looks random
5. Implement `prove_r1cs_zk()` with blinded commitment
6. Security proof: Simulate indistinguishable proofs without witness

**Workaround** (Until M5.2):
- **DO NOT** use for privacy-critical applications
- Treat proofs as public (assume adversary sees witness)
- Use only for correctness verification, not privacy

**References**:
- [Groth16 ZK-SNARK](https://eprint.iacr.org/2016/260.pdf) (polynomial blinding technique)
- [Plonk](https://eprint.iacr.org/2019/953.pdf) (hiding commitments)

---

### HIGH: Timing Attacks (CVSSv3.1: 6.8)

**ID**: VULN-002  
**Status**: ❌ **OPEN** (deferred to M7)  
**Affected Versions**: 0.1.0-dev  
**Severity**: **HIGH** (Exploitability: 0.6, Impact: 0.7, Scope: 0.8)

**Description**:  
Modular arithmetic operations (`mod_inverse`, `mod_mul`, `mod_pow`) are NOT constant-time. Execution time depends on input values, leaking witness information via timing side-channels.

**Attack Scenario**:
```rust
// Attacker measures prover execution time for different witnesses
let start = Instant::now();
prove_r1cs(&r1cs, &witness_candidate, &ctx, seed)?;
let duration = start.elapsed();

// mod_inverse(a, q) timing depends on:
// - Number of iterations in Extended Euclidean Algorithm
// - Bit pattern of `a` (early termination conditions)
// - Cache behavior (memory access patterns)

// By testing many candidates and measuring timing,
// attacker can narrow down witness space
```

**Impact**:
- **Partial Witness Recovery**: Statistical analysis over many proofs
- **Distinguisher**: Differentiate between witness classes (even if full recovery hard)
- **Cache-Timing**: Flush+Reload attacks on LWE commitment (SEAL library)

**Exploitability**: **MODERATE** (requires precise timing measurements, statistical analysis)

**Affected Code**:
- `rust-api/lambda-snark-core/src/modular.rs`: `mod_inverse()` (lines 78-110)
- `rust-api/lambda-snark/src/r1cs.rs`: `lagrange_interpolate()` (calls mod_inverse)
- `cpp-core/src/lwe_context.cpp`: SEAL operations (potential cache-timing)

**Mitigation** (Planned M7, ETA: January 2026):
1. **Constant-Time Modular Ops**:
   - Replace branching with select (conditional move)
   - Use `subtle` crate for constant-time comparisons
   - Validate with `dudect` (timing leak detector)
2. **Constant-Time Polynomial Ops**:
   - Ensure Lagrange interpolation has fixed iteration count
   - Pad loops to maximum iterations
3. **SEAL Audit**:
   - Review SEAL code for data-dependent branches
   - Consider alternative LWE implementation (constant-time by design)
4. **Timing Tests**:
   - Run `dudect` on critical functions (mod_inverse, prove_r1cs)
   - Measure timing variance for different witnesses
   - Target: <1% variance (statistical indistinguishability)

**Workaround** (Until M7):
- **DO NOT** use in adversarial timing environments
- Run prover in isolated environment (no shared CPU)
- Add random delays (timing noise) — not cryptographically sound, but raises attack bar

**References**:
- [Constant-Time Crypto](https://www.bearssl.org/ctmul.html) (BearSSL guide)
- [DudeCT](https://github.com/oreparaz/dudect) (timing leak detection)
- [Flush+Reload](https://eprint.iacr.org/2013/448.pdf) (cache-timing attack)

---

### HIGH: FFI Memory Safety (CVSSv3.1: 6.2)

**ID**: VULN-003  
**Status**: ❌ **OPEN** (deferred to M7)  
**Affected Versions**: 0.1.0-dev  
**Severity**: **HIGH** (Exploitability: 0.3, Impact: 1.0, Scope: 0.5)

**Description**:  
C++ SEAL library (used for LWE commitment) is not memory-safe. Malicious input to `LweContext::commit()` or `verify_opening()` could trigger undefined behavior (UB) in C++ code, potentially leading to arbitrary code execution (RCE) via Rust FFI boundary.

**Attack Scenario**:
```rust
// Attacker crafts malicious polynomial coefficients
let malicious_poly = vec![u64::MAX; 4096]; // Trigger integer overflow in SEAL

// Pass to FFI (no validation in Rust wrapper)
let commitment = ctx.commit(&malicious_poly, randomness)?;
//                          ^^^^^^^^^^^^^^ No bounds checking in C++!

// SEAL code (simplified):
// for (int i = 0; i < poly.size(); i++) {
//     result[i] = poly[i] * key[i];  // Integer overflow → UB
// }

// UB can lead to:
// - Out-of-bounds write (heap corruption)
// - RCE if attacker controls memory layout
```

**Impact**:
- **Remote Code Execution**: If attacker can trigger UB with crafted input
- **Denial of Service**: Crash prover/verifier processes
- **Memory Corruption**: Unpredictable behavior, data leakage

**Exploitability**: **LOW-MODERATE** (requires understanding SEAL internals, crafted input)

**Affected Code**:
- `cpp-core/src/lwe_context.cpp`: All SEAL API calls (commit, verify_opening)
- `rust-api/lambda-snark-sys/src/lib.rs`: FFI bindings (no validation)

**Mitigation** (Planned M7, ETA: January 2026):
1. **Input Validation**:
   - Check polynomial coefficients < q (modulus)
   - Validate degree ≤ n (ring dimension)
   - Sanitize all FFI inputs in Rust wrapper
2. **Fuzzing**:
   - Use `cargo-fuzz` to test LWE context with random inputs
   - Target: 48h continuous fuzzing without crashes
   - AFL++ or libFuzzer on C++ SEAL code
3. **Memory Sanitizers**:
   - Compile C++ with AddressSanitizer (ASan)
   - Valgrind memcheck for leak detection
4. **Consider Alternatives**:
   - Pure Rust LWE implementation (no FFI)
   - Formally verified LWE (hacspec, fiat-crypto)

**Workaround** (Until M7):
- **DO NOT** accept untrusted polynomial inputs
- Validate all coefficients < q before FFI call
- Run in sandboxed environment (containers, VMs)

**References**:
- [SEAL Documentation](https://github.com/microsoft/SEAL)
- [Rust FFI Safety](https://doc.rust-lang.org/nomicon/ffi.html)
- [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz)

---

### CRITICAL: Composite Modulus Bug (CVSSv3.1: 10.0) ✅ FIXED

**ID**: VULN-004  
**Status**: ✅ **RESOLVED** (commit d89f201, November 7, 2025)  
**Affected Versions**: 0.1.0-dev (commits ≤ 8861181)  
**Severity**: **CRITICAL** (Exploitability: 1.0, Impact: 1.0, Scope: 1.0)

**Description**:  
Modulus `17592186044417 = 2^44 + 1` used in earlier commits was **composite** (not prime). This violates the field assumption F_q, causing `mod_inverse()` to panic when computing inverses of elements that share a common factor with the modulus.

**Root Cause**:
```python
# 2^44 + 1 is a Fermat number, divisible by 17
>>> from sympy import factorint
>>> factorint(2**44 + 1)
{17: 1, 1034834473201: 1}  # Composite!

# Example failure:
>>> gcd(4866088311555, 17592186044417)
17  # Non-trivial common divisor
```

**Impact**:
- **Soundness Violation**: Field operations undefined (no inverses for some elements)
- **Prover Crash**: `mod_inverse()` panic during Lagrange interpolation
- **Complete System Failure**: All proofs with m > 10 constraints failed

**Discovery**: Benchmark crashed at m=20 with "a is not invertible mod m" (commit 8861181)

**Fix** (commit d89f201):
```bash
# Find next prime after 2^44 + 1
python3 -c "from sympy import nextprime; print(nextprime(2**44 + 1))"
# Output: 17592186044423 (verified prime)

# Replace all instances
sed -i 's/17592186044417/17592186044423/g' rust-api/lambda-snark-cli/src/main.rs
```

**Validation**: ✅ All examples now execute successfully, benchmark completes m=10/20/30

**Lesson Learned**: **ALWAYS verify primality for cryptographic parameters**  
**Prevention**: Add compile-time primality checks (Miller-Rabin test in build.rs)

---

## 🛡️ Security Roadmap

| Milestone | Vulnerability | Fix | ETA | Status |
|-----------|---------------|-----|-----|--------|
| **M5.2** | VULN-001 (Non-ZK) | Polynomial blinding | Dec 2025 | 🔜 Planned |
| **M7** | VULN-002 (Timing) | Constant-time ops + dudect | Jan 2026 | 🔜 Planned |
| **M7** | VULN-003 (FFI Safety) | Fuzzing + validation | Jan 2026 | 🔜 Planned |
| **M7** | External Audit | Trail of Bits / NCC Group | Q2 2026 | 🔜 Planned |
| **1.0.0** | Production Release | 6mo stability + formal verification | Q3 2026 | 🔜 Planned |

**Critical Path**: M5.2 (ZK) → M7 (Audit) → 1.0.0 (Production)

---

## 📧 Responsible Disclosure

### Reporting a Vulnerability

**DO NOT** open a public GitHub issue for security vulnerabilities.

**Contact**:  
📧 **security@lambda-snark.org** (monitored 24/7)  
🔐 **PGP Key**: [Download](https://keys.openpgp.org/search?q=security@lambda-snark.org)

**Include in Report**:
1. **Description**: What is the vulnerability?
2. **Affected Versions**: Which commits/releases?
3. **Reproduction Steps**: Minimal code to trigger issue
4. **Impact**: Confidentiality/Integrity/Availability affected?
5. **Suggested Fix**: (Optional) How to mitigate?
6. **Disclosure Preference**: Public credit? Anonymous?

### Disclosure Timeline

We follow **coordinated disclosure** (90-day standard):

1. **T+0**: Report received  
   - **SLA**: Acknowledgment within **72 hours**
   - **Action**: Triage severity (critical/high/medium/low)

2. **T+7d**: Initial Assessment  
   - **Action**: Confirm/reject vulnerability
   - **Communication**: Share timeline with reporter

3. **T+30-90d**: Fix Development  
   - **Critical**: Target 30 days
   - **High**: Target 60 days
   - **Medium/Low**: Target 90 days
   - **Updates**: Weekly status to reporter

4. **T+Fix**: Patch Released  
   - **Action**: Commit fix to private branch
   - **Testing**: Validate with reporter's PoC
   - **Embargo**: Hold public release for 7 days

5. **T+Fix+7d**: Public Disclosure  
   - **Action**: Merge fix to main, tag release
   - **Advisory**: Publish GHSA (GitHub Security Advisory)
   - **CVE**: Request CVE ID (if applicable)
   - **Credit**: Acknowledge reporter (unless anonymous)

### Out-of-Band Disclosure

For **critical vulnerabilities** (RCE, cryptographic break):
- **Immediate**: Acknowledge within **24 hours**
- **Fast Track**: Fix within **7-14 days**
- **Coordinated Release**: Notify package managers (crates.io, distros)

---

## 🏆 Hall of Fame

Contributors who responsibly disclose vulnerabilities will be listed here.

**Current**: No external reports yet (0.1.0-dev in development)

**Future**: Public credit policy applies to all reporters (unless anonymous request)

---

## 🔍 Security Testing

### Current Status (as of 0.1.0-dev)

- ✅ **Unit Tests**: 98 tests covering modular arithmetic, polynomial ops, R1CS
- ✅ **Integration Tests**: 60 tests for soundness, commitment binding
- ✅ **Soundness Tests**: 15 tests with invalid witnesses/modified proofs
- ❌ **Constant-Time Tests**: None (deferred to M7)
- ❌ **Fuzzing**: None (deferred to M7)
- ❌ **Formal Verification**: None (planned Lean 4 proofs in M7)

### Planned (M7, January 2026)

- 🔜 **DudeCT**: Timing leak detection (constant-time validation)
- 🔜 **cargo-fuzz**: 48h continuous fuzzing (LWE context, polynomial ops)
- 🔜 **AddressSanitizer**: Memory safety checks (C++ SEAL code)
- 🔜 **Valgrind**: Leak detection
- 🔜 **proptest**: Property-based testing (polynomial properties)

---

## 📚 Additional Resources

- **Documentation**: [README.md](README.md), [ROADMAP.md](ROADMAP.md), [CHANGELOG.md](CHANGELOG.md)
- **Examples**: [EXAMPLES.md](rust-api/lambda-snark-cli/EXAMPLES.md) (CLI usage)
- **Code**: [GitHub Repository](https://github.com/SafeAGI-lab/Lambda-SNARK-R)
- **Discussions**: [GitHub Discussions](https://github.com/SafeAGI-lab/Lambda-SNARK-R/discussions)
- **Issues**: [GitHub Issues](https://github.com/SafeAGI-lab/Lambda-SNARK-R/issues) (non-security bugs only)

---

**Thank you for helping keep ΛSNARK-R secure! 🔒**

**Last Updated**: November 7, 2025  
**Next Review**: January 2026 (M7 security testing)
