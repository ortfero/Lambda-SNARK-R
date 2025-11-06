# ΛSNARK-R Roadmap

## Overview

ΛSNARK-R development follows a **staged approach** prioritizing security, correctness, and performance in that order.

**Current Version**: 0.1.0-alpha  
**Target Production Release**: 1.0.0 (Q3 2026)

---

## Phase 1: Foundation (Q4 2025 - Q1 2026)

**Goal**: Working prototype with basic functionality.

### Milestones

- [x] **M1.1**: Repository setup (✅ November 2025)
  - Project structure
  - CI/CD pipeline
  - Documentation infrastructure

- [ ] **M1.2**: C++ Core (December 2025)
  - Full NTT implementation with NTL
  - Complete LWE commitment scheme
  - Gaussian sampling (constant-time)

- [ ] **M1.3**: Rust API (January 2026)
  - Safe FFI wrappers
  - R1CS data structures
  - Basic prover skeleton

- [ ] **M1.4**: Conformance Tests (January 2026)
  - TV-0: Linear checks
  - TV-1: Simple R1CS (multiplication)
  - TV-2: Physics constraints (Wilson loops)

**Deliverables**:
- Unstable API (0.2.0-alpha)
- Basic benchmark results
- Internal documentation

---

## Phase 2: Optimization (Q1 - Q2 2026)

**Goal**: Meet performance targets.

### Milestones

- [ ] **M2.1**: Prover Optimization (February 2026)
  - SIMD-optimized NTT (AVX-512)
  - Rayon parallelization
  - Memory pooling

- [ ] **M2.2**: Parameter Tuning (March 2026)
  - Integrate `lattice-estimator`
  - Optimize (q, σ) for profiles A/B
  - Security proof validation

- [ ] **M2.3**: Benchmarking (March 2026)
  - Profile critical paths
  - Achieve target: ≤20 min prover for M=10^6
  - Document bottlenecks

**Deliverables**:
- Beta release (0.5.0-beta)
- Performance report
- Parameter recommendations

**Success Criteria**:
- Prover: ≤20 minutes for M=10^6 constraints
- Verifier: ≤500 ms
- Proof size: ≤50 KB

---

## Phase 3: Hardening (Q2 - Q3 2026)

**Goal**: Production-ready security.

### Milestones

- [ ] **M3.1**: Security Audit (April-May 2026)
  - Engage Trail of Bits
  - Audit C++ core (6-8 weeks)
  - Fix findings

- [ ] **M3.2**: Formal Verification (May-June 2026)
  - Lean 4 soundness proof
  - Lean 4 zero-knowledge proof
  - Completeness theorem

- [ ] **M3.3**: Fuzzing & Testing (June 2026)
  - 90% coverage (cargo-tarpaulin)
  - 48h continuous fuzzing
  - Constant-time validation (dudect)

- [ ] **M3.4**: API Stabilization (July 2026)
  - Lock public API (semver 1.0.0)
  - Comprehensive documentation
  - Migration guides

**Deliverables**:
- Release candidate (1.0.0-rc1)
- Audit report (public)
- Formal proofs (Lean 4 repository)

**Success Criteria**:
- No critical/high severity findings in audit
- Formal proofs checked by Lean kernel
- API stable (breaking changes prohibited)

---

## Phase 4: Ecosystem (Q3 2026 onwards)

**Goal**: Integration and adoption.

### Milestones

- [ ] **M4.1**: Public Release (August 2026)
  - Version 1.0.0
  - Crates.io publication
  - Announcement (blog, RWC)

- [ ] **M4.2**: QGravity Integration (Q4 2026)
  - Wilson loops → R1CS
  - Plaquette constraints
  - Physics examples

- [ ] **M4.3**: Proof-Twin Integration (Q4 2026)
  - Export VK as Lean terms
  - Reverse-Explorer compatibility
  - Theorem graph visualization

- [ ] **M4.4**: Bounty Program (Q4 2026)
  - HackerOne setup
  - $20k initial budget
  - Disclosure policy

**Deliverables**:
- Production releases (1.x.x)
- Integration examples
- Academic publications

---

## Long-Term Vision (2027+)

### Research Directions

1. **Recursive Composition**: SNARK-of-SNARK for scalability
2. **Aggregation**: Batch verification for multiple proofs
3. **Hardware Acceleration**: FPGA/ASIC for NTT
4. **Trusted Setup Elimination**: Fully transparent setup

### Community Growth

- Conference talks (RWC, Eurocrypt, Crypto)
- Workshops and tutorials
- Industry partnerships
- Grant applications (NSF, EU Horizon)

---

## Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **NTL integration fails** | High | Low | Fallback to pure Rust (slower) |
| **Audit finds critical bugs** | High | Medium | 3-month buffer for fixes |
| **Parameters insecure** | Critical | Low | External review by lattice experts |
| **Performance targets missed** | Medium | Medium | Incremental optimization, relaxed targets |

---

## Decision Points

- **Q1 2026**: Go/No-Go for Phase 2 (based on M1.4 results)
- **Q2 2026**: Audit readiness review
- **Q3 2026**: Public release decision (based on audit)

---

## Get Involved

- **Contributors**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Feedback**: [GitHub Discussions](https://github.com/URPKS/lambda-snark-r/discussions)
- **Sponsorship**: Contact dev@lambda-snark.org

---

**Last Updated**: November 6, 2025  
**Next Review**: January 2026
