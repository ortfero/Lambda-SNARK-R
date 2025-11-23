# Security Policy

> Version: 0.1.0-alpha (M7 security snapshot)  
> Last Updated: 2025-11-23  
> Release Posture: Research prototype - not production ready  
> Contact: security@safeagi.dev

---

## Executive Summary
- ΛSNARK-R remains a research prototype. Deployments with adversarial input are out of scope until the external audit and constant-time hardening complete.
- The prover and verifier pass the M5 regression suite, but side-channel defenses and formal proofs are still in flight.
- Known high-risk gaps: residual modular reductions inside legacy SEAL FFI bindings; Rust R1CS/NTT/polynomial/sparse-matrix paths now reuse constant-time helpers end-to-end.
- Discrete Gaussian sampler now uses a branchless CDF lookup and passes dudect sanity checks (t-stat ≈ 0.30 at 20k traces); modular inverse and modular exponentiation share the constant-time ladder (|t| ≤ 1.6 at 20k traces).
- Dependency scans (cargo audit) report zero advisories as of 2025-11-23; monitoring continues via CI.

---

## Responsible Disclosure
- Report vulnerabilities privately to `security@safeagi.dev`. An acknowledgment will be sent within 72 hours.
- Please include a proof-of-concept, affected commits, and your disclosure preference. Encryption keys are available on request.
- Coordinated disclosure timeline: triage within 7 days, fix target within 30 days for critical issues and 60 days for high severity, public advisory within 7 days of patch release.
- Do not open public GitHub issues for security findings. For non-security bugs, use the normal issue tracker.

---

## Threat Model
- **Malicious prover**: attempts to convince the verifier of false statements. Mitigated by dual Fiat-Shamir challenges (soundness bound ε ≤ 2^-88 for m ≤ 32).
- **Malicious verifier**: attempts to recover the witness. Zero-knowledge blinding (M5.2) is implemented, pending formal proof in Lean.
- **Passive observer / side-channel adversary**: observes timing, cache, or EM leakage. Current implementation is not constant-time; assume leakage is possible.
- **Network attacker**: man-in-the-middle or replay attacks. Out of scope for the core library; applications must provide channel security (TLS, nonces).

Trusted components: Module-LWE hardness, SHAKE256 in the Random Oracle Model, and the correctness of the Microsoft SEAL backend. Untrusted components: prover inputs, verifier behavior, network transport.

---

## Cryptographic Foundations (M7 snapshot)
- Module-LWE parameters: n = 4096, k = 2, q = 17592169062401 (prime), σ = 3.19. Verified against the LWE estimator for 128-bit quantum security.
- Random Oracle assumption: SHAKE256 is used for both Fiat-Shamir challenges (α, β) with domain separation.
- Soundness: Schwartz-Zippel analysis with dual challenges yields ε ≤ (m/q)^2; empirical tests cover m up to 64.
- Completeness: All published test vectors (TV-R1CS-1/2) verify successfully in M5 regression runs.

---

## Residual Risks and Mitigations
| Risk | Description | Status | Planned Mitigation |
|------|-------------|--------|--------------------|
| Timing side-channels | Gaussian sampler and shared modular arithmetic helpers use branchless implementations (|t| ≤ 1.6 at 20k traces); remaining hotspots limited to commitment/SEAL FFI adapters. | Open (Medium) | Harden residual FFI code paths, extend dudect harness to cover commitment/opening (ETA 2026-01). |
| FFI safety gaps | Microsoft SEAL backend lacks exhaustive input validation and sanitizers. | Open (High) | Add bounds checks, enable ASan/UBSan, begin fuzzing harness in M7. |
| External audit | No third-party cryptographic review yet. | Open (High) | Engage Trail of Bits/NCC Group during M10 (ETA 2026-Q2). |
| Formal proofs | Lean 4 soundness proof complete; zero-knowledge proof pending. | In progress (Medium) | `lake build LambdaSNARK`; finalize ZK proof scripts tracked in `formal/`. |
| Artifact hygiene | Example runs may emit witness-dependent artifacts. | Managed (Medium) | Shared healthcare module writes to `artifacts/r1cs/healthcare.term`; delete or rotate after use. |

---

## Operational Guidance
- Run provers on dedicated hosts or inside confidentiality-preserving enclaves; avoid shared-tenancy environments until constant-time guarantees land.
- Enable compiler hardening (`-C panic=abort`, stack protector) when embedding the Rust crate.
- Treat files under `artifacts/` as sensitive. The healthcare example stores Lean terms derived from supplied witnesses; remove or secure these outputs after each run.
- Regenerate dudect timing reports with `cmake --build build --target dudect_sampler` followed by `./build/dudect_sampler` inside `cpp-core`; outputs land in `artifacts/dudect/`.
- Generate modular-arithmetic dudect sweeps via `cargo run --manifest-path rust-api/Cargo.toml -p lambda-snark --bin mod_arith_timing`; review `artifacts/dudect/mod_arith_report.md` for regressions.
- When integrating the C++ core, compile with AddressSanitizer/UndefinedBehaviorSanitizer during testing and keep `cpp-core/tests` in CI.
- Pin dependencies using the provided `Cargo.lock` and `vcpkg.json`. Run `cargo audit` and `pip-audit` before releases.

---

## Security Roadmap
| Milestone | Deliverable | ETA | Notes |
|-----------|------------|-----|-------|
| M7.4 | Expand test coverage to >200 cases, add property tests | 2025-12 | In-progress; see `TESTING.md`. |
| M7.5 | Constant-time modular arithmetic pass, dudect report | 2026-01 | Blocks any beta announcement. |
| M8 | Lean soundness proof scripts | 2026-04 | ✅ Completed (tracked in `formal/`). |
| M9 | Lean zero-knowledge proof scripts | 2026-05 | Requires completion of M8 prerequisites. |
| M10 | External security audit + hardening backlog | 2026-Q2 | Target vendors: Trail of Bits or NCC Group. |
| v1.0.0 | Production readiness review (CT, audit, 6mo stability) | 2026-Q3 | Publish final security whitepaper. |

---

## Recent Reviews
- Internal security review (M7.3) completed 2025-11-15: resolved composite modulus bug, documented timing and FFI gaps.
 - Dudect timing sweep for Gaussian sampler (20k traces) recorded 2025-11-23; report archived at `artifacts/dudect/gaussian_sampler_report.md`.
 - Dudect timing sweep for modular arithmetic (20k traces) recorded 2025-11-23; report archived at `artifacts/dudect/mod_arith_report.md` (|t_add_mod| ≈ 0.91, |t_sub_mod| ≈ 0.33, |t_mod_pow| ≈ 0.40, |t_mod_inverse| ≈ 0.12).
 - Extended dudect harness now benchmarks polynomial evaluation and sparse matrix multiplication pathways alongside modular helpers; results captured in the same report file.
- Latest `cargo audit` run: 2025-11-23, zero advisories, 162 crates checked.
- Lean build (`lake build LambdaSNARK`) passes as of 2025-11-23; soundness proof landed, ZK proof ongoing.

---

## Contact
- Security reports: `security@safeagi.dev`
- General questions: GitHub Discussions or SafeAGI Zulip `#lambda-snark`
- Emergency channel: request Signal bridge via the security email if the issue is high severity.

Please include commit hashes, platform details, and reproduction steps when contacting the team.
