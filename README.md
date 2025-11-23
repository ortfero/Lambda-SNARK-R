# ΛSNARK-R

[![Build status](https://github.com/SafeAGI-lab/Lambda-SNARK-R/actions/workflows/ci.yml/badge.svg)](https://github.com/SafeAGI-lab/Lambda-SNARK-R/actions/workflows/ci.yml)
[![Security policy](https://img.shields.io/badge/security-disclosure-blue.svg)](SECURITY.md)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-APACHE)

ΛSNARK-R is a research-first implementation of a post-quantum, zero-knowledge Λ-style succinct SNARK: Rust prover, C++ commitment kernel (Microsoft SEAL), and an in-progress Lean 4 formal verification layer. The goal is a reproducible argument system backed by Module-LWE assumptions with high assurance.

> Project status: **Alpha / M7 stabilization** — suitable for experimentation, not production-ready.

---

## Highlights (M7)
- R1CS→ΛSNARK pipeline with dual Fiat–Shamir challenges.
- Microsoft SEAL commitments exposed via safe Rust FFI and CLI workflows.
- Lean project `formal/` with completed soundness proof and active zero-knowledge development.
- Deterministic test vectors and a healthcare walkthrough that emit Lean artifacts.
- Constant-time discrete Gaussian sampler and modular arithmetic dudect sweeps (`artifacts/dudect/`).

---

## Layout
```
.
├── rust-api/           # Crates: lambda-snark, core, cli, sys
├── cpp-core/           # C++ NTT + commitments (SEAL)
├── formal/             # Lake + Lean 4 proofs
├── docs/               # MkDocs site, architecture notes
├── test-vectors/       # Canonical inputs/outputs
├── artifacts/          # Generated Lean/R1CS artifacts
└── scripts/            # Bootstrap and CI helpers
```

---

## Getting Started

### Prerequisites
- Rust 1.77+ (`rustup`, add `clippy`, `rustfmt`).
- C++20 toolchain, CMake 3.26+, Ninja (recommended).
- vcpkg (`./scripts/setup.sh`) for Microsoft SEAL dependencies.
- Python 3.11+ for docs and automation.
- Lean 4 via Lake (`formal/lean-toolchain`).

### Build & Test
```bash
# Rust workspace
cd rust-api
cargo build --workspace
cargo test --workspace

# C++ core
cd ../cpp-core
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build

# Dudect-style timing sanity check
cmake --build build --target dudect_sampler
./build/dudect_sampler  # writes artifacts/dudect/gaussian_sampler_report.md

# Modular arithmetic timing sweep
cargo run --manifest-path ../rust-api/Cargo.toml \
  -p lambda-snark --bin mod_arith_timing
# writes artifacts/dudect/mod_arith_report.md (add/sub/pow/inverse)
cargo run -p lambda-snark --bin mod_arith_timing \
  # writes artifacts/dudect/mod_arith_report.md (add/sub/pow/inverse)

# Lean proofs
cd ../formal
lake build LambdaSNARK
```

### CLI Example
```bash
cargo run -p lambda-snark-cli --release -- healthcare \
  --input ../test-vectors/tv-0-linear-system/input.json
# Output: artifacts/r1cs/healthcare.term (treat as sensitive if witnesses are real).
```

See `PROJECT_SETUP.md` and `TESTING.md` for extended guidance.

---

## Current Capabilities
- NTT-friendly modulus 17592169062401 (M7.2) with precomputed roots of unity.
- Zero-knowledge blinding (M5.2) validated via simulator tests.
- 150+ unit/integration checks across Rust, C++, and Lean layers; healthcare shared module in sync.
- Lean build (`lake build`) succeeds; soundness theorem proven in `formal/` (M8), zero-knowledge proof underway.

---

## Limitations
- Modular arithmetic is not constant-time (timing leakage) until M7.5.
- FFI sanitizers and fuzzing harnesses are queued (Q1 2026).
- External audit (Trail of Bits/NCC Group) planned for M10.
- Lean zero-knowledge proof is still in progress (`formal/`); soundness is proven (M8).

Risk register and mitigations are tracked in `SECURITY.md`.

---

## Standards
- Run `cargo fmt` and `cargo clippy --workspace --all-targets`.
- Execute `cargo test -p lambda-snark --no-run` before PRs.
- C++ checks: `ctest --test-dir cpp-core/build`.
- Lean: `lake build LambdaSNARK`.
- Do not commit sensitive artifacts outside `artifacts/`.

---

## Roadmap (excerpt)
- ✅ M5.1: O(m log m) NTT implementation.
- ✅ M5.2: Zero-knowledge blinding and simulator tests.
- ✅ M7.2: Prime modulus swap and soundness restoration.
- 🔄 M7.4: Expand to >200 tests (property, fuzz).
- 🔄 M7.5: Constant-time rewrite validated with dudect.
- ✅ M8: Lean soundness proof finalized; zero-knowledge proof continues in M9.
- 🔜 M9: Lean zero-knowledge proof scripts.
- 🔜 M10: External audit and hardening backlog.

Full milestone map lives in `ROADMAP.md`.

---

## Community
- Issues: GitHub tracker (non-security).
- Security: follow `SECURITY.md` (email, PGP).
- Discussions: GitHub Discussions, SafeAGI Zulip `#lambda-snark`.
- Contributing: read `CONTRIBUTING.md`; security-sensitive changes need dual review.

---

## License
Dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE); contributions imply agreement with both.

---

Last updated: 2025-11-23
