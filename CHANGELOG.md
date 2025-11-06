# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository structure
- C++ core with SEAL/NTL integration stubs
- Rust API with FFI bindings
- Basic commitment scheme interface
- GitHub Actions CI/CD
- Documentation infrastructure (mkdocs)
- Makefile for build automation

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0-alpha] - 2025-11-06

### Added
- Project initialization
- Hybrid architecture (C++ core + Rust API)
- Specification v0.1 (draft)

---

## Release Notes

### Version 0.1.0-alpha

**Release Date**: November 6, 2025  
**Status**: Pre-alpha, not for production

**Changes**:
- Initial scaffolding for ΛSNARK-R
- Stub implementations (LWE commitment, NTT)
- No security guarantees yet

**Known Issues**:
- NTT not implemented (uses identity transform)
- Commitment scheme incomplete (missing verification)
- No prover/verifier logic

**Next Milestone** (0.2.0-alpha):
- Full NTT implementation with NTL
- Complete LWE commitment + verification
- Basic R1CS prover (TV-0/1 tests passing)
