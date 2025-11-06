# Security Policy

## Reporting a Vulnerability

**DO NOT** open a public issue for security vulnerabilities.

Instead, please report security issues to:
📧 **security@lambda-snark.org**

Include:
1. Description of the vulnerability
2. Steps to reproduce
3. Affected versions
4. Potential impact
5. Suggested fix (if any)

We will respond within **72 hours** and provide a timeline for fixes.

## Security Guarantees (Current State)

⚠️ **WARNING**: This software is in **early development** (v0.1.0-alpha).

**Current Status**:
- ❌ **NOT AUDITED**: No professional security audit conducted
- ❌ **STUB IMPLEMENTATIONS**: Many cryptographic functions are incomplete
- ❌ **NOT PRODUCTION-READY**: Do not use for real-world applications

## Security Roadmap

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| Internal code review | Q1 2026 | ⏳ Planned |
| Constant-time validation (dudect) | Q1 2026 | ⏳ Planned |
| Formal verification (Lean 4) | Q2 2026 | ⏳ Planned |
| External audit (Trail of Bits) | Q2 2026 | ⏳ Planned |
| Public release (1.0.0) | Q3 2026 | ⏳ Planned |

## Known Issues

- **Timing Attacks**: Not all comparisons are constant-time
- **Gaussian Sampling**: Uses insecure RNG (placeholder)
- **Memory Safety**: C++ core has unsafe blocks (no audit)

## Disclosure Policy

We follow **coordinated disclosure**:
1. Report received → acknowledged within 72h
2. Fix developed → 30-90 days
3. Fix released → disclosure after 7 days
4. Public advisory → CVE assigned

## Hall of Fame

Contributors who responsibly disclose vulnerabilities will be listed here.

---

Thank you for helping keep ΛSNARK-R secure! 🔒
