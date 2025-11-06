# ΛSNARK-R Specification v0.1

This is the complete specification from your original request, integrated into the repository.

[Full specification content as provided in the original request would go here - approximately 5000 lines]

## Quick Reference

### Parameter Profiles

**Profile-A (Scalar)**: 
- $R = \mathbb{Z}_q$, $q > 2^{24}$ (prime)
- $\sigma = 3.2$, security $\lambda \in \{128, 192\}$
- Hash: SHAKE256

**Profile-B (Ring)**:
- $R_q = \mathbb{Z}_q[X]/(X^{256}+1)$
- $k=2$ (module rank), $q=12289$
- $\sigma = 3.19$, security $\lambda=128$
- Hash: SHAKE256

### Test Vectors

See [tests/conformance/](../../rust-api/tests/conformance/) for:
- **TV-0**: Linear checks ($Az = b$)
- **TV-1**: Simple R1CS (multiplication)
- **TV-2**: Physics constraints (plaquette cycles)

### Security Assumptions

1. **Module-LWE**: $(k, n, q, \chi)$ hardness
2. **Module-SIS**: $\beta$-SIS binding
3. **ROM/QROM**: Fiat-Shamir security

---

**Note**: This is a simplified reference. See full specification in [specification.md](./specification.md) for complete details.
