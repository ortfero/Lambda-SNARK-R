# Lean 4 Formal Verification

This directory hosts the Lean 4 project that validates the probabilistic
forking infrastructure of ΛSNARK-R. All principal lemmas are proved
constructively; the Lean kernel depends on no additional axioms.

## Layout

```
formal/
├── lakefile.lean          # Lake configuration
├── Main.lean              # Entry point that compiles the whole project
├── LambdaSNARK.lean       # Root module
└── LambdaSNARK/
    ├── ForkingInfrastructure.lean  # heavy-row, fork-success, extraction
    ├── Core.lean                    # R1CS, witnesses, adversary model
    ├── Polynomial.lean              # Polynomial toolkit and Z_H helpers
    ├── Soundness.lean               # Assembly of the knowledge theorem
    └── tests/                       # Certificate examples
```

## Build and Checks

```bash
# Build the library and regression lemmas
lake build LambdaSNARK

# Inspect a concrete module in interactive mode (example)
lake env lean LambdaSNARK/ForkingInfrastructure.lean

# Run Lean tests/examples
lake env lean tests/ForkingCertificateExample.lean
```

## Current Status (November 2025)

- ✅ Success probability lemma: `successProbability_eq_successfulRandomness_card_div`
- ✅ Heavy-row lemma: constructive reconstruction of "heavy" commitments
- ✅ Forking lower bound: `fork_success_bound` (ε²/2 − 1/|F|)
- ✅ Witness extraction: `tests/ForkingCertificateExample.lean`
- ✅ Healthcare quotient certificate: `tests/HealthcareQuotient.lean`
- ✅ Polynomial toolkit for division by Z_H and PMF links
- ⏳ Integration into `Soundness.lean` (full knowledge theorem)
- ⏳ Completeness and zero-knowledge (next milestones)

## Roadmap

- Connect the proven probabilistic lemmas with `Soundness.lean`
- Add new test circuits (PLAQUETTE, LWE, additional R1CS instances)
- Formalize completeness and zero-knowledge within Lean 4
- Keep CI running `lake build LambdaSNARK` on every commit

## Key Results

### Heavy-row and Fork Success

- `successProbability_eq_successfulRandomness_card_div`
- `heavy_row_lemma`
- `fork_success_bound`

These statements form the core of the forking argument and are already formally proved.

### Witness Extraction

```lean
theorem extraction_soundness … :=
  …
```

The complete linkage with concrete examples is available in
`tests/ForkingCertificateExample.lean` and `tests/HealthcareQuotient.lean`.

## References

- Lean 4 manual: https://leanprover.github.io/lean4/doc/
- Mathlib4: https://github.com/leanprover-community/mathlib4 (modules `PMF`, `ENNReal`, `Polynomial`)
- ΛSNARK-R project overview: ../README.md
