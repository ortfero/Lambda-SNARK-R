# Lean 4 Formal Verification

This directory contains formal proofs of ΛSNARK-R properties in Lean 4.

## Structure

```
formal/
├── lakefile.lean          # Lake build configuration
├── Main.lean              # Entry point
├── LambdaSNARK.lean       # Root module
└── LambdaSNARK/
    ├── Core.lean          # Core definitions
    ├── Soundness.lean     # Soundness theorem
    ├── ZeroKnowledge.lean # Zero-knowledge theorem (TODO)
    └── Completeness.lean  # Completeness theorem (TODO)
```

## Building

```bash
# Install Lean 4
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Build
lake build

# Check proofs
lake env lean LambdaSNARK/Soundness.lean
```

## Status

**Current (v0.1.0-alpha)**:
- ✅ Basic definitions (R1CS, Field, Params)
- ✅ Soundness statement (proof pending)
- ✅ Zero-knowledge implementation (M5.2, statement TODO)
- ⏳ Completeness statement (TODO)

**Target (v1.0.0)**:
- ✅ Soundness proof (under Module-SIS + ROM)
- ✅ Zero-knowledge proof (under Module-LWE + σ bounds)
- ✅ Completeness proof
- ✅ Integration with C++/Rust (extract VK as Lean terms)

## Theorems

### Soundness

```lean
theorem knowledge_soundness
  (pp : PP R) (vk : VK R) (A : Adversary)
  (h_sis : ModuleSIS_Hard pp.vc_pp)
  (h_rom : RandomOracle_Model pp.hash) :
  ∃ (E : Extractor), ∀ x,
    Pr[Verify vk x (A x)] ≥ ε →
    Pr[∃ w, satisfies vk.C x w ∧ E A x = some w] ≥ ε - negl(λ)
```

### Zero-Knowledge

```lean
theorem zero_knowledge
  (pp : PP R) (vk : VK R)
  (h_lwe : ModuleLWE_Hard pp.vc_pp) :
  ∃ (Sim : Simulator), ∀ (D : Distinguisher),
    |Pr[D (Real vk)] - Pr[D (Sim vk)]| ≤ negl(λ)
```

## References

- Lean 4 manual: https://leanprover.github.io/lean4/doc/
- Mathlib4: https://github.com/leanprover-community/mathlib4
- Functional correctness: https://lean-fro.org/
