import LambdaSNARK.Core
import LambdaSNARK.Polynomial
import LambdaSNARK.Soundness
import LambdaSNARK.Completeness

/-!
# ΛSNARK-R Formal Verification

Root module for ΛSNARK-R formal verification in Lean 4.

## Modules

- `Core`: Basic definitions (R1CS, Field, Commitment, Proof)
- `Polynomial`: Lagrange interpolation, polynomial division
- `Soundness`: Knowledge soundness theorem (Module-SIS + ROM)
- `Completeness`: Perfect completeness theorem

## Main Theorems

1. **Soundness** (`knowledge_soundness`): 
   Extractable witness from accepting proofs

2. **Completeness** (`completeness`): 
   Valid witnesses always produce accepting proofs

3. **Zero-Knowledge** (TODO): 
   Proofs reveal nothing beyond validity

## Usage

```lean
import LambdaSNARK

-- Example: Define an R1CS instance
def example_r1cs : R1CS (ZMod 97) := {
  nVars := 4,
  nCons := 1,
  nPub := 1,
  A := ⟨1, 4, [(0, 1, 1)]⟩,
  B := ⟨1, 4, [(0, 2, 1)]⟩,
  C := ⟨1, 4, [(0, 3, 1)]⟩,
  h_dim_A := by simp,
  h_dim_B := by simp,
  h_dim_C := by simp
}

-- Witness: [1, 2, 3, 6] where 2*3=6
def example_witness : Witness (ZMod 97) 4 :=
  ![1, 2, 3, 6]

-- Check satisfaction
#check satisfies example_r1cs example_witness
```
-/
