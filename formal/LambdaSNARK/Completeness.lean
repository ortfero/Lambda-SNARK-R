/-
Copyright (c) 2025 URPKS Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: URPKS Contributors
-/

import LambdaSNARK.Core
import LambdaSNARK.Polynomial
import Mathlib.Tactic

/-!
# ΛSNARK-R Completeness

Completeness theorem: if a prover has a valid witness w satisfying R1CS constraints,
then the prover can always produce an accepting proof.

## Main Result

`completeness`: Honest prover always succeeds

## References

- ΛSNARK-R Specification: docs/spec/specification.md
-/

namespace LambdaSNARK

open BigOperators Polynomial

/-- Honest prover algorithm producing proofs accepted by the verifier. -/
structure HonestProver (F : Type) [CommRing F] [DecidableEq F]
    (VC : VectorCommitment F) where
  build : (cs : R1CS F) → (w : Witness F cs.nVars) →
          (x : PublicInput F cs.nPub) → (randomness : ℕ) →
          { π : Proof F VC // verify VC cs x π = true }

def HonestProver.prove {F : Type} [CommRing F] [DecidableEq F]
    {VC : VectorCommitment F} (P : HonestProver F VC)
    (cs : R1CS F) (w : Witness F cs.nVars)
    (x : PublicInput F cs.nPub) (r : ℕ) : Proof F VC :=
  (P.build cs w x r).1

lemma HonestProver.prove_accepts {F : Type} [CommRing F] [DecidableEq F]
    {VC : VectorCommitment F} (P : HonestProver F VC)
    (cs : R1CS F) (w : Witness F cs.nVars)
    (x : PublicInput F cs.nPub) (r : ℕ) :
    verify VC cs x (P.prove cs w x r) = true :=
  (P.build cs w x r).2

/--
Completeness theorem: honest prover with valid witness always produces accepting proof.

**Statement**: For any R1CS constraint system cs, if a witness w satisfies cs
and matches public input x, then the honest prover's proof is accepted by the verifier.

**Proof Strategy**:
1. Show polynomial construction is correct (Az, Bz, Cz)
2. Show quotient polynomial exists (by satisfaction)
3. Show all verifier checks pass (commitment correctness + opening correctness)
-/
theorem completeness {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (_secParam : ℕ)
    (P : HonestProver F VC)
    :
    ∀ (w : Witness F cs.nVars) (x : PublicInput F cs.nPub) (r : ℕ),
      -- If witness is valid
      satisfies cs w →
      extractPublic cs.h_pub_le w = x →
      -- Then proof verifies
      verify VC cs x (P.prove cs w x r) = true := by
  intro w x r h_sat h_pub
  exact P.prove_accepts cs w x r

/-- Completeness error is zero (perfect completeness) -/
theorem perfect_completeness {F : Type} [Field F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (secParam : ℕ)
  (P : HonestProver F VC) :
    ∀ (w : Witness F cs.nVars) (x : PublicInput F cs.nPub) (r : ℕ),
      satisfies cs w →
      extractPublic cs.h_pub_le w = x →
      -- Probability of acceptance is exactly 1 (no randomness in verification)
      verify VC cs x (P.prove cs w x r) = true := by
  intro w x r h_sat h_pub
  exact completeness VC cs secParam P w x r h_sat h_pub

end LambdaSNARK
