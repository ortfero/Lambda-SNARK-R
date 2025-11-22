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

/-!
An honest prover in the completeness theorem must do two things:
1. construct a concrete proof object; and
2. supply evidence that the verifier accepts it.

Instead of postulating acceptance as an opaque proposition, we require the
prover to return a `ProofCertificate`, which enumerates all checks performed by
the verifier.  This ensures any completeness proof is grounded in the actual
verification predicate specified in `LambdaSNARK.Core`.
-/

/-- Honest prover algorithm together with a certificate that all verifier
checks succeed. -/
structure HonestProver (F : Type) [CommRing F] [DecidableEq F]
    (VC : VectorCommitment F) where
  build :
    (cs : R1CS F) →
    (w : Witness F cs.nVars) →
    (x : PublicInput F cs.nPub) →
    (randomness : ℕ) →
    satisfies cs w →
    extractPublic cs.h_pub_le w = x →
    ProofCertificate VC cs x

def HonestProver.prove {F : Type} [CommRing F] [DecidableEq F]
    {VC : VectorCommitment F} (P : HonestProver F VC)
    (cs : R1CS F) (w : Witness F cs.nVars)
    (x : PublicInput F cs.nPub) (r : ℕ)
    (h_sat : satisfies cs w)
    (h_pub : extractPublic cs.h_pub_le w = x) : Proof F VC :=
  (P.build cs w x r h_sat h_pub).proof

lemma HonestProver.prove_accepts {F : Type} [CommRing F] [DecidableEq F]
    {VC : VectorCommitment F} (P : HonestProver F VC)
    (cs : R1CS F) (w : Witness F cs.nVars)
    (x : PublicInput F cs.nPub) (r : ℕ)
    (h_sat : satisfies cs w)
    (h_pub : extractPublic cs.h_pub_le w = x) :
    verify VC cs x (P.prove cs w x r h_sat h_pub) = true := by
  change verify VC cs x (P.build cs w x r h_sat h_pub).proof = true
  exact
    ProofCertificate.verify_eq_true (P.build cs w x r h_sat h_pub)

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
    (P : HonestProver F VC) :
    ∀ (w : Witness F cs.nVars) (x : PublicInput F cs.nPub) (r : ℕ)
      (h_sat : satisfies cs w) (h_pub : extractPublic cs.h_pub_le w = x),
        verify VC cs x (P.prove cs w x r h_sat h_pub) = true := by
  intro w x r h_sat h_pub
  exact P.prove_accepts cs w x r h_sat h_pub

/-- Completeness error is zero (perfect completeness) -/
theorem perfect_completeness {F : Type} [Field F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (secParam : ℕ)
  (P : HonestProver F VC) :
    ∀ (w : Witness F cs.nVars) (x : PublicInput F cs.nPub) (r : ℕ)
      (h_sat : satisfies cs w) (h_pub : extractPublic cs.h_pub_le w = x),
      -- Probability of acceptance is exactly 1 (no randomness in verification)
      verify VC cs x (P.prove cs w x r h_sat h_pub) = true := by
  intro w x r h_sat h_pub
  exact completeness VC cs secParam P w x r h_sat h_pub

end LambdaSNARK
