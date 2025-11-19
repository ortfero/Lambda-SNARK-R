/-
Concrete `ProtocolForkingEquations` witness for the healthcare R1CS example.
The heavy algebraic equalities are currently bundled as axioms that will be
replaced by mechanised proofs in subsequent iterations.
-/

import LambdaSNARK.ForkingEquations
import LambdaSNARK.Tests.HealthcareCircuit
import LambdaSNARK.Tests.HealthcareWitnessData

namespace LambdaSNARK.Tests

open LambdaSNARK

noncomputable section

variable {VC : VectorCommitment HealthcareField}
variable (t1 t2 : Transcript HealthcareField VC)
variable (h_fork : is_valid_fork VC t1 t2)

/--

lemma healthcare_extract_quotient_eq_explicit
    (VC : VectorCommitment HealthcareField)
    (t1 t2 : Transcript HealthcareField VC)
    (h_fork : is_valid_fork VC t1 t2)
    (h_commit :
      VC.commit t1.pp healthcareQuotientCoeffs t1.quotient_rand = t1.comm_quotient) :
    extract_quotient_diff VC healthcareR1CS t1 t2 h_fork
      healthcareR1CS.nVars healthcareOmega =
        healthcareQuotientPolyExplicit := by
  classical
  have h_eq :
      VC.commit t1.pp (Polynomial.coeffList t1.quotient_poly) t1.quotient_rand =
        VC.commit t1.pp healthcareQuotientCoeffs t1.quotient_rand := by
    calc
      VC.commit t1.pp (Polynomial.coeffList t1.quotient_poly) t1.quotient_rand
          = t1.comm_quotient := t1.quotient_commitment_spec
      _ = VC.commit t1.pp healthcareQuotientCoeffs t1.quotient_rand :=
        h_commit.symm
  have h_list := (VC.commit_injective t1.pp t1.quotient_rand) h_eq
  have h_poly : t1.quotient_poly = healthcareQuotientPolyExplicit := by
    apply coeffList_injective
    simpa [healthcareQuotientPolyExplicit_coeffList] using h_list
  simpa [extract_quotient_diff, h_poly]

/-- Conjectured constraint interpolation for the healthcare circuit: the
polynomial extracted from any fork evaluates to the constraint residuals at all
domain points.  Marked as an axiom until the explicit interpolation proof is
formalised.
-/
axiom healthcare_quotient_eval
    (VC : VectorCommitment HealthcareField)
    (t1 t2 : Transcript HealthcareField VC)
    (h_fork : is_valid_fork VC t1 t2) :
    (x : PublicInput HealthcareField healthcareR1CS.nPub) →
    ∀ i : Fin healthcareR1CS.nCons,
      let q := extract_quotient_diff VC healthcareR1CS t1 t2 h_fork
        healthcareR1CS.nVars healthcareOmega
      let w := extract_witness VC healthcareR1CS q healthcareR1CS.nVars
        healthcareOmega rfl x
      q.eval (healthcareOmega ^ (i : ℕ)) =
        constraintPoly healthcareR1CS w i

/--
Conjectured vanishing relation for the healthcare circuit: the interpolating
polynomial has zero remainder when divided by the vanishing polynomial of the
execution domain.  Introduced as an axiom pending elimination via an explicit
divisibility argument.
-/
axiom healthcare_remainder_zero
    (VC : VectorCommitment HealthcareField)
    (t1 t2 : Transcript HealthcareField VC)
    (h_fork : is_valid_fork VC t1 t2) :
    (extract_quotient_diff VC healthcareR1CS t1 t2 h_fork
      healthcareR1CS.nVars healthcareOmega) %ₘ
        vanishing_poly healthcareR1CS.nVars healthcareOmega = 0

axiom healthcare_quotient_diff_natDegree_lt_domain
    (VC : VectorCommitment HealthcareField)
    (t1 t2 : Transcript HealthcareField VC)
    (h_fork : is_valid_fork VC t1 t2)
    (x : PublicInput HealthcareField healthcareR1CS.nPub) :
    (extract_quotient_diff VC healthcareR1CS t1 t2 h_fork
          healthcareR1CS.nVars healthcareOmega -
        LambdaSNARK.constraintNumeratorPoly healthcareR1CS
          (extract_witness VC healthcareR1CS
            (extract_quotient_diff VC healthcareR1CS t1 t2 h_fork
              healthcareR1CS.nVars healthcareOmega)
            healthcareR1CS.nVars healthcareOmega rfl x)
          healthcareOmega).natDegree < healthcareR1CS.nVars

/-- Witness packaging the healthcare-specific verifier equations. -/
noncomputable def healthcareForkingCore
    (VC : VectorCommitment HealthcareField)
    (t1 t2 : Transcript HealthcareField VC)
    (h_fork : is_valid_fork VC t1 t2) :
    ForkingVerifierEquationsCore VC healthcareR1CS t1 t2 h_fork :=
  { m := healthcareR1CS.nVars
    ω := healthcareOmega
    h_m_vars := rfl
    h_primitive := by
      simpa using healthcareOmega_isPrimitiveRoot
    quotient_eval := healthcare_quotient_eval VC t1 t2 h_fork
    quotient_diff_natDegree_lt_domain :=
      healthcare_quotient_diff_natDegree_lt_domain VC t1 t2 h_fork
    remainder_zero := healthcare_remainder_zero VC t1 t2 h_fork }

/-- The healthcare R1CS admits a square evaluation domain. -/
lemma healthcare_square :
    healthcareR1CS.nVars = healthcareR1CS.nCons := by decide

/-- Instance witnessing forking equations for the healthcare circuit. -/
noncomputable instance healthcareEquationWitness
    (VC : VectorCommitment HealthcareField) :
    ForkingEquationWitness VC healthcareR1CS where
  square := healthcare_square
  buildCore := healthcareForkingCore VC

/-- Convenience alias producing the protocol witness via typeclass search. -/
noncomputable def healthcareProtocol
    (VC : VectorCommitment HealthcareField) :
    ProtocolForkingEquations VC healthcareR1CS :=
  ForkingEquationWitness.protocolOf (VC := VC) (cs := healthcareR1CS)

/-- Convenience alias producing the provider required by the extractor. -/
noncomputable def healthcareProvider
    (VC : VectorCommitment HealthcareField) :
    ForkingEquationsProvider VC healthcareR1CS :=
  ForkingEquationWitness.providerOf (VC := VC) (cs := healthcareR1CS)

section Usage

variable (VC : VectorCommitment HealthcareField) (secParam : ℕ)
variable (A : Adversary HealthcareField VC) (ε : ℕ → ℝ)

/--
Application of `knowledge_soundness_of` specialised to the healthcare circuit.
Relies on the `ForkingEquationWitness` instance supplied above to discharge the
equation-provider requirement.
-/
lemma healthcare_knowledge_soundness
    (h_non_negl : NonNegligible ε)
    (h_mass : ε secParam * (Fintype.card HealthcareField : ℝ) ≥ 2)
    (h_sis : ModuleSIS_Hard 256 2 12289 1024)
    (h_rom : True) :
    ∃ (E : Extractor HealthcareField VC),
      E.poly_time ∧
      ∀ (x : PublicInput HealthcareField healthcareR1CS.nPub),
        (∃ π, verify VC healthcareR1CS x π = true) →
        (∃ w : Witness HealthcareField healthcareR1CS.nVars,
          satisfies healthcareR1CS w ∧
          extractPublic healthcareR1CS.h_pub_le w = x) := by
  simpa using
    (knowledge_soundness_of VC healthcareR1CS secParam A ε
      h_non_negl h_mass h_sis h_rom)

end Usage

end

end LambdaSNARK.Tests
