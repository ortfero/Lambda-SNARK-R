/-
Honest quotient polynomial for the healthcare R1CS instance.  Constructed via
Lagrange interpolation over the evaluation domain and equipped with evaluation
and vanishing proofs required for the forking equations.
-/

import LambdaSNARK.Core
import LambdaSNARK.Polynomial
import LambdaSNARK.Tests.HealthcareWitnessData
import Mathlib.Tactic

namespace LambdaSNARK.Tests

open LambdaSNARK

noncomputable section

open scoped BigOperators

/-- Evaluate `A` against the honest witness. -/
noncomputable def healthcareAzEval
    (i : Fin healthcareR1CS.nCons) : HealthcareField :=
  ∑ j : Fin healthcareR1CS.nVars,
    healthcareR1CS.A.toDense
      (Fin.cast healthcareR1CS.h_dim_A.1.symm i)
      (Fin.cast healthcareR1CS.h_dim_A.2.symm j) * healthcareWitness j

/-- Evaluate `B` against the honest witness. -/
noncomputable def healthcareBzEval
    (i : Fin healthcareR1CS.nCons) : HealthcareField :=
  ∑ j : Fin healthcareR1CS.nVars,
    healthcareR1CS.B.toDense
      (Fin.cast healthcareR1CS.h_dim_B.1.symm i)
      (Fin.cast healthcareR1CS.h_dim_B.2.symm j) * healthcareWitness j

/-- Evaluate `C` against the honest witness. -/
noncomputable def healthcareCzEval
    (i : Fin healthcareR1CS.nCons) : HealthcareField :=
  ∑ j : Fin healthcareR1CS.nVars,
    healthcareR1CS.C.toDense
      (Fin.cast healthcareR1CS.h_dim_C.1.symm i)
      (Fin.cast healthcareR1CS.h_dim_C.2.symm j) * healthcareWitness j

lemma healthcareConstraint_zero
    (i : Fin healthcareR1CS.nCons) :
    healthcareAzEval i * healthcareBzEval i = healthcareCzEval i := by
  classical
  have hsat :=
    (satisfies_iff_constraint_zero
      (cs := healthcareR1CS)
      (z := healthcareWitness)).mp healthcareWitness_satisfies i
  simpa [constraintPoly, healthcareAzEval, healthcareBzEval, healthcareCzEval]
    using hsat

/-- Interpolate `A_z` over the evaluation domain. -/
noncomputable def healthcareAzPoly : Polynomial HealthcareField :=
  lagrange_interpolate healthcareR1CS.nCons healthcareOmega
    (fun i => healthcareAzEval i)

/-- Interpolate `B_z` over the evaluation domain. -/
noncomputable def healthcareBzPoly : Polynomial HealthcareField :=
  lagrange_interpolate healthcareR1CS.nCons healthcareOmega
    (fun i => healthcareBzEval i)

/-- Interpolate `C_z` over the evaluation domain. -/
noncomputable def healthcareCzPoly : Polynomial HealthcareField :=
  lagrange_interpolate healthcareR1CS.nCons healthcareOmega
    (fun i => healthcareCzEval i)

lemma healthcareAzPoly_eval (i : Fin healthcareR1CS.nCons) :
    healthcareAzPoly.eval (healthcareOmega ^ (i : ℕ)) = healthcareAzEval i := by
  classical
  unfold healthcareAzPoly
  simpa using
    lagrange_interpolate_eval
      (m := healthcareR1CS.nCons)
      (ω := healthcareOmega)
      (hprim := healthcareOmega_isPrimitiveRoot)
      (evals := fun i => healthcareAzEval i)
      (i := i)

lemma healthcareBzPoly_eval (i : Fin healthcareR1CS.nCons) :
    healthcareBzPoly.eval (healthcareOmega ^ (i : ℕ)) = healthcareBzEval i := by
  classical
  unfold healthcareBzPoly
  simpa using
    lagrange_interpolate_eval
      (m := healthcareR1CS.nCons)
      (ω := healthcareOmega)
      (hprim := healthcareOmega_isPrimitiveRoot)
      (evals := fun i => healthcareBzEval i)
      (i := i)

lemma healthcareCzPoly_eval (i : Fin healthcareR1CS.nCons) :
    healthcareCzPoly.eval (healthcareOmega ^ (i : ℕ)) = healthcareCzEval i := by
  classical
  unfold healthcareCzPoly
  simpa using
    lagrange_interpolate_eval
      (m := healthcareR1CS.nCons)
      (ω := healthcareOmega)
      (hprim := healthcareOmega_isPrimitiveRoot)
      (evals := fun i => healthcareCzEval i)
      (i := i)

/-- Numerator polynomial `(A_z·B_z - C_z)(X)`. -/
noncomputable def healthcareNumeratorPoly : Polynomial HealthcareField :=
  healthcareAzPoly * healthcareBzPoly - healthcareCzPoly

lemma healthcareNumeratorPoly_eval
    (i : Fin healthcareR1CS.nCons) :
    healthcareNumeratorPoly.eval (healthcareOmega ^ (i : ℕ)) = 0 := by
  classical
  unfold healthcareNumeratorPoly
  simp [Polynomial.eval_mul, healthcareAzPoly_eval, healthcareBzPoly_eval,
        healthcareCzPoly_eval, healthcareConstraint_zero]

/-- Honest quotient polynomial obtained by dividing the numerator by `Z_H`. -/
noncomputable def healthcareQuotientPoly : Polynomial HealthcareField :=
  healthcareNumeratorPoly /ₘ
    vanishing_poly healthcareR1CS.nCons healthcareOmega

lemma healthcareQuotientPoly_remainder_zero :
    healthcareNumeratorPoly %ₘ
      vanishing_poly healthcareR1CS.nCons healthcareOmega = 0 := by
  classical
  have hzero :
      ∀ i : Fin healthcareR1CS.nCons,
        healthcareNumeratorPoly.eval (healthcareOmega ^ (i : ℕ)) = 0 :=
    healthcareNumeratorPoly_eval
  have :=
    (remainder_zero_iff_vanishing
      (F := HealthcareField)
      (f := healthcareNumeratorPoly)
      (m := healthcareR1CS.nCons)
      (ω := healthcareOmega)
      (hω := healthcareOmega_isPrimitiveRoot)).mpr hzero
  simpa using this

/-- Precomputed coefficient list for the honest quotient polynomial. -/
noncomputable def healthcareQuotientCoeffs : List HealthcareField :=
  [ 536591292
  , 151123296
  , 1268815861
  , 1689701572
  , 1641423289
  , 1200004351
  , 1233086762
  , 1322307170
  , 1694015127 ]

/-- The interpolated quotient polynomial matches the precomputed coefficients. -/
lemma healthcareQuotientPoly_coeffList :
    healthcareQuotientPoly.coeffList = healthcareQuotientCoeffs := by
  classical
  native_decide

/-- Honest quotient polynomial reconstructed directly from the coefficient table. -/
noncomputable def healthcareQuotientPolyExplicit : Polynomial HealthcareField :=
  Polynomial.ofList healthcareQuotientCoeffs

lemma healthcareQuotientPolyExplicit_coeffList :
    healthcareQuotientPolyExplicit.coeffList = healthcareQuotientCoeffs := by
  classical
  simpa [healthcareQuotientPolyExplicit] using
    (by decide :
      Polynomial.coeffList (Polynomial.ofList healthcareQuotientCoeffs)
        = healthcareQuotientCoeffs)

lemma healthcareQuotientPoly_eq_explicit :
    healthcareQuotientPoly = healthcareQuotientPolyExplicit := by
  classical
  apply coeffList_injective
  have h_interp := healthcareQuotientPoly_coeffList
  have h_exp := healthcareQuotientPolyExplicit_coeffList
  simpa [h_exp] using h_interp

lemma healthcareNumeratorPoly_div_eq_explicit :
    healthcareNumeratorPoly /ₘ
        vanishing_poly healthcareR1CS.nCons healthcareOmega =
      healthcareQuotientPolyExplicit := by
  classical
  simpa [healthcareQuotientPoly, healthcareQuotientPoly_eq_explicit]

lemma healthcareNumeratorPoly_remainder_zero :
    healthcareNumeratorPoly %ₘ
        vanishing_poly healthcareR1CS.nCons healthcareOmega = 0 :=
  healthcareQuotientPoly_remainder_zero

-- #eval (healthcareQuotientPoly.coeffList)

end

end LambdaSNARK.Tests
