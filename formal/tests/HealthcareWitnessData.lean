/-
Concrete data tables extracted from the healthcare R1CS example.
These vectors expose the honest witness, domain points, and constraint
residuals required for downstream interpolation of the quotient
polynomial.
-/

import LambdaSNARK.Tests.HealthcareCircuit
import Mathlib/Data/List/FinRange
import Mathlib/Data/Vector

namespace LambdaSNARK.Tests

open LambdaSNARK

noncomputable section

/-- Full witness values for the healthcare circuit as a vector. -/
noncomputable def healthcareWitnessVector :
    Vector HealthcareField healthcareR1CS.nVars :=
  ⟨(List.finRange healthcareR1CS.nVars).map
      (fun i => healthcareWitness i), by
    simpa using List.length_map _ (List.finRange healthcareR1CS.nVars)⟩

@[simp]
lemma healthcareWitnessVector_get (i : Fin healthcareR1CS.nVars) :
    healthcareWitnessVector.get i = healthcareWitness i := by
  classical
  simp [healthcareWitnessVector, Vector.get]

/-- Public input prefix recorded separately for convenience. -/
noncomputable def healthcarePublicVector :
    Vector HealthcareField healthcareR1CS.nPub :=
  ⟨(List.finRange healthcareR1CS.nPub).map
      (fun i => healthcarePublic i), by
    simpa using List.length_map _ (List.finRange healthcareR1CS.nPub)⟩

@[simp]
lemma healthcarePublicVector_get (i : Fin healthcareR1CS.nPub) :
    healthcarePublicVector.get i = healthcarePublic i := by
  classical
  simp [healthcarePublicVector, Vector.get]

/-- Evaluation domain points ω^i for 0 ≤ i < m. -/
noncomputable def healthcareDomainVector :
    Vector HealthcareField healthcareR1CS.nVars :=
  ⟨(List.finRange healthcareR1CS.nVars).map
      (fun i => healthcareOmega ^ (i : ℕ)), by
    simpa using List.length_map _ (List.finRange healthcareR1CS.nVars)⟩

@[simp]
lemma healthcareDomainVector_get (i : Fin healthcareR1CS.nVars) :
    healthcareDomainVector.get i = healthcareOmega ^ (i : ℕ) := by
  classical
  simp [healthcareDomainVector, Vector.get]

/-- Constraint residuals for the honest healthcare witness (all zero). -/
noncomputable def healthcareConstraintResiduals :
    Vector HealthcareField healthcareR1CS.nCons :=
  ⟨(List.finRange healthcareR1CS.nCons).map
      (fun i => constraintPoly healthcareR1CS healthcareWitness i), by
    simpa using List.length_map _ (List.finRange healthcareR1CS.nCons)⟩

@[simp]
lemma healthcareConstraintResiduals_get
    (i : Fin healthcareR1CS.nCons) :
    healthcareConstraintResiduals.get i =
      constraintPoly healthcareR1CS healthcareWitness i := by
  classical
  simp [healthcareConstraintResiduals, Vector.get]

lemma healthcareConstraintResiduals_zero
    (i : Fin healthcareR1CS.nCons) :
    healthcareConstraintResiduals.get i = 0 := by
  classical
  have h :=
    (satisfies_iff_constraint_zero (cs := healthcareR1CS)
      (z := healthcareWitness)).mp healthcareWitness_satisfies i
  simpa [healthcareConstraintResiduals_get] using h

end

end LambdaSNARK.Tests
