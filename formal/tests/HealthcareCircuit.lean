/-
Lean representation of the healthcare R1CS example exported from Rust.
This prepares concrete data for later integration into the forking
infrastructure proofs.
-/

import LambdaSNARK.Core
import Mathlib/Data/ZMod/PrimitiveRoot
import Mathlib/GroupTheory/OrderOfElement

open LambdaSNARK
open scoped BigOperators

namespace LambdaSNARK.Tests

noncomputable section

private def q : ℕ := 2013265921

/-- Finite field used by the healthcare circuit. -/
@[simp] abbrev HealthcareField : Type := ZMod q

/-- Sparse matrix A for the healthcare circuit. The final four rows are padded with zeros. -/
noncomputable def healthcareMatrixA : SparseMatrix HealthcareField :=
  { nRows := 10
    nCols := 10
    entries :=
      [ (0, 5, (1 : HealthcareField))
      , (1, 6, 1)
      , (2, 7, 1)
      , (3, 5, 1)
      , (4, 8, 1)
      , (5, 0, 1)
      , (5, 9, (2 : HealthcareField)) ] }

/-- Sparse matrix B for the healthcare circuit. The padded rows remain zero. -/
noncomputable def healthcareMatrixB : SparseMatrix HealthcareField :=
  { nRows := 10
    nCols := 10
    entries :=
      [ (0, 0, (-1 : HealthcareField))
      , (0, 5, 1)
      , (1, 0, (-1 : HealthcareField))
      , (1, 6, 1)
      , (2, 0, (-1 : HealthcareField))
      , (2, 7, 1)
      , (3, 6, 1)
      , (4, 7, 1)
      , (5, 0, 1) ] }

/-- Sparse matrix C for the healthcare circuit. -/
noncomputable def healthcareMatrixC : SparseMatrix HealthcareField :=
  { nRows := 10
    nCols := 10
    entries :=
      [ (3, 8, (1 : HealthcareField))
      , (4, 9, 1)
      , (5, 1, 1) ] }

/-- R1CS instance describing the healthcare risk circuit. -/
noncomputable def healthcareR1CS : R1CS HealthcareField :=
  { nVars := 10
    nCons := 10
    nPub := 2
    A := healthcareMatrixA
    B := healthcareMatrixB
    C := healthcareMatrixC
    h_dim_A := by
      constructor <;> decide
    h_dim_B := by
      constructor <;> decide
    h_dim_C := by
      constructor <;> decide
    h_pub_le := by decide }

/-- Witness vector witnessing high-risk diagnosis. -/
noncomputable def healthcareWitness : Witness HealthcareField healthcareR1CS.nVars
  | ⟨0, _⟩ => 1
  | ⟨1, _⟩ => 3
  | ⟨2, _⟩ => 142
  | ⟨3, _⟩ => 45
  | ⟨4, _⟩ => 31
  | ⟨5, _⟩ => 1
  | ⟨6, _⟩ => 1
  | ⟨7, _⟩ => 1
  | ⟨8, _⟩ => 1
  | ⟨9, _⟩ => 1

/-- Public inputs (constant 1 and reported risk score 3). -/
noncomputable def healthcarePublic : PublicInput HealthcareField healthcareR1CS.nPub :=
  extractPublic healthcareR1CS.h_pub_le healthcareWitness

lemma healthcarePublic_eval :
    healthcarePublic ⟨0, by decide⟩ = 1 ∧
    healthcarePublic ⟨1, by decide⟩ = 3 := by
  constructor <;> simp [healthcarePublic, healthcareWitness, extractPublic]

/-- The exported high-risk witness satisfies the healthcare R1CS instance. -/
lemma healthcareWitness_satisfies :
    satisfies healthcareR1CS healthcareWitness := by
  classical
  refine (satisfies_iff_constraint_zero _ _).2 ?h
  intro i
  fin_cases i <;> decide

/-- Exponent producing a 10th root of unity in the healthcare field. -/
private def healthcareOmegaExp : ℕ := (q - 1) / 10

lemma healthcareOmegaExp_ne_zero : healthcareOmegaExp ≠ 0 := by
  -- Computed value: (2013265921 - 1) / 10 = 201326592
  decide

lemma healthcareOmegaExp_dvd : healthcareOmegaExp ∣ (q - 1) := by
  -- 10 divides q - 1, hence the quotient divides q - 1
  simpa [healthcareOmegaExp, q] using (Nat.div_dvd_of_dvd (by decide : 10 ∣ q - 1))

/-- Concrete 10th primitive root of unity in the healthcare field. -/
noncomputable def healthcareOmega : HealthcareField :=
  (ZMod.primitiveRoot q) ^ healthcareOmegaExp

lemma healthcareOmega_isPrimitiveRoot : IsPrimitiveRoot healthcareOmega 10 := by
  classical
  have hprime : Nat.Prime q := by decide
  have hroot := ZMod.isPrimitiveRoot_primitiveRoot (R := HealthcareField) hprime
  have horder : orderOf (ZMod.primitiveRoot q : HealthcareField) = q - 1 :=
    hroot.orderOf
  have hdiv : healthcareOmegaExp ∣ orderOf (ZMod.primitiveRoot q : HealthcareField) := by
    simpa [horder] using healthcareOmegaExp_dvd
  have hpow :=
    orderOf_pow_of_dvd (x := (ZMod.primitiveRoot q : HealthcareField))
      (n := healthcareOmegaExp) healthcareOmegaExp_ne_zero hdiv
  have hten :
      orderOf ((ZMod.primitiveRoot q : HealthcareField) ^ healthcareOmegaExp) = 10 := by
    simpa [healthcareOmegaExp, q, horder] using hpow
  simpa [IsPrimitiveRoot, healthcareOmega, hten]

end

end LambdaSNARK.Tests
