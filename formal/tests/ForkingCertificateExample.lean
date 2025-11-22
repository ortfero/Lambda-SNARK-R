import LambdaSNARK.Core
import LambdaSNARK.ForkingInfrastructure
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Data.ZMod.Basic

namespace LambdaSNARK.Tests

open LambdaSNARK Polynomial

/-!
Illustrative instantiation of the forking infrastructure for a trivial
constraint system. The goal is to exercise the explicit remainder plumbing
introduced in the refactor and demonstrate that `extraction_soundness` applies
end-to-end on a toy circuit.
-/

/-- Dummy vector commitment that simply stores the committed list. -/
noncomputable def dummyVC (F : Type) [CommRing F] : VectorCommitment F where
  PP := Unit
  Commitment := List F
  Opening := Unit
  setup _ := ()
  commit _ v _ := v
  openProof _ _ _ _ := ()
  verify _ _ _ _ _ := true
  binding := by
    intro _ v₁ v₂ _ _ h_ne h_commit
    exact h_ne (by simpa using h_commit)
  correctness := by intro _ _ _ _; simp

/-- Sparse matrix with no non-zero entries. -/
def trivialSparseMatrix : SparseMatrix (ZMod 2) where
  nRows := 1
  nCols := 1
  entries := []

/-- Trivial R1CS instance whose single constraint is always satisfied. -/
def trivialR1CS : R1CS (ZMod 2) where
  nVars := 1
  nCons := 1
  nPub := 0
  A := trivialSparseMatrix
  B := trivialSparseMatrix
  C := trivialSparseMatrix
  h_dim_A := by simp [trivialSparseMatrix]
  h_dim_B := by simp [trivialSparseMatrix]
  h_dim_C := by simp [trivialSparseMatrix]
  h_pub_le := by decide

namespace Trivial

noncomputable instance : CommRing (ZMod 2) := inferInstance

noncomputable def VC : VectorCommitment (ZMod 2) := dummyVC (ZMod 2)

noncomputable def tStub : Transcript (ZMod 2) VC :=
{ pp := ()
  cs := trivialR1CS
  x := fun i => by cases i
  domainSize := 1
  omega := 1
  comm_Az := []
  comm_Bz := []
  comm_Cz := []
  comm_quotient := []
  quotient_poly := 0
  quotient_rand := 0
  quotient_commitment_spec := by simp [dummyVC]
  view := {
    alpha := 0
    Az_eval := 0
    Bz_eval := 0
    Cz_eval := 0
    quotient_eval := 0
    vanishing_eval := 0
    main_eq := verifierView_zero_eq (_F := ZMod 2)
  }
  challenge_β := 0
  opening_Az_α := ()
  opening_Bz_β := ()
  opening_Cz_α := ()
  opening_quotient_α := ()
  valid := true }

noncomputable def tStubFork : Transcript (ZMod 2) VC :=
{ tStub with
  view := { tStub.view with alpha := 1,
    main_eq := verifierView_zero_eq (_F := ZMod 2) } }

lemma stub_is_valid_fork : is_valid_fork VC tStub tStubFork := by
  classical
  unfold is_valid_fork tStub tStubFork
  simp [Transcript.ext_iff]

lemma stub_quotient_diff_zero :
    extract_quotient_diff VC trivialR1CS tStub tStubFork
      stub_is_valid_fork 1 1 = 0 := rfl

lemma constraintPoly_trivial_zero
    (z : Witness (ZMod 2) trivialR1CS.nVars) (i : Fin trivialR1CS.nCons) :
    constraintPoly trivialR1CS z i = 0 := by
  classical
  simp [constraintPoly, trivialR1CS, trivialSparseMatrix]

lemma stub_extract_witness_eq_zero
    (x : PublicInput (ZMod 2) trivialR1CS.nPub) :
    extract_witness VC trivialR1CS
      (extract_quotient_diff VC trivialR1CS tStub tStubFork
        stub_is_valid_fork 1 1)
      1 1 rfl x = fun _ => 0 := by
  classical
  funext i
  have hx : ¬ (i : ℕ) < trivialR1CS.nPub := by
    simp [trivialR1CS]
  simp [extract_witness, hx, stub_quotient_diff_zero]

lemma stub_constraint_zero
    (x : PublicInput (ZMod 2) trivialR1CS.nPub)
    (i : Fin trivialR1CS.nCons) :
    constraintPoly trivialR1CS
      (extract_witness VC trivialR1CS
        (extract_quotient_diff VC trivialR1CS tStub tStubFork
          stub_is_valid_fork 1 1)
        1 1 rfl x) i = 0 := by
  classical
  have hw := stub_extract_witness_eq_zero x
  simpa [hw] using constraintPoly_trivial_zero (z := fun _ => 0) i

lemma stub_constraint_numerator_poly_eq_zero
    (x : PublicInput (ZMod 2) trivialR1CS.nPub) :
    LambdaSNARK.constraintNumeratorPoly trivialR1CS
        (extract_witness VC trivialR1CS
          (extract_quotient_diff VC trivialR1CS tStub tStubFork
            stub_is_valid_fork 1 1)
          1 1 rfl x) 1 = 0 := by
  classical
  have hw := stub_extract_witness_eq_zero x
  simp [hw, LambdaSNARK.constraintNumeratorPoly, LambdaSNARK.constraintAzPoly,
    LambdaSNARK.constraintBzPoly, LambdaSNARK.constraintCzPoly, lagrange_interpolate,
    trivialR1CS, trivialSparseMatrix, evaluateConstraintA, evaluateConstraintB,
    evaluateConstraintC]

lemma stub_constraint_numerator_zero
    (x : PublicInput (ZMod 2) trivialR1CS.nPub) :
    (LambdaSNARK.constraintNumeratorPoly trivialR1CS
        (extract_witness VC trivialR1CS
          (extract_quotient_diff VC trivialR1CS tStub tStubFork
            stub_is_valid_fork 1 1)
          1 1 rfl x) 1).eval 1 = 0 := by
  classical
  have hpoly := stub_constraint_numerator_poly_eq_zero x
  simp [hpoly]

noncomputable def stubEquations :
    ForkingVerifierEquations VC trivialR1CS tStub tStubFork stub_is_valid_fork :=
{ m := 1
  ω := 1
  h_m_vars := rfl
  h_m_cons := rfl
  h_primitive := IsPrimitiveRoot.one
  quotient_eval := by
    intro x i
    have hi : (i : ℕ) = 0 := by decide
    subst hi
    simpa [stub_quotient_diff_zero] using stub_constraint_zero x ⟨0, by decide⟩
  quotient_diff_natDegree_lt_domain := by
    intro x
    have hnum := stub_constraint_numerator_poly_eq_zero x
    simp [stub_quotient_diff_zero, hnum] }

lemma stub_remainder_zero :
    (extract_quotient_diff VC trivialR1CS tStub tStubFork stub_is_valid_fork
        stubEquations.m stubEquations.ω)
        %ₘ vanishing_poly stubEquations.m stubEquations.ω = 0 := by
  classical
  simp [stubEquations, stub_quotient_diff_zero]

lemma stub_extraction_soundness
    (x : PublicInput (ZMod 2) trivialR1CS.nPub)
    (h_sis : ModuleSIS_Hard 256 2 12289 1024) :
    satisfies trivialR1CS
      (extract_witness VC trivialR1CS
        (extract_quotient_diff VC trivialR1CS tStub tStubFork
          stub_is_valid_fork stubEquations.m stubEquations.ω)
        stubEquations.m stubEquations.ω stubEquations.h_m_vars x) :=
  by
    let assumptions : LambdaSNARK.SoundnessCtx (ZMod 2) VC trivialR1CS :=
      LambdaSNARK.SoundnessAssumptions.simple (VC := VC) (cs := trivialR1CS) h_sis
    simpa using
      (extraction_soundness (VC := VC) (cs := trivialR1CS)
        (t1 := tStub) (t2 := tStubFork)
        stub_is_valid_fork stubEquations stub_remainder_zero
        assumptions) x

end Trivial

end LambdaSNARK.Tests
