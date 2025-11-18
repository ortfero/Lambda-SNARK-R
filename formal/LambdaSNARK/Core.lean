/-
Copyright (c) 2025 URPKS Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: URPKS Contributors
-/

import Mathlib.Data.Fintype.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic

/-!
# ΛSNARK-R Core Definitions

This file contains the fundamental definitions for the ΛSNARK-R formal verification:
- Parameter structures
- Field/Ring types
- R1CS representation
- Polynomial operations
- Commitment scheme interfaces
- Proof structures

## References

- Specification: docs/spec/specification.md
- C++ implementation: cpp-core/
- Rust implementation: rust-api/

## Implementation Notes

This formalization follows the ΛSNARK-R protocol with:
- R1CS constraints (Az ∘ Bz = Cz)
- Lagrange interpolation
- Polynomial commitments (Module-LWE based)
- Fiat-Shamir transformation (Random Oracle Model)
-/

namespace LambdaSNARK

open BigOperators Polynomial Matrix

/-- Security level in bits -/
inductive SecurityLevel
  | bits128
  | bits192
  | bits256
  deriving Repr, DecidableEq

/-- Parameter profile -/
inductive Profile
  | scalarA (q : ℕ) (sigma : ℝ)
  | ringB (n k : ℕ) (q : ℕ) (sigma : ℝ)

/-- Public parameters -/
structure Params where
  security_level : SecurityLevel
  profile : Profile

/-- Field element in Z_q -/
def FiniteField (q : ℕ) := ZMod q

/-- Polynomial over field F -/
abbrev FieldPoly (F : Type) [CommRing F] := Polynomial F

/-- Sparse matrix representation (row pointers, column indices, values) -/
structure SparseMatrix (F : Type) [Ring F] where
  nRows : ℕ
  nCols : ℕ
  entries : List (ℕ × ℕ × F)  -- (row, col, value)
  deriving Repr

/-- Convert sparse matrix to dense (for proofs) -/
def SparseMatrix.toDense {F : Type} [Ring F] [DecidableEq F] [Zero F]
    (M : SparseMatrix F) : Matrix (Fin M.nRows) (Fin M.nCols) F :=
  fun i j =>
    match M.entries.filter (fun ⟨r, c, _⟩ => r = i.val ∧ c = j.val) with
    | [] => 0
    | entries => entries.foldl (fun acc ⟨_, _, v⟩ => acc + v) 0

/-- R1CS constraint system -/
structure R1CS (F : Type) [CommRing F] where
  nVars : ℕ                          -- Total witness size (n)
  nCons : ℕ                          -- Number of constraints (m)
  nPub : ℕ                           -- Number of public inputs (l)
  A : SparseMatrix F
  B : SparseMatrix F
  C : SparseMatrix F
  h_dim_A : A.nRows = nCons ∧ A.nCols = nVars
  h_dim_B : B.nRows = nCons ∧ B.nCols = nVars
  h_dim_C : C.nRows = nCons ∧ C.nCols = nVars
  h_pub_le : nPub ≤ nVars            -- Public inputs are prefix of witness

/-- Witness vector (includes public inputs at start) -/
def Witness (F : Type) (n : ℕ) := Fin n → F

/-- Public input vector (first l elements of witness) -/
def PublicInput (F : Type) (l : ℕ) := Fin l → F

/-- Extract public inputs from witness -/
def extractPublic {F : Type} {n l : ℕ} (h : l ≤ n) (w : Witness F n) : PublicInput F l :=
  fun i => w ⟨i.val, Nat.lt_of_lt_of_le i.isLt h⟩

/-- Satisfaction predicate for R1CS -/
def satisfies {F : Type} [CommRing F] [DecidableEq F] (cs : R1CS F) (z : Witness F cs.nVars) : Prop :=
  ∀ i : Fin cs.nCons,
    let iA := Fin.cast cs.h_dim_A.1.symm i
    let iB := Fin.cast cs.h_dim_B.1.symm i
    let iC := Fin.cast cs.h_dim_C.1.symm i
    (∑ j : Fin cs.nVars, cs.A.toDense iA (Fin.cast cs.h_dim_A.2.symm j) * z j) *
    (∑ j : Fin cs.nVars, cs.B.toDense iB (Fin.cast cs.h_dim_B.2.symm j) * z j) =
    (∑ j : Fin cs.nVars, cs.C.toDense iC (Fin.cast cs.h_dim_C.2.symm j) * z j)

/-- Hadamard (elementwise) product of vectors -/
def hadamard {F : Type} [CommRing F] {n : ℕ} (u v : Fin n → F) : Fin n → F :=
  fun i => u i * v i

/-- Constraint evaluation polynomial: Az(i) * Bz(i) - Cz(i) -/
def constraintPoly {F : Type} [CommRing F] [DecidableEq F] (cs : R1CS F) (z : Witness F cs.nVars)
    : Fin cs.nCons → F :=
  fun i =>
    let iA := Fin.cast cs.h_dim_A.1.symm i
    let iB := Fin.cast cs.h_dim_B.1.symm i
    let iC := Fin.cast cs.h_dim_C.1.symm i
    (∑ j : Fin cs.nVars, cs.A.toDense iA (Fin.cast cs.h_dim_A.2.symm j) * z j) *
    (∑ j : Fin cs.nVars, cs.B.toDense iB (Fin.cast cs.h_dim_B.2.symm j) * z j) -
    (∑ j : Fin cs.nVars, cs.C.toDense iC (Fin.cast cs.h_dim_C.2.symm j) * z j)

/-- Satisfaction is equivalent to constraint poly being zero -/
theorem satisfies_iff_constraint_zero {F : Type} [CommRing F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) :
    satisfies cs z ↔ ∀ i, constraintPoly cs z i = 0 := by
  unfold satisfies constraintPoly
  simp only []
  constructor
  · intro h i
    exact sub_eq_zero.mpr (h i)
  · intro h i
    exact sub_eq_zero.mp (h i)

/-- Vector commitment scheme (abstract interface) -/
structure VectorCommitment (F : Type) [CommRing F] where
  PP : Type                    -- Public parameters
  Commitment : Type            -- Commitment type
  Opening : Type               -- Opening information

  setup : (setupParam : ℕ) → PP
  commit : PP → (v : List F) → (randomness : ℕ) → Commitment
  openProof : PP → (v : List F) → (randomness : ℕ) → (α : F) → Opening
  verify : PP → Commitment → F → F → Opening → Bool

  -- Binding property: two different messages can't have same commitment (with high prob)
  binding : ∀ (pp : PP) (v₁ v₂ : List F) (r₁ r₂ : ℕ),
    v₁ ≠ v₂ →
    commit pp v₁ r₁ = commit pp v₂ r₂ →
    False  -- Negligible probability (formalized via concrete security)

  -- Correctness: honest opening always verifies
  correctness : ∀ (pp : PP) (v : List F) (r : ℕ) (α : F),
    let c := commit pp v r
    let π := openProof pp v r α
    -- Evaluation would use polynomial constructed from v
    verify pp c α α π = true  -- Simplified for now

/-- Any two openings that agree on the same randomness and message list map to identical commitments. -/
lemma VectorCommitment.commit_eq_iff {F : Type} [CommRing F]
    (VC : VectorCommitment F) (pp : VC.PP) (r : ℕ)
    (v₁ v₂ : List F)
    (h : VC.commit pp v₁ r = VC.commit pp v₂ r) : v₁ = v₂ := by
  classical
  by_contra h_ne
  have := VC.binding pp v₁ v₂ r r h_ne h
  exact this.elim

/-- Commitments collide only when the underlying messages coincide. -/
lemma VectorCommitment.commit_injective {F : Type} [CommRing F]
    (VC : VectorCommitment F) (pp : VC.PP) (r : ℕ) :
    Function.Injective fun v : List F => VC.commit pp v r := by
  intro v₁ v₂ h_eq
  exact VC.commit_eq_iff pp r v₁ v₂ h_eq

/-- Proof structure for ΛSNARK-R -/
structure Proof (F : Type) [CommRing F] (VC : VectorCommitment F) where
  -- Commitment parameters and public inputs
  pp : VC.PP
  public_input : List F

  -- Commitments to witness polynomials
  comm_Az : VC.Commitment
  comm_Bz : VC.Commitment
  comm_Cz : VC.Commitment

  -- Challenge points (Fiat-Shamir)
  challenge_α : F
  challenge_β : F

  -- Claimed polynomial evaluations at challenges
  eval_Az_α : F
  eval_Bz_α : F
  eval_Bz_β : F
  eval_Cz_α : F
  constraint_eval : F
  vanishing_at_α : F

  -- Polynomial openings at challenges
  opening_Az_α : VC.Opening
  opening_Bz_α : VC.Opening
  opening_Bz_β : VC.Opening
  opening_Cz_α : VC.Opening

  -- Quotient polynomial commitment, evaluation, and opening
  comm_quotient : VC.Commitment
  opening_quotient_α : VC.Opening
  quotient_poly : Polynomial F

  -- Domain information for vanishing polynomial
  domain_size : ℕ
  primitive_root : F

/-- Canonical list representation of the first `cs.nPub` public inputs. -/
def publicInputToList {F : Type} [CommRing F]
    {cs : R1CS F} (x : PublicInput F cs.nPub) : List F :=
  List.ofFn fun i : Fin cs.nPub => x i

lemma extractPublic_toList {F : Type} [CommRing F]
    {cs : R1CS F} (w : Witness F cs.nVars) :
    publicInputToList (x := extractPublic cs.h_pub_le w) =
      List.ofFn fun i : Fin cs.nPub => w ⟨i.val, Nat.lt_of_lt_of_le i.isLt cs.h_pub_le⟩ := by
  rfl

/-- Evaluation of the `A` constraint polynomial at index `i`. -/
def evaluateConstraintA {F : Type} [CommRing F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (i : Fin cs.nCons) : F :=
  let iA := Fin.cast cs.h_dim_A.1.symm i
  ∑ j : Fin cs.nVars, cs.A.toDense iA (Fin.cast cs.h_dim_A.2.symm j) * z j

/-- Evaluation of the `B` constraint polynomial at index `i`. -/
def evaluateConstraintB {F : Type} [CommRing F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (i : Fin cs.nCons) : F :=
  let iB := Fin.cast cs.h_dim_B.1.symm i
  ∑ j : Fin cs.nVars, cs.B.toDense iB (Fin.cast cs.h_dim_B.2.symm j) * z j

/-- Evaluation of the `C` constraint polynomial at index `i`. -/
def evaluateConstraintC {F : Type} [CommRing F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (i : Fin cs.nCons) : F :=
  let iC := Fin.cast cs.h_dim_C.1.symm i
  ∑ j : Fin cs.nVars, cs.C.toDense iC (Fin.cast cs.h_dim_C.2.symm j) * z j

/-- Evaluate the constraint residual at index `i`. -/
def evaluateConstraintResidual {F : Type} [CommRing F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (i : Fin cs.nCons) : F :=
  evaluateConstraintA cs z i * evaluateConstraintB cs z i -
    evaluateConstraintC cs z i

lemma constraintPoly_eq_evaluateResidual {F : Type} [CommRing F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (i : Fin cs.nCons) :
    constraintPoly cs z i = evaluateConstraintResidual cs z i := by
  classical
  unfold constraintPoly evaluateConstraintResidual evaluateConstraintA
    evaluateConstraintB evaluateConstraintC
  simp [Fin.cast]

lemma satisfies_iff_residual_zero {F : Type} [CommRing F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) :
    satisfies cs z ↔ ∀ i, evaluateConstraintResidual cs z i = 0 := by
  simpa [constraintPoly_eq_evaluateResidual] using
    satisfies_iff_constraint_zero cs z


/-- Verifier's decision predicate -/
def verify {F : Type} [CommRing F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (x : PublicInput F cs.nPub) (π : Proof F VC) : Bool :=
  let inputsOk := decide (π.public_input = publicInputToList x)
  let domainOk := decide (π.domain_size = cs.nCons)
  let openAz := VC.verify π.pp π.comm_Az π.challenge_α π.eval_Az_α π.opening_Az_α
  let openBα := VC.verify π.pp π.comm_Bz π.challenge_α π.eval_Bz_α π.opening_Bz_α
  let openBβ := VC.verify π.pp π.comm_Bz π.challenge_β π.eval_Bz_β π.opening_Bz_β
  let openC := VC.verify π.pp π.comm_Cz π.challenge_α π.eval_Cz_α π.opening_Cz_α
  let quotEval := π.quotient_poly.eval π.challenge_α
  let openQ := VC.verify π.pp π.comm_quotient π.challenge_α quotEval π.opening_quotient_α
  let constraintOk := decide (π.eval_Az_α * π.eval_Bz_α - π.eval_Cz_α = π.constraint_eval)
  let vanishingOk := decide (π.constraint_eval = quotEval * π.vanishing_at_α)
  inputsOk && domainOk && openAz && openBα && openBβ && openC && openQ && constraintOk && vanishingOk

lemma verify_eq_true_of_spec {F : Type} [CommRing F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (x : PublicInput F cs.nPub) (π : Proof F VC)
    (h_inputs : π.public_input = publicInputToList x)
    (h_domain : π.domain_size = cs.nCons)
    (h_openAz : VC.verify π.pp π.comm_Az π.challenge_α π.eval_Az_α π.opening_Az_α = true)
    (h_openBα : VC.verify π.pp π.comm_Bz π.challenge_α π.eval_Bz_α π.opening_Bz_α = true)
    (h_openBβ : VC.verify π.pp π.comm_Bz π.challenge_β π.eval_Bz_β π.opening_Bz_β = true)
    (h_openC : VC.verify π.pp π.comm_Cz π.challenge_α π.eval_Cz_α π.opening_Cz_α = true)
    (h_openQ : VC.verify π.pp π.comm_quotient π.challenge_α
      (π.quotient_poly.eval π.challenge_α) π.opening_quotient_α = true)
    (h_constraint : π.eval_Az_α * π.eval_Bz_α - π.eval_Cz_α = π.constraint_eval)
    (h_vanishing : π.constraint_eval =
      π.quotient_poly.eval π.challenge_α * π.vanishing_at_α) :
    verify VC cs x π = true := by
  classical
  simp [verify, publicInputToList, h_inputs, h_domain, h_openAz, h_openBα,
    h_openBβ, h_openC, h_openQ, h_constraint, h_vanishing]

/-- Data certificate ensuring a proof passes all verifier checks. -/
structure ProofCertificate {F : Type} [CommRing F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (x : PublicInput F cs.nPub) where
  proof : Proof F VC
  h_public : proof.public_input = publicInputToList x
  h_domain : proof.domain_size = cs.nCons
  h_openAz : VC.verify proof.pp proof.comm_Az proof.challenge_α proof.eval_Az_α
    proof.opening_Az_α = true
  h_openBα : VC.verify proof.pp proof.comm_Bz proof.challenge_α proof.eval_Bz_α
    proof.opening_Bz_α = true
  h_openBβ : VC.verify proof.pp proof.comm_Bz proof.challenge_β proof.eval_Bz_β
    proof.opening_Bz_β = true
  h_openC : VC.verify proof.pp proof.comm_Cz proof.challenge_α proof.eval_Cz_α
    proof.opening_Cz_α = true
  h_openQ : VC.verify proof.pp proof.comm_quotient proof.challenge_α
    (proof.quotient_poly.eval proof.challenge_α) proof.opening_quotient_α = true
  h_constraint : proof.eval_Az_α * proof.eval_Bz_α - proof.eval_Cz_α =
    proof.constraint_eval
  h_vanishing : proof.constraint_eval =
    proof.quotient_poly.eval proof.challenge_α * proof.vanishing_at_α

namespace ProofCertificate

variable {F : Type} [CommRing F] [DecidableEq F]
variable {VC : VectorCommitment F} {cs : R1CS F}
variable {x : PublicInput F cs.nPub}

@[simp] lemma verify_eq_true (cert : ProofCertificate VC cs x) :
    verify VC cs x cert.proof = true :=
  verify_eq_true_of_spec VC cs x cert.proof cert.h_public cert.h_domain
    cert.h_openAz cert.h_openBα cert.h_openBβ cert.h_openC cert.h_openQ
    cert.h_constraint cert.h_vanishing

end ProofCertificate

/-- Predicate: verification passes and quotient polynomial is correct -/
def verify_with_quotient {F : Type} [Field F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (x : PublicInput F cs.nPub) (π : Proof F VC)
    (m : ℕ) (ω : F) (hω : IsPrimitiveRoot ω m) (h_m : m = cs.nCons) : Prop :=
  let _ := hω
  let _ := h_m
  verify VC cs x π = true ∧
  -- Quotient polynomial opening is valid at challenge α
  (∃ pp, VC.verify pp π.comm_quotient π.challenge_α
    (π.quotient_poly.eval π.challenge_α) π.opening_quotient_α = true)

/-- Placeholder for Module-LWE hardness assumption -/
axiom ModuleLWE_Hard (n k : ℕ) (q : ℕ) (σ : ℝ) : Prop

/-- Placeholder for Module-SIS hardness assumption -/
axiom ModuleSIS_Hard (n k : ℕ) (q : ℕ) (β : ℕ) : Prop

end LambdaSNARK
