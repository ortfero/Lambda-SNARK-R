import LambdaSNARK.Soundness
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Polynomial.Basic

namespace LambdaSNARK.Tests

open LambdaSNARK

noncomputable section

/-- Simple vector commitment used to instantiate the probability lemmas in tests. -/
noncomputable def testVectorCommitment (F : Type) [CommRing F] : VectorCommitment F where
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

/-- Trivial adversary state with constant openings, adequate for exercising
`fork_success_bound`. -/
noncomputable def testAdversaryState {F : Type} [Field F] [DecidableEq F] :
    AdversaryState F (testVectorCommitment F) :=
  { randomness := 0
    pp := ()
    comm_Az := []
    comm_Bz := []
    comm_Cz := []
    comm_quotient := []
    quotient_poly := 0
    quotient_rand := 0
    quotient_commitment_spec := by
      simp [testVectorCommitment, Polynomial.coeffList_zero]
    domainSize := 1
    omega := 1
    respond := fun _ _ =>
      { Az_eval := 0
        Bz_eval := 0
        Cz_eval := 0
        quotient_eval := 0
        vanishing_eval := 0
        opening_Az_α := ()
        opening_Bz_β := ()
        opening_Cz_α := ()
        opening_quotient_α := () } }

section ZMod2

open scoped BigOperators

variable (valid : Finset (ZMod 2))

lemma univ_card_zmod2 : (Finset.univ : Finset (ZMod 2)).card = 2 := by simp

lemma real_card_zmod2 : ((Finset.univ : Finset (ZMod 2)).card : ℝ) = 2 := by simp

/-- Edge-case check: for `|F| = 2` and `ε = 1`, the fork success bound reduces to the
non-negative inequality `1 ≥ 0`. -/
lemma fork_success_bound_zmod2_eps_one :
    1 ≥ (1 : ℝ) ^ 2 / 2 - 1 / (2 : ℝ) := by
  classical
  let VC := testVectorCommitment (ZMod 2)
  let state := testAdversaryState (F := ZMod 2)
  let valid_challenges : Finset (ZMod 2) := Finset.univ
  have h_heavy : (valid_challenges.card : ℝ) ≥ (1 : ℝ) * (Fintype.card (ZMod 2) : ℝ) := by
    simp [valid_challenges]
  have h_field : (Fintype.card (ZMod 2) : ℝ) ≥ 2 := by simp
  have h_nonempty : valid_challenges.card ≥ 2 := by
    simpa [valid_challenges] using (by decide : Fintype.card (ZMod 2) ≥ 2)
  have h_bound :=
    fork_success_bound (VC := VC)
      (state := state)
      (valid_challenges := valid_challenges)
      (ε := 1)
      h_heavy
      (by norm_num)
      (by norm_num)
      h_field
      h_nonempty
  have h_pairs :
      (Nat.choose valid_challenges.card 2 : ℝ) / (Nat.choose (Fintype.card (ZMod 2)) 2 : ℝ) = 1 := by
    simp [valid_challenges]
  have h_rhs : (1 : ℝ) ^ 2 / 2 - 1 / (Fintype.card (ZMod 2) : ℝ) = 0 := by
    simp
  simpa [h_pairs, h_rhs]

/-- Edge-case check: with `ε = 1/2` and `|F| = 2` we still retain a positive lower
bound from `fork_success_bound`. -/
lemma fork_success_bound_zmod2_eps_half :
    1 ≥ (1 / (2 : ℝ)) ^ 2 / 2 - 1 / (2 : ℝ) := by
  classical
  let VC := testVectorCommitment (ZMod 2)
  let state := testAdversaryState (F := ZMod 2)
  let valid_challenges : Finset (ZMod 2) := Finset.univ
  have h_heavy : (valid_challenges.card : ℝ) ≥ (1 / (2 : ℝ)) * (Fintype.card (ZMod 2) : ℝ) := by
    simp [valid_challenges]
  have h_field : (Fintype.card (ZMod 2) : ℝ) ≥ 2 := by simp
  have h_nonempty : valid_challenges.card ≥ 2 := by
    simpa [valid_challenges] using (by decide : Fintype.card (ZMod 2) ≥ 2)
  have h_bound :=
    fork_success_bound (VC := VC)
      (state := state)
      (valid_challenges := valid_challenges)
      (ε := 1 / (2 : ℝ))
      h_heavy
      (by norm_num)
      (by norm_num)
      h_field
      h_nonempty
  have h_pairs :
      (Nat.choose valid_challenges.card 2 : ℝ) / (Nat.choose (Fintype.card (ZMod 2)) 2 : ℝ) = 1 := by
    simp [valid_challenges]
  have h_rhs : (1 / (2 : ℝ)) ^ 2 / 2 - 1 / (Fintype.card (ZMod 2) : ℝ) = -3 / 8 := by
    simp [Fintype.card_zmod, one_div, pow_two]
  simpa [h_pairs, h_rhs]

end ZMod2

end

end LambdaSNARK.Tests
