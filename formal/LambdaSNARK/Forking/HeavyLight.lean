/-
Copyright (c) 2025 URPKS Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: URPKS Contributors
-/

import LambdaSNARK.Forking.Probability
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Choose.Cast
import Mathlib.Data.Nat.Choose.Central
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Heavy-Light Commitment Decomposition

This module isolates the reasoning about ε-heavy commitment tuples and the
associated probability mass inequalities that appear in the heavy row lemma.

It contains the following ingredients:

* Definition of ε-heavy commitment tuples and their complement.
* Aggregation lemmas relating conditional success mass to global success
  probability.
* Bookkeeping infrastructure for challenges that contribute positive success
  mass under a fixed commitment (`successfulChallenges`).
* Combinatorial bounds connecting the heaviness predicate to the cardinality of
  successful challenges.
-/

namespace LambdaSNARK

open scoped BigOperators
open BigOperators Polynomial
open LambdaSNARK

/-
Auxiliary combinatorial estimates used to convert cardinality bounds into real
inequalities during the forking analysis.
-/
section Combinatorics

open Nat

lemma choose_two_cast (n : ℕ) :
    (Nat.choose n 2 : ℝ) = (n : ℝ) * (n - 1) / 2 := by
  simpa using (Nat.cast_choose_two (K := ℝ) n)

lemma choose_two_pos {n : ℕ} (hn : 2 ≤ n) :
    (Nat.choose n 2 : ℝ) > 0 := by
  have h_nat : 0 < Nat.choose n 2 := Nat.choose_pos (k := 2) hn
  exact_mod_cast h_nat

lemma choose_two_mono {m n : ℕ} (h : m ≤ n) :
    (Nat.choose m 2 : ℝ) ≤ Nat.choose n 2 := by
  exact_mod_cast (Nat.choose_le_choose (c := 2) h)

lemma two_mul_sub_three_add_inv_nonneg {n : ℝ} (hn_two : (2 : ℝ) ≤ n) :
    0 ≤ 2 * n - 3 + 1 / n := by
  have hn_pos : 0 < n := lt_of_lt_of_le (show (0 : ℝ) < 2 by norm_num) hn_two
  have hn_ne : n ≠ 0 := ne_of_gt hn_pos
  have h_num_nonneg : 0 ≤ (2 * n - 1) * (n - 1) := by
    have h_left : 0 ≤ 2 * n - 1 := by
      have h_sum : (4 : ℝ) ≤ n + n := by
        have := add_le_add hn_two hn_two
        convert this using 1
        · norm_num
      have h_two_n_ge_four : (4 : ℝ) ≤ 2 * n := by
        simpa [two_mul, add_comm] using h_sum
      have h_two_n_ge_one : (1 : ℝ) ≤ 2 * n := le_trans (by norm_num) h_two_n_ge_four
      exact sub_nonneg.mpr h_two_n_ge_one
    have h_right : 0 ≤ n - 1 :=
      sub_nonneg.mpr ((show (1 : ℝ) ≤ 2 by norm_num).trans hn_two)
    exact mul_nonneg h_left h_right
  have h_div : 0 ≤ ((2 * n - 1) * (n - 1)) / n :=
    div_nonneg h_num_nonneg (le_of_lt hn_pos)
  have h_cast :
      ((2 * n - 1) * (n - 1)) / n = 2 * n - 3 + 1 / n := by
    field_simp [hn_ne, one_div]; ring
  simpa [h_cast] using h_div

lemma eps_mul_sub_one_over_ge {ε n : ℝ}
    (hn_two : (2 : ℝ) ≤ n) (h_prod : 1 ≤ ε * n) :
    ε * (ε * n - 1) / (n - 1) ≥ ε ^ 2 / 2 - 1 / n := by
  classical
  have hn_pos : 0 < n := lt_of_lt_of_le (show (0 : ℝ) < 2 by norm_num) hn_two
  have h_one_lt : (1 : ℝ) < n := lt_of_lt_of_le (show (1 : ℝ) < 2 by norm_num) hn_two
  have hn_sub_pos : 0 < n - 1 := sub_pos.mpr h_one_lt
  have hn_ne : n ≠ 0 := ne_of_gt hn_pos
  have hn_sub_ne : n - 1 ≠ 0 := sub_ne_zero.mpr (ne_of_gt h_one_lt)
  have h_eps_lower : 1 / n ≤ ε := by
    have h_inv_pos : 0 < (1 / n : ℝ) := by
      simpa [one_div] using inv_pos.mpr hn_pos
    have := mul_le_mul_of_nonneg_right h_prod (le_of_lt h_inv_pos)
    simpa [one_div, hn_ne, mul_comm, mul_left_comm, mul_assoc] using this
  let δ : ℝ := ε - 1 / n
  have h_delta_nonneg : 0 ≤ δ := sub_nonneg.mpr h_eps_lower
  have h_const_nonneg : 0 ≤ 2 * n - 3 + 1 / n :=
    two_mul_sub_three_add_inv_nonneg hn_two
  have h_coeff_nonneg : 0 ≤ n ^ 2 + n := by
    have : 0 ≤ n := le_of_lt hn_pos
    have : 0 ≤ n ^ 2 := sq_nonneg n
    exact add_nonneg this ‹0 ≤ n›
  have h_expand :
      (n ^ 2 + n) * ε ^ 2 - 2 * n * ε + 2 * (n - 1)
        = (n ^ 2 + n) * δ ^ 2 + 2 * δ + (2 * n - 3 + 1 / n) := by
    have hε : ε = δ + 1 / n := by simp [δ]
    have htmp :
        (n ^ 2 + n) * (δ + 1 / n) ^ 2 - 2 * n * (δ + 1 / n) + 2 * (n - 1)
          - ((n ^ 2 + n) * δ ^ 2 + 2 * δ + (2 * n - 3 + 1 / n)) = 0 := by
      have hn_ne' : (n : ℝ) ≠ 0 := hn_ne
      field_simp [pow_two, one_div, hn_ne']
      ring
    have h_eq := sub_eq_zero.mp htmp
    simpa [hε]
      using h_eq
  have h_sq_nonneg : 0 ≤ (n ^ 2 + n) * δ ^ 2 :=
    mul_nonneg h_coeff_nonneg (sq_nonneg δ)
  have h_lin_nonneg : 0 ≤ 2 * δ :=
    mul_nonneg (show 0 ≤ (2 : ℝ) by norm_num) h_delta_nonneg
  have h_numer_nonneg :
      0 ≤ (n ^ 2 + n) * ε ^ 2 - 2 * n * ε + 2 * (n - 1) := by
    have h_total := add_nonneg (add_nonneg h_sq_nonneg h_lin_nonneg) h_const_nonneg
    simpa [h_expand] using h_total
  have h_denom_pos : 0 < 2 * n * (n - 1) :=
    mul_pos (mul_pos (show (0 : ℝ) < 2 by norm_num) hn_pos) hn_sub_pos
  have h_denom_nonneg : 0 ≤ 2 * n * (n - 1) := le_of_lt h_denom_pos
  have h_mul_eq :
      (ε * (ε * n - 1) / (n - 1) - (ε ^ 2 / 2 - 1 / n)) * (2 * n * (n - 1))
        - ((n ^ 2 + n) * ε ^ 2 - 2 * n * ε + 2 * (n - 1)) = 0 := by
    field_simp [pow_two, one_div, hn_ne, hn_sub_ne, sub_eq_add_neg]
    ring
  have h_denom_ne : 2 * n * (n - 1) ≠ 0 := by
    have h_two_ne : (2 : ℝ) ≠ 0 := by norm_num
    have h_mul_ne : 2 * n ≠ 0 := mul_ne_zero h_two_ne hn_ne
    exact mul_ne_zero h_mul_ne hn_sub_ne
  have h_delta_eq :
      ε * (ε * n - 1) / (n - 1) - (ε ^ 2 / 2 - 1 / n)
        = ((n ^ 2 + n) * ε ^ 2 - 2 * n * ε + 2 * (n - 1)) / (2 * n * (n - 1)) := by
    have h_eq := sub_eq_zero.mp h_mul_eq
    exact (eq_div_iff_mul_eq h_denom_ne).2 h_eq
  have h_fraction := div_nonneg h_numer_nonneg h_denom_nonneg
  have h_diff_nonneg :
      0 ≤ ε * (ε * n - 1) / (n - 1) - (ε ^ 2 / 2 - 1 / n) := by
    simpa only [h_delta_eq] using h_fraction
  have h_main := sub_nonneg.mp h_diff_nonneg
  have h_le : ε ^ 2 / 2 ≤ ε * (ε * n - 1) / (n - 1) + 1 / n := by
    have := add_le_add_right h_main (1 / n)
    simpa using this
  exact (sub_le_iff_le_add).2 h_le

end Combinatorics

/-!
### Heavy commitments

We say a commitment tuple is heavy when the conditional success mass is at least
`ε`.
-/

section HeavyCommitmentDecomposition

variable {F : Type} [Field F] [Fintype F] [DecidableEq F]
variable (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
variable (x : PublicInput F cs.nPub) (secParam : ℕ)

def heavyCommitments
    (ε : ℝ) :
    Set (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :=
  {comm_tuple |
    0 < commitMass VC cs A x secParam comm_tuple ∧
      successMassGivenCommit VC cs A x secParam comm_tuple ≥
        ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple}

noncomputable def heavyCommitMass (ε : ℝ) : ENNReal := by
  classical
  exact ∑' comm_tuple,
    if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
      commitMass VC cs A x secParam comm_tuple
    else
      0

noncomputable def heavyCommitSeedWeight (ε : ℝ)
    (rand : Fin secParam.succ) : ENNReal := by
  classical
  exact
    if commitTupleOfRandomness VC cs A x secParam rand ∈
        heavyCommitments VC cs A x secParam ε then
      (uniform_pmf : PMF (Fin (secParam.succ))) rand
    else
      0

noncomputable def heavyRandomnessSet (ε : ℝ) : Set (Fin secParam.succ) :=
  {rand |
    commitTupleOfRandomness VC cs A x secParam rand ∈
      heavyCommitments VC cs A x secParam ε}

lemma heavyRandomnessSet_finite (ε : ℝ) :
    (heavyRandomnessSet (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε).Finite := by
  classical
  have h_univ :
      (Set.univ : Set (Fin secParam.succ)).Finite :=
    Set.finite_univ
  refine h_univ.subset ?_
  intro rand _
  trivial

noncomputable def heavyRandomnessFinset (ε : ℝ) :
    Finset (Fin secParam.succ) :=
  (heavyRandomnessSet_finite (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)).toFinset

lemma mem_heavyRandomnessFinset (ε : ℝ) {rand : Fin secParam.succ} :
    rand ∈ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε ↔
      commitTupleOfRandomness VC cs A x secParam rand ∈
        heavyCommitments VC cs A x secParam ε := by
  classical
  unfold heavyRandomnessFinset
  simp [heavyRandomnessSet]

lemma heavyRandomnessFinset_card (ε : ℝ) :
    (heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε).card =
      (heavyRandomnessSet (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε).ncard := by
  classical
  have := Set.ncard_eq_toFinset_card
    (s := heavyRandomnessSet (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε)
    (hs := heavyRandomnessSet_finite (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε))
  simpa [heavyRandomnessFinset]
    using this.symm

lemma heavyRandomnessFinset_card_cast (ε : ℝ) :
    ((heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε).card : ℝ)
      = (Set.ncard
          (heavyRandomnessSet (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam) ε) : ℝ) := by
  classical
  have h_card :=
    heavyRandomnessFinset_card (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)
  simpa using congrArg (fun n : ℕ => (n : ℝ)) h_card

noncomputable def lightCommitMass (ε : ℝ) : ENNReal := by
  classical
  exact ∑' comm_tuple,
    if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
      0
    else
      commitMass VC cs A x secParam comm_tuple

/-- Mass of heavy commitments when paired with an independent uniformly sampled
challenge.  We measure it directly in the joint distribution produced by
`run_adversary_commit_uniform_challenge`. -/
noncomputable def heavyCommitChallengeMass (ε : ℝ) : ENNReal := by
  classical
  let C := VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment
  exact ∑' comm : C,
    ∑' α : F,
      if comm ∈ heavyCommitments VC cs A x secParam ε then
        run_adversary_commit_uniform_challenge VC cs A x secParam (comm, α)
      else
        0

/-- Contribution of a randomness seed and a challenge value to the heavy mass
after the challenge is sampled uniformly. -/
noncomputable def heavyCommitChallengeWeight (ε : ℝ)
    (rand : Fin secParam.succ) (α : F) : ENNReal := by
  classical
  let comm := commitTupleOfRandomness VC cs A x secParam rand
  exact
    if comm ∈ heavyCommitments VC cs A x secParam ε then
      (uniform_pmf : PMF (Fin (secParam.succ))) rand *
        (uniform_pmf : PMF F) α
    else
      0

lemma heavyCommitChallengeWeight_le_heavyCommitSeedWeight
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (ε : ℝ) (rand : Fin secParam.succ) (α : F) :
    heavyCommitChallengeWeight VC cs A x secParam ε rand α ≤
      heavyCommitSeedWeight VC cs A x secParam ε rand := by
  classical
  by_cases h_mem :
      commitTupleOfRandomness VC cs A x secParam rand ∈
        heavyCommitments VC cs A x secParam ε
  · have h_nonneg : ∀ β ∈ (Finset.univ : Finset F),
        0 ≤ (uniform_pmf : PMF F) β := by
      intro β _; exact bot_le
    have h_le_sum :=
      Finset.single_le_sum h_nonneg (Finset.mem_univ α)
    have h_sum : ∑ β : F, (uniform_pmf : PMF F) β = 1 := by
      simpa [tsum_fintype] using (uniform_pmf : PMF F).tsum_coe
    have h_le_one : (uniform_pmf : PMF F) α ≤ (1 : ENNReal) := by
      simpa [h_sum] using h_le_sum
    have h_mul :=
      mul_le_mul'
        (le_of_eq (rfl :
            (uniform_pmf : PMF (Fin (secParam.succ))) rand =
              (uniform_pmf : PMF (Fin (secParam.succ))) rand))
        h_le_one
    have h_weight :
        heavyCommitChallengeWeight VC cs A x secParam ε rand α =
          (uniform_pmf : PMF (Fin (secParam.succ))) rand *
            (uniform_pmf : PMF F) α := by
      simp [heavyCommitChallengeWeight, h_mem]
    have h_seed :
        heavyCommitSeedWeight VC cs A x secParam ε rand =
          (uniform_pmf : PMF (Fin (secParam.succ))) rand := by
      simp [heavyCommitSeedWeight, h_mem]
    have h_goal :
        heavyCommitChallengeWeight VC cs A x secParam ε rand α ≤
          (uniform_pmf : PMF (Fin (secParam.succ))) rand := by
      simpa [h_weight] using h_mul
    have h_seed_symm := h_seed.symm
    exact (calc
      heavyCommitChallengeWeight VC cs A x secParam ε rand α ≤
          (uniform_pmf : PMF (Fin (secParam.succ))) rand := h_goal
      _ = heavyCommitSeedWeight VC cs A x secParam ε rand := h_seed_symm)
  · simp [heavyCommitChallengeWeight, heavyCommitSeedWeight, h_mem]


lemma successMassGivenCommit_le_successMass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    successMassGivenCommit VC cs A x secParam comm_tuple ≤
      successMass VC cs A x secParam := by
  classical
  unfold successMass successMassGivenCommit
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_pointwise :
      (fun t => if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple then p t else 0)
        ≤ fun t => if success_event VC cs x t then p t else 0 := by
    intro t
    by_cases hSucc : success_event VC cs x t
    · by_cases hComm : transcriptCommitTuple VC t = comm_tuple
      · simp [hSucc, hComm]
      · simp [hSucc, hComm]
    · simp [hSucc]
  simpa [hp] using ENNReal.tsum_le_tsum h_pointwise

lemma successMassGivenCommit_le_commitMass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    successMassGivenCommit VC cs A x secParam comm_tuple ≤
      commitMass VC cs A x secParam comm_tuple := by
  classical
  unfold successMassGivenCommit commitMass
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_pointwise :
      (fun t => if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple then p t else 0)
        ≤ fun t => if transcriptCommitTuple VC t = comm_tuple then p t else 0 := by
    intro t
    by_cases hSucc : success_event VC cs x t
    · by_cases hComm : transcriptCommitTuple VC t = comm_tuple
      · simp [hSucc, hComm]
      · simp [hSucc, hComm]
    · simp [hSucc]
  simpa [hp] using ENNReal.tsum_le_tsum h_pointwise

lemma successMassGivenCommit_le_run_adversary_commit_tuple
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    successMassGivenCommit VC cs A x secParam comm_tuple ≤
      run_adversary_commit_tuple VC cs A x secParam comm_tuple := by
  have h :=
    successMassGivenCommit_le_commitMass (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (comm_tuple := comm_tuple)
  simpa [commitMass_eq_run_adversary_commit_tuple (VC := VC) (cs := cs) (A := A)
    (x := x) (secParam := secParam) (comm_tuple := comm_tuple)] using h

lemma heavyCommitMass_eq_uniform_randomness_sum
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (ε : ℝ) :
    heavyCommitMass VC cs A x secParam ε =
      ∑ rand : Fin secParam.succ,
        heavyCommitSeedWeight VC cs A x secParam ε rand := by
  classical
  let C := VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment
  have h_term :
      ∀ comm : C,
        (if comm ∈ heavyCommitments VC cs A x secParam ε then
            commitMass VC cs A x secParam comm
          else
            0)
          =
          ∑' rand : Fin secParam.succ,
            (if comm ∈ heavyCommitments VC cs A x secParam ε then
                commitSeedWeight VC cs A x secParam comm rand
              else
                0) := by
    intro comm
    by_cases h_mem : comm ∈ heavyCommitments VC cs A x secParam ε
    · simp [h_mem, commitMass_eq_uniform_randomness_sum, tsum_fintype]
    · simp [h_mem]
  have h_sum :
      heavyCommitMass VC cs A x secParam ε =
        ∑' comm : C,
          ∑' rand : Fin secParam.succ,
            (if comm ∈ heavyCommitments VC cs A x secParam ε then
                commitSeedWeight VC cs A x secParam comm rand
              else
                0) := by
    unfold heavyCommitMass
    have :
        (∑' comm : C,
            (if comm ∈ heavyCommitments VC cs A x secParam ε then
                commitMass VC cs A x secParam comm
              else
                0))
          =
            ∑' comm : C,
              ∑' rand : Fin secParam.succ,
                (if comm ∈ heavyCommitments VC cs A x secParam ε then
                    commitSeedWeight VC cs A x secParam comm rand
                  else
                    0) := by
      refine tsum_congr ?_
      intro comm
      exact h_term comm
    simpa [tsum_fintype]
      using this
  have h_swap :
      (∑' comm : C,
          ∑' rand : Fin secParam.succ,
            (if comm ∈ heavyCommitments VC cs A x secParam ε then
                commitSeedWeight VC cs A x secParam comm rand
              else
                0))
        = ∑' rand : Fin secParam.succ,
            ∑' comm : C,
              (if comm ∈ heavyCommitments VC cs A x secParam ε then
                  commitSeedWeight VC cs A x secParam comm rand
                else
                  0) := by
    simpa using
      ENNReal.tsum_comm
        (f := fun (comm : C) (rand : Fin secParam.succ) =>
          if comm ∈ heavyCommitments VC cs A x secParam ε then
            commitSeedWeight VC cs A x secParam comm rand
          else
            0)
  have h_inner :
      ∀ rand : Fin secParam.succ,
        (∑' comm : C,
            (if comm ∈ heavyCommitments VC cs A x secParam ε then
                commitSeedWeight VC cs A x secParam comm rand
              else
                0))
          = heavyCommitSeedWeight VC cs A x secParam ε rand := by
    intro rand
    classical
    set c := commitTupleOfRandomness VC cs A x secParam rand
    have h_single :
        (∑' comm : C,
            (if comm ∈ heavyCommitments VC cs A x secParam ε then
                commitSeedWeight VC cs A x secParam comm rand
              else
                0))
          =
            (if c ∈ heavyCommitments VC cs A x secParam ε then
                commitSeedWeight VC cs A x secParam c rand
              else
                0) := by
      refine
        tsum_eq_single c
          (by
            intro comm h_comm_ne
            by_cases h_mem : comm ∈ heavyCommitments VC cs A x secParam ε
            · have h_ne' : comm ≠ commitTupleOfRandomness VC cs A x secParam rand := by
                simpa [c] using h_comm_ne
              simp [commitSeedWeight, h_mem, h_ne']
            · simp [h_mem])
    have h_eval :
        (if c ∈ heavyCommitments VC cs A x secParam ε then
            commitSeedWeight VC cs A x secParam c rand
          else
            0)
          = heavyCommitSeedWeight VC cs A x secParam ε rand := by
      simp [heavyCommitSeedWeight, commitSeedWeight, c]
    exact h_single.trans h_eval
  calc
    heavyCommitMass VC cs A x secParam ε
        = ∑' comm : C,
            ∑' rand : Fin secParam.succ,
              (if comm ∈ heavyCommitments VC cs A x secParam ε then
                  commitSeedWeight VC cs A x secParam comm rand
                else
                  0) := h_sum
    _ = ∑' rand : Fin secParam.succ,
            ∑' comm : C,
              (if comm ∈ heavyCommitments VC cs A x secParam ε then
                  commitSeedWeight VC cs A x secParam comm rand
                else
                  0) := h_swap
    _ = ∑' rand : Fin secParam.succ,
            heavyCommitSeedWeight VC cs A x secParam ε rand := by
          refine tsum_congr ?_
          intro rand
          exact h_inner rand
    _ = ∑ rand : Fin secParam.succ,
            heavyCommitSeedWeight VC cs A x secParam ε rand := by
          simp [tsum_fintype]

lemma heavyCommitChallengeMass_eq_heavyCommitMass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    [DecidableEq
      (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)]
    (ε : ℝ) :
    heavyCommitChallengeMass VC cs A x secParam ε =
      heavyCommitMass VC cs A x secParam ε := by
  classical
  let C := VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment
  have h_term :
      ∀ comm : C,
        (∑' α : F,
            if comm ∈ heavyCommitments VC cs A x secParam ε then
              run_adversary_commit_uniform_challenge VC cs A x secParam (comm, α)
            else
              0)
          =
            (if comm ∈ heavyCommitments VC cs A x secParam ε then
                commitMass VC cs A x secParam comm
              else
                0) := by
    intro comm
    by_cases h_mem : comm ∈ heavyCommitments VC cs A x secParam ε
    · simp [h_mem,
        tsum_run_adversary_commit_uniform_challenge_over_challenges
          (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
          (comm_tuple := comm)]
    · simp [h_mem]
  unfold heavyCommitChallengeMass
  have h_congr :
      (∑' comm : C,
          ∑' α : F,
            if comm ∈ heavyCommitments VC cs A x secParam ε then
              run_adversary_commit_uniform_challenge VC cs A x secParam (comm, α)
            else
              0)
        = ∑' comm : C,
            (if comm ∈ heavyCommitments VC cs A x secParam ε then
                commitMass VC cs A x secParam comm
              else
                0) := by
    refine tsum_congr ?_
    intro comm
    exact h_term comm
  simpa [heavyCommitMass] using h_congr

lemma heavyCommitChallengeMass_eq_uniform_randomness_challenge_sum
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    [DecidableEq
      (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)]
    (ε : ℝ) :
    heavyCommitChallengeMass VC cs A x secParam ε =
      ∑ rand : Fin secParam.succ,
        ∑ α : F,
          heavyCommitChallengeWeight VC cs A x secParam ε rand α := by
  classical
  have h_inner :
      ∀ rand : Fin secParam.succ,
        (∑ α : F,
            heavyCommitChallengeWeight VC cs A x secParam ε rand α)
          = heavyCommitSeedWeight VC cs A x secParam ε rand := by
    intro rand
    have h_uniform :
        ∑ α : F, (uniform_pmf : PMF F) α = (1 : ENNReal) := by
      simpa [tsum_fintype] using (uniform_pmf : PMF F).tsum_coe
    by_cases h_mem :
        commitTupleOfRandomness VC cs A x secParam rand ∈
          heavyCommitments VC cs A x secParam ε
    · classical
      have h_sum :
          (∑ α : F,
              heavyCommitChallengeWeight VC cs A x secParam ε rand α)
            =
              (uniform_pmf : PMF (Fin (secParam.succ))) rand *
                ∑ α : F, (uniform_pmf : PMF F) α := by
        have h_mul :=
          (Finset.mul_sum
              (a := (uniform_pmf : PMF (Fin (secParam.succ))) rand)
              (s := (Finset.univ : Finset F))
              (f := fun α : F => (uniform_pmf : PMF F) α)).symm
        simpa [heavyCommitChallengeWeight, h_mem] using h_mul
      calc
        (∑ α : F,
            heavyCommitChallengeWeight VC cs A x secParam ε rand α)
            = (uniform_pmf : PMF (Fin (secParam.succ))) rand *
                ∑ α : F, (uniform_pmf : PMF F) α := h_sum
        _ = (uniform_pmf : PMF (Fin (secParam.succ))) rand * 1 := by
              simp [h_uniform]
        _ = heavyCommitSeedWeight VC cs A x secParam ε rand := by
              simp [heavyCommitSeedWeight, h_mem]
    · simp [heavyCommitChallengeWeight, heavyCommitSeedWeight, h_mem]
  have h_outer :
      (∑ rand : Fin secParam.succ,
          ∑ α : F,
            heavyCommitChallengeWeight VC cs A x secParam ε rand α)
        = ∑ rand : Fin secParam.succ,
            heavyCommitSeedWeight VC cs A x secParam ε rand := by
    refine Finset.sum_congr rfl ?_
    intro rand h_rand
    exact h_inner rand
  calc
    heavyCommitChallengeMass VC cs A x secParam ε
        = heavyCommitMass VC cs A x secParam ε :=
          heavyCommitChallengeMass_eq_heavyCommitMass
            (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
            (ε := ε)
    _ = ∑ rand : Fin secParam.succ,
          heavyCommitSeedWeight VC cs A x secParam ε rand :=
          heavyCommitMass_eq_uniform_randomness_sum
            (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
            (ε := ε)
    _ = ∑ rand : Fin secParam.succ,
          ∑ α : F,
            heavyCommitChallengeWeight VC cs A x secParam ε rand α := by
          simpa using h_outer.symm

lemma heavyCommitMass_mul_randomnessCard_eq_heavySeedCount
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ) :
    heavyCommitMass VC cs A x secParam ε * (secParam.succ : ENNReal) =
      (Nat.cast
        (Set.ncard
          {rand : Fin secParam.succ |
            commitTupleOfRandomness VC cs A x secParam rand ∈
              heavyCommitments VC cs A x secParam ε}) : ENNReal) := by
  classical
  set heavySet :=
    heavyRandomnessSet (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε
  have h_finite : heavySet.Finite :=
    heavyRandomnessSet_finite (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)
  set heavyFinset : Finset (Fin secParam.succ) := h_finite.toFinset
  have h_mem :
      ∀ rand : Fin secParam.succ,
        rand ∈ heavyFinset ↔ rand ∈ heavySet :=
    fun rand => h_finite.mem_toFinset (a := rand)
  have h_weights :
      ∀ rand : Fin secParam.succ,
        heavyCommitSeedWeight VC cs A x secParam ε rand =
          if rand ∈ heavyFinset then (secParam.succ : ENNReal)⁻¹ else 0 := by
    intro rand
    by_cases h_heavy :
        commitTupleOfRandomness VC cs A x secParam rand ∈
          heavyCommitments VC cs A x secParam ε
    · have h_set : rand ∈ heavyFinset :=
        (h_mem rand).2 (by
          change commitTupleOfRandomness VC cs A x secParam rand ∈
            heavyCommitments VC cs A x secParam ε
          exact h_heavy)
      have h_uniform :
          (uniform_pmf : PMF (Fin (secParam.succ))) rand =
            (↑secParam + 1 : ENNReal)⁻¹ := by
        unfold uniform_pmf
        change (Fintype.card (Fin (secParam.succ)) : ENNReal)⁻¹ =
          (↑secParam + 1 : ENNReal)⁻¹
        simp [Fintype.card_fin, Nat.cast_succ]
      simp [heavyCommitSeedWeight, h_heavy, h_set, h_uniform, Nat.cast_succ]
    · have h_set : rand ∉ heavyFinset := by
        intro h_mem_fin
        have h_set_mem : rand ∈ heavySet := (h_mem rand).1 h_mem_fin
        change commitTupleOfRandomness VC cs A x secParam rand ∈
          heavyCommitments VC cs A x secParam ε at h_set_mem
        exact h_heavy h_set_mem
      simp [heavyCommitSeedWeight, h_heavy, h_set]
  have h_sum_indicator :
      (∑ rand : Fin secParam.succ,
          heavyCommitSeedWeight VC cs A x secParam ε rand) =
        ∑ rand : Fin secParam.succ,
            if rand ∈ heavyFinset then (secParam.succ : ENNReal)⁻¹ else 0 := by
    refine Finset.sum_congr rfl ?_
    intro rand _
    exact h_weights rand
  have h_filter :
      (Finset.univ.filter
          fun rand : Fin secParam.succ => rand ∈ heavyFinset) = heavyFinset := by
    ext rand
    by_cases h : rand ∈ heavyFinset
    · simp [h]
    · simp [h]
  have h_sum_const :
      (∑ rand : Fin secParam.succ,
          if rand ∈ heavyFinset then (secParam.succ : ENNReal)⁻¹ else 0) =
        (heavyFinset.card : ENNReal) * (secParam.succ : ENNReal)⁻¹ := by
    have h_filtered_aux :=
      (Finset.sum_filter
          (s := (Finset.univ : Finset (Fin secParam.succ)))
          (p := fun rand : Fin secParam.succ => rand ∈ heavyFinset)
          (f := fun _ => (secParam.succ : ENNReal)⁻¹)).symm
    have h_sum_filter :=
      congrArg
        (fun s : Finset (Fin secParam.succ) =>
          ∑ rand ∈ s, (secParam.succ : ENNReal)⁻¹)
        h_filter
    have h_filtered :
        (∑ rand : Fin secParam.succ,
            if rand ∈ heavyFinset then (secParam.succ : ENNReal)⁻¹ else 0) =
          ∑ rand ∈ heavyFinset, (secParam.succ : ENNReal)⁻¹ :=
      h_filtered_aux.trans h_sum_filter
    have h_sum :
        ∑ rand ∈ heavyFinset, (secParam.succ : ENNReal)⁻¹ =
          (heavyFinset.card : ENNReal) * (secParam.succ : ENNReal)⁻¹ := by
      simp [Finset.sum_const, nsmul_eq_mul]
    exact h_filtered.trans h_sum
  have h_card_nat : heavySet.ncard = heavyFinset.card :=
    Set.ncard_eq_toFinset_card (s := heavySet) (hs := h_finite)
  have h_card_eq :
      ((Set.ncard heavySet : ℕ) : ENNReal) =
        (heavyFinset.card : ENNReal) :=
    congrArg (fun n : ℕ => (n : ENNReal)) h_card_nat
  have h_succ_ne_zero : (secParam.succ : ENNReal) ≠ 0 := by
    exact_mod_cast Nat.succ_ne_zero secParam
  have h_succ_ne_top : (secParam.succ : ENNReal) ≠ (⊤ : ENNReal) :=
    ENNReal.natCast_ne_top _
  have h_mass_sum :
      (∑ rand : Fin secParam.succ,
          heavyCommitSeedWeight VC cs A x secParam ε rand) =
        (heavyFinset.card : ENNReal) * (secParam.succ : ENNReal)⁻¹ := by
    calc
      (∑ rand : Fin secParam.succ,
          heavyCommitSeedWeight VC cs A x secParam ε rand)
          = ∑ rand : Fin secParam.succ,
              if rand ∈ heavyFinset then (secParam.succ : ENNReal)⁻¹ else 0 :=
                h_sum_indicator
      _ = (heavyFinset.card : ENNReal) * (secParam.succ : ENNReal)⁻¹ :=
        h_sum_const
  have h_cancel_raw :
      (secParam.succ : ENNReal) * (secParam.succ : ENNReal)⁻¹ = 1 :=
    ENNReal.mul_inv_cancel h_succ_ne_zero h_succ_ne_top
  have h_cancel :
      (secParam.succ : ENNReal)⁻¹ * (secParam.succ : ENNReal) = 1 := by
    simpa [mul_comm] using h_cancel_raw
  calc
    heavyCommitMass VC cs A x secParam ε * (secParam.succ : ENNReal)
        =
          (∑ rand : Fin secParam.succ,
              heavyCommitSeedWeight VC cs A x secParam ε rand)
            * (secParam.succ : ENNReal) := by
              simp [heavyCommitMass_eq_uniform_randomness_sum]
    _ =
        ((heavyFinset.card : ENNReal) * (secParam.succ : ENNReal)⁻¹) *
          (secParam.succ : ENNReal) := by
            rw [h_mass_sum]
    _ = (heavyFinset.card : ENNReal) := by
          have h_mul_assoc :
              ((heavyFinset.card : ENNReal) * (secParam.succ : ENNReal)⁻¹) *
                  (secParam.succ : ENNReal)
                  = (heavyFinset.card : ENNReal) *
                      ((secParam.succ : ENNReal)⁻¹ *
                        (secParam.succ : ENNReal)) := by
            simpa using
              (mul_assoc (heavyFinset.card : ENNReal)
                ((secParam.succ : ENNReal)⁻¹)
                (secParam.succ : ENNReal))
          have h_value :=
            congrArg (fun t : ENNReal => (heavyFinset.card : ENNReal) * t) h_cancel
          have h_value' :
              (heavyFinset.card : ENNReal) *
                  ((secParam.succ : ENNReal)⁻¹ * (secParam.succ : ENNReal)) =
                (heavyFinset.card : ENNReal) :=
            h_value.trans (by simp)
          exact h_mul_assoc.trans h_value'
    _ = (Nat.cast (Set.ncard heavySet) : ENNReal) := h_card_eq.symm

lemma sum_uniform_pmf_card
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (S : Finset α) :
    ∑ a ∈ S, (uniform_pmf : PMF α) a
      = (S.card : ENNReal) * (Fintype.card α : ENNReal)⁻¹ := by
  classical
  change ∑ _ ∈ S, (Fintype.card α : ENNReal)⁻¹
      = (S.card : ENNReal) * (Fintype.card α : ENNReal)⁻¹
  simp [Finset.sum_const, nsmul_eq_mul]

lemma heavyCommitChallengeWeight_sum_finset
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (ε : ℝ) (rand : Fin secParam.succ) (S : Finset F) :
    ∑ α ∈ S, heavyCommitChallengeWeight VC cs A x secParam ε rand α =
      heavyCommitSeedWeight VC cs A x secParam ε rand *
        ((S.card : ENNReal) * (Fintype.card F : ENNReal)⁻¹) := by
  classical
  by_cases h_mem :
      commitTupleOfRandomness VC cs A x secParam rand ∈
        heavyCommitments VC cs A x secParam ε
  · have h_seed :
      heavyCommitSeedWeight VC cs A x secParam ε rand =
        (uniform_pmf : PMF (Fin (secParam.succ))) rand := by
        simp [heavyCommitSeedWeight, h_mem]
    have h_uniform_sum :
        ∑ α ∈ S,
            heavyCommitChallengeWeight VC cs A x secParam ε rand α
          = (uniform_pmf : PMF (Fin (secParam.succ))) rand *
              ((S.card : ENNReal) * (Fintype.card F : ENNReal)⁻¹) := by
        calc
          ∑ α ∈ S,
              heavyCommitChallengeWeight VC cs A x secParam ε rand α
              = ∑ α ∈ S,
                  (uniform_pmf : PMF (Fin (secParam.succ))) rand *
                    (uniform_pmf : PMF F) α := by
                    simp [heavyCommitChallengeWeight, h_mem]
          _ = (uniform_pmf : PMF (Fin (secParam.succ))) rand *
                ∑ α ∈ S, (uniform_pmf : PMF F) α := by
                  simpa [mul_comm, mul_left_comm, mul_assoc]
                    using (Finset.mul_sum
                      (a := (uniform_pmf : PMF (Fin (secParam.succ))) rand)
                      (s := S)
                      (f := fun α => (uniform_pmf : PMF F) α)).symm
          _ = (uniform_pmf : PMF (Fin (secParam.succ))) rand *
                ((S.card : ENNReal) * (Fintype.card F : ENNReal)⁻¹) := by
                  simp [sum_uniform_pmf_card (S := S), mul_comm, mul_assoc]
    have h_final :
        ∑ α ∈ S,
            heavyCommitChallengeWeight VC cs A x secParam ε rand α
          = heavyCommitSeedWeight VC cs A x secParam ε rand *
              ((S.card : ENNReal) * (Fintype.card F : ENNReal)⁻¹) := by
      simpa [h_seed, mul_comm, mul_left_comm, mul_assoc] using h_uniform_sum
    exact h_final
  · simp [heavyCommitChallengeWeight, heavyCommitSeedWeight, h_mem]

lemma tsum_commitMass_eq_one
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    ∑' comm_tuple,
        commitMass VC cs A x secParam comm_tuple = 1 := by
  classical
  have h_tsum := (run_adversary_commit_tuple VC cs A x secParam).tsum_coe
  have h_congr :
      ∑' comm_tuple,
          commitMass VC cs A x secParam comm_tuple
            = ∑' comm_tuple,
                run_adversary_commit_tuple VC cs A x secParam comm_tuple := by
    refine tsum_congr ?_
    intro comm_tuple
    simp [commitMass_eq_run_adversary_commit_tuple (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (comm_tuple := comm_tuple)]
  calc
    ∑' comm_tuple, commitMass VC cs A x secParam comm_tuple
        = ∑' comm_tuple,
            run_adversary_commit_tuple VC cs A x secParam comm_tuple := h_congr
    _ = 1 := h_tsum

lemma commitMass_le_one
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    commitMass VC cs A x secParam comm_tuple ≤ 1 := by
  classical
  unfold commitMass
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_pointwise :
      (fun t => if transcriptCommitTuple VC t = comm_tuple then p t else 0)
        ≤ fun t => p t := by
    intro t
    by_cases hComm : transcriptCommitTuple VC t = comm_tuple
    · simp [hComm]
    · simp [hComm]
  have h_le := ENNReal.tsum_le_tsum h_pointwise
  have h_total : ∑' t, p t = 1 := p.tsum_coe
  simpa [commitMass, hp, h_total]
    using h_le

lemma successProbability_nonneg
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    0 ≤ successProbability VC cs A x secParam := by
  unfold successProbability
  exact ENNReal.toReal_nonneg

lemma successProbability_le_one
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    successProbability VC cs A x secParam ≤ 1 := by
  classical
  unfold successProbability
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  set S := {t : Transcript F VC | success_event VC cs x t} with hS
  set U : Set (Transcript F VC) := Set.univ
  have h_le : p.toOuterMeasure S ≤ p.toOuterMeasure U :=
    p.toOuterMeasure_mono (fun t _ => Set.mem_univ _)
  have h_univ : p.toOuterMeasure U = 1 := by
    have h_support_subset : p.support ⊆ U := fun _ _ => trivial
    simpa [U] using (p.toOuterMeasure_apply_eq_one_iff (s := U)).2 h_support_subset
  have h_le_one : p.toOuterMeasure S ≤ ENNReal.ofReal 1 := by
    simpa [U, h_univ, ENNReal.ofReal_one] using h_le
  have h_toReal := ENNReal.toReal_le_of_le_ofReal zero_le_one h_le_one
  simpa [successProbability, hp, hS, U, ENNReal.ofReal_one] using h_toReal

lemma successMass_le_one
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    successMass VC cs A x secParam ≤ 1 := by
  classical
  unfold successMass
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_pointwise :
      (fun t => if success_event VC cs x t then p t else 0)
        ≤ fun t => p t := by
    intro t
    by_cases hSucc : success_event VC cs x t
    · simp [hSucc]
    · simp [hSucc]
  have h_le := ENNReal.tsum_le_tsum h_pointwise
  have h_total : ∑' t, p t = 1 := p.tsum_coe
  simpa [hp, h_total] using h_le

lemma successProbability_toReal_successMass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    successProbability VC cs A x secParam =
      (successMass VC cs A x secParam).toReal := by
  classical
  unfold successProbability successMass
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_outer :=
    p.toOuterMeasure_apply {t : Transcript F VC | success_event VC cs x t}
  have h_indicator :
      (∑' t,
          Set.indicator {t : Transcript F VC | success_event VC cs x t} p t)
        = ∑' t, ite (success_event VC cs x t) (p t) 0 := by
    refine tsum_congr ?_
    intro t
    by_cases h_t : success_event VC cs x t
    · simp [Set.indicator, h_t]
    · simp [Set.indicator, h_t]
  have h_indicator' :
      ∑' t, ite (success_event VC cs x t) (p t) 0
        = ∑' t, (if success_event VC cs x t then p t else 0) := by
    simp
  simpa [hp, h_indicator, h_indicator', Set.mem_setOf_eq]
    using congrArg ENNReal.toReal h_outer

lemma successMass_ofReal_le_of_successProbability_le
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    {ε : ℝ}
    (h_prob : ε ≤ successProbability VC cs A x secParam) :
    ENNReal.ofReal ε ≤ successMass VC cs A x secParam := by
  classical
  have h_eq := successProbability_toReal_successMass
    (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
  have h_toReal : ε ≤ (successMass VC cs A x secParam).toReal := by
    simpa [h_eq] using h_prob
  have h_lt_top : successMass VC cs A x secParam < (⊤ : ENNReal) :=
    lt_of_le_of_lt (successMass_le_one (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam)) ENNReal.one_lt_top
  have h_ne_top : successMass VC cs A x secParam ≠ (⊤ : ENNReal) :=
    (lt_top_iff_ne_top).1 h_lt_top
  have h_coe := ENNReal.ofReal_toReal h_ne_top
  calc
    ENNReal.ofReal ε
        ≤ ENNReal.ofReal (successMass VC cs A x secParam).toReal :=
          ENNReal.ofReal_le_ofReal h_toReal
    _ = successMass VC cs A x secParam := by simpa using h_coe

lemma commitChallengeMass_eq_run_adversary_commit_challenge
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) (α : F) :
    commitChallengeMass VC cs A x secParam comm_tuple α =
      run_adversary_commit_challenge VC cs A x secParam (comm_tuple, α) := by
  classical
  unfold commitChallengeMass
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  change (∑' t,
            (if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0))
      = run_adversary_commit_challenge VC cs A x secParam (comm_tuple, α)
  have h_map :
      run_adversary_commit_challenge VC cs A x secParam (comm_tuple, α)
        = ∑' t,
            (if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0) := by
    simp [run_adversary_commit_challenge, transcriptCommitChallenge, hp.symm, PMF.map_apply,
      eq_comm]
  simpa using h_map.symm

lemma exists_success_transcript_of_heavyCommitment
    {ε : ℝ} (h_ε_pos : 0 < ε)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm_tuple ∈ heavyCommitments VC cs A x secParam ε) :
    ∃ t : Transcript F VC,
      success_event VC cs x t ∧
      transcriptCommitTuple VC t = comm_tuple ∧
      0 < (run_adversary_transcript (VC := VC) (cs := cs) A x secParam) t := by
  classical
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_heavy :
      0 < commitMass VC cs A x secParam comm_tuple ∧
        successMassGivenCommit VC cs A x secParam comm_tuple ≥
          ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple := by
    simpa [heavyCommitments] using h_mem
  have h_commit_pos : 0 < commitMass VC cs A x secParam comm_tuple := h_heavy.1
  have h_success_ge := h_heavy.2
  have h_eps_pos : 0 < ENNReal.ofReal ε := ENNReal.ofReal_pos.mpr h_ε_pos
  have h_sum_pos :
      0 < successMassGivenCommit VC cs A x secParam comm_tuple := by
    have h_prod_ne_zero :
        ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple ≠ 0 :=
      mul_ne_zero (ne_of_gt h_eps_pos) (ne_of_gt h_commit_pos)
    have h_prod_pos :
        0 < ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple :=
      lt_of_le_of_ne'
        (bot_le : (0 : ENNReal) ≤ ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple)
        (by simpa [eq_comm] using h_prod_ne_zero)
    exact lt_of_lt_of_le h_prod_pos h_success_ge
  have h_sum_pos' :
      0 < (∑'
        t,
          (if
              success_event VC cs x t ∧
                  transcriptCommitTuple VC t = comm_tuple
            then p t
            else 0)) := by
    simpa [successMassGivenCommit, hp] using h_sum_pos
  refine by_contra ?_
  intro h_none
  have h_all_zero :
      ∀ t : Transcript F VC,
        success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple → p t = 0 := by
    intro t h_pred
    have h_not_pos : ¬ 0 < p t := by
      intro h_pos
      exact h_none ⟨t, h_pred.1, h_pred.2, h_pos⟩
    have h_le_zero : p t ≤ 0 := not_lt.mp h_not_pos
    exact le_antisymm h_le_zero bot_le
  have h_zero :
      (∑'
        t,
          (if
              success_event VC cs x t ∧
                  transcriptCommitTuple VC t = comm_tuple
            then p t
            else 0)) = 0 := by
    refine ENNReal.tsum_eq_zero.mpr ?_
    intro t
    by_cases h_pred : success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple
    · have : p t = 0 := h_all_zero t h_pred
      simp [h_pred, this]
    · simp [h_pred]
  exact (ne_of_gt h_sum_pos') h_zero

lemma exists_success_transcript_of_successMassGivenCommitAndChallenge_pos
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    {α : F}
    (h_pos : 0 <
      successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α) :
    ∃ t : Transcript F VC,
      success_event VC cs x t ∧
      transcriptCommitTuple VC t = comm_tuple ∧
      t.view.alpha = α ∧
      0 < (run_adversary_transcript (VC := VC) (cs := cs) A x secParam) t := by
  classical
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_sum_pos :
      0 < (∑'
        t,
          (if
              success_event VC cs x t ∧
                transcriptCommitTuple VC t = comm_tuple ∧
                t.view.alpha = α then p t else 0)) := by
    simpa [successMassGivenCommitAndChallenge, hp] using h_pos
  by_contra h_none
  have h_forall := h_none
  push_neg at h_forall
  have h_all_zero :
      ∀ t : Transcript F VC,
        success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α →
          p t = 0 := by
    intro t h_pred
    have h_le_zero : p t ≤ 0 := h_forall t h_pred.1 h_pred.2.1 h_pred.2.2
    exact le_antisymm h_le_zero bot_le
  have h_zero :
      (∑'
        t,
          (if
              success_event VC cs x t ∧
                transcriptCommitTuple VC t = comm_tuple ∧
                t.view.alpha = α then p t else 0)) = 0 := by
    refine ENNReal.tsum_eq_zero.mpr ?_
    intro t
    by_cases h_pred :
        success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α
    · have : p t = 0 := h_all_zero t h_pred
      simp [h_pred, this]
    · simp [h_pred]
  exact (ne_of_gt h_sum_pos) h_zero

lemma heavyCommitments_mem_iff (ε : ℝ)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment} :
    comm_tuple ∈ heavyCommitments VC cs A x secParam ε ↔
      0 < commitMass VC cs A x secParam comm_tuple ∧
        successMassGivenCommit VC cs A x secParam comm_tuple ≥
          ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple := by
  classical
  rfl

lemma successMassGivenCommit_lt_of_not_heavy (ε : ℝ)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_not : comm_tuple ∉ heavyCommitments VC cs A x secParam ε) :
    commitMass VC cs A x secParam comm_tuple = 0 ∨
      successMassGivenCommit VC cs A x secParam comm_tuple <
        ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple := by
  classical
  have h :=
    (heavyCommitments_mem_iff (VC := VC) (cs := cs) (A := A) (x := x)
      (secParam := secParam) (ε := ε) (comm_tuple := comm_tuple)).not.mp h_not
  by_cases h_mass : commitMass VC cs A x secParam comm_tuple = 0
  · exact Or.inl h_mass
  · have h_mass_pos : 0 < commitMass VC cs A x secParam comm_tuple := by
      refine lt_of_le_of_ne' (bot_le : (0 : ENNReal) ≤ _ ) ?_
      simpa [eq_comm] using h_mass
    have : ¬ successMassGivenCommit VC cs A x secParam comm_tuple ≥
        ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple :=
      (not_and.mp h) h_mass_pos
    exact Or.inr (lt_of_not_ge this)

lemma successMassGivenCommit_le_of_not_heavy (ε : ℝ)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_not : comm_tuple ∉ heavyCommitments VC cs A x secParam ε) :
    successMassGivenCommit VC cs A x secParam comm_tuple ≤
      ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple := by
  classical
  obtain h_zero | h_lt :=
    successMassGivenCommit_lt_of_not_heavy (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε (comm_tuple := comm_tuple) h_not
  · have h_success_zero :
        successMassGivenCommit VC cs A x secParam comm_tuple = 0 := by
      have := successMassGivenCommit_le_commitMass (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) comm_tuple
      have : successMassGivenCommit VC cs A x secParam comm_tuple ≤ 0 := by
        simpa [h_zero] using this
      exact le_antisymm this bot_le
    simp [h_zero, h_success_zero]
  · exact (le_of_lt h_lt)

lemma exists_commitMass_pos :
    ∃ comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment,
      0 < commitMass VC cs A x secParam comm_tuple := by
  classical
  have h_total := tsum_commitMass_eq_one (VC := VC) (cs := cs) (A := A)
    (x := x) (secParam := secParam)
  by_contra h_none
  push_neg at h_none
  have h_zero : (∑'
      comm_tuple,
        commitMass VC cs A x secParam comm_tuple) = 0 :=
    ENNReal.tsum_eq_zero.2 (by intro comm_tuple; simpa using h_none comm_tuple)
  have h_contra : (0 : ENNReal) = 1 := h_zero.symm.trans h_total
  exact (zero_ne_one : (0 : ENNReal) ≠ 1) h_contra

/- Разложение условной массы успеха по вызовам. -/
lemma successMassGivenCommit_eq_tsum_successMassGivenCommitChallenge
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    successMassGivenCommit VC cs A x secParam comm_tuple =
      ∑' α, successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
  classical
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_pointwise :
      ∀ t : Transcript F VC,
        (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple then p t else 0)
          = ∑' α,
              (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧
                  t.view.alpha = α then p t else 0) := by
    intro t
    by_cases hSucc : success_event VC cs x t
    · by_cases hComm : transcriptCommitTuple VC t = comm_tuple
      · have h_single :
          ∑' α, (if t.view.alpha = α then p t else 0) = p t := by
          refine (tsum_eq_single (t.view.alpha)
              (fun α hne => ?_)).trans ?_
          · have hneq : t.view.alpha ≠ α := by
              simpa [eq_comm] using hne
            simp [hneq]
          · simp
        have h_congr :
            ∑' α,
                (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧
                    t.view.alpha = α then p t else 0)
              = ∑' α, (if t.view.alpha = α then p t else 0) := by
          refine tsum_congr ?_
          intro α
          simp [hSucc, hComm]
        have h_sum := h_congr.trans h_single
        simpa [hSucc, hComm] using h_sum.symm
      · simp [hSucc, hComm]
    · simp [hSucc]
  have h_split :
      ∑' t,
          (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple then p t else 0)
        = ∑' t,
            ∑' α,
              (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧
                  t.view.alpha = α then p t else 0) := by
    refine tsum_congr ?_
    intro t
    exact h_pointwise t
  have h_commute :=
    ENNReal.tsum_comm
      (f :=
        fun α (t : Transcript F VC) =>
          if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧
              t.view.alpha = α then p t else 0)
  have h_swapped :
      ∑' t,
          ∑' α,
            (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧
                t.view.alpha = α then p t else 0)
        = ∑' α,
            ∑' t,
              (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧
                  t.view.alpha = α then p t else 0) := by
    simpa using h_commute.symm
  have h_left : successMassGivenCommit VC cs A x secParam comm_tuple
      = ∑' t,
          (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple then p t else 0)
          := by
    simp [successMassGivenCommit, hp]
  have h_right :
      ∑' α, successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α
        = ∑' α,
            ∑' t,
              (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧
                  t.view.alpha = α then p t else 0) := by
    refine tsum_congr ?_
    intro α
    simp [successMassGivenCommitAndChallenge, hp]
  exact h_left.trans (h_split.trans (h_swapped.trans h_right.symm))

/-- Finite version of the previous decomposition. -/
lemma successMassGivenCommit_eq_sum_successMassGivenCommitChallenge
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    successMassGivenCommit VC cs A x secParam comm_tuple =
      ∑ α ∈ (Finset.univ : Finset F),
        successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
  classical
  simpa [tsum_fintype] using
    successMassGivenCommit_eq_tsum_successMassGivenCommitChallenge (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (comm_tuple := comm_tuple)

/-- Разложение общей массы успеха на сумму условных масс по кортежам коммитов. -/
lemma successMass_eq_tsum_successMassGivenCommit {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    successMass VC cs A x secParam =
      ∑' comm_tuple,
        successMassGivenCommit VC cs A x secParam comm_tuple := by
  classical
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_pointwise :
      ∀ t : Transcript F VC,
        (if success_event VC cs x t then p t else 0)
          = ∑' comm,
              (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm then p t else 0) := by
    intro t
    by_cases hSucc : success_event VC cs x t
    · have h_single :
        ∑' comm,
          (if comm = transcriptCommitTuple VC t then p t else 0) = p t := by
        refine (tsum_eq_single (transcriptCommitTuple VC t)
          (fun comm hne => if_neg hne)).trans ?_
        simp
      have h_sum :
          ∑' comm,
              (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm then p t else 0)
                = p t := by
        have h_congr :
            ∑' comm,
                (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm then p t else 0)
              = ∑' comm,
                  (if comm = transcriptCommitTuple VC t then p t else 0) := by
          refine tsum_congr ?_
          intro comm
          simp [hSucc, eq_comm]
        exact h_congr.trans h_single
      simpa [hSucc] using h_sum.symm
    · simp [hSucc]
  have h_split :
      ∑' t, (if success_event VC cs x t then p t else 0)
        = ∑' t,
            ∑' comm,
              (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm then p t else 0) := by
    refine tsum_congr ?_
    intro t
    exact h_pointwise t
  have h_commute :=
    ENNReal.tsum_comm
      (f := fun comm (t : Transcript F VC) =>
        if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm then p t else 0)
  have h_swapped :
      ∑' t,
          ∑' comm,
            (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm then p t else 0)
        = ∑' comm,
            ∑' t,
              (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm then p t else 0) := by
    simpa using h_commute.symm
  have h_final := h_split.trans h_swapped
  have h_left : successMass VC cs A x secParam
      = ∑' t, (if success_event VC cs x t then p t else 0) := by
    simp [successMass, hp]
  have h_right :
      ∑' comm_tuple,
          successMassGivenCommit VC cs A x secParam comm_tuple
        = ∑' comm,
            ∑' t,
              (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm then p t else 0) := by
    refine tsum_congr ?_
    intro comm
    simp [successMassGivenCommit, hp]
  exact h_left.trans (h_final.trans h_right.symm)

/-!
#### Heavy commitment bounds

These inequalities turn the pointwise heaviness predicate into statements about
the total success mass.  They assert that if no commitment tuple is heavy, then
overall success is bounded by `ε`, and conversely any larger success mass forces
the existence of a heavy tuple.
-/

lemma successMass_le_of_all_not_heavy
    (ε : ℝ)
    (h_not : ∀ comm_tuple,
      comm_tuple ∉ heavyCommitments VC cs A x secParam ε) :
    successMass VC cs A x secParam ≤ ENNReal.ofReal ε := by
  classical
  have h_pointwise : ∀ comm_tuple,
      successMassGivenCommit VC cs A x secParam comm_tuple ≤
        ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple := by
    intro comm_tuple
    exact successMassGivenCommit_le_of_not_heavy (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)
      (comm_tuple := comm_tuple) (h_not comm_tuple)
  have h_sum := ENNReal.tsum_le_tsum h_pointwise
  have h_success := successMass_eq_tsum_successMassGivenCommit
    (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
  have h_comm := tsum_commitMass_eq_one (VC := VC) (cs := cs) (A := A)
    (x := x) (secParam := secParam)
  have h_rhs :
      (∑'
        comm_tuple,
          ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple)
        = ENNReal.ofReal ε := by
    have :=
      ENNReal.tsum_mul_right
        (f := fun comm_tuple => commitMass VC cs A x secParam comm_tuple)
        (a := ENNReal.ofReal ε)
    simpa [mul_comm, mul_left_comm, mul_assoc, h_comm] using this
  simpa [h_success, h_rhs] using h_sum

lemma exists_heavyCommitment_of_successMass_lt
  (ε : ℝ)
  (h_mass : ENNReal.ofReal ε < successMass VC cs A x secParam) :
    ∃ comm_tuple,
      comm_tuple ∈ heavyCommitments VC cs A x secParam ε := by
  classical
  by_contra h_none
  push_neg at h_none
  have h_le := successMass_le_of_all_not_heavy (VC := VC) (cs := cs) (A := A)
    (x := x) (secParam := secParam) (ε := ε) h_none
  exact (not_le_of_gt h_mass) h_le

lemma exists_heavyCommitment_of_successProbability_lt
  (ε : ℝ)
  (h_prob : ε < successProbability VC cs A x secParam) :
    ∃ comm_tuple,
      comm_tuple ∈ heavyCommitments VC cs A x secParam ε := by
  classical
  by_cases h_nonneg : 0 ≤ ε
  · by_contra h_none
    push_neg at h_none
    have h_mass_le := successMass_le_of_all_not_heavy (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_none
    have h_eq := successProbability_toReal_successMass (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)
    have h_toReal_le := ENNReal.toReal_le_of_le_ofReal h_nonneg h_mass_le
    have h_prob_le : successProbability VC cs A x secParam ≤ ε := by
      simpa [h_eq] using h_toReal_le
    exact (not_le_of_gt h_prob) h_prob_le
  · have h_neg : ε < 0 := lt_of_not_ge h_nonneg
    have h_ofReal_zero : ENNReal.ofReal ε = 0 := by
      simpa using ENNReal.ofReal_of_nonpos (le_of_lt h_neg)
    obtain ⟨comm_tuple, h_mass_pos⟩ := exists_commitMass_pos (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)
    refine ⟨comm_tuple, ?_⟩
    have h_success_ge :
        successMassGivenCommit VC cs A x secParam comm_tuple ≥ (0 : ENNReal) :=
      bot_le
    have h_ge :
        successMassGivenCommit VC cs A x secParam comm_tuple ≥
          ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple := by
      convert h_success_ge using 1
      simp [h_ofReal_zero]
    exact (heavyCommitments_mem_iff (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) (comm_tuple := comm_tuple)).2
        ⟨h_mass_pos, h_ge⟩


lemma heavyCommitMass_mul_eps_le_successMass
  (ε : ℝ) :
    ENNReal.ofReal ε *
        heavyCommitMass VC cs A x secParam ε ≤
      successMass VC cs A x secParam := by
  classical
  set C := VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment
  have h_pointwise :
      (fun comm_tuple : C =>
        if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
          ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple
        else
          0)
        ≤ fun comm_tuple : C =>
            successMassGivenCommit VC cs A x secParam comm_tuple := by
    intro comm_tuple
    by_cases h_mem :
        comm_tuple ∈ heavyCommitments VC cs A x secParam ε
    · have h_heavy :=
        (heavyCommitments_mem_iff (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) (ε := ε)
          (comm_tuple := comm_tuple)).1 h_mem
      have h_bound := h_heavy.2
      simpa [h_mem]
        using h_bound
    · have h_nonneg :
          0 ≤ successMassGivenCommit VC cs A x secParam comm_tuple := bot_le
      simp [h_mem, h_nonneg]
  have h_sum := ENNReal.tsum_le_tsum h_pointwise
  have h_congr_fun :
      (∑' (comm_tuple : C),
          if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
            ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple
          else
            0)
        = ∑' (comm_tuple : C),
            ENNReal.ofReal ε *
              (if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
                commitMass VC cs A x secParam comm_tuple
              else
                0) := by
    refine tsum_congr ?_
    intro comm_tuple
    by_cases h_mem :
        comm_tuple ∈ heavyCommitments VC cs A x secParam ε
    · simp [h_mem]
    · simp [h_mem]
  have h_left :
      (∑' (comm_tuple : C),
          if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
            ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple
          else
            0)
        = heavyCommitMass VC cs A x secParam ε * ENNReal.ofReal ε := by
    have := ENNReal.tsum_mul_right
      (f := fun comm_tuple : C =>
        if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
          commitMass VC cs A x secParam comm_tuple
        else
          0)
      (a := ENNReal.ofReal ε)
    simpa [h_congr_fun, heavyCommitMass, mul_comm, mul_left_comm, mul_assoc]
      using this
  have h_right :
      (∑' (comm_tuple : C),
          successMassGivenCommit VC cs A x secParam comm_tuple)
        = successMass VC cs A x secParam :=
    (successMass_eq_tsum_successMassGivenCommit (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)).symm
  have h_goal :
      heavyCommitMass VC cs A x secParam ε * ENNReal.ofReal ε ≤
        successMass VC cs A x secParam := by
    simpa [h_left, h_right]
      using h_sum
  simpa [mul_comm, mul_left_comm, mul_assoc]
    using h_goal


lemma heavyCommitMass_le_one
  (ε : ℝ) :
    heavyCommitMass VC cs A x secParam ε ≤ 1 := by
  classical
  set C := VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment
  have h_pointwise :
      (fun comm_tuple : C =>
        if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
          commitMass VC cs A x secParam comm_tuple
        else
          0)
        ≤ fun comm_tuple : C => commitMass VC cs A x secParam comm_tuple := by
    intro comm_tuple
    by_cases h_mem :
        comm_tuple ∈ heavyCommitments VC cs A x secParam ε
    · simp [h_mem]
    · simp [h_mem]
  have h_sum := ENNReal.tsum_le_tsum h_pointwise
  have h_total :=
    tsum_commitMass_eq_one (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam)
  have h_bound :
      heavyCommitMass VC cs A x secParam ε ≤
        ∑' (comm_tuple : C), commitMass VC cs A x secParam comm_tuple := by
    simpa [heavyCommitMass]
      using h_sum
  have h_sum_eq :
      ∑' (comm_tuple : C), commitMass VC cs A x secParam comm_tuple = 1 := by
    simpa using h_total
  simpa [h_sum_eq] using h_bound

lemma heavyCommitMass_toReal_mul_randomnessCard_eq_heavySeedCount
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ) :
    (heavyCommitMass VC cs A x secParam ε).toReal * (secParam.succ : ℝ) =
      (Set.ncard
        (heavyRandomnessSet (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) ε) : ℝ) := by
  classical
  have h_eq :=
    heavyCommitMass_mul_randomnessCard_eq_heavySeedCount (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε)
  have h_mass_lt_top :
      heavyCommitMass VC cs A x secParam ε < (⊤ : ENNReal) := by
    have h_le_one :=
      heavyCommitMass_le_one (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε)
    exact lt_of_le_of_lt h_le_one ENNReal.one_lt_top
  have h_mass_ne_top :
      heavyCommitMass VC cs A x secParam ε ≠ (⊤ : ENNReal) :=
    (lt_top_iff_ne_top).1 h_mass_lt_top
  have h_succ_ne_top :
      (secParam.succ : ENNReal) ≠ (⊤ : ENNReal) :=
    ENNReal.natCast_ne_top _
  have h_toReal := congrArg ENNReal.toReal h_eq
  simpa [ENNReal.toReal_mul, h_mass_ne_top, h_succ_ne_top]
    using h_toReal


lemma lightCommitMass_le_one
  (ε : ℝ) :
    lightCommitMass VC cs A x secParam ε ≤ 1 := by
  classical
  set C := VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment
  have h_pointwise :
      (fun comm_tuple : C =>
        if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
          0
        else
          commitMass VC cs A x secParam comm_tuple)
        ≤ fun comm_tuple : C => commitMass VC cs A x secParam comm_tuple := by
    intro comm_tuple
    by_cases h_mem :
        comm_tuple ∈ heavyCommitments VC cs A x secParam ε
    · simp [h_mem]
    · simp [h_mem]
  have h_sum := ENNReal.tsum_le_tsum h_pointwise
  have h_total :=
    tsum_commitMass_eq_one (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam)
  have h_bound :
      lightCommitMass VC cs A x secParam ε ≤
        ∑' (comm_tuple : C), commitMass VC cs A x secParam comm_tuple := by
    simpa [lightCommitMass]
      using h_sum
  have h_sum_eq :
      ∑' (comm_tuple : C), commitMass VC cs A x secParam comm_tuple = 1 := by
    simpa using h_total
  simpa [h_sum_eq] using h_bound


lemma heavyCommitMass_add_lightCommitMass_eq_one
  (ε : ℝ) :
    heavyCommitMass VC cs A x secParam ε +
        lightCommitMass VC cs A x secParam ε = 1 := by
  classical
  set C := VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment
  let f : C → ENNReal :=
    fun comm_tuple =>
      if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
        commitMass VC cs A x secParam comm_tuple
      else
        0
  let g : C → ENNReal :=
    fun comm_tuple =>
      if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
        0
      else
        commitMass VC cs A x secParam comm_tuple
  have h_partition :
      (∑' (c : C), (f c + g c))
        = ∑' (comm_tuple : C), commitMass VC cs A x secParam comm_tuple := by
    refine tsum_congr ?_
    intro comm_tuple
    by_cases h_mem :
        comm_tuple ∈ heavyCommitments VC cs A x secParam ε
    · simp [f, g, h_mem]
    · simp [f, g, h_mem]
  have h_total :=
    tsum_commitMass_eq_one (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam)
  have h_split :
      (∑' (comm_tuple : C), f comm_tuple) +
        (∑' (comm_tuple : C), g comm_tuple)
        = ∑' (comm_tuple : C), (f comm_tuple + g comm_tuple) := by
    simpa [f, g] using (ENNReal.tsum_add (f := f) (g := g)).symm
  have h_partition' := h_partition
  calc
    heavyCommitMass VC cs A x secParam ε +
        lightCommitMass VC cs A x secParam ε
        = (∑' (comm_tuple : C), f comm_tuple) +
            (∑' (comm_tuple : C), g comm_tuple) := by rfl
    _ = ∑' (comm_tuple : C), (f comm_tuple + g comm_tuple) := h_split
    _ = 1 := by
          simpa [h_partition', h_total]


lemma successMass_le_heavyMass_add_eps_mul_light
    (ε : ℝ) :
    successMass VC cs A x secParam ≤
      heavyCommitMass VC cs A x secParam ε +
        ENNReal.ofReal ε * lightCommitMass VC cs A x secParam ε := by
  classical
  set C := VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment
  have h_pointwise :
      ∀ comm_tuple : C,
        successMassGivenCommit VC cs A x secParam comm_tuple ≤
          (if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
            commitMass VC cs A x secParam comm_tuple
          else
            ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple) := by
    intro comm_tuple
    by_cases h_mem :
        comm_tuple ∈ heavyCommitments VC cs A x secParam ε
    · have h_le :=
        successMassGivenCommit_le_commitMass (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) (comm_tuple := comm_tuple)
      simpa [h_mem]
        using h_le
    · have h_le :=
        successMassGivenCommit_le_of_not_heavy (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) (ε := ε)
          (comm_tuple := comm_tuple) h_mem
      simpa [h_mem]
        using h_le
  have h_sum := ENNReal.tsum_le_tsum h_pointwise
  have h_success := successMass_eq_tsum_successMassGivenCommit
    (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
  let f : C → ENNReal :=
    fun comm_tuple =>
      if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
        commitMass VC cs A x secParam comm_tuple
      else
        0
  let g : C → ENNReal :=
    fun comm_tuple =>
      if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
        0
      else
        commitMass VC cs A x secParam comm_tuple
  have h_integrand :
      (fun comm_tuple : C =>
        (if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
            commitMass VC cs A x secParam comm_tuple
          else
            ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple))
        = fun comm_tuple : C =>
            f comm_tuple + ENNReal.ofReal ε * g comm_tuple := by
    funext comm_tuple
    by_cases h_mem :
        comm_tuple ∈ heavyCommitments VC cs A x secParam ε
    · simp [f, g, h_mem]
    · simp [f, g, h_mem]
  have h_split :=
    ENNReal.tsum_add (f := f) (g := fun comm_tuple : C => ENNReal.ofReal ε * g comm_tuple)
  have h_left :
      (∑' (comm_tuple : C), f comm_tuple)
        = heavyCommitMass VC cs A x secParam ε := rfl
  have h_right :
      (∑' (comm_tuple : C), ENNReal.ofReal ε * g comm_tuple)
        = ENNReal.ofReal ε * lightCommitMass VC cs A x secParam ε := by
    have := ENNReal.tsum_mul_right (f := g) (a := ENNReal.ofReal ε)
    simpa [g, lightCommitMass, mul_comm, mul_left_comm, mul_assoc] using this
  have h_decompose :
      (∑' (comm_tuple : C),
          (if comm_tuple ∈ heavyCommitments VC cs A x secParam ε then
            commitMass VC cs A x secParam comm_tuple
          else
            ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple))
        = heavyCommitMass VC cs A x secParam ε +
            ENNReal.ofReal ε * lightCommitMass VC cs A x secParam ε := by
    simpa [h_integrand, h_left, h_right]
      using h_split
  have h_goal :
      successMass VC cs A x secParam ≤
        heavyCommitMass VC cs A x secParam ε +
          ENNReal.ofReal ε * lightCommitMass VC cs A x secParam ε := by
    simpa [h_success, h_decompose]
      using h_sum
  exact h_goal


lemma successMass_le_heavyMass_add_eps
  (ε : ℝ) :
    successMass VC cs A x secParam ≤
      heavyCommitMass VC cs A x secParam ε + ENNReal.ofReal ε := by
  classical
  have h_base := successMass_le_heavyMass_add_eps_mul_light
    (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam) (ε := ε)
  have h_light := lightCommitMass_le_one (VC := VC) (cs := cs) (A := A)
    (x := x) (secParam := secParam) (ε := ε)
  have h_mul_le :
      ENNReal.ofReal ε * lightCommitMass VC cs A x secParam ε ≤ ENNReal.ofReal ε := by
    have :=
      mul_le_mul_of_nonneg_left h_light (show 0 ≤ ENNReal.ofReal ε from bot_le)
    simpa [mul_comm, mul_left_comm, mul_assoc]
      using this
  exact h_base.trans (add_le_add_left h_mul_le _)

/-- Число тяжёлых случайностей оценивает запас вероятности успеха. -/
lemma heavyRandomness_card_lower_bound
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_nonneg : 0 ≤ ε) :
    (successProbability VC cs A x secParam - ε) * (secParam.succ : ℝ) ≤
      (Set.ncard
        (heavyRandomnessSet (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) ε) : ℝ) := by
  classical
  have h_mass_bound :=
    successMass_le_heavyMass_add_eps (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε)
  have h_success_lt_top :=
    lt_of_le_of_lt
      (successMass_le_one (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam)) ENNReal.one_lt_top
  have h_success_ne_top :
      successMass VC cs A x secParam ≠ (⊤ : ENNReal) :=
    (lt_top_iff_ne_top).1 h_success_lt_top
  have h_heavy_lt_top :=
    lt_of_le_of_lt
      (heavyCommitMass_le_one (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε)) ENNReal.one_lt_top
  have h_heavy_ne_top :
      heavyCommitMass VC cs A x secParam ε ≠ (⊤ : ENNReal) :=
    (lt_top_iff_ne_top).1 h_heavy_lt_top
  have h_ofReal_ne_top : ENNReal.ofReal ε ≠ (⊤ : ENNReal) := by simp
  have h_sum_ne_top :
      heavyCommitMass VC cs A x secParam ε + ENNReal.ofReal ε ≠ (⊤ : ENNReal) :=
    ENNReal.add_ne_top.mpr ⟨h_heavy_ne_top, h_ofReal_ne_top⟩
  have h_toReal :=
    (ENNReal.toReal_le_toReal h_success_ne_top h_sum_ne_top).mpr h_mass_bound
  have h_success_eq :=
    successProbability_toReal_successMass (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam)
  have h_sum_toReal :=
    ENNReal.toReal_add h_heavy_ne_top h_ofReal_ne_top
  have h_ofReal_eval := ENNReal.toReal_ofReal h_nonneg
  have h_success_prob_le :
      successProbability VC cs A x secParam ≤
        (heavyCommitMass VC cs A x secParam ε).toReal + ε := by
    simpa [h_success_eq, h_sum_toReal, h_ofReal_eval]
      using h_toReal
  have h_sub :
      successProbability VC cs A x secParam - ε ≤
        (heavyCommitMass VC cs A x secParam ε).toReal :=
    (sub_le_iff_le_add).2 h_success_prob_le
  have h_succ_pos : 0 < (secParam.succ : ℝ) := by
    exact_mod_cast Nat.succ_pos secParam
  have h_mul :=
    mul_le_mul_of_nonneg_right h_sub (le_of_lt h_succ_pos)
  have h_card_eq :=
    heavyCommitMass_toReal_mul_randomnessCard_eq_heavySeedCount
      (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
      (ε := ε)
  have h_card_eq' :
      (heavyCommitMass VC cs A x secParam ε).toReal *
          ((secParam : ℝ) + 1) =
        (Set.ncard
          (heavyRandomnessSet (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam) ε) : ℝ) := by
    simpa [Nat.cast_succ, add_comm, add_left_comm, add_assoc]
      using h_card_eq
  have h_goal :
      (successProbability VC cs A x secParam - ε) * (secParam.succ : ℝ) ≤
        (Set.ncard
          (heavyRandomnessSet (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam) ε) : ℝ) := by
    simpa [h_card_eq', Nat.cast_succ, add_comm, add_left_comm, add_assoc]
      using h_mul
  exact h_goal

lemma heavyRandomness_finset_lower_bound
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_nonneg : 0 ≤ ε) :
    (successProbability VC cs A x secParam - ε) * (secParam.succ : ℝ) ≤
      (heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε).card := by
  classical
  have h :=
    heavyRandomness_card_lower_bound (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_nonneg
  have h_cast :=
    heavyRandomnessFinset_card_cast (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)
  simpa [h_cast]
    using h

/-- Ненулевая прибавка к вероятности успеха гарантирует существование тяжёлой случайности. -/
lemma heavyRandomnessFinset_nonempty_of_successProbability_gt
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_nonneg : 0 ≤ ε)
    (h_gap : ε < successProbability VC cs A x secParam) :
    (heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε).Nonempty := by
  classical
  have h_lower :=
    heavyRandomness_finset_lower_bound (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_nonneg
  have h_prod_pos :
      0 < (successProbability VC cs A x secParam - ε) * (secParam.succ : ℝ) := by
    have h_diff_pos :
        0 < successProbability VC cs A x secParam - ε :=
      sub_pos.mpr h_gap
    have h_succ_pos : 0 < (secParam.succ : ℝ) := by
      exact_mod_cast Nat.succ_pos secParam
    exact mul_pos h_diff_pos h_succ_pos
  have h_card_real_pos :
      0 < ((heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε).card : ℝ) :=
    lt_of_lt_of_le h_prod_pos h_lower
  have h_card_pos :
      0 < (heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε).card := by
    exact_mod_cast h_card_real_pos
  exact Finset.card_pos.mp h_card_pos

/-- Существует конкретная случайность, делающая коммит тяжелым, если вероятность успеха превосходит ε. -/
lemma exists_heavyRandomness_of_successProbability_gt
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_nonneg : 0 ≤ ε)
    (h_gap : ε < successProbability VC cs A x secParam) :
    ∃ rand : Fin secParam.succ,
      commitTupleOfRandomness VC cs A x secParam rand ∈
        heavyCommitments VC cs A x secParam ε := by
  classical
  obtain ⟨rand, h_rand⟩ :=
    heavyRandomnessFinset_nonempty_of_successProbability_gt (VC := VC)
      (cs := cs) (A := A) (x := x) (secParam := secParam) (ε := ε)
      h_nonneg h_gap
  refine ⟨rand, ?_⟩
  have h_mem :=
    (mem_heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)).1 h_rand
  exact h_mem


lemma exists_success_transcript_of_successProbability_lt
  {ε : ℝ}
  (h_ε_pos : 0 < ε)
  (h_prob : ε < successProbability VC cs A x secParam) :
    ∃ t : Transcript F VC,
      success_event VC cs x t ∧
      0 < (run_adversary_transcript (VC := VC) (cs := cs) A x secParam) t := by
  classical
  obtain ⟨comm_tuple, h_heavy⟩ :=
    exists_heavyCommitment_of_successProbability_lt (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_prob
  obtain ⟨t, h_success, _, h_pos⟩ :=
    exists_success_transcript_of_heavyCommitment (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_ε_pos h_heavy
  exact ⟨t, h_success, h_pos⟩

lemma successMass_pos_of_successProbability_lt
  {ε : ℝ}
  (h_ε_pos : 0 < ε)
  (h_prob : ε < successProbability VC cs A x secParam) :
    0 < successMass VC cs A x secParam := by
  classical
  have h_success_pos : 0 < successProbability VC cs A x secParam :=
    lt_trans h_ε_pos h_prob
  by_contra h_not_pos
  have h_le_zero : successMass VC cs A x secParam ≤ 0 := not_lt.mp h_not_pos
  have h_zero : successMass VC cs A x secParam = 0 :=
    le_antisymm h_le_zero bot_le
  have h_prob_zero : successProbability VC cs A x secParam = 0 := by
    simp [successProbability_toReal_successMass (VC := VC) (cs := cs)
        (A := A) (x := x) (secParam := secParam), h_zero]
  exact (ne_of_gt h_success_pos) (by simp [h_prob_zero])

/-- Полное разложение массы успеха по кортежам коммитов и вызовам. -/
lemma successMass_eq_tsum_successMassGivenCommitAndChallenge :
    successMass VC cs A x secParam =
      ∑' comm_tuple,
        ∑' α,
          successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
  classical
  have h_comm :=
    successMass_eq_tsum_successMassGivenCommit (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)
  have h_refine :
      ∑' comm_tuple,
          successMassGivenCommit VC cs A x secParam comm_tuple
        = ∑' comm_tuple,
            ∑' α,
              successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
    refine tsum_congr ?_
    intro comm_tuple
    simpa using
      successMassGivenCommit_eq_tsum_successMassGivenCommitChallenge (VC := VC) (cs := cs)
        (A := A) (x := x) (secParam := secParam) (comm_tuple := comm_tuple)
  exact h_comm.trans h_refine

lemma exists_commit_with_successMass_pos_of_successProbability_lt
  {ε : ℝ}
  (h_ε_pos : 0 < ε)
  (h_prob : ε < successProbability VC cs A x secParam) :
    ∃ comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment,
      0 < successMassGivenCommit VC cs A x secParam comm_tuple := by
  classical
  obtain ⟨t, h_success, h_pos⟩ :=
    exists_success_transcript_of_successProbability_lt (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_ε_pos h_prob
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  refine ⟨transcriptCommitTuple VC t, ?_⟩
  let f : Transcript F VC → ENNReal := fun s =>
    if success_event VC cs x s ∧ transcriptCommitTuple VC s = transcriptCommitTuple VC t
      then p s else 0
  have hf_def :
      successMassGivenCommit VC cs A x secParam (transcriptCommitTuple VC t)
        = ∑' s, f s := by
    simp [f, successMassGivenCommit, hp]
  let g : Transcript F VC → ENNReal := fun s => if s = t then f s else 0
  have hg_le : g ≤ f := by
    intro s
    by_cases hst : s = t
    · subst hst
      simp [g]
    · have h_nonneg : 0 ≤ f s := by
        dsimp [f]
        by_cases h_cond : success_event VC cs x s ∧
            transcriptCommitTuple VC s = transcriptCommitTuple VC t
        · have : 0 ≤ p s := bot_le
          simp [h_cond, this]
        · simp [h_cond]
      simp [g, hst, h_nonneg]
  have hg_sum : (∑' s, g s) = f t := by
    classical
    refine (tsum_eq_single t (by
      intro s hst
      simp [g, hst])).trans ?_
    simp [g]
  have h_f_t : f t = p t := by
    simp [f, h_success]
  have h_tsum_le := ENNReal.tsum_le_tsum hg_le
  have h_rhs : (∑' s, f s)
      = successMassGivenCommit VC cs A x secParam (transcriptCommitTuple VC t) := by
    simp [hf_def]
  have h_lhs : (∑' s, g s) = p t := by simp [hg_sum, h_f_t]
  have h_ge : p t ≤ successMassGivenCommit VC cs A x secParam (transcriptCommitTuple VC t) := by
    simpa [h_lhs, h_rhs] using h_tsum_le
  exact lt_of_lt_of_le h_pos h_ge

lemma exists_successMassGivenCommitAndChallenge_pos_of_successMassGivenCommit_pos
  {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_pos : 0 < successMassGivenCommit VC cs A x secParam comm_tuple) :
    ∃ α : F,
      0 < successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
  classical
  have h_tsum :=
    successMassGivenCommit_eq_tsum_successMassGivenCommitChallenge (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (comm_tuple := comm_tuple)
  by_contra h_none
  push_neg at h_none
  have h_all_zero :
      ∀ α : F,
        successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α = 0 := by
    intro α
    have h_le_zero := h_none α
    exact le_antisymm h_le_zero bot_le
  have h_sum_zero :
      ∑' α,
        successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α = 0 := by
    refine ENNReal.tsum_eq_zero.mpr ?_
    intro α
    simpa using h_all_zero α
  have h_zero : successMassGivenCommit VC cs A x secParam comm_tuple = 0 := by
    simpa [h_sum_zero] using h_tsum
  exact (ne_of_gt h_pos) h_zero

lemma successMassGivenCommit_pos_of_heavyCommitment
  {ε : ℝ}
    (h_ε_pos : 0 < ε)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm_tuple ∈ heavyCommitments VC cs A x secParam ε) :
    0 < successMassGivenCommit VC cs A x secParam comm_tuple := by
  classical
  have h_heavy :
      0 < commitMass VC cs A x secParam comm_tuple ∧
        successMassGivenCommit VC cs A x secParam comm_tuple ≥
          ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple := by
    simpa [heavyCommitments] using h_mem
  have h_commit_pos : 0 < commitMass VC cs A x secParam comm_tuple := h_heavy.1
  have h_ge := h_heavy.2
  have h_eps_pos : 0 < ENNReal.ofReal ε := ENNReal.ofReal_pos.mpr h_ε_pos
  have h_prod_ne_zero :
      ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple ≠ 0 :=
    mul_ne_zero (ne_of_gt h_eps_pos) (ne_of_gt h_commit_pos)
  have h_prod_pos :
      0 < ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple :=
    lt_of_le_of_ne' (bot_le : (0 : ENNReal) ≤
        ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple)
      (by simpa [eq_comm] using h_prod_ne_zero)
  exact lt_of_lt_of_le h_prod_pos h_ge

lemma exists_successMassGivenCommitAndChallenge_pos_of_heavyCommitment
  {ε : ℝ}
    (h_ε_pos : 0 < ε)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm_tuple ∈ heavyCommitments VC cs A x secParam ε) :
    ∃ α : F,
      0 < successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
  refine exists_successMassGivenCommitAndChallenge_pos_of_successMassGivenCommit_pos
    (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
    (comm_tuple := comm_tuple)
    (successMassGivenCommit_pos_of_heavyCommitment (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_ε_pos h_mem)

lemma exists_success_transcript_with_challenge_of_heavyCommitment
  {ε : ℝ}
    (h_ε_pos : 0 < ε)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm_tuple ∈ heavyCommitments VC cs A x secParam ε) :
    ∃ (α : F) (t : Transcript F VC),
      success_event VC cs x t ∧
      transcriptCommitTuple VC t = comm_tuple ∧
      t.view.alpha = α ∧
      0 < (run_adversary_transcript (VC := VC) (cs := cs) A x secParam) t := by
  classical
  obtain ⟨α, h_pos⟩ :=
    exists_successMassGivenCommitAndChallenge_pos_of_heavyCommitment (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε) h_ε_pos h_mem
  obtain ⟨t, h_success, h_comm, h_alpha, h_run_pos⟩ :=
    exists_success_transcript_of_successMassGivenCommitAndChallenge_pos (VC := VC)
      (cs := cs) (A := A) (x := x) (secParam := secParam)
      (comm_tuple := comm_tuple) (α := α) h_pos
  exact ⟨α, t, h_success, h_comm, h_alpha, h_run_pos⟩

/-!
#### Successful challenge bookkeeping

The `successfulChallenges` helper captures the subset of challenges that
contribute positive conditional success mass for a fixed commitment tuple.
-/
section SuccessfulChallenges

/-- Set of challenges that yield positive conditional success mass for a fixed commitment. -/
noncomputable def successfulChallenges
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    Finset F :=
  (Finset.univ : Finset F).filter
    (fun α => 0 < successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α)

lemma successfulChallenges_mem
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    {α : F}
    (h_pos : 0 < successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α) :
    α ∈ successfulChallenges VC cs A x secParam comm_tuple := by
  classical
  simp [successfulChallenges, h_pos]

lemma successfulChallenges_pos_of_mem
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    {α : F}
    (h_mem : α ∈ successfulChallenges VC cs A x secParam comm_tuple) :
    0 < successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
  classical
  obtain ⟨_, h_pos⟩ := Finset.mem_filter.1 h_mem
  exact h_pos

lemma heavyCommitChallengeWeight_sum_successfulChallenges
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (ε : ℝ) (rand : Fin secParam.succ) :
    ∑ α ∈ successfulChallenges VC cs A x secParam
        (commitTupleOfRandomness VC cs A x secParam rand),
          heavyCommitChallengeWeight VC cs A x secParam ε rand α
      = heavyCommitSeedWeight VC cs A x secParam ε rand *
          ((successfulChallenges VC cs A x secParam
              (commitTupleOfRandomness VC cs A x secParam rand)).card : ENNReal) *
          (Fintype.card F : ENNReal)⁻¹ := by
  classical
  simpa [mul_comm, mul_left_comm, mul_assoc]
    using
      (heavyCommitChallengeWeight_sum_finset
        (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
        (ε := ε) (rand := rand)
        (S := successfulChallenges VC cs A x secParam
          (commitTupleOfRandomness VC cs A x secParam rand)))

lemma successProbability_le_heavyCommitMass_add
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (hε : 0 ≤ ε) :
    successProbability VC cs A x secParam ≤
      (heavyCommitMass VC cs A x secParam ε).toReal + ε := by
  classical
  have h_bound := successMass_le_heavyMass_add_eps
    (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam) (ε := ε)
  have h_succ_ne_top :
      successMass VC cs A x secParam ≠ (⊤ : ENNReal) := by
    have h_le_one := successMass_le_one (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam)
    exact (lt_top_iff_ne_top).1
      (lt_of_le_of_lt h_le_one ENNReal.one_lt_top)
  have h_heavy_ne_top :
      heavyCommitMass VC cs A x secParam ε ≠ (⊤ : ENNReal) := by
    have h_le_one := heavyCommitMass_le_one (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)
    exact (lt_top_iff_ne_top).1
      (lt_of_le_of_lt h_le_one ENNReal.one_lt_top)
  have h_sum_ne_top :
      heavyCommitMass VC cs A x secParam ε + ENNReal.ofReal ε ≠ (⊤ : ENNReal) := by
    refine ENNReal.add_ne_top.mpr ⟨h_heavy_ne_top, ENNReal.ofReal_ne_top⟩
  have h_toReal :=
    (ENNReal.toReal_le_toReal h_succ_ne_top h_sum_ne_top).mpr h_bound
  have h_toReal_add_raw :=
    ENNReal.toReal_add (a := heavyCommitMass VC cs A x secParam ε)
      (b := ENNReal.ofReal ε) h_heavy_ne_top ENNReal.ofReal_ne_top
  have h_toReal_add :
      (heavyCommitMass VC cs A x secParam ε + ENNReal.ofReal ε).toReal =
        (heavyCommitMass VC cs A x secParam ε).toReal + ε := by
    simpa [ENNReal.toReal_ofReal, hε] using h_toReal_add_raw
  have h_prob_eq := successProbability_toReal_successMass (VC := VC) (cs := cs)
    (A := A) (x := x) (secParam := secParam)
  simpa [h_prob_eq, h_toReal_add, ENNReal.toReal_ofReal]
    using h_toReal

lemma heavyCommitMass_toReal_ge_successProbability_sub
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (hε : 0 ≤ ε) :
    successProbability VC cs A x secParam - ε ≤
      (heavyCommitMass VC cs A x secParam ε).toReal := by
  classical
  have h_upper := successProbability_le_heavyCommitMass_add
    (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
    (ε := ε) (hε := hε)
  exact (sub_le_iff_le_add).2 h_upper

lemma exists_successfulChallenge_of_heavyCommitment
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
    (h_ε_pos : 0 < ε)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm_tuple ∈ heavyCommitments VC cs A x secParam ε) :
    ∃ α ∈ successfulChallenges VC cs A x secParam comm_tuple,
      0 < successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
  classical
  obtain ⟨α, h_pos⟩ :=
    exists_successMassGivenCommitAndChallenge_pos_of_heavyCommitment (VC := VC)
      (cs := cs) (A := A) (x := x) (secParam := secParam) (ε := ε)
      h_ε_pos h_mem
  refine ⟨α, ?_, h_pos⟩
  simpa [successfulChallenges] using h_pos

lemma exists_transcript_of_successfulChallenge
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    {α : F}
    (h_mem : α ∈ successfulChallenges VC cs A x secParam comm_tuple) :
    ∃ t : Transcript F VC,
      success_event VC cs x t ∧
      transcriptCommitTuple VC t = comm_tuple ∧
      t.view.alpha = α ∧
      0 < (run_adversary_transcript (VC := VC) (cs := cs) A x secParam) t := by
  classical
  have h_pos :=
    successfulChallenges_pos_of_mem (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (comm_tuple := comm_tuple) h_mem
  obtain ⟨t, h_success, h_comm, h_alpha, h_run_pos⟩ :=
    exists_success_transcript_of_successMassGivenCommitAndChallenge_pos (VC := VC)
      (cs := cs) (A := A) (x := x) (secParam := secParam)
      (comm_tuple := comm_tuple) (α := α) h_pos
  exact ⟨t, h_success, h_comm, h_alpha, h_run_pos⟩

lemma successfulChallenges_nonempty_of_heavyCommitment
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
    (h_ε_pos : 0 < ε)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm_tuple ∈ heavyCommitments VC cs A x secParam ε) :
    (successfulChallenges VC cs A x secParam comm_tuple).Nonempty := by
  classical
  obtain ⟨α, hα_mem, -⟩ :=
    exists_successfulChallenge_of_heavyCommitment (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_ε_pos h_mem
  exact ⟨α, hα_mem⟩

end SuccessfulChallenges

lemma commitChallengeMass_le_commitMass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)
    (α : F) :
    commitChallengeMass VC cs A x secParam comm_tuple α ≤
      commitMass VC cs A x secParam comm_tuple := by
  classical
  unfold commitChallengeMass commitMass
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_pointwise :
      (fun t =>
          if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0)
        ≤ fun t => if transcriptCommitTuple VC t = comm_tuple then p t else 0 := by
    intro t
    by_cases hComm : transcriptCommitTuple VC t = comm_tuple
    · by_cases hAlpha : t.view.alpha = α
      · simp [hComm, hAlpha]
      · simp [hComm, hAlpha]
    · simp [hComm]
  have h_le := ENNReal.tsum_le_tsum h_pointwise
  simpa [hp]

lemma successMassGivenCommitAndChallenge_eq_zero_of_not_mem_successfulChallenges
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)
    (α : F)
    (h_not : α ∉ successfulChallenges VC cs A x secParam comm_tuple) :
    successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α = 0 := by
  classical
  have h_not_pos : ¬ 0 <
      successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
    simpa [successfulChallenges] using h_not
  have h_le_zero := not_lt.mp h_not_pos
  exact le_antisymm h_le_zero bot_le

lemma successMassGivenCommit_eq_sum_successfulChallenges
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    successMassGivenCommit VC cs A x secParam comm_tuple =
      ∑ α ∈ successfulChallenges VC cs A x secParam comm_tuple,
        successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
  classical
  have h_branch :
      ∀ α,
        (if 0 <
            successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α
          then
            successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α
          else 0)
          = successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
    intro α
    by_cases h_pos :
        0 < successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α
    · simp [h_pos]
    · have h_zero :
        successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α = 0 := by
        refine le_antisymm (not_lt.mp h_pos) bot_le
      simp [h_zero]
  have h_filtered :
      ∑ α ∈ successfulChallenges VC cs A x secParam comm_tuple,
          successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α
        =
          ∑ α ∈ (Finset.univ : Finset F),
            (if
                0 < successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α
              then
                successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α
              else 0) := by
    simpa [successfulChallenges] using
      Finset.sum_filter (s := (Finset.univ : Finset F))
        (f := fun α =>
          successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α)
        (p :=
          fun α =>
            0 < successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α)
  have h_filtered' :
      ∑ α ∈ successfulChallenges VC cs A x secParam comm_tuple,
          successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α
        =
          ∑ α ∈ (Finset.univ : Finset F),
            successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α := by
    simpa [h_branch] using h_filtered
  have h_total :=
    successMassGivenCommit_eq_sum_successMassGivenCommitChallenge (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (comm_tuple := comm_tuple)
  exact h_total.trans h_filtered'.symm

lemma successMassGivenCommitAndChallenge_le_commitChallengeMass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) (α : F) :
    successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α ≤
      commitChallengeMass VC cs A x secParam comm_tuple α := by
  classical
  unfold successMassGivenCommitAndChallenge commitChallengeMass
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_pointwise :
      (fun t =>
          if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α
            then p t else 0)
        ≤ fun t =>
            if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0 := by
    intro t
    by_cases hSucc : success_event VC cs x t
    · by_cases hComm : transcriptCommitTuple VC t = comm_tuple
      · by_cases hAlpha : t.view.alpha = α
        · simp [hSucc, hComm, hAlpha]
        · simp [hSucc, hComm, hAlpha]
      · simp [hSucc, hComm]
    · simp [hSucc]
  have h_le := ENNReal.tsum_le_tsum h_pointwise
  simpa [hp]

lemma successMassGivenCommitAndChallenge_le_commitMass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)
    (α : F) :
    successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α ≤
      commitMass VC cs A x secParam comm_tuple :=
  (successMassGivenCommitAndChallenge_le_commitChallengeMass (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (comm_tuple := comm_tuple) (α := α)).trans
    (commitChallengeMass_le_commitMass (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (comm_tuple := comm_tuple) (α := α))

lemma successMassGivenCommit_le_card_successfulChallenges_smul_commitMass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    successMassGivenCommit VC cs A x secParam comm_tuple ≤
      (successfulChallenges VC cs A x secParam comm_tuple).card •
        commitMass VC cs A x secParam comm_tuple := by
  classical
  have h_sum :=
    successMassGivenCommit_eq_sum_successfulChallenges (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (comm_tuple := comm_tuple)
  have h_le :
      ∑ α ∈ successfulChallenges VC cs A x secParam comm_tuple,
          successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α
        ≤
          ∑ α ∈ successfulChallenges VC cs A x secParam comm_tuple,
            commitMass VC cs A x secParam comm_tuple :=
    Finset.sum_le_sum fun α hα =>
      successMassGivenCommitAndChallenge_le_commitMass (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (comm_tuple := comm_tuple) (α := α)
  have h_const :
      ∑ α ∈ successfulChallenges VC cs A x secParam comm_tuple,
          commitMass VC cs A x secParam comm_tuple
        =
          (successfulChallenges VC cs A x secParam comm_tuple).card •
            commitMass VC cs A x secParam comm_tuple := by
    simp [Finset.sum_const]
  have h_bound := h_sum.trans_le (h_le.trans_eq h_const)
  simpa using h_bound

lemma successfulChallenges_card_ENNReal_ge_of_heavyCommitment
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm_tuple ∈ heavyCommitments VC cs A x secParam ε) :
    ENNReal.ofReal ε ≤
      ((successfulChallenges VC cs A x secParam comm_tuple).card : ENNReal) := by
  classical
  set mass := commitMass VC cs A x secParam comm_tuple with h_mass_def
  have h_heavy :=
    (heavyCommitments_mem_iff (VC := VC) (cs := cs) (A := A) (x := x)
        (secParam := secParam) (ε := ε) (comm_tuple := comm_tuple)).1 h_mem
  have h_mass_pos : 0 < mass := by simpa [h_mass_def] using h_heavy.1
  have h_mass_ne_zero : mass ≠ 0 := ne_of_gt h_mass_pos
  have h_mass_le_one : mass ≤ (1 : ENNReal) :=
    by simpa [h_mass_def] using
      commitMass_le_one (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
        comm_tuple
  have h_mass_ne_top : mass ≠ (⊤ : ENNReal) := by
    have : mass < (⊤ : ENNReal) := lt_of_le_of_lt h_mass_le_one ENNReal.one_lt_top
    exact (lt_top_iff_ne_top).1 this
  have h_upper :=
    successMassGivenCommit_le_card_successfulChallenges_smul_commitMass (VC := VC)
      (cs := cs) (A := A) (x := x) (secParam := secParam) (comm_tuple := comm_tuple)
  have h_lower := h_heavy.2
  have h_main : ENNReal.ofReal ε * mass ≤
      ((successfulChallenges VC cs A x secParam comm_tuple).card : ENNReal) * mass := by
    have := le_trans h_lower h_upper
    simpa [h_mass_def, nsmul_eq_mul, mul_comm, mul_left_comm, mul_assoc] using this
  have h_mul :=
    mul_le_mul' h_main (le_of_eq (rfl : mass⁻¹ = mass⁻¹))
  have h_cancel : mass * mass⁻¹ = (1 : ENNReal) :=
    ENNReal.mul_inv_cancel h_mass_ne_zero h_mass_ne_top
  have h_left : (ENNReal.ofReal ε * mass) * mass⁻¹ = ENNReal.ofReal ε := by
    simp [mul_assoc, h_cancel]
  have h_right :
      (((successfulChallenges VC cs A x secParam comm_tuple).card : ENNReal) * mass)
          * mass⁻¹
        = ((successfulChallenges VC cs A x secParam comm_tuple).card : ENNReal) := by
    calc
      (((successfulChallenges VC cs A x secParam comm_tuple).card : ENNReal) * mass) * mass⁻¹
          = ((successfulChallenges VC cs A x secParam comm_tuple).card : ENNReal)
              * (mass * mass⁻¹) := by
                simp [mul_comm, mul_left_comm, mul_assoc]
      _ = ((successfulChallenges VC cs A x secParam comm_tuple).card : ENNReal) * 1 := by
        simp [h_cancel]
      _ = ((successfulChallenges VC cs A x secParam comm_tuple).card : ENNReal) := by simp
  simpa [h_left, h_right] using h_mul

lemma heavyCommitChallengeWeight_sum_successfulChallenges_ge
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (ε : ℝ) (rand : Fin secParam.succ)
    (h_mem : commitTupleOfRandomness VC cs A x secParam rand ∈
      heavyCommitments VC cs A x secParam ε) :
    heavyCommitSeedWeight VC cs A x secParam ε rand *
        ENNReal.ofReal ε * (Fintype.card F : ENNReal)⁻¹ ≤
      ∑ α ∈ successfulChallenges VC cs A x secParam
          (commitTupleOfRandomness VC cs A x secParam rand),
        heavyCommitChallengeWeight VC cs A x secParam ε rand α := by
  classical
  have h_card :=
    successfulChallenges_card_ENNReal_ge_of_heavyCommitment (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε)
      (comm_tuple := commitTupleOfRandomness VC cs A x secParam rand) h_mem
  have h_sum :=
    heavyCommitChallengeWeight_sum_successfulChallenges (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε) (rand := rand)
  have h_mul_left :
      heavyCommitSeedWeight VC cs A x secParam ε rand * ENNReal.ofReal ε ≤
        heavyCommitSeedWeight VC cs A x secParam ε rand *
          ((successfulChallenges VC cs A x secParam
            (commitTupleOfRandomness VC cs A x secParam rand)).card : ENNReal) := by
    exact (mul_le_mul_left' h_card _)
  have h_mul_right :=
    mul_le_mul_right' h_mul_left ((Fintype.card F : ENNReal)⁻¹)
  simpa [h_sum, mul_comm, mul_left_comm, mul_assoc] using h_mul_right

lemma heavyCommitChallengeWeight_sum_heavyRandomness_ge
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (ε : ℝ)
    (S : Finset (Fin secParam.succ))
    (h_subset : S ⊆ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε) :
    ∑ rand ∈ S,
        ∑ α ∈ successfulChallenges VC cs A x secParam
            (commitTupleOfRandomness VC cs A x secParam rand),
          heavyCommitChallengeWeight VC cs A x secParam ε rand α
      ≥ ENNReal.ofReal ε * (Fintype.card F : ENNReal)⁻¹ *
          ∑ rand ∈ S, heavyCommitSeedWeight VC cs A x secParam ε rand := by
  classical
  set c := ENNReal.ofReal ε * (Fintype.card F : ENNReal)⁻¹
  have h_aux :
      ∀ rand ∈ S,
        heavyCommitSeedWeight VC cs A x secParam ε rand * c ≤
          ∑ α ∈ successfulChallenges VC cs A x secParam
              (commitTupleOfRandomness VC cs A x secParam rand),
            heavyCommitChallengeWeight VC cs A x secParam ε rand α := by
    intro rand h_rand
    have h_mem : commitTupleOfRandomness VC cs A x secParam rand ∈
        heavyCommitments VC cs A x secParam ε := by
      have h_rand_heavy : rand ∈ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) ε := h_subset h_rand
      exact (mem_heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) (ε := ε) (rand := rand)).1 h_rand_heavy
    have := heavyCommitChallengeWeight_sum_successfulChallenges_ge
      (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
      (ε := ε) (rand := rand) h_mem
    simpa [c, mul_comm, mul_left_comm, mul_assoc]
      using this
  have h_sum_le :=
    Finset.sum_le_sum fun rand h_rand => h_aux rand h_rand
  have h_left_eq :
      ∑ rand ∈ S, heavyCommitSeedWeight VC cs A x secParam ε rand * c =
        c * ∑ rand ∈ S, heavyCommitSeedWeight VC cs A x secParam ε rand := by
    have h_mul_sum :=
      (Finset.mul_sum (s := S)
          (f := fun rand => heavyCommitSeedWeight VC cs A x secParam ε rand) c)
    simpa [mul_comm, mul_left_comm, mul_assoc] using h_mul_sum.symm
  have h_goal :
      c * ∑ rand ∈ S, heavyCommitSeedWeight VC cs A x secParam ε rand ≤
        ∑ rand ∈ S,
            ∑ α ∈ successfulChallenges VC cs A x secParam
                (commitTupleOfRandomness VC cs A x secParam rand),
              heavyCommitChallengeWeight VC cs A x secParam ε rand α := by
    simpa [c, h_left_eq] using h_sum_le
  simpa [c, mul_comm, mul_left_comm, mul_assoc] using h_goal

lemma heavyCommitSeedWeight_sum_heavyRandomnessFinset
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ) :
    ∑ rand ∈ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε,
        heavyCommitSeedWeight VC cs A x secParam ε rand
      = ∑ rand : Fin secParam.succ,
          heavyCommitSeedWeight VC cs A x secParam ε rand := by
  classical
  set S := heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε
  have h_subset : S ⊆ (Finset.univ : Finset (Fin secParam.succ)) := by
    intro rand _; exact Finset.mem_univ _
  have h_zero :
      ∀ rand ∈ (Finset.univ : Finset (Fin secParam.succ)), rand ∉ S →
        heavyCommitSeedWeight VC cs A x secParam ε rand = 0 := by
    intro rand _ h_not
    have h_not_mem : rand ∉
        heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) ε := by
      simpa [S] using h_not
    have h_not_heavy :
        commitTupleOfRandomness VC cs A x secParam rand ∉
          heavyCommitments VC cs A x secParam ε := by
      have h_iff :=
        mem_heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) (ε := ε) (rand := rand)
      exact (not_congr h_iff).1 h_not_mem
    simp [heavyCommitSeedWeight, h_not_heavy]
  have h_sum :=
    Finset.sum_subset h_subset h_zero
  have h_sum' :
      ∑ rand : Fin secParam.succ, heavyCommitSeedWeight VC cs A x secParam ε rand =
        ∑ rand ∈ S, heavyCommitSeedWeight VC cs A x secParam ε rand := by
    simpa [S] using h_sum.symm
  simpa [S] using h_sum'.symm

lemma heavyCommitChallengeWeight_sum_heavyRandomness_ge_mass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (ε : ℝ) :
    ∑ rand ∈ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε,
        ∑ α ∈ successfulChallenges VC cs A x secParam
            (commitTupleOfRandomness VC cs A x secParam rand),
          heavyCommitChallengeWeight VC cs A x secParam ε rand α
      ≥ ENNReal.ofReal ε * (Fintype.card F : ENNReal)⁻¹ *
          heavyCommitMass VC cs A x secParam ε := by
  classical
  set S := heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε
  have h_subset : S ⊆ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε := by intro rand h_rand; simpa [S] using h_rand
  have h_lower :=
    heavyCommitChallengeWeight_sum_heavyRandomness_ge (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε)
      (S := S) h_subset
  have h_seed_sum :=
    heavyCommitSeedWeight_sum_heavyRandomnessFinset (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε)
  have h_mass :=
    heavyCommitMass_eq_uniform_randomness_sum (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)
  have := h_lower
  simpa [S, h_seed_sum, h_mass] using this

lemma heavyCommitChallengeWeight_sum_successfulChallenges_le_seedWeight
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (ε : ℝ) (rand : Fin secParam.succ) :
    ∑ α ∈ successfulChallenges VC cs A x secParam
        (commitTupleOfRandomness VC cs A x secParam rand),
        heavyCommitChallengeWeight VC cs A x secParam ε rand α ≤
      heavyCommitSeedWeight VC cs A x secParam ε rand := by
  classical
  have h_card_le_nat :
      (successfulChallenges VC cs A x secParam
          (commitTupleOfRandomness VC cs A x secParam rand)).card ≤
        Fintype.card F := by
    simpa [successfulChallenges]
      using Finset.card_filter_le (Finset.univ : Finset F)
        (fun α =>
          0 < successMassGivenCommitAndChallenge VC cs A x secParam
              (commitTupleOfRandomness VC cs A x secParam rand) α)
  have h_card_le :
      ((successfulChallenges VC cs A x secParam
          (commitTupleOfRandomness VC cs A x secParam rand)).card : ENNReal)
        ≤ (Fintype.card F : ENNReal) := by
    exact_mod_cast h_card_le_nat
  have h_sum :=
    heavyCommitChallengeWeight_sum_successfulChallenges (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε) (rand := rand)
  have h_ratio_le_one :
      ((successfulChallenges VC cs A x secParam
          (commitTupleOfRandomness VC cs A x secParam rand)).card : ENNReal) *
          (Fintype.card F : ENNReal)⁻¹ ≤ 1 := by
    have h_c_nonneg : 0 ≤ (Fintype.card F : ENNReal)⁻¹ := by simp
    have h_mul_le :=
      mul_le_mul_of_nonneg_right h_card_le h_c_nonneg
    have h_le_one :
        ((successfulChallenges VC cs A x secParam
            (commitTupleOfRandomness VC cs A x secParam rand)).card : ENNReal) *
            (Fintype.card F : ENNReal)⁻¹
          ≤ (Fintype.card F : ENNReal) * (Fintype.card F : ENNReal)⁻¹ := by
      simpa [mul_comm, mul_left_comm, mul_assoc] using h_mul_le
    have h_card_pos_nat : 0 < Fintype.card F := Fintype.card_pos
    have h_card_pos : (Fintype.card F : ENNReal) ≠ 0 := by
      exact_mod_cast (ne_of_gt h_card_pos_nat : Fintype.card F ≠ 0)
    calc
      ((successfulChallenges VC cs A x secParam
          (commitTupleOfRandomness VC cs A x secParam rand)).card : ENNReal) *
          (Fintype.card F : ENNReal)⁻¹
          ≤ (Fintype.card F : ENNReal) * (Fintype.card F : ENNReal)⁻¹ := h_le_one
      _ = 1 := by
        simp [ENNReal.mul_inv_cancel h_card_pos]
  have h_seed_nonneg : 0 ≤ heavyCommitSeedWeight VC cs A x secParam ε rand := by
    exact bot_le
  have h_mul_le :=
    mul_le_mul_of_nonneg_left h_ratio_le_one h_seed_nonneg
  have h_le :
      heavyCommitSeedWeight VC cs A x secParam ε rand *
          (((successfulChallenges VC cs A x secParam
              (commitTupleOfRandomness VC cs A x secParam rand)).card : ENNReal) *
            (Fintype.card F : ENNReal)⁻¹)
        ≤ heavyCommitSeedWeight VC cs A x secParam ε rand := by
    calc
      heavyCommitSeedWeight VC cs A x secParam ε rand *
          (((successfulChallenges VC cs A x secParam
              (commitTupleOfRandomness VC cs A x secParam rand)).card : ENNReal) *
            (Fintype.card F : ENNReal)⁻¹)
          ≤ heavyCommitSeedWeight VC cs A x secParam ε rand * 1 := by
            simpa [mul_comm, mul_left_comm, mul_assoc] using h_mul_le
      _ = heavyCommitSeedWeight VC cs A x secParam ε rand := by simp
  simpa [h_sum, mul_comm, mul_left_comm, mul_assoc] using h_le

lemma heavyCommitChallengeWeight_sum_heavyRandomness_le_mass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (ε : ℝ) :
    ∑ rand ∈ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε,
        ∑ α ∈ successfulChallenges VC cs A x secParam
            (commitTupleOfRandomness VC cs A x secParam rand),
          heavyCommitChallengeWeight VC cs A x secParam ε rand α ≤
      heavyCommitMass VC cs A x secParam ε := by
  classical
  set S := heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε
  have h_inner_le :
      ∀ rand ∈ S,
        ∑ α ∈ successfulChallenges VC cs A x secParam
            (commitTupleOfRandomness VC cs A x secParam rand),
          heavyCommitChallengeWeight VC cs A x secParam ε rand α ≤
          heavyCommitSeedWeight VC cs A x secParam ε rand := by
    intro rand _
    simpa using
      heavyCommitChallengeWeight_sum_successfulChallenges_le_seedWeight
        (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
        (ε := ε) (rand := rand)
  have h_sum_le :
      ∑ rand ∈ S,
          ∑ α ∈ successfulChallenges VC cs A x secParam
              (commitTupleOfRandomness VC cs A x secParam rand),
            heavyCommitChallengeWeight VC cs A x secParam ε rand α ≤
        ∑ rand ∈ S,
          heavyCommitSeedWeight VC cs A x secParam ε rand :=
    Finset.sum_le_sum h_inner_le
  have h_seed_sum :=
    heavyCommitSeedWeight_sum_heavyRandomnessFinset (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε)
  have h_mass :=
    heavyCommitMass_eq_uniform_randomness_sum (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)
  have h_rhs :
      ∑ rand ∈ S, heavyCommitSeedWeight VC cs A x secParam ε rand =
        heavyCommitMass VC cs A x secParam ε := by
    calc
      ∑ rand ∈ S, heavyCommitSeedWeight VC cs A x secParam ε rand =
          ∑ rand : Fin secParam.succ,
            heavyCommitSeedWeight VC cs A x secParam ε rand := by
              simpa [S] using h_seed_sum
      _ = heavyCommitMass VC cs A x secParam ε := by
          simpa using h_mass.symm
  have := h_sum_le.trans_eq h_rhs
  simpa [S] using this


lemma successfulChallenges_card_ge_of_heavyCommitment
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
    (h_ε_pos : 0 < ε)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm_tuple ∈ heavyCommitments VC cs A x secParam ε) :
    ε ≤ ((successfulChallenges VC cs A x secParam comm_tuple).card : ℝ) := by
  classical
  have h_card :=
    successfulChallenges_card_ENNReal_ge_of_heavyCommitment (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε) h_mem
  have h_card' :
      ENNReal.ofReal ε ≤ ENNReal.ofReal
        ((successfulChallenges VC cs A x secParam comm_tuple).card : ℝ) := by
    simpa using h_card
  have h_nonneg : 0 ≤ ((successfulChallenges VC cs A x secParam comm_tuple).card : ℝ) := by
    exact_mod_cast (successfulChallenges VC cs A x secParam comm_tuple).card.zero_le
  have h_le := ENNReal.toReal_le_of_le_ofReal h_nonneg h_card'
  have h_toReal : (ENNReal.ofReal ε).toReal = ε := by simp [h_ε_pos.le]
  simpa [h_toReal] using h_le

lemma heavyCommitment_witnesses_successfulChallenges
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
    (h_ε_pos : 0 < ε)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm_tuple ∈ heavyCommitments VC cs A x secParam ε) :
    ∃ valid_challenges : Finset F,
      (∀ α ∈ valid_challenges,
        ∃ t : Transcript F VC,
          success_event VC cs x t ∧
          transcriptCommitTuple VC t = comm_tuple ∧
          t.view.alpha = α ∧
          0 < (run_adversary_transcript (VC := VC) (cs := cs) A x secParam) t) ∧
      (valid_challenges.card : ℝ) ≥ ε := by
  classical
  refine ⟨successfulChallenges VC cs A x secParam comm_tuple, ?_, ?_⟩
  · intro α hα
    obtain ⟨t, h_success, h_comm, h_alpha, h_run_pos⟩ :=
      exists_transcript_of_successfulChallenge (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (comm_tuple := comm_tuple) (α := α) hα
    exact ⟨t, h_success, h_comm, h_alpha, h_run_pos⟩
  · exact successfulChallenges_card_ge_of_heavyCommitment (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε) h_ε_pos h_mem

lemma exists_heavyRandomness_successfulChallenge_of_successProbability_gt
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_pos : 0 < ε)
    (h_gap : ε < successProbability VC cs A x secParam) :
    ∃ (rand : Fin secParam.succ) (α : F) (t : Transcript F VC),
      commitTupleOfRandomness VC cs A x secParam rand ∈
        heavyCommitments VC cs A x secParam ε ∧
      success_event VC cs x t ∧
      transcriptCommitTuple VC t =
        commitTupleOfRandomness VC cs A x secParam rand ∧
      t.view.alpha = α ∧
      0 < (run_adversary_transcript (VC := VC) (cs := cs) A x secParam) t := by
  classical
  have h_nonneg : 0 ≤ ε := le_of_lt h_pos
  obtain ⟨rand, h_heavy_rand⟩ :=
    exists_heavyRandomness_of_successProbability_gt (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_nonneg h_gap
  have h_witness :=
    heavyCommitment_witnesses_successfulChallenges (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_pos h_heavy_rand
  rcases h_witness with ⟨valid_challenges, h_valid, h_card⟩
  have h_card_pos_real :
      0 < (valid_challenges.card : ℝ) := lt_of_lt_of_le h_pos h_card
  have h_card_pos : 0 < valid_challenges.card := by
    exact_mod_cast h_card_pos_real
  obtain ⟨α, hα_mem⟩ := Finset.card_pos.mp h_card_pos
  obtain ⟨t, h_success, h_comm, h_alpha, h_run_pos⟩ := h_valid α hα_mem
  refine ⟨rand, α, t, h_heavy_rand, h_success, ?_, h_alpha, h_run_pos⟩
  simpa using h_comm

lemma heavyCommitment_scaled_witnesses
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
    (h_ε_pos : 0 < ε)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm_tuple ∈
      heavyCommitments VC cs A x secParam (ε * (Fintype.card F : ℝ))) :
    ∃ valid_challenges : Finset F,
      (∀ α ∈ valid_challenges,
        ∃ t : Transcript F VC,
          success_event VC cs x t ∧
          transcriptCommitTuple VC t = comm_tuple ∧
          t.view.alpha = α) ∧
      (valid_challenges.card : ℝ) ≥ ε * (Fintype.card F : ℝ) := by
  classical
  have h_card_nat : 0 < Fintype.card F := Fintype.card_pos
  have h_card_pos : 0 < (Fintype.card F : ℝ) := by exact_mod_cast h_card_nat
  have h_scaled_pos : 0 < ε * (Fintype.card F : ℝ) :=
    mul_pos h_ε_pos h_card_pos
  obtain ⟨valid_challenges, h_witness, h_card_ge⟩ :=
    heavyCommitment_witnesses_successfulChallenges (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε * (Fintype.card F : ℝ))
      h_scaled_pos h_mem
  refine ⟨valid_challenges, ?_, ?_⟩
  · intro α hα
    obtain ⟨t, h_success, h_comm, h_alpha, _⟩ := h_witness α hα
    exact ⟨t, h_success, h_comm, h_alpha⟩
  · simpa [mul_comm, mul_left_comm, mul_assoc] using h_card_ge

lemma successMassGivenCommitAndChallenge_le_run_adversary_commit_challenge
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) (α : F) :
    successMassGivenCommitAndChallenge VC cs A x secParam comm_tuple α ≤
      run_adversary_commit_challenge VC cs A x secParam (comm_tuple, α) := by
  have h :=
    successMassGivenCommitAndChallenge_le_commitChallengeMass (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (comm_tuple := comm_tuple) (α := α)
  simpa [commitChallengeMass_eq_run_adversary_commit_challenge (VC := VC) (cs := cs)
    (A := A) (x := x) (secParam := secParam) (comm_tuple := comm_tuple) (α := α)] using h

lemma tsum_commitChallengeMass_eq_one {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
  ∑' cc : (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F,
    commitChallengeMass VC cs A x secParam cc.1 cc.2 = 1 := by
  classical
  have h_tsum := (run_adversary_commit_challenge VC cs A x secParam).tsum_coe
  have h_congr :
      ∑' cc : (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F,
          commitChallengeMass VC cs A x secParam cc.1 cc.2
            = ∑' cc,
                run_adversary_commit_challenge VC cs A x secParam cc := by
    refine tsum_congr ?_
    intro cc
    rcases cc with ⟨comm_tuple, α⟩
    simp [commitChallengeMass_eq_run_adversary_commit_challenge (VC := VC) (cs := cs)
        (A := A) (x := x) (secParam := secParam) (comm_tuple := comm_tuple) (α := α)]
  calc
    ∑'
        cc : (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F,
          commitChallengeMass VC cs A x secParam cc.1 cc.2
        = ∑' cc,
            run_adversary_commit_challenge VC cs A x secParam cc := h_congr
    _ = 1 := h_tsum

lemma commitMass_eq_tsum_commitChallengeMass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    commitMass VC cs A x secParam comm_tuple =
      ∑' α, commitChallengeMass VC cs A x secParam comm_tuple α := by
  classical
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_pointwise :
      ∀ t : Transcript F VC,
        (if transcriptCommitTuple VC t = comm_tuple then p t else 0)
          = ∑' α,
              (if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0) := by
    intro t
    by_cases hComm : transcriptCommitTuple VC t = comm_tuple
    · have h_single :
        ∑' α, (if t.view.alpha = α then p t else 0) = p t := by
        refine (tsum_eq_single (t.view.alpha)
            (fun α hne => ?_)).trans ?_
        · have hneq : t.view.alpha ≠ α := by
            simpa [eq_comm] using hne
          simp [hneq]
        · simp
      have h_congr :
          ∑' α,
              (if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0)
            = ∑' α, (if t.view.alpha = α then p t else 0) := by
        refine tsum_congr ?_
        intro α
        by_cases hα : t.view.alpha = α
        · simp [hComm, hα]
        · simp [hComm, hα]
      have h_sum := h_congr.trans h_single
      simpa [hComm] using h_sum.symm
    · simp [hComm]
  have h_split :
      ∑' t, (if transcriptCommitTuple VC t = comm_tuple then p t else 0)
        = ∑' t,
            ∑' α,
              (if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0) := by
    refine tsum_congr ?_
    intro t
    exact h_pointwise t
  have h_commute :=
    ENNReal.tsum_comm
      (f := fun α (t : Transcript F VC) =>
        if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0)
  have h_swapped :
      ∑' t,
          ∑' α,
            (if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0)
        = ∑' α,
            ∑' t,
              (if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0) := by
    simpa using h_commute.symm
  have h_left : commitMass VC cs A x secParam comm_tuple
      = ∑' t, (if transcriptCommitTuple VC t = comm_tuple then p t else 0) := by
    simp [commitMass, hp]
  have h_right :
      ∑' α, commitChallengeMass VC cs A x secParam comm_tuple α
        = ∑' α,
            ∑' t,
              (if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0) := by
    refine tsum_congr ?_
    intro α
    simp [commitChallengeMass, hp]
  exact h_left.trans (h_split.trans (h_swapped.trans h_right.symm))

lemma commitMass_eq_sum_commitChallengeMass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    commitMass VC cs A x secParam comm_tuple =
      (∑ α ∈ (Finset.univ : Finset F),
        commitChallengeMass VC cs A x secParam comm_tuple α) := by
  classical
  have :=
    commitMass_eq_tsum_commitChallengeMass (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (comm_tuple := comm_tuple)
  simpa [tsum_fintype] using this

end HeavyCommitmentDecomposition

end LambdaSNARK
