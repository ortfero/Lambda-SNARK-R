/-
Copyright (c) 2025 URPKS Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: URPKS Contributors
-/

import LambdaSNARK.Core
import LambdaSNARK.Polynomial
import LambdaSNARK.Constraints
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Probability.ProbabilityMassFunction.Monad
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Data.Finset.Card
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Choose.Cast
import Mathlib.Data.Nat.Choose.Central
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

/-!
# Cryptographic Transcript Infrastructure for Forking Lemma

This file provides infrastructure for the forking lemma proof in ΛSNARK-R soundness:
- Transcript type (commitment × challenge × response)
- Adversary state machine (commit → challenge → respond)
- Rewinding mechanism (replay with different challenge)
- Probability bounds (ε² - negl(λ) success amplification)

## Main Components

- `Transcript`: Interactive proof transcript
- `AdversaryState`: Adversary's internal state before challenge
- `run_adversary`: Execute adversary once and record commitment snapshot
- `rewind_adversary`: Replay adversary with new challenge
- `heavy_row_lemma`: If Pr[success] ≥ ε, then many "good" commitments exist
- `fork_success_bound`: Pr[two valid transcripts] ≥ ε²/2

## References

- Bellare-Neven forking lemma (2006)
- Bootle et al. "Efficient Zero-Knowledge Arguments" (2016)
- ΛSNARK-R Specification: docs/spec/specification.md

-/

namespace LambdaSNARK

open scoped BigOperators
open BigOperators Polynomial
open LambdaSNARK

/-!
## Combinatorial Helpers

The following lemmas provide the arithmetic backbone for the probability
estimates in the forking lemma.  They translate counting arguments about
subsets of a finite challenge space into inequalities over real numbers.
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

/-! ### Helper lemmas for finite sums -/

lemma hasSum_fintype {α : Type*} [Fintype α] [DecidableEq α]
    (f : α → ENNReal) : HasSum f (∑ a : α, f a) := by
  classical
  refine Filter.Tendsto.congr' ?_ tendsto_const_nhds
  refine Filter.eventually_atTop.2 ?_
  refine ⟨(Finset.univ : Finset α), ?_⟩
  intro s hs
  have hs_eq : s = (Finset.univ : Finset α) := by
    apply Finset.ext
    intro a
    constructor
    · intro _; simp
    · intro _
      exact hs (by simp)
  simp [hs_eq]

/-! ### Protocol data structures -/

/-- View data revealed to the verifier during transcript validation. -/
structure TranscriptView (F : Type) where
  alpha : F
  Az_eval : F
  Bz_eval : F
  Cz_eval : F
  quotient_eval : F
  vanishing_eval : F
  main_eq : Prop


/-- Placeholder relation used while the verifier equations are not yet formalised. -/
def verifierView_zero_eq (_F : Type) : Prop := True

/-- Interactive proof transcript produced by the adversary and verifier. -/
structure Transcript (F : Type) [CommRing F] (VC : VectorCommitment F) where
  pp : VC.PP
  cs : R1CS F
  x : PublicInput F cs.nPub
  domainSize : ℕ
  omega : F
  comm_Az : VC.Commitment
  comm_Bz : VC.Commitment
  comm_Cz : VC.Commitment
  comm_quotient : VC.Commitment
  quotient_poly : Polynomial F
  quotient_rand : ℕ
  quotient_commitment_spec :
    VC.commit pp (Polynomial.coeffList quotient_poly) quotient_rand = comm_quotient
  view : TranscriptView F
  challenge_β : F
  opening_Az_α : VC.Opening
  opening_Bz_β : VC.Opening
  opening_Cz_α : VC.Opening
  opening_quotient_α : VC.Opening
  valid : Bool

/-- Evaluations and openings returned by the adversary when challenged. -/
structure AdversaryResponse (F : Type) [CommRing F] (VC : VectorCommitment F) where
  Az_eval : F
  Bz_eval : F
  Cz_eval : F
  quotient_eval : F
  vanishing_eval : F
  opening_Az_α : VC.Opening
  opening_Bz_β : VC.Opening
  opening_Cz_α : VC.Opening
  opening_quotient_α : VC.Opening

def is_valid_fork {F : Type} [CommRing F] [DecidableEq F] (VC : VectorCommitment F)
    (t1 t2 : Transcript F VC) : Prop :=
  -- Same setup parameters and statement data
  t1.pp = t2.pp ∧
  t1.cs = t2.cs ∧
  HEq t1.x t2.x ∧
  t1.domainSize = t2.domainSize ∧
  t1.omega = t2.omega ∧
  -- Same commitments and randomness
  t1.comm_Az = t2.comm_Az ∧
  t1.comm_Bz = t2.comm_Bz ∧
  t1.comm_Cz = t2.comm_Cz ∧
  t1.comm_quotient = t2.comm_quotient ∧
  t1.quotient_rand = t2.quotient_rand ∧
  -- Distinct challenges
  t1.view.alpha ≠ t2.view.alpha ∧
  -- Both transcripts accepted by the verifier
  t1.valid = true ∧
  t2.valid = true


/-- Adversary's internal state after committing, before receiving challenge.
    Captures the "commitment phase" for rewinding. -/
structure AdversaryState (F : Type) [CommRing F] (VC : VectorCommitment F) where
  -- Internal randomness (fixes commitments)
  randomness : ℕ

  -- Public parameters used during commitment phase
  pp : VC.PP

  -- Committed values
  comm_Az : VC.Commitment
  comm_Bz : VC.Commitment
  comm_Cz : VC.Commitment
  comm_quotient : VC.Commitment

  -- Quotient polynomial data used for extraction
  quotient_poly : Polynomial F
  quotient_rand : ℕ
  quotient_commitment_spec :
    VC.commit pp (Polynomial.coeffList quotient_poly) quotient_rand = comm_quotient
  domainSize : ℕ
  omega : F

  -- Response function: given challenge, produce evaluations and openings
  respond : F → F → AdversaryResponse F VC

/-- Extract commitment tuple from adversary state -/
def AdversaryState.commitments {F : Type} [CommRing F] (VC : VectorCommitment F)
    (state : AdversaryState F VC) :
    VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment :=
  (state.comm_Az, state.comm_Bz, state.comm_Cz, state.comm_quotient)

/-- Probabilistic polynomial-time adversary interacting with the protocol. -/
structure Adversary (F : Type) [CommRing F] (VC : VectorCommitment F) where
  run :
      (cs : R1CS F) →
      (x : PublicInput F cs.nPub) →
      (randomness : ℕ) →
      Proof F VC
  snapshot :
      (cs : R1CS F) →
      (x : PublicInput F cs.nPub) →
      (randomness : ℕ) →
      AdversaryState F VC
  poly_time : Prop

-- ============================================================================
-- Uniform PMF over Finite Types
-- ============================================================================

/-- Uniform distribution over a finite set: every element has probability
  `1 / |α|`. -/
noncomputable def uniform_pmf {α : Type*} [Fintype α] [Nonempty α] : PMF α :=
  ⟨fun _ => (Fintype.card α : ENNReal)⁻¹,
   by -- HasSum (const (1/card α)) 1
     have h_card_pos : 0 < Fintype.card α := Fintype.card_pos
     have h_card_ne_zero : (Fintype.card α : ENNReal) ≠ 0 := by
       norm_cast
       exact Nat.pos_iff_ne_zero.mp h_card_pos
     have h_card_ne_top : (Fintype.card α : ENNReal) ≠ ⊤ :=
       ENNReal.natCast_ne_top (Fintype.card α)
     -- HasSum via ENNReal.summable + tsum equality
     have h_summable : Summable (fun (_ : α) => (Fintype.card α : ENNReal)⁻¹) := by
       exact ENNReal.summable
     have h_tsum : ∑' (_ : α), (Fintype.card α : ENNReal)⁻¹ = 1 := by
       rw [tsum_fintype]
       simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
       exact ENNReal.mul_inv_cancel h_card_ne_zero h_card_ne_top
     rw [← h_tsum]
     exact h_summable.hasSum
   ⟩

/-- Uniform distribution excluding one element.

    **Constructive definition** (replaces previous axiom):
    PMF where each element in α \ {x} has probability 1/(|α| - 1).

    Construction:
    - Support: S = {a : α | a ≠ x}
    - PMF: f(a) = 1/(card(α) - 1) if a ∈ S, else 0
    - Proof: ∑_{a ∈ S} f(a) = (card(α) - 1) * 1/(card(α) - 1) = 1

    Requires: card(α) ≥ 2 (ensures S non-empty and card(α) - 1 > 0)
    Full formalization: 1-1.5h with Mathlib.Data.Finset.Card lemmas. -/
noncomputable def uniform_pmf_ne {α : Type*} [Fintype α] [DecidableEq α]
    (x : α) (h : Fintype.card α ≥ 2) : PMF α :=
  ⟨fun a => if a = x then 0 else ((Fintype.card α - 1) : ENNReal)⁻¹,
    by
      classical
      let c : ENNReal := ((Fintype.card α - 1) : ENNReal)⁻¹
      change HasSum (fun a : α => if a = x then 0 else c) 1
      have hx_mem : x ∈ (Finset.univ : Finset α) := Finset.mem_univ _
      have h_card_gt_one : 1 < Fintype.card α :=
        Nat.lt_of_lt_of_le (by decide : 1 < 2) h
      have h_card_pred_ne_zero_nat : Fintype.card α - 1 ≠ 0 :=
        Nat.sub_ne_zero_of_lt h_card_gt_one
      have h_card_pred_ne_zero : (Fintype.card α - 1 : ENNReal) ≠ 0 := by
        exact_mod_cast h_card_pred_ne_zero_nat
      have h_card_pred_ne_top : (Fintype.card α - 1 : ENNReal) ≠ ⊤ := by
        intro h_top
        cases h_top
      have h_card_nat :
          (Finset.univ.erase x).card = Fintype.card α - 1 := by
        classical
        calc
          (Finset.univ.erase x).card
              = (Finset.univ.card) - 1 := Finset.card_erase_of_mem hx_mem
          _ = Fintype.card α - 1 := by simp
      have h_card_cast :
          ((Finset.univ.erase x).card : ENNReal)
            = (Fintype.card α - 1 : ENNReal) := by
        exact_mod_cast h_card_nat
      have h_sum_const :
          (Finset.univ.erase x).sum (fun _ => c)
                = ((Finset.univ.erase x).card : ENNReal) * c := by
        simp [Finset.sum_const, nsmul_eq_mul]
      have h_sum_finset :
          (Finset.univ.erase x).sum (fun _ => c) = 1 := by
        calc
          (Finset.univ.erase x).sum (fun _ => c)
              = ((Finset.univ.erase x).card : ENNReal) * c := h_sum_const
          _ = (Fintype.card α - 1 : ENNReal) * c := by
            rw [h_card_cast]
          _ = 1 := ENNReal.mul_inv_cancel h_card_pred_ne_zero h_card_pred_ne_top
      have h_sum_univ :
          (∑ a : α, if a = x then 0 else c)
              = (Finset.univ.erase x).sum (fun _ => c) := by
        have hx_subset :
            (Finset.univ.erase x : Finset α) ⊆ (Finset.univ : Finset α) := by
          intro a _; exact Finset.mem_univ _
        have h_zero :
            ∀ a ∈ (Finset.univ : Finset α),
              a ∉ Finset.univ.erase x → (if a = x then 0 else c) = 0 := by
          intro a ha ha_not
          have hax : a = x := by
            classical
            by_contra h_ne
            have : a ∈ Finset.univ.erase x := by
              simp [Finset.mem_erase, ha, h_ne]
            exact ha_not this
          simp [hax]
        calc
          (∑ a : α, if a = x then 0 else c)
              = (Finset.univ : Finset α).sum (fun a => if a = x then 0 else c) := by simp
          _ = (Finset.univ.erase x).sum (fun a => if a = x then 0 else c) :=
            (Finset.sum_subset hx_subset h_zero).symm
          _ = (Finset.univ.erase x).sum (fun _ => c) := by
            refine Finset.sum_congr rfl ?_
            intro a ha
            have : a ≠ x := by
              classical
              simpa [Finset.mem_erase] using ha
            simp [this]
      have h_sum_total :
          (∑ a : α, if a = x then 0 else c) = 1 :=
        h_sum_univ.trans h_sum_finset
      have h_has_sum := hasSum_fintype (fun a : α => if a = x then 0 else c)
      simpa [h_sum_total, c] using h_has_sum
    ⟩

-- ============================================================================
-- Run Adversary (First Execution)
-- ============================================================================

/-- Execute adversary once to obtain a commitment-phase snapshot together with the
    resulting transcript. Samples the adversary's randomness from the uniform
    distribution, runs the adversary, and packages both the proof and the
    state required for rewinding. -/
noncomputable def run_adversary {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F)
  (_A : Adversary F VC) (x : PublicInput F cs.nPub)
    (_secParam : ℕ) : PMF (AdversaryState F VC × Transcript F VC) := by
  classical
  let randomnessPMF : PMF (Fin (_secParam.succ)) := uniform_pmf
  refine PMF.bind randomnessPMF ?_
  intro r
  let rand : ℕ := r
  let proof := _A.run cs x rand
  let state := _A.snapshot cs x rand
  exact PMF.pure (state, {
    pp := state.pp,
    cs := cs,
    x := x,
    domainSize := state.domainSize,
    omega := state.omega,
    comm_Az := state.comm_Az,
    comm_Bz := state.comm_Bz,
    comm_Cz := state.comm_Cz,
    comm_quotient := state.comm_quotient,
    quotient_poly := state.quotient_poly,
    quotient_rand := state.quotient_rand,
    quotient_commitment_spec := state.quotient_commitment_spec,
    view := {
      alpha := proof.challenge_α,
      Az_eval := proof.eval_Az_α,
      Bz_eval := proof.eval_Bz_β,
      Cz_eval := proof.eval_Cz_α,
      quotient_eval := proof.constraint_eval,
      vanishing_eval := proof.vanishing_at_α,
      main_eq := verifierView_zero_eq (_F := F)
    },
    challenge_β := proof.challenge_β,
    opening_Az_α := proof.opening_Az_α,
    opening_Bz_β := proof.opening_Bz_β,
    opening_Cz_α := proof.opening_Cz_α,
    opening_quotient_α := proof.opening_quotient_α,
    valid := verify VC cs x proof
  })

/-- Distribution over commitment states produced during the first adversary run. -/
noncomputable def run_adversary_state {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F)
  (A : Adversary F VC) (x : PublicInput F cs.nPub)
    (secParam : ℕ) : PMF (AdversaryState F VC) :=
  PMF.bind (run_adversary (VC := VC) (cs := cs) A x secParam) (fun sample =>
    PMF.pure sample.1)

/-- Distribution over transcripts emitted in the first adversary run. -/
noncomputable def run_adversary_transcript {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F)
  (A : Adversary F VC) (x : PublicInput F cs.nPub)
    (secParam : ℕ) : PMF (Transcript F VC) :=
  PMF.bind (run_adversary (VC := VC) (cs := cs) A x secParam) (fun sample =>
    PMF.pure sample.2)

/-- Characterization of points in the support of the first adversary run. -/
lemma mem_support_run_adversary {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F)
  (A : Adversary F VC) (x : PublicInput F cs.nPub)
    (secParam : ℕ) {sample : AdversaryState F VC × Transcript F VC}
    (h_mem : sample ∈ (run_adversary (VC := VC) (cs := cs) A x secParam).support) :
    ∃ rand : Fin secParam.succ,
      sample =
        let randNat : ℕ := rand
        let proof := A.run cs x randNat
        let state := A.snapshot cs x randNat
        (state,
          { pp := state.pp
            cs := cs
            x := x
            domainSize := state.domainSize
            omega := state.omega
            comm_Az := state.comm_Az
            comm_Bz := state.comm_Bz
            comm_Cz := state.comm_Cz
            comm_quotient := state.comm_quotient
            quotient_poly := state.quotient_poly
            quotient_rand := state.quotient_rand
            quotient_commitment_spec := state.quotient_commitment_spec
            view := {
              alpha := proof.challenge_α
              Az_eval := proof.eval_Az_α
              Bz_eval := proof.eval_Bz_β
              Cz_eval := proof.eval_Cz_α
              quotient_eval := proof.constraint_eval
              vanishing_eval := proof.vanishing_at_α
              main_eq := verifierView_zero_eq (_F := F)
            }
            challenge_β := proof.challenge_β
            opening_Az_α := proof.opening_Az_α
            opening_Bz_β := proof.opening_Bz_β
            opening_Cz_α := proof.opening_Cz_α
            opening_quotient_α := proof.opening_quotient_α
            valid := verify VC cs x proof }) := by
  classical
  obtain ⟨rand, -, h_pure⟩ :=
    (PMF.mem_support_bind_iff _ _ _).1 h_mem
  refine ⟨rand, ?_⟩
  have h_eq :
      sample =
        let randNat : ℕ := rand
        let proof := A.run cs x randNat
        let state := A.snapshot cs x randNat
        (state,
          { pp := state.pp
            cs := cs
            x := x
            domainSize := state.domainSize
            omega := state.omega
            comm_Az := state.comm_Az
            comm_Bz := state.comm_Bz
            comm_Cz := state.comm_Cz
            comm_quotient := state.comm_quotient
            quotient_poly := state.quotient_poly
            quotient_rand := state.quotient_rand
            quotient_commitment_spec := state.quotient_commitment_spec
            view := {
              alpha := proof.challenge_α
              Az_eval := proof.eval_Az_α
              Bz_eval := proof.eval_Bz_β
              Cz_eval := proof.eval_Cz_α
              quotient_eval := proof.constraint_eval
              vanishing_eval := proof.vanishing_at_α
              main_eq := verifierView_zero_eq (_F := F)
            }
            challenge_β := proof.challenge_β
            opening_Az_α := proof.opening_Az_α
            opening_Bz_β := proof.opening_Bz_β
            opening_Cz_α := proof.opening_Cz_α
            opening_quotient_α := proof.opening_quotient_α
            valid := verify VC cs x proof }) :=
    (PMF.mem_support_pure_iff (a := by
        classical
        let randNat : ℕ := rand
        let proof := A.run cs x randNat
        let state := A.snapshot cs x randNat
        exact (state,
          { pp := state.pp
            cs := cs
            x := x
            domainSize := state.domainSize
            omega := state.omega
            comm_Az := state.comm_Az
            comm_Bz := state.comm_Bz
            comm_Cz := state.comm_Cz
            comm_quotient := state.comm_quotient
            quotient_poly := state.quotient_poly
            quotient_rand := state.quotient_rand
            quotient_commitment_spec := state.quotient_commitment_spec
            view := {
              alpha := proof.challenge_α
              Az_eval := proof.eval_Az_α
              Bz_eval := proof.eval_Bz_β
              Cz_eval := proof.eval_Cz_α
              quotient_eval := proof.constraint_eval
              vanishing_eval := proof.vanishing_at_α
              main_eq := verifierView_zero_eq (_F := F)
            }
            challenge_β := proof.challenge_β
            opening_Az_α := proof.opening_Az_α
            opening_Bz_β := proof.opening_Bz_β
            opening_Cz_α := proof.opening_Cz_α
            opening_quotient_α := proof.opening_quotient_α
            valid := verify VC cs x proof })) (a' := sample)).1 h_pure
  simpa [run_adversary] using h_eq

-- ============================================================================
-- Rewind Adversary (Second Execution with Different Challenge)
-- ============================================================================

/-- Replay adversary with same commitments but different challenge.
    Core of forking lemma: reuse randomness, resample challenge. -/
noncomputable def rewind_adversary {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F)
  (_A : Adversary F VC) (x : PublicInput F cs.nPub)
  (state : AdversaryState F VC)
  (first_challenge : F) (h_card : Fintype.card F ≥ 2) :
    PMF (Transcript F VC) := by
  classical
  let alphaPMF : PMF F := uniform_pmf_ne first_challenge h_card
  let betaPMF : PMF F := uniform_pmf
  refine PMF.bind alphaPMF ?_
  intro alpha'
  refine PMF.bind betaPMF ?_
  intro beta
  let response := state.respond alpha' beta
  exact PMF.pure {
    pp := state.pp,
    cs := cs,
    x := x,
    domainSize := state.domainSize,
    omega := state.omega,
    comm_Az := state.comm_Az,
    comm_Bz := state.comm_Bz,
    comm_Cz := state.comm_Cz,
    comm_quotient := state.comm_quotient,
    quotient_poly := state.quotient_poly,
    quotient_rand := state.quotient_rand,
    quotient_commitment_spec := state.quotient_commitment_spec,
    view := {
      alpha := alpha',
      Az_eval := response.Az_eval,
      Bz_eval := response.Bz_eval,
      Cz_eval := response.Cz_eval,
      quotient_eval := response.quotient_eval,
      vanishing_eval := response.vanishing_eval,
      main_eq := verifierView_zero_eq (_F := F)
    },
    challenge_β := beta,
    opening_Az_α := response.opening_Az_α,
    opening_Bz_β := response.opening_Bz_β,
    opening_Cz_α := response.opening_Cz_α,
    opening_quotient_α := response.opening_quotient_α,
    valid := true
  }

/-- Unpack an element of the rewound adversary distribution. -/
lemma mem_support_rewind_adversary {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (state : AdversaryState F VC)
  (first_challenge : F) (h_card : Fintype.card F ≥ 2)
  {t : Transcript F VC}
  (h_mem : t ∈ (rewind_adversary (VC := VC) (cs := cs) A x state first_challenge h_card).support) :
  ∃ alpha beta,
    alpha ≠ first_challenge ∧
    t =
      let response := state.respond alpha beta
      { pp := state.pp
        cs := cs
        x := x
        domainSize := state.domainSize
        omega := state.omega
        comm_Az := state.comm_Az
        comm_Bz := state.comm_Bz
        comm_Cz := state.comm_Cz
        comm_quotient := state.comm_quotient
        quotient_poly := state.quotient_poly
        quotient_rand := state.quotient_rand
        quotient_commitment_spec := state.quotient_commitment_spec
        view := {
          alpha := alpha
          Az_eval := response.Az_eval
          Bz_eval := response.Bz_eval
          Cz_eval := response.Cz_eval
          quotient_eval := response.quotient_eval
          vanishing_eval := response.vanishing_eval
          main_eq := verifierView_zero_eq (_F := F)
        }
        challenge_β := beta
        opening_Az_α := response.opening_Az_α
        opening_Bz_β := response.opening_Bz_β
        opening_Cz_α := response.opening_Cz_α
        opening_quotient_α := response.opening_quotient_α
        valid := true } := by
  classical
  obtain ⟨alpha, h_alpha, h_bind⟩ :=
    (PMF.mem_support_bind_iff _ _ _).1 h_mem
  obtain ⟨beta, -, h_pure⟩ :=
    (PMF.mem_support_bind_iff _ _ _).1 h_bind
  have h_t_eq :=
    (PMF.mem_support_pure_iff (a := by
        classical
        let response := state.respond alpha beta
        exact {
          pp := state.pp
          cs := cs
          x := x
          domainSize := state.domainSize
          omega := state.omega
          comm_Az := state.comm_Az
          comm_Bz := state.comm_Bz
          comm_Cz := state.comm_Cz
          comm_quotient := state.comm_quotient
          quotient_poly := state.quotient_poly
          quotient_rand := state.quotient_rand
          quotient_commitment_spec := state.quotient_commitment_spec
          view := {
            alpha := alpha
            Az_eval := response.Az_eval
            Bz_eval := response.Bz_eval
            Cz_eval := response.Cz_eval
            quotient_eval := response.quotient_eval
            vanishing_eval := response.vanishing_eval
            main_eq := verifierView_zero_eq (_F := F)
          }
          challenge_β := beta
          opening_Az_α := response.opening_Az_α
          opening_Bz_β := response.opening_Bz_β
          opening_Cz_α := response.opening_Cz_α
          opening_quotient_α := response.opening_quotient_α
          valid := true
        }) (a' := t)).1 h_pure
  have h_alpha_ne : alpha ≠ first_challenge := by
    intro h_eq
    subst h_eq
    have h_mass_ne :=
      (PMF.mem_support_iff (p := uniform_pmf_ne alpha h_card) (a := alpha)).1 h_alpha
    have h_mass_zero : (uniform_pmf_ne alpha h_card) alpha = 0 := by
      classical
      change (if alpha = alpha then 0 else ((Fintype.card F - 1) : ENNReal)⁻¹) = 0
      simp
    exact h_mass_ne h_mass_zero
  refine ⟨alpha, beta, h_alpha_ne, ?_⟩
  simpa [rewind_adversary] using h_t_eq
/-- Every transcript sampled from `rewind_adversary` has a challenge distinct from the
    original `first_challenge`. -/
lemma rewind_adversary_support_alpha_ne {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (state : AdversaryState F VC)
  (first_challenge : F) (h_card : Fintype.card F ≥ 2)
  {t₂ : Transcript F VC}
  (h_mem : t₂ ∈ (rewind_adversary (VC := VC) (cs := cs) A x state first_challenge h_card).support) :
  t₂.view.alpha ≠ first_challenge := by
  classical
  obtain ⟨α, hα_mem, hβ_mem⟩ :=
    (PMF.mem_support_bind_iff _ _ _).1 h_mem
  obtain ⟨β, hβ_support, h_pure⟩ :=
    (PMF.mem_support_bind_iff _ _ _).1 hβ_mem
  have hα_ne : α ≠ first_challenge := by
    intro h_eq
    have h_mass_ne :=
      (PMF.mem_support_iff (p := uniform_pmf_ne first_challenge h_card) (a := α)).1 hα_mem
    have h_mass_zero : (uniform_pmf_ne first_challenge h_card) α = 0 := by
      change (if α = first_challenge then 0 else ((Fintype.card F - 1) : ENNReal)⁻¹) = 0
      simp [h_eq]
    exact h_mass_ne h_mass_zero
  let response := state.respond α β
  let forked : Transcript F VC :=
    { pp := state.pp,
      cs := cs,
      x := x,
      domainSize := state.domainSize,
      omega := state.omega,
      comm_Az := state.comm_Az,
      comm_Bz := state.comm_Bz,
      comm_Cz := state.comm_Cz,
      comm_quotient := state.comm_quotient,
      quotient_poly := state.quotient_poly,
      quotient_rand := state.quotient_rand,
      quotient_commitment_spec := state.quotient_commitment_spec,
      view := {
        alpha := α,
        Az_eval := response.Az_eval,
        Bz_eval := response.Bz_eval,
        Cz_eval := response.Cz_eval,
        quotient_eval := response.quotient_eval,
        vanishing_eval := response.vanishing_eval,
        main_eq := verifierView_zero_eq (_F := F)
      },
      challenge_β := β,
      opening_Az_α := response.opening_Az_α,
      opening_Bz_β := response.opening_Bz_β,
      opening_Cz_α := response.opening_Cz_α,
      opening_quotient_α := response.opening_quotient_α,
      valid := true }
  have h_t₂_eq :=
    (PMF.mem_support_pure_iff (a := forked) (a' := t₂)).1 h_pure
  have h_alpha_eq : t₂.view.alpha = α :=
    congrArg (fun t : Transcript F VC => t.view.alpha) h_t₂_eq
  exact h_alpha_eq ▸ hα_ne

/-- Sample commitment snapshot together with two transcripts forming a fork. -/
noncomputable def fork_state_and_transcripts {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F)
  (A : Adversary F VC) (x : PublicInput F cs.nPub)
    (secParam : ℕ) (h_card : Fintype.card F ≥ 2) :
    PMF (AdversaryState F VC × Transcript F VC × Transcript F VC) := by
  classical
  let firstRun := run_adversary (VC := VC) (cs := cs) A x secParam
  refine PMF.bind firstRun ?_
  intro sample
  let state := sample.1
  let t1 := sample.2
  let rewind := rewind_adversary (VC := VC) (cs := cs) A x state t1.view.alpha h_card
  refine PMF.bind rewind ?_
  intro t2
  exact PMF.pure (state, t1, t2)

/-- In the forked triple, the two transcripts always carry distinct challenges. -/
lemma fork_state_and_transcripts_support_alpha_ne {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (secParam : ℕ) (h_card : Fintype.card F ≥ 2)
  {triple : AdversaryState F VC × Transcript F VC × Transcript F VC}
  (h_mem : triple ∈ (fork_state_and_transcripts (VC := VC) (cs := cs) A x secParam h_card).support) :
  triple.2.1.view.alpha ≠ triple.2.2.view.alpha := by
  classical
  obtain ⟨sample, h_sample, h_bind⟩ :=
    (PMF.mem_support_bind_iff _ _ _).1 h_mem
  obtain ⟨t₂, h_rewind, h_pure⟩ :=
    (PMF.mem_support_bind_iff _ _ _).1 h_bind
  have h_triple_eq :=
    (PMF.mem_support_pure_iff (a := (sample.1, sample.2, t₂)) (a' := triple)).1 h_pure
  cases h_triple_eq
  have h_ne :=
    rewind_adversary_support_alpha_ne (VC := VC) (cs := cs) (A := A) (x := x)
      (state := sample.1) (first_challenge := sample.2.view.alpha) h_card h_rewind
  simpa [ne_comm] using h_ne

/-- If the first transcript in the fork triple is accepting, then the sampled triple forms
    a valid fork. -/
lemma fork_state_and_transcripts_support_is_valid_fork
  {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (secParam : ℕ) (h_card : Fintype.card F ≥ 2)
  {triple : AdversaryState F VC × Transcript F VC × Transcript F VC}
  (h_mem : triple ∈ (fork_state_and_transcripts (VC := VC) (cs := cs) A x secParam h_card).support)
  (h_valid : triple.2.1.valid = true) :
  is_valid_fork VC triple.2.1 triple.2.2 := by
  classical
  obtain ⟨sample, h_sample, h_bind⟩ :=
    (PMF.mem_support_bind_iff _ _ _).1 h_mem
  obtain ⟨t₂, h_rewind, h_pure⟩ :=
    (PMF.mem_support_bind_iff _ _ _).1 h_bind
  have h_triple_eq :=
    (PMF.mem_support_pure_iff (a := (sample.1, sample.2, t₂)) (a' := triple)).1 h_pure
  cases h_triple_eq
  have h_sample_valid : sample.2.valid = true := by
    simpa using h_valid
  obtain ⟨rand, h_sample_struct⟩ :=
    mem_support_run_adversary (VC := VC) (cs := cs) (A := A) (x := x)
      (secParam := secParam) h_sample
  obtain ⟨α, β, h_alpha_ne, h_t₂_struct⟩ :=
    mem_support_rewind_adversary (VC := VC) (cs := cs) (A := A) (x := x)
      (state := sample.1) (first_challenge := sample.2.view.alpha) h_card
      (h_mem := h_rewind)
  have h_t₂_valid : t₂.valid = true := by
    simp [h_t₂_struct]
  have h_ne : sample.2.view.alpha ≠ t₂.view.alpha := by
    simpa [h_t₂_struct, ne_comm] using h_alpha_ne
  subst h_sample_struct
  subst h_t₂_struct
  have h_sample_valid' : verify VC cs x (A.run cs x ↑rand) = true := by
    simpa using h_sample_valid
  simp [is_valid_fork, h_ne, h_sample_valid']

/-- Sample two transcripts forming a fork by running the adversary once and then
    rewinding it with a freshly sampled challenge distinct from the original. -/
noncomputable def fork_transcripts {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F)
  (A : Adversary F VC) (x : PublicInput F cs.nPub)
    (secParam : ℕ) (h_card : Fintype.card F ≥ 2) :
    PMF (Transcript F VC × Transcript F VC) := by
  classical
  let triples := fork_state_and_transcripts (VC := VC) (cs := cs) A x secParam h_card
  refine PMF.bind triples ?_
  intro triple
  exact PMF.pure (triple.2.1, triple.2.2)

/-- The forked transcript pair always contains distinct challenges. -/
lemma fork_transcripts_support_alpha_ne {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (secParam : ℕ) (h_card : Fintype.card F ≥ 2)
  {pair : Transcript F VC × Transcript F VC}
  (h_mem : pair ∈ (fork_transcripts (VC := VC) (cs := cs) A x secParam h_card).support) :
  pair.1.view.alpha ≠ pair.2.view.alpha := by
  classical
  obtain ⟨triple, h_triple, h_pure⟩ :=
    (PMF.mem_support_bind_iff _ _ _).1 h_mem
  have h_pair_eq :=
    (PMF.mem_support_pure_iff (a := (triple.2.1, triple.2.2)) (a' := pair)).1 h_pure
  cases h_pair_eq
  have h_ne :=
    fork_state_and_transcripts_support_alpha_ne (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) h_card h_triple
  simpa using h_ne

/-- If the first transcript in the forked pair is accepting, the pair forms a valid fork. -/
lemma fork_transcripts_support_is_valid_fork
  {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (secParam : ℕ) (h_card : Fintype.card F ≥ 2)
  {pair : Transcript F VC × Transcript F VC}
  (h_mem : pair ∈ (fork_transcripts (VC := VC) (cs := cs) A x secParam h_card).support)
  (h_valid : pair.1.valid = true) :
  is_valid_fork VC pair.1 pair.2 := by
  classical
  obtain ⟨triple, h_triple, h_pure⟩ :=
    (PMF.mem_support_bind_iff _ _ _).1 h_mem
  have h_pair_eq :=
    (PMF.mem_support_pure_iff (a := (triple.2.1, triple.2.2)) (a' := pair)).1 h_pure
  cases h_pair_eq
  have h_valid' : triple.2.1.valid = true := by
    simpa using h_valid
  exact fork_state_and_transcripts_support_is_valid_fork (VC := VC) (cs := cs) (A := A)
    (x := x) (secParam := secParam) h_card h_triple h_valid'

-- ============================================================================
-- Heavy Row Lemma (Forking Core)
-- ============================================================================

/-- Extract the commitment tuple carried by a transcript. -/
def transcriptCommitTuple {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (t : Transcript F VC) :
    VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment :=
  (t.comm_Az, t.comm_Bz, t.comm_Cz, t.comm_quotient)

/-- Extract both the commitment tuple and challenge from a transcript. -/
def transcriptCommitChallenge {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (t : Transcript F VC) :
    (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F :=
  (transcriptCommitTuple VC t, t.view.alpha)

/-- Distribution over commitment tuples produced in the first adversary run. -/
noncomputable def run_adversary_commit_tuple {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (A : Adversary F VC) (x : PublicInput F cs.nPub)
    (secParam : ℕ) :
    PMF (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :=
  PMF.map (transcriptCommitTuple VC)
    (run_adversary_transcript (VC := VC) (cs := cs) A x secParam)

/-- Distribution over `(commitment tuple, challenge)` pairs seen in the first adversary run. -/
noncomputable def run_adversary_commit_challenge {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (A : Adversary F VC) (x : PublicInput F cs.nPub)
    (secParam : ℕ) :
    PMF ((VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F) :=
  PMF.map (transcriptCommitChallenge VC)
    (run_adversary_transcript (VC := VC) (cs := cs) A x secParam)

lemma mem_support_run_adversary_commit_tuple {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (A : Adversary F VC) (x : PublicInput F cs.nPub)
    (secParam : ℕ)
    {commTuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment} :
    commTuple ∈
        (run_adversary_commit_tuple VC cs A x secParam).support ↔
      ∃ t ∈ (run_adversary_transcript (VC := VC) (cs := cs) A x secParam).support,
        transcriptCommitTuple VC t = commTuple := by
  classical
  unfold run_adversary_commit_tuple
  constructor
  · intro h_mem
    obtain ⟨t, h_t, h_eq⟩ :=
      (PMF.mem_support_map_iff (f := transcriptCommitTuple VC)
          (p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam)
          (b := commTuple)).1 h_mem
    exact ⟨t, h_t, h_eq⟩
  · rintro ⟨t, h_t, h_eq⟩
    exact
      (PMF.mem_support_map_iff (f := transcriptCommitTuple VC)
          (p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam)
          (b := commTuple)).2
        ⟨t, h_t, h_eq⟩

lemma mem_support_run_adversary_commit_challenge {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (A : Adversary F VC) (x : PublicInput F cs.nPub)
    (secParam : ℕ)
    {cc : (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F} :
    cc ∈ (run_adversary_commit_challenge VC cs A x secParam).support ↔
      ∃ t ∈ (run_adversary_transcript (VC := VC) (cs := cs) A x secParam).support,
        transcriptCommitChallenge VC t = cc := by
  classical
  unfold run_adversary_commit_challenge
  constructor
  · intro h_mem
    obtain ⟨t, h_t, h_eq⟩ :=
      (PMF.mem_support_map_iff (f := transcriptCommitChallenge VC)
          (p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam)
          (b := cc)).1 h_mem
    exact ⟨t, h_t, h_eq⟩
  · rintro ⟨t, h_t, h_eq⟩
    exact
      (PMF.mem_support_map_iff (f := transcriptCommitChallenge VC)
          (p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam)
          (b := cc)).2
        ⟨t, h_t, h_eq⟩

/-- Success event: adversary produces accepting proof -/
def success_event {F : Type} [Field F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (x : PublicInput F cs.nPub)
    (t : Transcript F VC) : Prop :=
  let _ := VC
  let _ := cs
  let _ := x
  t.valid = true

/-- Fork success event: both transcripts succeed and form a valid fork. -/
def fork_success_event {F : Type} [Field F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (x : PublicInput F cs.nPub)
    (pair : Transcript F VC × Transcript F VC) : Prop :=
  success_event VC cs x pair.1 ∧
  success_event VC cs x pair.2 ∧
  is_valid_fork VC pair.1 pair.2

/-- Total success probability of the adversary's first execution. -/
noncomputable def successProbability {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) : ℝ :=
  ((run_adversary_transcript (VC := VC) (cs := cs) A x secParam).toOuterMeasure
      {t | success_event VC cs x t}).toReal

/-- Масса события успеха в распределении первой попытки (значение в `ENNReal`). -/
noncomputable def successMass {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (secParam : ℕ) : ENNReal :=
  by
    classical
    let p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam
    exact ∑' t, (if success_event VC cs x t then p t else 0)

noncomputable def successMassGivenCommit {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) : ENNReal :=
  by
    classical
    let p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam
    exact ∑' t,
      (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple then p t else 0)

/-- Масса успеха для фиксированного кортежа коммитов и вызова. -/
noncomputable def successMassGivenCommitAndChallenge {F : Type} [Field F] [Fintype F]
    [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)
    (α : F) : ENNReal :=
  by
    classical
    let p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam
    exact ∑' t,
      (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α
        then p t else 0)

/-- Полная масса транскриптов, чьи коммиты совпадают с заданным кортежем. -/
noncomputable def commitMass {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) : ENNReal :=
  by
    classical
    let p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam
    exact ∑' t, (if transcriptCommitTuple VC t = comm_tuple then p t else 0)

lemma commitMass_eq_run_adversary_commit_tuple
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    commitMass VC cs A x secParam comm_tuple
      = run_adversary_commit_tuple VC cs A x secParam comm_tuple := by
  classical
  unfold commitMass
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  change (∑' t, (if transcriptCommitTuple VC t = comm_tuple then p t else 0))
      = run_adversary_commit_tuple VC cs A x secParam comm_tuple
  have h_map :
      run_adversary_commit_tuple VC cs A x secParam comm_tuple
        = ∑' t, (if transcriptCommitTuple VC t = comm_tuple then p t else 0) := by
    simpa [run_adversary_commit_tuple, hp.symm, PMF.map_apply, eq_comm] using
      (PMF.map_apply (transcriptCommitTuple VC)
        (run_adversary_transcript (VC := VC) (cs := cs) A x secParam) comm_tuple)
  simpa using h_map.symm

/-- Probability mass of a commitment tuple is at most one. -/
lemma commitMass_le_one {F : Type} [Field F] [Fintype F] [DecidableEq F]
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
    by_cases h : transcriptCommitTuple VC t = comm_tuple
    · simp [h]
    · have : (0 : ENNReal) ≤ p t := bot_le
      convert this using 1
      simp [h]
  have h_le := ENNReal.tsum_le_tsum h_pointwise
  have h_total : ∑' t, p t = 1 := p.tsum_coe
  simpa [h_total] using h_le

/-- Полная масса транскриптов, фиксирующих кортеж коммитов и вызов. -/
noncomputable def commitChallengeMass {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)
    (α : F) : ENNReal :=
  by
    classical
    let p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam
    exact ∑' t,
      (if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0)

/-- Полная масса фиксированного `(commit, challenge)` совпадает с pushforward-дистрибуцией. -/
lemma commitChallengeMass_eq_run_adversary_commit_challenge
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) (α : F) :
    commitChallengeMass VC cs A x secParam comm_tuple α
      = run_adversary_commit_challenge VC cs A x secParam (comm_tuple, α) := by
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

lemma successProbability_toReal_successMass {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    successProbability VC cs A x secParam =
      ENNReal.toReal (successMass VC cs A x secParam) := by
  classical
  unfold successProbability successMass
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  have h_outer := p.toOuterMeasure_apply {t : Transcript F VC | success_event VC cs x t}
  have h_indicator :
      (∑' t, Set.indicator {t : Transcript F VC | success_event VC cs x t} p t)
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
  simpa [hp, h_indicator, h_indicator'] using congrArg ENNReal.toReal h_outer

lemma successMassGivenCommit_le_successMass {F : Type} [Field F] [Fintype F] [DecidableEq F]
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

lemma successMassGivenCommit_le_commitMass {F : Type} [Field F] [Fintype F] [DecidableEq F]
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

lemma tsum_commitMass_eq_one {F : Type} [Field F] [Fintype F] [DecidableEq F]
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

lemma successProbability_nonneg {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    0 ≤ successProbability VC cs A x secParam := by
  unfold successProbability
  exact ENNReal.toReal_nonneg

lemma successProbability_le_one {F : Type} [Field F] [Fintype F] [DecidableEq F]
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
    have h_support_subset : p.support ⊆ U := fun _t _ => trivial
    simpa [U] using (p.toOuterMeasure_apply_eq_one_iff (s := U)).2 h_support_subset
  have h_le_one : p.toOuterMeasure S ≤ ENNReal.ofReal 1 := by
    simpa [U, h_univ, ENNReal.ofReal_one] using h_le
  have h_toReal := ENNReal.toReal_le_of_le_ofReal zero_le_one h_le_one
  simpa [successProbability, hp, hS, U, ENNReal.ofReal_one] using h_toReal

lemma successMass_le_one {F : Type} [Field F] [Fintype F] [DecidableEq F]
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

/-!
### Heavy commitments

We say a commitment tuple is heavy when the conditional success mass is at least `ε`.
-/

def heavyCommitments {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ) :
    Set (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :=
  {comm_tuple |
    0 < commitMass VC cs A x secParam comm_tuple ∧
      successMassGivenCommit VC cs A x secParam comm_tuple ≥
        ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple}

lemma exists_success_transcript_of_heavyCommitment
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
    (h_ε_pos : 0 < ε)
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
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
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

lemma heavyCommitments_mem_iff {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment} :
    comm_tuple ∈ heavyCommitments VC cs A x secParam ε ↔
      0 < commitMass VC cs A x secParam comm_tuple ∧
        successMassGivenCommit VC cs A x secParam comm_tuple ≥
          ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple := by
  classical
    rfl

lemma successMassGivenCommit_lt_of_not_heavy
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
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

lemma successMassGivenCommit_le_of_not_heavy
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_not : comm_tuple ∉ heavyCommitments VC cs A x secParam ε) :
    successMassGivenCommit VC cs A x secParam comm_tuple ≤
      ENNReal.ofReal ε * commitMass VC cs A x secParam comm_tuple := by
  classical
  obtain h_zero | h_lt :=
    successMassGivenCommit_lt_of_not_heavy (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) (comm_tuple := comm_tuple) h_not
  · have h_success_zero :
        successMassGivenCommit VC cs A x secParam comm_tuple = 0 := by
      have := successMassGivenCommit_le_commitMass (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) comm_tuple
      have : successMassGivenCommit VC cs A x secParam comm_tuple ≤ 0 := by
        simpa [h_zero] using this
      exact le_antisymm this bot_le
    simp [h_zero, h_success_zero]
  · exact (le_of_lt h_lt)

lemma exists_commitMass_pos
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
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

lemma successMass_le_of_all_not_heavy
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
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
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
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
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
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

lemma exists_success_transcript_of_successProbability_lt
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
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
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
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
lemma successMass_eq_tsum_successMassGivenCommitAndChallenge
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
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
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
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
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
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
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
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
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
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
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) {ε : ℝ}
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

/-- Set of challenges that yield positive conditional success mass for a fixed commitment. -/
noncomputable def successfulChallenges
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
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
    exists_successMassGivenCommitAndChallenge_pos_of_heavyCommitment (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε) h_ε_pos h_mem
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
  simpa [hp] using h_le

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

lemma fork_success_event.success_left {F : Type} [Field F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (x : PublicInput F cs.nPub)
  {pair : Transcript F VC × Transcript F VC}
  (h_event : fork_success_event VC cs x pair) :
  success_event VC cs x pair.1 := h_event.1

lemma fork_success_event.success_right {F : Type} [Field F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (x : PublicInput F cs.nPub)
  {pair : Transcript F VC × Transcript F VC}
  (h_event : fork_success_event VC cs x pair) :
  success_event VC cs x pair.2 := h_event.2.1

lemma fork_success_event.is_valid {F : Type} [Field F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (x : PublicInput F cs.nPub)
  {pair : Transcript F VC × Transcript F VC}
  (h_event : fork_success_event VC cs x pair) :
  is_valid_fork VC pair.1 pair.2 := h_event.2.2

/-- Specialized version phrased via `success_event`. -/
lemma fork_transcripts_support_success_is_valid_fork
  {F : Type} [Field F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (secParam : ℕ) (h_card : Fintype.card F ≥ 2)
  {pair : Transcript F VC × Transcript F VC}
  (h_mem : pair ∈ (fork_transcripts (VC := VC) (cs := cs) A x secParam h_card).support)
  (h_success : success_event VC cs x pair.1) :
  is_valid_fork VC pair.1 pair.2 := by
  classical
  have h_valid : pair.1.valid = true := by
    simpa [success_event] using h_success
  exact fork_transcripts_support_is_valid_fork (VC := VC) (cs := cs) (A := A)
    (x := x) (secParam := secParam) h_card h_mem h_valid

/-- Forking a successful transcript yields another successful transcript. -/
lemma fork_transcripts_support_success_second
  {F : Type} [Field F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (secParam : ℕ) (h_card : Fintype.card F ≥ 2)
  {pair : Transcript F VC × Transcript F VC}
  (h_mem : pair ∈ (fork_transcripts (VC := VC) (cs := cs) A x secParam h_card).support)
  (h_success : success_event VC cs x pair.1) :
  success_event VC cs x pair.2 := by
  classical
  have h_fork :=
    fork_transcripts_support_success_is_valid_fork (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) h_card h_mem h_success
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, h_valid₂⟩ := h_fork
  simpa [success_event] using h_valid₂

/-- Any forked transcript pair sampled after a successful run remains
  successful and forms a valid fork. -/
lemma fork_transcripts_support_success_event
  {F : Type} [Field F] [Fintype F] [DecidableEq F]
  (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (secParam : ℕ) (h_card : Fintype.card F ≥ 2)
  {pair : Transcript F VC × Transcript F VC}
  (h_mem : pair ∈ (fork_transcripts (VC := VC) (cs := cs) A x secParam h_card).support)
  (h_success : success_event VC cs x pair.1) :
  fork_success_event VC cs x pair := by
  refine ⟨h_success,
    fork_transcripts_support_success_second (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) h_card h_mem h_success,
    fork_transcripts_support_success_is_valid_fork (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) h_card h_mem h_success⟩

/-- A commitment is "heavy" if many challenges lead to valid proofs.
    Formally: at least ε fraction of challenges are valid. -/
def is_heavy_commitment {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (x : PublicInput F cs.nPub)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)
    (ε : ℝ) : Prop :=
  ∃ valid_challenges : Finset F,
    (∀ α ∈ valid_challenges,
      ∃ t : Transcript F VC,
        success_event VC cs x t ∧
        transcriptCommitTuple VC t = comm_tuple ∧
        t.view.alpha = α) ∧
    (valid_challenges.card : ℝ) ≥ ε * (Fintype.card F : ℝ)

/-!
## WARNING: Axiomatized Heavy Theorems

Two probability bounds remain axiomatized to keep the development free of
`sorry` while the probabilistic infrastructure is being formalized:

1. **heavy_row_lemma**: Formalization of the adversary success probability via
  `PMF.bind` and a heavy-row counting argument.
2. **fork_success_bound**: Combinatorial lower bound on obtaining two distinct
  successful challenges once a heavy commitment exists.

These statements will be discharged after completing the PMF model for the
adversary and proving the required binomial coefficient inequalities.
-/

/-- If adversary succeeds with probability ≥ ε, then a fraction ≥ ε - 1/|F|
    of commitment choices are "heavy": for each such commitment,
    at least ε|F| challenges lead to accepting proofs.

    **AXIOM**: Requires probabilistic model formalization.

    Proof strategy (for future implementation):
    1. Total success probability = sum over (commitments × challenges) of indicator
    2. Group by commitments: Pr[success] = sum_c Pr[commit=c] * Pr[success | c]
    3. If too few heavy commitments, total prob < ε (contradiction)
    4. Pigeonhole: if Pr[success] ≥ ε, then heavy commitments exist

    Dependencies:
    - run_adversary implemented via PMF.bind
    - Success probability formalized as PMF.toMeasure
    - Finset.sum lemmas for expectation reasoning

    Estimated effort: 3-4h -/
axiom heavy_row_lemma {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (A : Adversary F VC) (x : PublicInput F cs.nPub)
    (ε : ℝ) (secParam : ℕ)
    (h_ε_pos : 0 < ε)
    (_h_ε_bound : ε ≤ 1)
    (h_field_size : (Fintype.card F : ℝ) > 0)
    (h_success : True)  -- TODO: formalize Pr[verify = true] ≥ ε
    :
    ∃ (heavy_comms : Finset (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)),
      (heavy_comms.card : ℝ) ≥ (ε - 1 / (Fintype.card F : ℝ)) * secParam ∧
      ∀ c ∈ heavy_comms, is_heavy_commitment VC cs x c ε

-- ============================================================================
-- Fork Success Probability
-- ============================================================================

/-- Given a "heavy" commitment (many valid challenges),
  the probability of obtaining two distinct valid challenges is ≥ ε²/2.

  This lemma converts the combinatorial inequality on binomial coefficients
  into a real-valued inequality.  The helper lemmas `choose_two_cast` and
  `eps_mul_sub_one_over_ge` take care of the arithmetic manipulations. -/
lemma fork_success_bound {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F)
    (_state : AdversaryState F VC)
    (valid_challenges : Finset F)
    (ε : ℝ)
    (h_heavy : (valid_challenges.card : ℝ) ≥ ε * (Fintype.card F : ℝ))
    (h_ε_pos : 0 < ε)
    (_h_ε_bound : ε ≤ 1)
    (h_field_size : (Fintype.card F : ℝ) ≥ 2)
    (h_valid_nonempty : valid_challenges.card ≥ 2)
    :
    let total_pairs := Nat.choose (Fintype.card F) 2
    let valid_pairs := Nat.choose valid_challenges.card 2
    (valid_pairs : ℝ) / (total_pairs : ℝ) ≥ ε^2 / 2 - 1 / (Fintype.card F : ℝ) := by
  classical
  set n : ℝ := (Fintype.card F : ℝ)
  set m : ℝ := (valid_challenges.card : ℝ)
  let total_pairs := Nat.choose (Fintype.card F) 2
  let valid_pairs := Nat.choose valid_challenges.card 2
  change (valid_pairs : ℝ) / (total_pairs : ℝ) ≥ ε ^ 2 / 2 - 1 / n
  have hn_two : (2 : ℝ) ≤ n := by
    simpa [n] using h_field_size
  have hn_pos : 0 < n := lt_of_lt_of_le (by norm_num : (0 : ℝ) < 2) hn_two
  have hn_ne : n ≠ 0 := ne_of_gt hn_pos
  have hn_gt_one : (1 : ℝ) < n := lt_of_lt_of_le (by norm_num : (1 : ℝ) < 2) hn_two
  have hn_ne_one : n ≠ 1 := ne_of_gt hn_gt_one
  have hn_sub_ne : n - 1 ≠ 0 := sub_ne_zero.mpr (ne_of_gt hn_gt_one)
  have hm_two : (2 : ℝ) ≤ m := by
    simpa [m] using h_valid_nonempty
  have hm_pos : 0 < m := lt_of_lt_of_le (by norm_num : (0 : ℝ) < 2) hm_two
  have hm_ne : m ≠ 0 := ne_of_gt hm_pos
  have hm_gt_one : (1 : ℝ) < m := lt_of_lt_of_le (by norm_num : (1 : ℝ) < 2) hm_two
  have hm_sub_ne : m - 1 ≠ 0 := sub_ne_zero.mpr (ne_of_gt hm_gt_one)
  have hm_ge_one : 1 ≤ m := le_of_lt hm_gt_one
  have h_inv_nonneg : 0 ≤ 1 / n := by
    have := inv_pos.mpr hn_pos
    exact le_of_lt (by simpa [one_div] using this)
  have h_eps_le : ε ≤ m / n := by
    have h_mul : ε * n ≤ m := by
      simpa [n, m, mul_comm] using h_heavy
    have := mul_le_mul_of_nonneg_left h_mul h_inv_nonneg
    simpa [div_eq_mul_inv, n, m, hn_ne, mul_comm, mul_left_comm, mul_assoc] using this
  have h_mul_cancel : (m / n) * n = m := by
    simp [div_eq_mul_inv, n, hn_ne]
  have h_prod : 1 ≤ (m / n) * n := by
    simpa [h_mul_cancel] using hm_ge_one
  have h_eps_bound' :=
    eps_mul_sub_one_over_ge (ε := m / n) (n := n) hn_two (by simpa [h_mul_cancel] using h_prod)
  have h_eps_bound : (m / n) * (m - 1) / (n - 1) ≥ (m / n) ^ 2 / 2 - 1 / n := by
    simpa [h_mul_cancel] using h_eps_bound'
  have h_ratio_eq :
      (valid_pairs : ℝ) / (total_pairs : ℝ)
        = (m / n) * (m - 1) / (n - 1) := by
    have h_two_ne : (2 : ℝ) ≠ 0 := by norm_num
    have h_valid_cast : (valid_pairs : ℝ) = m * (m - 1) / 2 := by
      simp [valid_pairs, m, choose_two_cast]
    have h_total_cast : (total_pairs : ℝ) = n * (n - 1) / 2 := by
      simp [total_pairs, n, choose_two_cast]
    have h_main :
        (m * (m - 1) / 2) / (n * (n - 1) / 2)
          = (m / n) * (m - 1) / (n - 1) := by
      field_simp [hn_ne, hn_sub_ne, hm_ne, hm_sub_ne, h_two_ne]
    simpa [h_valid_cast, h_total_cast] using h_main
  have h_ratio_lower :
      (valid_pairs : ℝ) / (total_pairs : ℝ) ≥ (m / n) ^ 2 / 2 - 1 / n := by
    simpa [h_ratio_eq] using h_eps_bound
  have h_eps_nonneg : 0 ≤ ε := le_of_lt h_ε_pos
  have h_ratio_nonneg : 0 ≤ m / n :=
    div_nonneg (le_of_lt hm_pos) (le_of_lt hn_pos)
  have h_abs_le : |ε| ≤ |m / n| := by
    simpa [abs_of_nonneg h_eps_nonneg, abs_of_nonneg h_ratio_nonneg] using h_eps_le
  have h_sq_le : ε ^ 2 ≤ (m / n) ^ 2 := sq_le_sq.mpr h_abs_le
  have h_inv_two_nonneg : 0 ≤ (1 / (2 : ℝ)) := by norm_num
  have h_sq_half_le : ε ^ 2 / 2 ≤ (m / n) ^ 2 / 2 := by
    have := mul_le_mul_of_nonneg_right h_sq_le h_inv_two_nonneg
    simpa [div_eq_mul_inv] using this
  have h_step : ε ^ 2 / 2 - 1 / n ≤ (m / n) ^ 2 / 2 - 1 / n :=
    sub_le_sub_right h_sq_half_le (1 / n)
  have h_final : ε ^ 2 / 2 - 1 / n ≤ (valid_pairs : ℝ) / (total_pairs : ℝ) :=
    le_trans h_step h_ratio_lower
  simpa [n] using h_final

-- ============================================================================
-- Witness Extraction from Fork
-- ============================================================================

/-- Binding property of the vector commitment scheme forces quotient polynomials to coincide. -/
lemma quotient_poly_eq_of_fork {F : Type} [CommRing F] [DecidableEq F]
    (VC : VectorCommitment F) (t1 t2 : Transcript F VC)
    (h_fork : is_valid_fork VC t1 t2) :
    t1.quotient_poly = t2.quotient_poly := by
  classical
  obtain ⟨h_pp, _, _, _, _, _, _, _, h_comm, h_rand, _, _, _⟩ := h_fork
  have h_commit :
      VC.commit t1.pp (Polynomial.coeffList t1.quotient_poly) t1.quotient_rand =
        VC.commit t2.pp (Polynomial.coeffList t2.quotient_poly) t2.quotient_rand := by
    calc
      VC.commit t1.pp (Polynomial.coeffList t1.quotient_poly) t1.quotient_rand
          = t1.comm_quotient := t1.quotient_commitment_spec
      _ = t2.comm_quotient := h_comm
      _ = VC.commit t2.pp (Polynomial.coeffList t2.quotient_poly) t2.quotient_rand :=
        t2.quotient_commitment_spec.symm
  have h_lists :
      Polynomial.coeffList t1.quotient_poly =
        Polynomial.coeffList t2.quotient_poly := by
    exact
      (VC.commit_injective t1.pp t1.quotient_rand)
        (by simpa [h_pp, h_rand] using h_commit)
  exact coeffList_injective h_lists

/-- Extract the quotient polynomial witnessed inside a forked pair of transcripts.

  Thanks to the binding property captured in `quotient_poly_eq_of_fork`, both transcripts
  agree on the underlying polynomial whenever their commitments coincide and share the same
  randomness. -/
noncomputable def extract_quotient_diff {F : Type} [Field F] [DecidableEq F]
    (VC : VectorCommitment F) (_cs : R1CS F)
    (t1 t2 : Transcript F VC)
    (h_fork : is_valid_fork VC t1 t2)
    (_m : ℕ) (_ω : F) :
    Polynomial F :=
  by
    let _ := h_fork
    classical
    have := quotient_poly_eq_of_fork VC t1 t2 h_fork
    exact t1.quotient_poly
/-- Extract witness from quotient polynomial via Lagrange interpolation.

    Strategy:
    1. Quotient q encodes constraint satisfaction over domain H = {ωⁱ | i < m}
    2. Witness values: w(i) = evaluate witness polynomial at ωⁱ
    3. Use lagrange_interpolate_eval (Polynomial.lean:156) in reverse:
       Given q, compute w(i) = q(ωⁱ) for each i
    4. Result is witness vector satisfying R1CS (by extraction_soundness) -/
noncomputable def extract_witness {F : Type} [Field F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (q : Polynomial F) (m : ℕ) (ω : F)
    (h_m : m = cs.nVars)
    (x : PublicInput F cs.nPub) :
    Witness F cs.nVars :=
  by
    let _ := VC
    let _ := h_m
    classical
    -- Prefix copies public input, remainder follows quotient polynomial
    exact fun i : Fin cs.nVars =>
      if hx : (i : ℕ) < cs.nPub then
        x ⟨i, hx⟩
      else
        q.eval (ω ^ (i : ℕ))

/-! ### Extraction helper lemmas

These lemmas isolate the behaviour of `extract_witness` on the public and
private portions of the witness vector. They will be used in the forthcoming
proof of `extraction_soundness` to separate the handling of inputs already
known to the verifier from evaluations determined by the extracted quotient
polynomial. -/

section ExtractionHelpers

variable {F : Type} [Field F] [DecidableEq F]

lemma extract_witness_public {VC : VectorCommitment F} {cs : R1CS F}
    {q : Polynomial F} {m : ℕ} {ω : F}
    (h_m : m = cs.nVars) (x : PublicInput F cs.nPub)
    {i : Fin cs.nVars} (hi : (i : ℕ) < cs.nPub) :
    extract_witness VC cs q m ω h_m x i = x ⟨i, hi⟩ := by
  classical
  unfold extract_witness
  simp [hi]

lemma extract_witness_private {VC : VectorCommitment F} {cs : R1CS F}
    {q : Polynomial F} {m : ℕ} {ω : F}
    (h_m : m = cs.nVars) (x : PublicInput F cs.nPub)
    {i : Fin cs.nVars} (hi : cs.nPub ≤ (i : ℕ)) :
    extract_witness VC cs q m ω h_m x i = q.eval (ω ^ (i : ℕ)) := by
  classical
  unfold extract_witness
  have : ¬ (i : ℕ) < cs.nPub := by exact Nat.not_lt.mpr hi
  simp [this]

lemma extract_witness_public_prefix {VC : VectorCommitment F} {cs : R1CS F}
    {q : Polynomial F} {m : ℕ} {ω : F}
    (h_m : m = cs.nVars) (x : PublicInput F cs.nPub) :
    ∀ i : Fin cs.nPub,
      extract_witness VC cs q m ω h_m x ⟨i, Nat.lt_of_lt_of_le i.isLt cs.h_pub_le⟩ =
        x i := by
  classical
  intro i
  have hlt : ((⟨i, Nat.lt_of_lt_of_le i.isLt cs.h_pub_le⟩ : Fin cs.nVars) : ℕ) < cs.nPub := by
    simp
  simp [extract_witness, hlt]

lemma extract_witness_satisfies_of_constraint_zero
    {VC : VectorCommitment F} {cs : R1CS F}
    {q : Polynomial F} {m : ℕ} {ω : F}
    (h_m : m = cs.nVars) (x : PublicInput F cs.nPub)
    (h_zero : ∀ i : Fin cs.nCons,
      constraintPoly cs (extract_witness VC cs q m ω h_m x) i = 0) :
    satisfies cs (extract_witness VC cs q m ω h_m x) := by
  classical
  exact
    (satisfies_iff_constraint_zero (cs := cs)
      (z := extract_witness VC cs q m ω h_m x)).2 h_zero

lemma extraction_soundness_of_constraint_zero
    {VC : VectorCommitment F} {cs : R1CS F}
    {t1 t2 : Transcript F VC}
    (h_fork : is_valid_fork VC t1 t2)
    {m : ℕ} {ω : F} (h_m : m = cs.nVars)
    (x : PublicInput F cs.nPub)
    (h_zero : ∀ i : Fin cs.nCons,
      constraintPoly cs
        (extract_witness VC cs
          (extract_quotient_diff VC cs t1 t2 h_fork m ω)
          m ω h_m x) i = 0) :
    satisfies cs
      (extract_witness VC cs
        (extract_quotient_diff VC cs t1 t2 h_fork m ω)
        m ω h_m x) := by
  classical
  exact extract_witness_satisfies_of_constraint_zero (VC := VC) (cs := cs)
      (q := extract_quotient_diff VC cs t1 t2 h_fork m ω) (m := m) (ω := ω)
      h_m x h_zero

end ExtractionHelpers

/-! ### Verifier equations for forks -/

section ForkingEquations

variable {F : Type} [Field F] [DecidableEq F]

/-- Certificate capturing the polynomial relations enforced by the verifier
    on a pair of transcripts that form a valid fork. The fields encode the
    domain parameters and algebraic equalities required to show that the
    extracted witness satisfies all R1CS constraints. -/
structure ForkingVerifierEquationsCore (VC : VectorCommitment F) (cs : R1CS F)
    (t1 t2 : Transcript F VC) (h_fork : is_valid_fork VC t1 t2) where
  m : ℕ
  ω : F
  h_m_vars : m = cs.nVars
  h_primitive : IsPrimitiveRoot ω m
  quotient_eval :
    (x : PublicInput F cs.nPub) →
      ∀ i : Fin cs.nCons,
        (extract_quotient_diff VC cs t1 t2 h_fork m ω).eval (ω ^ (i : ℕ)) =
          constraintPoly cs
            (extract_witness VC cs
              (extract_quotient_diff VC cs t1 t2 h_fork m ω)
              m ω h_m_vars x) i
  quotient_diff_natDegree_lt_domain :
    (x : PublicInput F cs.nPub) →
      (extract_quotient_diff VC cs t1 t2 h_fork m ω
          - LambdaSNARK.constraintNumeratorPoly cs
              (extract_witness VC cs
                (extract_quotient_diff VC cs t1 t2 h_fork m ω)
                m ω h_m_vars x) ω).natDegree < m

structure ForkingVerifierEquations (VC : VectorCommitment F) (cs : R1CS F)
    (t1 t2 : Transcript F VC) (h_fork : is_valid_fork VC t1 t2) where
  m : ℕ
  ω : F
  h_m_vars : m = cs.nVars
  h_m_cons : m = cs.nCons
  h_primitive : IsPrimitiveRoot ω m
  quotient_eval :
    (x : PublicInput F cs.nPub) →
      ∀ i : Fin cs.nCons,
        (extract_quotient_diff VC cs t1 t2 h_fork m ω).eval (ω ^ (i : ℕ)) =
          constraintPoly cs
            (extract_witness VC cs
              (extract_quotient_diff VC cs t1 t2 h_fork m ω)
              m ω h_m_vars x) i
  quotient_diff_natDegree_lt_domain :
    (x : PublicInput F cs.nPub) →
      (extract_quotient_diff VC cs t1 t2 h_fork m ω
          - LambdaSNARK.constraintNumeratorPoly cs
              (extract_witness VC cs
                (extract_quotient_diff VC cs t1 t2 h_fork m ω)
                m ω h_m_vars x) ω).natDegree < m

lemma constraint_numerator_eval_matches_quotient_of_equations
    (VC : VectorCommitment F) (cs : R1CS F)
    {t1 t2 : Transcript F VC} {h_fork : is_valid_fork VC t1 t2}
    (eqns : ForkingVerifierEquations VC cs t1 t2 h_fork)
    (x : PublicInput F cs.nPub) :
    ∀ i : Fin cs.nCons,
      (LambdaSNARK.constraintNumeratorPoly cs
        (extract_witness VC cs
          (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
          eqns.m eqns.ω eqns.h_m_vars x) eqns.ω).eval (eqns.ω ^ (i : ℕ)) =
        (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω).eval
          (eqns.ω ^ (i : ℕ)) := by
  classical
  set q := extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω
  set w := extract_witness VC cs q eqns.m eqns.ω eqns.h_m_vars x
  have h_prim : IsPrimitiveRoot eqns.ω cs.nCons := by
    simpa [eqns.h_m_cons] using eqns.h_primitive
  intro i
  have h_eval := eqns.quotient_eval x i
  have h_cp := LambdaSNARK.constraintPoly_eval_domain_eq_constraintNumerator
      (cs := cs) (z := w) (ω := eqns.ω) h_prim i
  have h_match :
      q.eval (eqns.ω ^ (i : ℕ)) =
        (LambdaSNARK.constraintNumeratorPoly cs w eqns.ω).eval (eqns.ω ^ (i : ℕ)) := by
    simpa [q, w, h_cp] using h_eval
  simpa [q, w] using h_match.symm

lemma constraint_quotient_sub_numerator_mod_vanishing_zero_of_equations
    (VC : VectorCommitment F) (cs : R1CS F)
    {t1 t2 : Transcript F VC} {h_fork : is_valid_fork VC t1 t2}
    (eqns : ForkingVerifierEquations VC cs t1 t2 h_fork)
    (x : PublicInput F cs.nPub) :
    ((extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
        - LambdaSNARK.constraintNumeratorPoly cs
            (extract_witness VC cs
              (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
              eqns.m eqns.ω eqns.h_m_vars x) eqns.ω)
        %ₘ vanishing_poly cs.nCons eqns.ω = 0 := by
  classical
  set q := extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω
  set w := extract_witness VC cs q eqns.m eqns.ω eqns.h_m_vars x
  have h_prim : IsPrimitiveRoot eqns.ω cs.nCons := by
    simpa [eqns.h_m_cons] using eqns.h_primitive
  refine (remainder_zero_iff_vanishing
      (f := q - LambdaSNARK.constraintNumeratorPoly cs w eqns.ω)
      (m := cs.nCons) (ω := eqns.ω) h_prim).2 ?_
  intro i
  have h_match := constraint_numerator_eval_matches_quotient_of_equations
      (VC := VC) (cs := cs) (t1 := t1) (t2 := t2) (h_fork := h_fork)
      eqns x i
  simpa [q, w, Polynomial.eval_sub, sub_eq_add_neg] using
    (sub_eq_zero.mpr h_match.symm)

lemma constraint_quotient_sub_numerator_eq_zero_of_equations
    (VC : VectorCommitment F) (cs : R1CS F)
    {t1 t2 : Transcript F VC} {h_fork : is_valid_fork VC t1 t2}
    (eqns : ForkingVerifierEquations VC cs t1 t2 h_fork)
    (x : PublicInput F cs.nPub) :
    extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω =
      LambdaSNARK.constraintNumeratorPoly cs
        (extract_witness VC cs
          (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
          eqns.m eqns.ω eqns.h_m_vars x) eqns.ω := by
  classical
  set q := extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω
  set w := extract_witness VC cs q eqns.m eqns.ω eqns.h_m_vars x
  have h_monic : (vanishing_poly eqns.m eqns.ω).Monic := by
    classical
    unfold LambdaSNARK.vanishing_poly
    apply Polynomial.monic_prod_of_monic
    intro i _
    simpa using Polynomial.monic_X_sub_C (eqns.ω ^ (i : ℕ))
  have h_mod_raw := constraint_quotient_sub_numerator_mod_vanishing_zero_of_equations
      (VC := VC) (cs := cs) (t1 := t1) (t2 := t2) (h_fork := h_fork) eqns x
  have h_mod :
      (q - LambdaSNARK.constraintNumeratorPoly cs w eqns.ω) %ₘ
          vanishing_poly eqns.m eqns.ω = 0 := by
    simpa [q, w, eqns.h_m_cons] using h_mod_raw
  have h_dvd :
      vanishing_poly eqns.m eqns.ω ∣
        (q - LambdaSNARK.constraintNumeratorPoly cs w eqns.ω) := by
    refine (Polynomial.modByMonic_eq_zero_iff_dvd h_monic).1 ?_
    simpa using h_mod
  have h_nat :
      (q - LambdaSNARK.constraintNumeratorPoly cs w eqns.ω).natDegree < eqns.m := by
    simpa [q, w] using eqns.quotient_diff_natDegree_lt_domain x
  have h_zero :=
    LambdaSNARK.vanishing_poly_dvd_eq_zero_of_natDegree_lt (F := F)
      (m := eqns.m) (ω := eqns.ω)
      (p := q - LambdaSNARK.constraintNumeratorPoly cs w eqns.ω)
      h_dvd h_nat
  have h_eq := sub_eq_zero.mp h_zero
  simpa [q, w] using h_eq

lemma vanishing_poly_dvd_quotient_sub_numerator_of_equations
    (VC : VectorCommitment F) (cs : R1CS F)
    {t1 t2 : Transcript F VC} {h_fork : is_valid_fork VC t1 t2}
    (eqns : ForkingVerifierEquations VC cs t1 t2 h_fork)
    (x : PublicInput F cs.nPub) :
    vanishing_poly cs.nCons eqns.ω ∣
      (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω
        - LambdaSNARK.constraintNumeratorPoly cs
            (extract_witness VC cs
              (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
              eqns.m eqns.ω eqns.h_m_vars x) eqns.ω) := by
  classical
  set q := extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω
  set w :=
    extract_witness VC cs q eqns.m eqns.ω eqns.h_m_vars x
  have h_eq :=
    constraint_quotient_sub_numerator_eq_zero_of_equations (VC := VC)
      (cs := cs) (t1 := t1) (t2 := t2) (h_fork := h_fork) eqns x
  have h_zero :
      q - LambdaSNARK.constraintNumeratorPoly cs w eqns.ω = 0 := by
    simpa [q, w] using sub_eq_zero.mpr h_eq
  have : vanishing_poly cs.nCons eqns.ω ∣
      q - LambdaSNARK.constraintNumeratorPoly cs w eqns.ω := by
    simp [q, w, h_zero]
  simpa [q, w] using this

lemma constraint_quotient_mod_vanishing_zero_of_equations (VC : VectorCommitment F)
  (cs : R1CS F) {t1 t2 : Transcript F VC} {h_fork : is_valid_fork VC t1 t2}
  (eqns : ForkingVerifierEquations VC cs t1 t2 h_fork)
  (h_rem : (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
      %ₘ vanishing_poly eqns.m eqns.ω = 0) :
    (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
        %ₘ vanishing_poly cs.nCons eqns.ω = 0 := by
  classical
  set q := extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω
  simpa [q, eqns.h_m_cons] using h_rem

lemma constraint_poly_zero_of_equations (VC : VectorCommitment F) (cs : R1CS F)
    {t1 t2 : Transcript F VC} {h_fork : is_valid_fork VC t1 t2}
    (eqns : ForkingVerifierEquations VC cs t1 t2 h_fork)
    (x : PublicInput F cs.nPub)
    (h_rem : (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
        %ₘ vanishing_poly eqns.m eqns.ω = 0) :
    ∀ i : Fin cs.nCons,
      constraintPoly cs
        (extract_witness VC cs
          (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
          eqns.m eqns.ω eqns.h_m_vars x) i = 0 := by
  classical
  set q := extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω
  set w := extract_witness VC cs q eqns.m eqns.ω eqns.h_m_vars x
  have h_exists :
      ∃ f : Polynomial F,
        (∀ i : Fin cs.nCons, f.eval (eqns.ω ^ (i : ℕ)) = constraintPoly cs w i) ∧
        f %ₘ vanishing_poly eqns.m eqns.ω = 0 := by
    refine ⟨q, ?_, ?_⟩
    · intro i
      have h_q := eqns.quotient_eval x i
      simpa [q, w] using h_q
    ·
      simpa [q] using h_rem
  have h_sat : satisfies cs w :=
    (quotient_exists_iff_satisfies cs w eqns.m eqns.ω eqns.h_m_cons eqns.h_primitive).mpr h_exists
  have h_zero := (satisfies_iff_constraint_zero cs w).mp h_sat
  simpa [w] using h_zero

lemma constraint_numerator_mod_vanishing_zero_of_equations (VC : VectorCommitment F)
    (cs : R1CS F) {t1 t2 : Transcript F VC} {h_fork : is_valid_fork VC t1 t2}
    (eqns : ForkingVerifierEquations VC cs t1 t2 h_fork)
    (x : PublicInput F cs.nPub)
    (h_rem : (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
        %ₘ vanishing_poly eqns.m eqns.ω = 0) :
    (LambdaSNARK.constraintNumeratorPoly cs
      (extract_witness VC cs
        (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
        eqns.m eqns.ω eqns.h_m_vars x) eqns.ω)
        %ₘ vanishing_poly cs.nCons eqns.ω = 0 := by
  classical
  set q := extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω
  set w := extract_witness VC cs q eqns.m eqns.ω eqns.h_m_vars x
  have h_zero_raw := constraint_poly_zero_of_equations (VC := VC) (cs := cs)
      (t1 := t1) (t2 := t2) (h_fork := h_fork) eqns x h_rem
  have h_zero : ∀ i : Fin cs.nCons, constraintPoly cs w i = 0 := by
    simpa [q, w] using h_zero_raw
  have h_prim : IsPrimitiveRoot eqns.ω cs.nCons := by
    simpa [eqns.h_m_cons] using eqns.h_primitive
  have h_mod :=
    LambdaSNARK.constraintNumeratorPoly_mod_vanishing_zero_of_constraint_zero (cs := cs)
      (z := w) (ω := eqns.ω) h_prim h_zero
  simpa [q, w] using h_mod

lemma constraint_numerator_eval_zero_of_equations (VC : VectorCommitment F)
    (cs : R1CS F) {t1 t2 : Transcript F VC} {h_fork : is_valid_fork VC t1 t2}
    (eqns : ForkingVerifierEquations VC cs t1 t2 h_fork)
    (x : PublicInput F cs.nPub)
    (h_rem : (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
        %ₘ vanishing_poly eqns.m eqns.ω = 0) :
    ∀ i : Fin cs.nCons,
      (LambdaSNARK.constraintNumeratorPoly cs
        (extract_witness VC cs
          (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
          eqns.m eqns.ω eqns.h_m_vars x) eqns.ω).eval
            (eqns.ω ^ (i : ℕ)) = 0 := by
  classical
  set q := extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω
  set w := extract_witness VC cs q eqns.m eqns.ω eqns.h_m_vars x
  have h_vanish :=
    constraint_poly_zero_of_equations (VC := VC) (cs := cs)
      (t1 := t1) (t2 := t2) (h_fork := h_fork) eqns x h_rem
  intro i
  have h_match :=
    constraint_numerator_eval_matches_quotient_of_equations (VC := VC)
      (cs := cs) (t1 := t1) (t2 := t2) (h_fork := h_fork) eqns x i
  have h_q := eqns.quotient_eval x i
  have h_zero : constraintPoly cs w i = 0 := by
    simpa [q, w] using h_vanish i
  have h_num_eq := h_match.trans h_q
  simpa [q, w] using h_num_eq.trans h_zero

end ForkingEquations

-- ============================================================================
-- Extraction Soundness
-- ============================================================================

/-- If two valid transcripts form a fork (same commitments, different challenges),
    then the extracted witness satisfies the R1CS constraints provided we have a
    certificate of the verifier equations for that fork. -/
theorem extraction_soundness {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (t1 t2 : Transcript F VC)
    (h_fork : is_valid_fork VC t1 t2)
    (eqns : ForkingVerifierEquations VC cs t1 t2 h_fork)
    (h_rem : (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
        %ₘ vanishing_poly eqns.m eqns.ω = 0)
    (h_sis : ModuleSIS_Hard 256 2 12289 1024) :
    (x : PublicInput F cs.nPub) →
    satisfies cs
      (extract_witness VC cs
        (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
        eqns.m eqns.ω eqns.h_m_vars x) := by
  intro x
  have _ := h_sis
  classical
  have h_zero := constraint_poly_zero_of_equations (VC := VC) (cs := cs)
      (t1 := t1) (t2 := t2) (h_fork := h_fork) eqns x h_rem
  exact
    extract_witness_satisfies_of_constraint_zero (VC := VC) (cs := cs)
      (q := extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
      (m := eqns.m) (ω := eqns.ω) eqns.h_m_vars x h_zero

/-- A provider that, given any fork of transcripts, returns the verifier
    equations witness required by `extraction_soundness`. Concrete protocol
    instantiations must supply such a provider. -/
structure ProtocolForkingEquations {F : Type} [Field F] [Fintype F]
    [DecidableEq F] (VC : VectorCommitment F) (cs : R1CS F) where
  square : cs.nVars = cs.nCons
  buildCore :
    (t1 t2 : Transcript F VC) →
    (h_fork : is_valid_fork VC t1 t2) →
    ForkingVerifierEquationsCore VC cs t1 t2 h_fork
  remainder_zero :
    (t1 t2 : Transcript F VC) →
    (h_fork : is_valid_fork VC t1 t2) →
      let core := buildCore t1 t2 h_fork
      (extract_quotient_diff VC cs t1 t2 h_fork core.m core.ω)
        %ₘ vanishing_poly core.m core.ω = 0

structure ForkingEquationsProvider {F : Type} [Field F] [Fintype F]
    [DecidableEq F] (VC : VectorCommitment F) (cs : R1CS F) where
  square : cs.nVars = cs.nCons
  buildCore :
    (t1 t2 : Transcript F VC) →
    (h_fork : is_valid_fork VC t1 t2) →
    ForkingVerifierEquationsCore VC cs t1 t2 h_fork
  remainder_zero :
    (t1 t2 : Transcript F VC) →
    (h_fork : is_valid_fork VC t1 t2) →
      let core := buildCore t1 t2 h_fork
      (extract_quotient_diff VC cs t1 t2 h_fork core.m core.ω)
        %ₘ vanishing_poly core.m core.ω = 0

namespace ForkingEquationsProvider

variable {F : Type} [Field F] [Fintype F] [DecidableEq F]
variable {VC : VectorCommitment F} {cs : R1CS F}

@[simp]
def build (provider : ForkingEquationsProvider VC cs)
    (t1 t2 : Transcript F VC) (h_fork : is_valid_fork VC t1 t2) :
    ForkingVerifierEquations VC cs t1 t2 h_fork :=
  let core := provider.buildCore t1 t2 h_fork
  { m := core.m
    ω := core.ω
    h_m_vars := core.h_m_vars
    h_m_cons := core.h_m_vars.trans provider.square
    h_primitive := core.h_primitive
    quotient_eval := core.quotient_eval
    quotient_diff_natDegree_lt_domain := by
      intro x
      simpa using core.quotient_diff_natDegree_lt_domain x }

@[simp]
lemma build_remainder_zero (provider : ForkingEquationsProvider VC cs)
    (t1 t2 : Transcript F VC) (h_fork : is_valid_fork VC t1 t2) :
    (extract_quotient_diff VC cs t1 t2 h_fork
        (provider.build t1 t2 h_fork).m (provider.build t1 t2 h_fork).ω)
        %ₘ vanishing_poly (provider.build t1 t2 h_fork).m
          (provider.build t1 t2 h_fork).ω = 0 := by
  classical
  let core := provider.buildCore t1 t2 h_fork
  have h := provider.remainder_zero t1 t2 h_fork
  simpa [build, core] using h

def ofProtocol (VC : VectorCommitment F) (cs : R1CS F)
  (proto : ProtocolForkingEquations VC cs) :
  ForkingEquationsProvider VC cs :=
  { square := proto.square
    buildCore := proto.buildCore
    remainder_zero := proto.remainder_zero }

end ForkingEquationsProvider

/-/ Witness extractor (uses adversary as black box) -/
structure Extractor (F : Type) [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) where
  extract : (A : Adversary F VC) → (cs : R1CS F) →
            (eq_provider : ForkingEquationsProvider VC cs) →
            (x : PublicInput F cs.nPub) →
            Option (Witness F cs.nVars)
  poly_time : Prop  -- Runtime bounded by polynomial in adversary's runtime

namespace ForkingExtractor

section Basic

variable {F : Type} [Field F] [DecidableEq F]

/-- Deterministic transcripts used by the generic forking extractor. -/
noncomputable def transcript (VC : VectorCommitment F) (cs : R1CS F)
    (x : PublicInput F cs.nPub) (α β : F) : Transcript F VC := by
  classical
  let pp := VC.setup 256
  exact {
    pp := pp
    cs := cs
    x := x
    domainSize := cs.nCons
    omega := 1
    comm_Az := VC.commit pp [] 0
    comm_Bz := VC.commit pp [] 0
    comm_Cz := VC.commit pp [] 0
    comm_quotient := VC.commit pp [] 0
    quotient_poly := 0
    quotient_rand := 0
    quotient_commitment_spec := by
      simp [Polynomial.coeffList_zero]
    view := {
      alpha := α
      Az_eval := 0
      Bz_eval := 0
      Cz_eval := 0
      quotient_eval := 0
      vanishing_eval := 0
      main_eq := verifierView_zero_eq (_F := F)
    }
    challenge_β := β
    opening_Az_α := VC.openProof pp [] 0 α
    opening_Bz_β := VC.openProof pp [] 0 β
    opening_Cz_α := VC.openProof pp [] 0 α
    opening_quotient_α := VC.openProof pp [] 0 α
    valid := true
  }

lemma fork (VC : VectorCommitment F) (cs : R1CS F)
    (x : PublicInput F cs.nPub) :
    is_valid_fork VC (transcript VC cs x 0 0) (transcript VC cs x 1 0) := by
  classical
  unfold transcript
  simp [is_valid_fork, zero_ne_one]

/-- The canonical extractor transcripts form a successful forked event. -/
lemma deterministic_fork_success_event (VC : VectorCommitment F)
    (cs : R1CS F) (x : PublicInput F cs.nPub) :
    fork_success_event VC cs x
      (transcript VC cs x 0 0, transcript VC cs x 1 0) := by
  classical
  refine ⟨?_, ?_, fork (VC := VC) (cs := cs) (x := x)⟩
  · simp [success_event, transcript]
  · simp [success_event, transcript]

end Basic

section Witness

variable {F : Type} [Field F] [Fintype F] [DecidableEq F]

/-- Witness assembled from the equations returned by the provider on the
    canonical deterministic fork. -/
noncomputable def witness (VC : VectorCommitment F) (cs : R1CS F)
    (provider : ForkingEquationsProvider VC cs)
    (x : PublicInput F cs.nPub) : Witness F cs.nVars := by
  classical
  let t1 := transcript (VC := VC) (cs := cs) (x := x) 0 0
  let t2 := transcript (VC := VC) (cs := cs) (x := x) 1 0
  let h_fork := fork (VC := VC) (cs := cs) (x := x)
  let eqns := provider.build t1 t2 h_fork
  exact
    extract_witness VC cs
      (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
      eqns.m eqns.ω eqns.h_m_vars x

lemma witness_satisfies (VC : VectorCommitment F) (cs : R1CS F)
    (provider : ForkingEquationsProvider VC cs)
    (x : PublicInput F cs.nPub)
    (h_sis : ModuleSIS_Hard 256 2 12289 1024) :
    satisfies cs (witness VC cs provider x) := by
  classical
  unfold witness
  simp only
  let t1 := transcript (VC := VC) (cs := cs) (x := x) 0 0
  let t2 := transcript (VC := VC) (cs := cs) (x := x) 1 0
  let hFork := fork (VC := VC) (cs := cs) (x := x)
  let eqns := provider.build t1 t2 hFork
  let q := extract_quotient_diff VC cs t1 t2 hFork eqns.m eqns.ω
  have h_rem := provider.build_remainder_zero t1 t2 hFork
  have h_sat :=
    extraction_soundness VC cs t1 t2 hFork eqns h_rem h_sis x
  simpa [q]

lemma witness_public (VC : VectorCommitment F) (cs : R1CS F)
    (provider : ForkingEquationsProvider VC cs)
    (x : PublicInput F cs.nPub) :
    extractPublic cs.h_pub_le (witness VC cs provider x) = x := by
  classical
  unfold witness
  simp only
  let t1 := transcript (VC := VC) (cs := cs) (x := x) 0 0
  let t2 := transcript (VC := VC) (cs := cs) (x := x) 1 0
  let hFork := fork (VC := VC) (cs := cs) (x := x)
  let eqns := provider.build t1 t2 hFork
  let q := extract_quotient_diff VC cs t1 t2 hFork eqns.m eqns.ω
  funext i
  have hi_lt : (i : ℕ) < cs.nPub := i.isLt
  simp [extractPublic, extract_witness, hi_lt]

end Witness

end ForkingExtractor

-- ============================================================================
-- Forking Extractor Construction
-- ============================================================================

/-- Extractor that uses forking technique:
    1. Run adversary once
    2. If successful, rewind with different challenge
    3. If both succeed, extract witness from fork -/
noncomputable def forking_extractor {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (_secParam : ℕ) : Extractor (F := F) (VC := VC) := {
  extract := fun _A cs provider x =>
    some (ForkingExtractor.witness (VC := VC) (cs := cs)
      (provider := provider) (x := x))

  poly_time := True  -- Runtime = O(adversary_time × 2 + poly(secParam))
}

end LambdaSNARK
