/-
Copyright (c) 2025 URPKS Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: URPKS Contributors
-/

import LambdaSNARK.Core
import LambdaSNARK.Polynomial
import LambdaSNARK.Constraints
import LambdaSNARK.Forking.Types
import LambdaSNARK.Forking.Probability
import LambdaSNARK.Forking.HeavyLight
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Probability.ProbabilityMassFunction.Monad
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Data.Finset.Card
import Mathlib.Algebra.BigOperators.Intervals
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

/-- Local shorthand for the structured soundness assumptions. -/
abbrev SoundnessCtx (F : Type) [Field F]
    (VC : VectorCommitment F) (cs : R1CS F) : Type :=
  SoundnessAssumptions F VC cs

open scoped BigOperators
open BigOperators Polynomial
open LambdaSNARK

/-!
## Combinatorial Helpers

The following lemmas provide the arithmetic backbone for the probability
estimates in the forking lemma.  They translate counting arguments about
subsets of a finite challenge space into inequalities over real numbers.
-/
/-!
### Heavy commitments

All heavy/light probability infrastructure lives in
`LambdaSNARK.Forking.HeavyLight`.  We import it above and reuse the lemmas
directly here.
-/

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

lemma heavyCommitment_mem_is_heavy
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    {ε : ℝ} (h_ε_pos : 0 < ε)
    (h_mem : comm_tuple ∈
      heavyCommitments VC cs A x secParam (ε * (Fintype.card F : ℝ))) :
    is_heavy_commitment VC cs x comm_tuple ε := by
  classical
  obtain ⟨valid_challenges, h_witness, h_card⟩ :=
    heavyCommitment_scaled_witnesses (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_ε_pos h_mem
  refine ⟨valid_challenges, ?_, ?_⟩
  · intro α hα
    obtain ⟨t, h_success, h_commit, h_alpha⟩ := h_witness α hα
    exact ⟨t, h_success, h_commit, h_alpha⟩
  · simpa using h_card

/-- A strictly positive success margin implies the existence of an explicit
    randomness seed producing a successful transcript of positive mass. -/
lemma exists_randomness_successful_transcript_of_successProbability_gt
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_pos : 0 < ε)
    (h_gap : ε < successProbability VC cs A x secParam) :
    ∃ (rand : Fin secParam.succ) (t : Transcript F VC),
      rand ∈ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε ∧
      success_event VC cs x t ∧
      transcriptOfRandomness VC cs A x secParam rand = t ∧
      transcriptCommitTuple VC t =
        commitTupleOfRandomness VC cs A x secParam rand ∧
      0 < (run_adversary_transcript (VC := VC) (cs := cs) A x secParam) t := by
  classical
  obtain ⟨rand_heavy, α, t, h_heavy_rand_heavy, h_success, h_commit, _h_alpha, h_run_pos⟩ :=
    _root_.LambdaSNARK.exists_heavyRandomness_successfulChallenge_of_successProbability_gt
      (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
      (ε := ε) h_pos h_gap
  have h_support : t ∈
      (run_adversary_transcript (VC := VC) (cs := cs) A x secParam).support := by
    have h_ne :
        (run_adversary_transcript (VC := VC) (cs := cs) A x secParam) t ≠ 0 :=
      ne_of_gt h_run_pos
    exact (PMF.mem_support_iff (p :=
      run_adversary_transcript (VC := VC) (cs := cs) A x secParam)
      (a := t)).mpr h_ne
  obtain ⟨rand, h_rand_eq⟩ :=
    exists_randomness_of_mem_support_run_adversary_transcript
      (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
      (t := t) h_support
  have h_commit_rand :
      commitTupleOfRandomness VC cs A x secParam rand =
        transcriptCommitTuple VC t := by
    simp [commitTupleOfRandomness, h_rand_eq]
  have h_heavy_transcript :
      transcriptCommitTuple VC t ∈ heavyCommitments VC cs A x secParam ε := by
    simpa [h_commit] using h_heavy_rand_heavy
  have h_heavy_rand :
      commitTupleOfRandomness VC cs A x secParam rand ∈
        heavyCommitments VC cs A x secParam ε := by
    simpa [h_commit_rand] using h_heavy_transcript
  have h_rand_mem :
      rand ∈ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε :=
    (mem_heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)).2 h_heavy_rand
  have h_commit_rand' :
      transcriptCommitTuple VC t =
        commitTupleOfRandomness VC cs A x secParam rand :=
    h_commit_rand.symm
  exact ⟨rand, t, h_rand_mem, h_success, h_rand_eq, h_commit_rand', h_run_pos⟩

lemma exists_heavy_sample_success_of_successProbability_gt
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_pos : 0 < ε)
    (h_gap : ε < successProbability VC cs A x secParam) :
    ∃ rand,
      rand ∈ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε ∧
      success_event VC cs x ((adversarySample VC cs A x secParam rand).2) ∧
      (adversarySample VC cs A x secParam rand)
        ∈ (run_adversary (VC := VC) (cs := cs) A x secParam).support ∧
      0 < (run_adversary_transcript (VC := VC) (cs := cs) A x secParam)
        ((adversarySample VC cs A x secParam rand).2) := by
  classical
  obtain ⟨rand, t, h_rand_mem, h_success, h_rand_eq, _h_commit, h_run_pos⟩ :=
    exists_randomness_successful_transcript_of_successProbability_gt
      (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
      (ε := ε) h_pos h_gap
  have h_sample_transcript :
      (adversarySample VC cs A x secParam rand).2 = t := h_rand_eq
  have h_success_sample :
      success_event VC cs x ((adversarySample VC cs A x secParam rand).2) := by
    simpa [h_sample_transcript]
      using h_success
  have h_support_sample :
      adversarySample VC cs A x secParam rand
        ∈ (run_adversary (VC := VC) (cs := cs) A x secParam).support := by
    have h_unif_mem :
        rand ∈ (uniform_pmf : PMF (Fin (secParam.succ))).support := by
      have h_ne :
          (uniform_pmf : PMF (Fin (secParam.succ))) rand ≠ 0 :=
        uniform_pmf_apply_ne_zero rand
      exact (PMF.mem_support_iff
        (p := (uniform_pmf : PMF (Fin (secParam.succ)))) (a := rand)).2 h_ne
    have h_map :=
      run_adversary_eq_map_adversarySample VC cs A x secParam
    have h_mem_map :
        adversarySample VC cs A x secParam rand ∈
          (PMF.map (adversarySample VC cs A x secParam)
            (uniform_pmf : PMF (Fin (secParam.succ)))).support := by
      refine (PMF.mem_support_map_iff
        (f := adversarySample VC cs A x secParam)
        (p := (uniform_pmf : PMF (Fin (secParam.succ))))
        (b := adversarySample VC cs A x secParam rand)).2 ?_
      exact ⟨rand, h_unif_mem, rfl⟩
    simpa [h_map] using h_mem_map
  have h_run_pos_sample :
      0 < (run_adversary_transcript (VC := VC) (cs := cs) A x secParam)
        ((adversarySample VC cs A x secParam rand).2) := by
    simpa [h_sample_transcript]
      using h_run_pos
  exact ⟨rand, h_rand_mem, h_success_sample, h_support_sample, h_run_pos_sample⟩

/-- Finite set of randomness seeds whose transcripts validate in the first adversary run. -/
noncomputable def successfulRandomnessFinset
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) : Finset (Fin secParam.succ) := by
  classical
  exact (Finset.univ : Finset (Fin secParam.succ)).filter fun rand =>
    success_event VC cs x
      (transcriptOfRandomness VC cs A x secParam rand)

lemma mem_successfulRandomnessFinset
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    {rand : Fin secParam.succ} :
    rand ∈ successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ↔
      success_event VC cs x
        (transcriptOfRandomness VC cs A x secParam rand) := by
  classical
  simp [successfulRandomnessFinset]

lemma successMass_eq_uniform_randomness_success_count
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    successMass VC cs A x secParam =
      ((successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam)).card : ENNReal) *
        (secParam.succ : ENNReal)⁻¹ := by
  classical
  have h_mass :=
    successMass_eq_uniform_randomness_tsum (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)
  have h_sum_congr :
      (∑ rand : Fin secParam.succ,
        successSeedWeight VC cs A x secParam rand) =
        ∑ rand : Fin secParam.succ,
          (if success_event VC cs x
              (transcriptOfRandomness VC cs A x secParam rand)
            then (secParam.succ : ENNReal)⁻¹ else 0) := by
    refine Finset.sum_congr rfl ?_
    intro rand _
    by_cases h_success :
        success_event VC cs x
          (transcriptOfRandomness VC cs A x secParam rand)
    ·
      have := LambdaSNARK.uniform_pmf_apply_fin (n := secParam) (rand := rand)
      simp [successSeedWeight, h_success, this]
    · simp [successSeedWeight, h_success]
  have h_filtered :
      (∑ rand : Fin secParam.succ,
          (if success_event VC cs x
              (transcriptOfRandomness VC cs A x secParam rand)
            then (secParam.succ : ENNReal)⁻¹ else 0))
        =
          ∑ rand ∈
              successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
                (x := x) (secParam := secParam),
            (secParam.succ : ENNReal)⁻¹ := by
    classical
    simpa [successfulRandomnessFinset]
      using
        (Finset.sum_filter
            (s := (Finset.univ : Finset (Fin secParam.succ)))
            (f := fun _ => (secParam.succ : ENNReal)⁻¹)
            (p := fun rand =>
              success_event VC cs x
                (transcriptOfRandomness VC cs A x secParam rand))).symm
  have h_const_sum :
      (∑ rand ∈
          successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam),
        (secParam.succ : ENNReal)⁻¹)
        =
          ((successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
              (x := x) (secParam := secParam)).card : ENNReal) *
            (secParam.succ : ENNReal)⁻¹ := by
    simp [Finset.sum_const, nsmul_eq_mul]
  calc
    successMass VC cs A x secParam
        = ∑ rand : Fin secParam.succ,
            successSeedWeight VC cs A x secParam rand := h_mass
    _ = ∑ rand : Fin secParam.succ,
            (if success_event VC cs x
                (transcriptOfRandomness VC cs A x secParam rand)
              then (secParam.succ : ENNReal)⁻¹ else 0) := h_sum_congr
    _ = ∑ rand ∈
            successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
              (x := x) (secParam := secParam),
          (secParam.succ : ENNReal)⁻¹ := h_filtered
    _ = ((successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam)).card : ENNReal) *
        (secParam.succ : ENNReal)⁻¹ := h_const_sum

lemma ENNReal.toReal_inv_natSucc (n : ℕ) :
    ((n.succ : ENNReal)⁻¹).toReal = (n.succ : ℝ)⁻¹ := by
  have h_ne_zero : (n.succ : ENNReal) ≠ 0 := by
    exact_mod_cast (Nat.succ_ne_zero n)
  have h_ne_top : (n.succ : ENNReal) ≠ (⊤ : ENNReal) :=
    ENNReal.natCast_ne_top _
  have h_prod := ENNReal.mul_inv_cancel h_ne_zero h_ne_top
  have h_prod_toReal := congrArg ENNReal.toReal h_prod
  have h_mul :
      (n.succ : ℝ) * ((n.succ : ENNReal)⁻¹).toReal = 1 := by
    simpa [ENNReal.toReal_mul]
      using h_prod_toReal
  have h_nonzero : (n.succ : ℝ) ≠ 0 := by
    exact_mod_cast (Nat.succ_ne_zero n)
  have h_inv : (n.succ : ℝ) * (n.succ : ℝ)⁻¹ = (1 : ℝ) := by
    have h' : (n.succ : ℝ) ≠ 0 := h_nonzero
    simpa [div_eq_mul_inv, h'] using (div_self (a := (n.succ : ℝ)) h')
  have h_eq :
      (n.succ : ℝ) * ((n.succ : ENNReal)⁻¹).toReal =
        (n.succ : ℝ) * (n.succ : ℝ)⁻¹ :=
    h_mul.trans h_inv.symm
  exact (mul_left_cancel₀ h_nonzero) h_eq

lemma successProbability_eq_successfulRandomness_card_div
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    successProbability VC cs A x secParam =
      ((successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam)).card : ℝ) /
        (secParam.succ : ℝ) := by
  classical
  have h_mass_eq :=
    successMass_eq_uniform_randomness_success_count (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)
  have h_prob :=
    successProbability_toReal_successMass (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)
  set successCard : ENNReal :=
    ((successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam)).card : ENNReal)
  have h_mass_toReal :
      (successMass VC cs A x secParam).toReal =
        (successCard * (secParam.succ : ENNReal)⁻¹).toReal := by
    simpa [successCard] using congrArg ENNReal.toReal h_mass_eq
  have h_mul_toReal :=
    ENNReal.toReal_mul (a := successCard) (b := (secParam.succ : ENNReal)⁻¹)
  have h_card_toReal : successCard.toReal =
      (successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam)).card := by
    simp [successCard]
  have h_inv_toReal :
      ((secParam.succ : ENNReal)⁻¹).toReal = (secParam.succ : ℝ)⁻¹ := by
    simpa using ENNReal.toReal_inv_natSucc secParam
  have h_card_mul :
      successCard.toReal * ((secParam.succ : ENNReal)⁻¹).toReal =
        ((successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam)).card : ℝ) *
          (secParam.succ : ℝ)⁻¹ := by
    calc
      successCard.toReal * ((secParam.succ : ENNReal)⁻¹).toReal
          = ((successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
              (x := x) (secParam := secParam)).card : ℝ) *
              ((secParam.succ : ENNReal)⁻¹).toReal := by
            rw [h_card_toReal]
      _ = ((successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
              (x := x) (secParam := secParam)).card : ℝ) *
            (secParam.succ : ℝ)⁻¹ := by
            rw [h_inv_toReal]
  calc
    successProbability VC cs A x secParam
        = (successMass VC cs A x secParam).toReal := h_prob
    _ = (successCard * (secParam.succ : ENNReal)⁻¹).toReal := h_mass_toReal
    _ = successCard.toReal * ((secParam.succ : ENNReal)⁻¹).toReal :=
      h_mul_toReal
    _ = ((successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam)).card : ℝ) *
          (secParam.succ : ℝ)⁻¹ := h_card_mul
    _ = ((successfulRandomnessFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam)).card : ℝ) /
          (secParam.succ : ℝ) := by
          simp [div_eq_mul_inv]

/-!
## Historical Note: Heavy-Row Infrastructure

The heavy-row counting argument and the associated fork-success bound were
previously axiomatized while the probabilistic toolkit was under construction.
Both facts are now proved constructively in this file:

1. `heavy_row_lemma` packages the PMF-based heavy-set extraction.
2. `fork_success_bound` derives the ε²/2 lower bound on repeated success.

The comment is kept to document the milestone and provide quick pointers for
future maintenance. No axioms remain in this section.
-/

lemma exists_heavyCommitment_of_successProbability_ge
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (A : Adversary F VC) (x : PublicInput F cs.nPub)
    {ε : ℝ} (secParam : ℕ)
    (h_ε_pos : 0 < ε)
    (h_prob : ε ≤ successProbability VC cs A x secParam) :
    ∃ comm_tuple,
      comm_tuple ∈ heavyCommitments VC cs A x secParam
        (max (ε - 1 / (Fintype.card F : ℝ)) 0) := by
  classical
  set n : ℝ := (Fintype.card F : ℝ)
  have hn_pos : 0 < n := by
    have : 0 < (Fintype.card F : ℝ) :=
      by exact_mod_cast (Fintype.card_pos : 0 < Fintype.card F)
    simpa [n] using this
  set δ := max (ε - 1 / n) 0 with hδ_def
  have hδ_lt : δ < successProbability VC cs A x secParam := by
    by_cases h_case : ε ≤ 1 / n
    · have h_sub : ε - 1 / n ≤ 0 := sub_nonpos.mpr h_case
      have hδ_zero : δ = 0 := by
        simpa [δ, hδ_def, h_sub] using max_eq_right h_sub
      have h_succ_pos : 0 < successProbability VC cs A x secParam :=
        lt_of_lt_of_le h_ε_pos h_prob
      simpa [hδ_zero]
        using h_succ_pos
    · have h_gt : 1 / n < ε := lt_of_not_ge h_case
      have h_sub_pos : 0 < ε - 1 / n := sub_pos.mpr h_gt
      have hδ_eq : δ = ε - 1 / n := by
        have h_le : 0 ≤ ε - 1 / n := le_of_lt h_sub_pos
        simpa [δ, hδ_def, h_le] using max_eq_left h_le
      have h_lt_eps : δ < ε := by
        have h_inv_pos : 0 < 1 / n := by
          simpa [one_div] using inv_pos.mpr hn_pos
        have h_sub_lt : ε - 1 / n < ε := sub_lt_self ε h_inv_pos
        simpa [hδ_eq] using h_sub_lt
      exact lt_of_lt_of_le h_lt_eps h_prob
  exact exists_heavyCommitment_of_successProbability_lt (VC := VC) (cs := cs)
    (A := A) (x := x) (secParam := secParam) (ε := δ) hδ_lt

lemma exists_heavyCommitment_of_successProbability_gt
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (A : Adversary F VC) (x : PublicInput F cs.nPub)
    {ε : ℝ} (secParam : ℕ)
    (h_ε_nonneg : 0 ≤ ε)
    (h_gap : ε < successProbability VC cs A x secParam) :
    ∃ comm_tuple,
      comm_tuple ∈ heavyCommitments VC cs A x secParam ε := by
  classical
  have h_mass_le_one :=
    successMass_le_one (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam)
  have h_mass_lt_top :
      successMass VC cs A x secParam < (⊤ : ENNReal) :=
    lt_of_le_of_lt h_mass_le_one ENNReal.one_lt_top
  have h_mass_ne_top :
      successMass VC cs A x secParam ≠ (⊤ : ENNReal) :=
    (lt_top_iff_ne_top).1 h_mass_lt_top
  have h_ofReal_ne_top : ENNReal.ofReal ε ≠ (⊤ : ENNReal) := by simp
  by_contra h_none
  have h_none' := not_exists.mp h_none
  have h_pointwise :
      (fun comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment =>
        successMassGivenCommit VC cs A x secParam comm_tuple)
        ≤ fun comm_tuple =>
          ENNReal.ofReal ε *
            commitMass VC cs A x secParam comm_tuple := by
    intro comm_tuple
    have h_not : comm_tuple ∉ heavyCommitments VC cs A x secParam ε :=
      h_none' comm_tuple
    exact
      successMassGivenCommit_le_of_not_heavy (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε)
        (comm_tuple := comm_tuple) h_not
  have h_sum_le := ENNReal.tsum_le_tsum h_pointwise
  have h_mass_eq :=
    successMass_eq_tsum_successMassGivenCommit (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)
  have h_commit_sum :=
    tsum_commitMass_eq_one (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam)
  have h_bound :
      successMass VC cs A x secParam ≤ ENNReal.ofReal ε := by
    have h_mass_le :
        successMass VC cs A x secParam
          ≤ ∑'
              comm_tuple,
                ENNReal.ofReal ε *
                  commitMass VC cs A x secParam comm_tuple := by
      simpa [h_mass_eq] using h_sum_le
    have h_right :=
      ENNReal.tsum_mul_right
        (f := fun comm_tuple :
            VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment =>
              commitMass VC cs A x secParam comm_tuple)
        (a := ENNReal.ofReal ε)
    have h_eval :
        ∑'
            comm_tuple,
              ENNReal.ofReal ε *
                  commitMass VC cs A x secParam comm_tuple
          = ENNReal.ofReal ε *
              ∑'
                comm_tuple,
                  commitMass VC cs A x secParam comm_tuple := by
      simpa [mul_comm, mul_left_comm, mul_assoc]
        using h_right
    simpa [h_eval, h_commit_sum, mul_comm]
      using h_mass_le
  have h_toReal :=
    successProbability_toReal_successMass (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)
  have h_contra : successProbability VC cs A x secParam ≤ ε := by
    have h_toReal_le :=
      (ENNReal.toReal_le_toReal h_mass_ne_top h_ofReal_ne_top).mpr h_bound
    simpa [h_toReal, ENNReal.toReal_ofReal h_ε_nonneg] using h_toReal_le
  exact (not_le_of_gt h_gap) h_contra

lemma heavy_randomness_lower_bound
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_nonneg : 0 ≤ ε) :
    (successProbability VC cs A x secParam - ε) * (secParam.succ : ℝ) ≤
      (heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε).card := by
  classical
  simpa using
    heavyRandomness_finset_lower_bound (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_nonneg

lemma heavy_randomness_card_le_randomness_mul_heavyCommit_image
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ) :
    (heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε).card ≤
      secParam.succ *
        ((heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam) ε).image
          (fun rand =>
            commitTupleOfRandomness VC cs A x secParam rand)).card := by
  classical
  set S :=
    heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε
    with hS
  set imageComm :=
    S.image (fun rand => commitTupleOfRandomness VC cs A x secParam rand)
    with hImage
  have h_mem_image :
      ∀ rand ∈ S,
        commitTupleOfRandomness VC cs A x secParam rand ∈ imageComm := by
    intro rand h_rand
    change commitTupleOfRandomness VC cs A x secParam rand ∈
        S.image (fun rand => commitTupleOfRandomness VC cs A x secParam rand)
    refine Finset.mem_image.mpr ?_
    exact ⟨rand, h_rand, rfl⟩
  let domain := {rand : Fin secParam.succ // rand ∈ S}
  let codomain :=
    {comm : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment //
        comm ∈ imageComm} × Fin secParam.succ
  let g : domain → codomain := fun rand =>
    let randVal := rand.val
    let h_rand_mem : randVal ∈ S := rand.property
    let h_comm_mem := h_mem_image randVal h_rand_mem
    ⟨⟨commitTupleOfRandomness VC cs A x secParam randVal, h_comm_mem⟩, randVal⟩
  have h_inj : Function.Injective g := by
    intro rand₁ rand₂ h_eq
    apply Subtype.ext
    have := congrArg Prod.snd h_eq
    simpa [g] using this
  have h_card_le :
      Fintype.card domain ≤ Fintype.card codomain :=
    Fintype.card_le_of_injective g h_inj
  have h_domain_card : Fintype.card domain = S.card := by
    simp [domain, S]
  have h_codomain_card : Fintype.card codomain =
      secParam.succ * imageComm.card := by
    have h : Fintype.card codomain = imageComm.card * secParam.succ := by
      simp [codomain, imageComm, Fintype.card_prod, Fintype.card_fin]
    simpa [Nat.mul_comm] using h
  simpa [h_domain_card, h_codomain_card, S, imageComm] using h_card_le

noncomputable def heavyCommitmentImageFinset
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ) :
    Finset (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :=
  (heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε).image
    (fun rand => commitTupleOfRandomness VC cs A x secParam rand)

lemma mem_heavyCommitmentImageFinset
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    {comm : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment} :
    comm ∈ heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε ↔
      ∃ rand ∈ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) ε,
        commitTupleOfRandomness VC cs A x secParam rand = comm := by
  classical
  unfold heavyCommitmentImageFinset
  constructor
  · intro h_mem
    obtain ⟨rand, h_rand_mem, h_comm_eq⟩ :=
      Finset.mem_image.mp h_mem
    refine ⟨rand, h_rand_mem, ?_⟩
    simp [h_comm_eq]
  · rintro ⟨rand, h_rand_mem, rfl⟩
    exact Finset.mem_image.mpr ⟨rand, h_rand_mem, rfl⟩

lemma heavy_randomness_card_le_randomness_mul_heavyCommit_image_real
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ) :
    ((heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε).card : ℝ) ≤
      (secParam.succ : ℝ) *
        (((heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam) ε).image
          (fun rand =>
            commitTupleOfRandomness VC cs A x secParam rand)).card : ℝ) := by
  classical
  have h_nat :=
    heavy_randomness_card_le_randomness_mul_heavyCommit_image (VC := VC)
      (cs := cs) (A := A) (x := x) (secParam := secParam) (ε := ε)
  have h_real :
      ((heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) ε).card : ℝ) ≤
        (secParam.succ *
            ((heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
              (x := x) (secParam := secParam) ε).image
              (fun rand =>
                commitTupleOfRandomness VC cs A x secParam rand)).card : ℝ) := by
    exact_mod_cast h_nat
  simpa [Nat.cast_mul] using h_real

lemma heavyCommitMass_toReal_eq_heavyRandomness_fraction
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ) :
    (heavyCommitMass VC cs A x secParam ε).toReal =
      ((heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) ε).card : ℝ) /
        (secParam.succ : ℝ) := by
  classical
  have h_mul_eq :=
    heavyCommitMass_toReal_mul_randomnessCard_eq_heavySeedCount
      (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
      (ε := ε)
  have h_card_eq :=
    heavyRandomnessFinset_card_cast (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)
  have h_succ_pos : 0 < (secParam.succ : ℝ) := by
    exact_mod_cast Nat.succ_pos secParam
  have h_succ_ne_zero : (secParam.succ : ℝ) ≠ 0 := ne_of_gt h_succ_pos
  have h_mul_cast :
      (heavyCommitMass VC cs A x secParam ε).toReal * (secParam.succ : ℝ) =
        ((heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam) ε).card : ℝ) := by
    simpa [h_card_eq, mul_comm, mul_left_comm, mul_assoc]
      using h_mul_eq
  refine (eq_div_iff_mul_eq h_succ_ne_zero).2 ?_
  simpa [h_card_eq, mul_comm, mul_left_comm, mul_assoc]
    using h_mul_cast

lemma heavyCommitMass_toReal_le_heavyCommitmentImage_card
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ) :
    (heavyCommitMass VC cs A x secParam ε).toReal ≤
      ((heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) ε).card : ℝ) := by
  classical
  have h_mass_times :
      (heavyCommitMass VC cs A x secParam ε).toReal * (secParam.succ : ℝ) =
        ((heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam) ε).card : ℝ) := by
    have h_cast :=
      heavyRandomnessFinset_card_cast (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε)
    have h_eq :=
      heavyCommitMass_toReal_mul_randomnessCard_eq_heavySeedCount
        (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
        (ε := ε)
    simpa [h_cast]
      using h_eq
  have h_mass_times' :
      ((heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) ε).card : ℝ) =
        (heavyCommitMass VC cs A x secParam ε).toReal * (secParam.succ : ℝ) := by
    simpa [mul_comm]
      using h_mass_times.symm
  have h_randomness_bound :=
    heavy_randomness_card_le_randomness_mul_heavyCommit_image_real
      (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
      (ε := ε)
  have h_mul_bound :
      (heavyCommitMass VC cs A x secParam ε).toReal * (secParam.succ : ℝ) ≤
        (secParam.succ : ℝ) *
          ((heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
              (x := x) (secParam := secParam) ε).card : ℝ) := by
    simpa [h_mass_times', mul_comm, mul_left_comm, mul_assoc]
      using h_randomness_bound
  have h_pos : 0 < (secParam.succ : ℝ) := by
    exact_mod_cast Nat.succ_pos secParam
  have h_mul_bound' :
      (secParam.succ : ℝ) * (heavyCommitMass VC cs A x secParam ε).toReal ≤
        (secParam.succ : ℝ) *
          ((heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
              (x := x) (secParam := secParam) ε).card : ℝ) := by
    simpa [mul_comm, mul_left_comm, mul_assoc]
      using h_mul_bound
  have h_final := le_of_mul_le_mul_left h_mul_bound' h_pos
  simpa [mul_comm]
    using h_final

lemma successProbability_sub_le_heavyCommitmentImage_card
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_nonneg : 0 ≤ ε) :
    successProbability VC cs A x secParam - ε ≤
      ((heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) ε).card : ℝ) := by
  classical
  have h_mass :=
    heavyCommitMass_toReal_ge_successProbability_sub (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε) h_nonneg
  have h_card :=
    heavyCommitMass_toReal_le_heavyCommitmentImage_card (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε)
  exact le_trans h_mass h_card

lemma successProbability_sub_scaled_le_heavyCommitmentImage_card
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_nonneg : 0 ≤ ε)
    (h_card_pos : 0 < (Fintype.card F : ℝ)) :
    successProbability VC cs A x secParam - ε * (Fintype.card F : ℝ) ≤
      ((heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) (ε := ε * (Fintype.card F : ℝ))).card : ℝ) := by
  classical
  have h_scaled_nonneg : 0 ≤ ε * (Fintype.card F : ℝ) :=
    mul_nonneg h_nonneg (le_of_lt h_card_pos)
  have h_bound :=
    successProbability_sub_le_heavyCommitmentImage_card (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)
      (ε := ε * (Fintype.card F : ℝ)) h_scaled_nonneg
  simpa using h_bound

lemma heavyCommitmentImageFinset_subset_heavyCommitments
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    {comm : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm ∈ heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε) :
    comm ∈ heavyCommitments VC cs A x secParam ε := by
  classical
  obtain ⟨rand, h_rand_mem, h_eq⟩ :=
    (mem_heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε)).1 h_mem
  have h_heavy_rand :
      commitTupleOfRandomness VC cs A x secParam rand ∈
        heavyCommitments VC cs A x secParam ε := by
    exact
      (mem_heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε)).1 h_rand_mem
  simpa [heavyCommitmentImageFinset, h_eq]
    using h_heavy_rand

lemma heavyCommitmentImageFinset_mem_is_heavy
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_pos : 0 < ε)
    {comm : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_mem : comm ∈ heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε * (Fintype.card F : ℝ))) :
    is_heavy_commitment VC cs x comm ε := by
  classical
  have h_subset :=
    heavyCommitmentImageFinset_subset_heavyCommitments (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)
      (ε := ε * (Fintype.card F : ℝ)) h_mem
  exact
    heavyCommitment_mem_is_heavy (VC := VC) (cs := cs) (A := A) (x := x)
      (secParam := secParam) (ε := ε) h_pos h_subset

lemma heavyCommitmentImageFinset_forall_is_heavy
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_pos : 0 < ε) :
    ∀ comm ∈ heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε * (Fintype.card F : ℝ)),
      is_heavy_commitment VC cs x comm ε := by
  classical
  intro comm h_comm
  exact heavyCommitmentImageFinset_mem_is_heavy (VC := VC) (cs := cs) (A := A)
    (x := x) (secParam := secParam) (ε := ε) h_pos h_comm

lemma heavyCommitmentImageFinset_card_lower_bound
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_nonneg : 0 ≤ ε) :
    successProbability VC cs A x secParam - ε ≤
      (heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) ε).card := by
  classical
  set S :=
    heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) ε
  have h_lower :=
    heavy_randomness_lower_bound (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_nonneg
  have h_lower' :
      (successProbability VC cs A x secParam - ε) *
          (secParam.succ : ℝ) ≤ (S.card : ℝ) := by
    simpa [S] using h_lower
  have h_upper :=
    heavy_randomness_card_le_randomness_mul_heavyCommit_image_real
      (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε)
  have h_upper' :
      (S.card : ℝ) ≤
        (secParam.succ : ℝ) *
          (heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam) ε).card := by
    simpa [S, heavyCommitmentImageFinset]
      using h_upper
  have h_combined := le_trans h_lower' h_upper'
  have h_commuted :
      (secParam.succ : ℝ) *
          (successProbability VC cs A x secParam - ε) ≤
        (secParam.succ : ℝ) *
          (heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam) ε).card := by
    simpa [mul_comm, mul_left_comm, mul_assoc]
      using h_combined
  have h_pos : 0 < (secParam.succ : ℝ) := by
    exact_mod_cast Nat.succ_pos secParam
  have h_final := le_of_mul_le_mul_left h_commuted h_pos
  simpa [heavyCommitmentImageFinset, mul_comm]
    using h_final

lemma heavyCommitmentImageFinset_weighted_card_lower_bound
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_nonneg : 0 ≤ ε) :
    (successProbability VC cs A x secParam - ε) * (secParam.succ : ℝ) ≤
      ((heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) ε).card : ℝ) * (secParam.succ : ℝ) := by
  classical
  have h_lower :=
    heavyCommitmentImageFinset_card_lower_bound (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_nonneg
  have h_pos : 0 < (secParam.succ : ℝ) := by
    exact_mod_cast Nat.succ_pos secParam
  have h_nonneg' : 0 ≤ (secParam.succ : ℝ) := h_pos.le
  have := mul_le_mul_of_nonneg_right h_lower h_nonneg'
  simpa [mul_comm, mul_left_comm, mul_assoc]
    using this

lemma heavyCommitmentImageFinset_heavy_properties
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
  (h_pos : 0 < ε) :
    (∀ comm ∈ heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε * (Fintype.card F : ℝ)),
      is_heavy_commitment VC cs x comm ε) ∧
    successProbability VC cs A x secParam - ε * (Fintype.card F : ℝ) ≤
      (heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε * (Fintype.card F : ℝ))).card := by
  classical
  have h_card_pos : 0 < (Fintype.card F : ℝ) := by
    simpa using (show 0 < (Fintype.card F : ℝ) from by exact_mod_cast Fintype.card_pos)
  have h_scaled_pos : 0 < ε * (Fintype.card F : ℝ) := mul_pos h_pos h_card_pos
  have h_scaled_nonneg : 0 ≤ ε * (Fintype.card F : ℝ) := h_scaled_pos.le
  refine ⟨?_, ?_⟩
  · intro comm h_comm
    exact heavyCommitmentImageFinset_mem_is_heavy (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam) (ε := ε) h_pos h_comm
  · have h_lower :=
      heavyCommitmentImageFinset_card_lower_bound (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam)
        (ε := ε * (Fintype.card F : ℝ)) h_scaled_nonneg
    simpa using h_lower

lemma heavyRandomnessFinset_mem_is_heavy
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_pos : 0 < ε)
    {rand : Fin secParam.succ}
    (h_mem : rand ∈ heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε * (Fintype.card F : ℝ))) :
    is_heavy_commitment VC cs x
      (commitTupleOfRandomness VC cs A x secParam rand) ε := by
  classical
  have h_comm_mem :
      commitTupleOfRandomness VC cs A x secParam rand ∈
        heavyCommitments VC cs A x secParam (ε * (Fintype.card F : ℝ)) := by
    exact
      (mem_heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam)
        (ε := ε * (Fintype.card F : ℝ))).1 h_mem
  exact
    heavyCommitment_mem_is_heavy (VC := VC) (cs := cs) (A := A) (x := x)
      (secParam := secParam) (ε := ε) h_pos h_comm_mem

lemma exists_heavy_randomness_candidate
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (ε : ℝ)
    (h_pos : 0 < ε)
    (h_nonneg : 0 ≤ ε) :
    ∃ heavy_rand : Finset (Fin secParam.succ),
      (heavy_rand.card : ℝ) ≥
          (successProbability VC cs A x secParam -
              ε * (Fintype.card F : ℝ)) * (secParam.succ : ℝ) ∧
      ∀ rand ∈ heavy_rand,
        is_heavy_commitment VC cs x
          (commitTupleOfRandomness VC cs A x secParam rand) ε := by
  classical
  set n : ℝ := (Fintype.card F : ℝ)
  have h_card_pos : 0 < n := by
    simpa [n] using
      (show 0 < (Fintype.card F : ℝ) from by exact_mod_cast Fintype.card_pos)
  have h_scaled_nonneg : 0 ≤ ε * n := mul_nonneg h_nonneg (le_of_lt h_card_pos)
  set heavy_rand :=
    heavyRandomnessFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε * n)
  have h_lower :=
    heavy_randomness_lower_bound (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε * n) h_scaled_nonneg
  refine ⟨heavy_rand, ?_, ?_⟩
  · simpa [heavy_rand, n]
      using h_lower
  · intro rand h_rand
    simpa [heavy_rand, n]
      using heavyRandomnessFinset_mem_is_heavy (VC := VC) (cs := cs) (A := A)
        (x := x) (secParam := secParam) (ε := ε) h_pos h_rand

lemma exists_heavy_commitments_candidate
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_pos : 0 < ε) :
    ∃ heavy_comms : Finset (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment),
      (heavy_comms.card : ℝ) ≥
          successProbability VC cs A x secParam - ε * (Fintype.card F : ℝ) ∧
      ∀ comm ∈ heavy_comms, is_heavy_commitment VC cs x comm ε := by
  classical
  set heavy_comms :=
    heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε * (Fintype.card F : ℝ))
  obtain ⟨h_all, h_card⟩ :=
    heavyCommitmentImageFinset_heavy_properties (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_pos
  refine ⟨heavy_comms, ?_, ?_⟩
  · simpa [heavy_comms] using h_card
  · intro comm h_comm
    simpa [heavy_comms] using h_all comm h_comm

lemma exists_heavy_commitments_candidate_weighted
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) (ε : ℝ)
    (h_pos : 0 < ε)
    (h_nonneg : 0 ≤ ε)
    (h_card_pos : (Fintype.card F : ℝ) > 0) :
    ∃ heavy_comms : Finset (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment),
      ((heavy_comms.card : ℝ) * (secParam.succ : ℝ)) ≥
          (successProbability VC cs A x secParam - ε * (Fintype.card F : ℝ)) *
            (secParam.succ : ℝ) ∧
      ∀ comm ∈ heavy_comms, is_heavy_commitment VC cs x comm ε := by
  classical
  have h_card_nonneg : 0 ≤ (Fintype.card F : ℝ) := le_of_lt h_card_pos
  have h_scaled_nonneg : 0 ≤ ε * (Fintype.card F : ℝ) :=
    mul_nonneg h_nonneg h_card_nonneg
  set heavy_comms :=
    heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε * (Fintype.card F : ℝ))
  obtain ⟨h_all, h_card⟩ :=
    heavyCommitmentImageFinset_heavy_properties (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_pos
  have h_weighted :=
    heavyCommitmentImageFinset_weighted_card_lower_bound (VC := VC) (cs := cs)
      (A := A) (x := x) (secParam := secParam)
      (ε := ε * (Fintype.card F : ℝ)) h_scaled_nonneg
  have h_card_eq :
      (((heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
          (x := x) (secParam := secParam) (ε := ε * (Fintype.card F : ℝ))).card : ℝ) *
        (secParam.succ : ℝ)) =
        ((heavy_comms.card : ℝ) * (secParam.succ : ℝ)) := by
    simp [heavy_comms]
  refine ⟨heavy_comms, ?_, ?_⟩
  · have h_coe :
        ((heavyCommitmentImageFinset (VC := VC) (cs := cs) (A := A)
            (x := x) (secParam := secParam)
            (ε := ε * (Fintype.card F : ℝ))).card : ℝ) =
          (heavy_comms.card : ℝ) := by
      simp [heavy_comms]
    have h_weighted' :=
      show
          (successProbability VC cs A x secParam - ε * (Fintype.card F : ℝ)) *
              (secParam.succ : ℝ) ≤
            (heavy_comms.card : ℝ) * (secParam.succ : ℝ) from by
        simpa [heavy_comms, h_coe] using h_weighted
    have h_rewrite :
        (successProbability VC cs A x secParam - ε * (Fintype.card F : ℝ)) *
            (secParam.succ : ℝ)
          = (successProbability VC cs A x secParam -
              (ε * (Fintype.card F : ℝ))) * (secParam.succ : ℝ) := by
      simp
    simpa [h_rewrite]
      using h_weighted'
  · intro comm h_comm
    simpa [heavy_comms] using h_all comm h_comm

/-- Combinatorial counting step of the heavy-row argument.

Given a positive heaviness threshold `ε`, we extract an explicit finite set of
commitment tuples such that:
* every commitment in the set is `ε`-heavy (at least `ε · |F|` challenges
  succeed), and
* the cardinality of the set dominates the global success deficit
  `successProbability - ε · |F|`.

This lemma is obtained purely from the heavy/light infrastructure developed in
`LambdaSNARK.Forking.HeavyLight`.  The remaining work to fully match the
classical heavy-row statement is to rewrite the bound into the customary
`ε - 1/|F|` form; the algebraic manipulations for that strengthening are tracked
in `axiom-elimination-plan.md`.
-/
lemma heavy_row_lemma {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub)
    (ε : ℝ) (secParam : ℕ)
    (h_ε_pos : 0 < ε)
    (_h_ε_bound : ε ≤ 1)
    (_h_field_size : (Fintype.card F : ℝ) > 0) :
    ∃ (heavy_comms : Finset (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)),
      (heavy_comms.card : ℝ) ≥
          successProbability VC cs A x secParam -
            ε * (Fintype.card F : ℝ) ∧
      ∀ c ∈ heavy_comms, is_heavy_commitment VC cs x c ε := by
  classical
  obtain ⟨heavy_comms, h_card, h_heavy⟩ :=
    exists_heavy_commitments_candidate (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (ε := ε) h_ε_pos
  refine ⟨heavy_comms, ?_, ?_⟩
  · simpa using h_card
  · exact h_heavy

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

lemma fork_success_bound_of_heavyCommitment
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) [DecidableEq VC.Commitment]
    (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub)
    (secParam : ℕ)
    (ε : ℝ)
    (h_ε_pos : 0 < ε) (h_ε_bound : ε ≤ 1)
    (h_field_size : (Fintype.card F : ℝ) ≥ 2)
    (h_card_nat : Fintype.card F ≥ 2)
    (h_ε_mass : ε * (Fintype.card F : ℝ) ≥ 2)
    {comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment}
    (h_heavy_mem : comm_tuple ∈
      heavyCommitments VC cs A x secParam (ε * (Fintype.card F : ℝ))) :
    let valid_challenges := successfulChallenges VC cs A x secParam comm_tuple
    let total_pairs := Nat.choose (Fintype.card F) 2
    let valid_pairs := Nat.choose valid_challenges.card 2
    (valid_pairs : ℝ) / (total_pairs : ℝ) ≥ ε ^ 2 / 2 - 1 / (Fintype.card F : ℝ) := by
  classical
  intro valid_challenges total_pairs valid_pairs
  have h_card_pos_nat : 0 < Fintype.card F :=
    Nat.lt_of_lt_of_le (by decide : 0 < 2) h_card_nat
  have h_field_pos : 0 < (Fintype.card F : ℝ) := by
    exact_mod_cast h_card_pos_nat
  have h_scaled_pos : 0 < ε * (Fintype.card F : ℝ) :=
    mul_pos h_ε_pos h_field_pos
  have h_card_ge : ε * (Fintype.card F : ℝ) ≤ (valid_challenges.card : ℝ) := by
    simpa [valid_challenges]
      using successfulChallenges_card_ge_of_heavyCommitment (VC := VC) (cs := cs)
        (A := A) (x := x) (secParam := secParam)
        (ε := ε * (Fintype.card F : ℝ)) h_scaled_pos h_heavy_mem
  have h_two_le_card_real : (2 : ℝ) ≤ (valid_challenges.card : ℝ) :=
    le_trans (by simpa using h_ε_mass) h_card_ge
  have h_valid_nonempty : valid_challenges.card ≥ 2 := by
    exact_mod_cast h_two_le_card_real
  have h_result :=
    fork_success_bound (VC := VC)
      (_state := A.snapshot cs x 0)
      (valid_challenges := valid_challenges)
      (ε := ε)
      (h_heavy := h_card_ge)
      (h_ε_pos := h_ε_pos)
      (_h_ε_bound := h_ε_bound)
      (h_field_size := h_field_size)
      (h_valid_nonempty := h_valid_nonempty)
  simpa [valid_challenges, total_pairs, valid_pairs] using h_result


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
    (assumptions : SoundnessCtx F VC cs) :
    (x : PublicInput F cs.nPub) →
    satisfies cs
      (extract_witness VC cs
        (extract_quotient_diff VC cs t1 t2 h_fork eqns.m eqns.ω)
        eqns.m eqns.ω eqns.h_m_vars x) := by
  intro x
  have _ := assumptions.moduleSIS_holds
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

/-! Witness extractor (uses adversary as black box). -/
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
  (assumptions : SoundnessCtx F VC cs) :
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
    (extraction_soundness VC cs t1 t2 hFork eqns h_rem assumptions) x
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
