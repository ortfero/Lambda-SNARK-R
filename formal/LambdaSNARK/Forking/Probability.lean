import LambdaSNARK.Core
import LambdaSNARK.Forking.Types
import LambdaSNARK.Polynomial
import LambdaSNARK.Constraints
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Data.Finset.Card
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Probability.ProbabilityMassFunction.Monad

/-!
# Forking Probability Layer

Probability distributions and mass functions supporting the ΛSNARK forking
analysis.  This module packages the single-run adversary experiment together
with its pushforward distributions (transcript, commitments, commitment and
challenge pairs) and various derived probability masses used in the
infrastructure proofs.

## Main definitions

- `run_adversary_state`/`run_adversary_transcript`: distributions obtained from a
  single adversary execution.
- `run_adversary_commit_tuple`/`run_adversary_commit_challenge`: pushforwards to
  commitment data.
- `successProbability`/`successMass*`: probability mass helpers for success
  events conditioned on commitments and challenges.
- `commitMass`/`commitChallengeMass`: marginal masses on commitments and
  commitment/challenge pairs.
-/

namespace LambdaSNARK

open scoped BigOperators
open BigOperators Polynomial

namespace PMF

lemma bind_pure_map {α β} (μ : PMF α) (f : α → β) :
    PMF.bind μ (fun a => PMF.pure (f a)) = PMF.map f μ := by
  classical
  ext b
  simp [PMF.bind, PMF.map, PMF.pure]

lemma map_map_comp {α β γ} (μ : PMF α) (f : α → β) (g : β → γ) :
    PMF.map g (PMF.map f μ) = PMF.map (g ∘ f) μ := by
  classical
  calc
    PMF.map g (PMF.map f μ)
        = PMF.bind (PMF.map f μ) (fun b => PMF.pure (g b)) := by
            simpa using (bind_pure_map (μ := PMF.map f μ) (f := g)).symm
        _ = PMF.bind (PMF.bind μ (fun a => PMF.pure (f a))) (fun b => PMF.pure (g b)) := by
          simp [(bind_pure_map (μ := μ) (f := f)).symm]
        _ = PMF.bind μ (fun a => PMF.bind (PMF.pure (f a)) (fun b => PMF.pure (g b))) := by
          simp [PMF.bind_bind (p := μ)
            (f := fun a => PMF.pure (f a))
            (g := fun b => PMF.pure (g b))]
        _ = PMF.bind μ (fun a => PMF.pure (g (f a))) := by
          simp
        _ = PMF.map (g ∘ f) μ := by
          simpa [Function.comp] using bind_pure_map (μ := μ) (f := g ∘ f)

lemma bind_const {α β} (μ : PMF α) (ν : PMF β) :
    PMF.bind μ (fun _ => ν) = ν := by
  classical
  ext b
  simp

lemma bind_pure {α} [DecidableEq α] (μ : PMF α) :
    PMF.bind μ (fun a => PMF.pure a) = μ := by
  classical
  ext a
  simp

lemma map_const {α β} [DecidableEq β] (μ : PMF α) (b : β) :
    PMF.map (fun _ : α => b) μ = PMF.pure b := by
  classical
  have h_map :=
    (bind_pure_map (μ := μ) (f := fun _ : α => b)).symm
  have h_bind := bind_const (μ := μ) (ν := PMF.pure b)
  simpa using h_map.trans h_bind

lemma map_id {α} [DecidableEq α] (μ : PMF α) :
    PMF.map (fun a : α => a) μ = μ := by
  classical
  have h_map :=
    (bind_pure_map (μ := μ) (f := fun a : α => a)).symm
  have h_bind := bind_pure (μ := μ)
  simpa using h_map.trans h_bind

lemma map_id' {α} [DecidableEq α] (μ : PMF α) : PMF.map (id : α → α) μ = μ := by
  simpa [id] using map_id (μ := μ)

end PMF

section SingleRun

variable {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
variable (VC : VectorCommitment F) (cs : R1CS F)
variable (A : Adversary F VC) (x : PublicInput F cs.nPub) (secParam : ℕ)

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

/-- Uniform distribution over a finite set: every element has probability
  `1 / |α|`. -/
noncomputable def uniform_pmf {α : Type*} [Fintype α] [Nonempty α] : PMF α :=
  ⟨fun _ => (Fintype.card α : ENNReal)⁻¹,
   by
     classical
     have h_card_pos : 0 < Fintype.card α := Fintype.card_pos
     have h_card_ne_zero : (Fintype.card α : ENNReal) ≠ 0 := by
       norm_cast
       exact Nat.pos_iff_ne_zero.mp h_card_pos
     have h_card_ne_top : (Fintype.card α : ENNReal) ≠ ⊤ :=
       ENNReal.natCast_ne_top (Fintype.card α)
     have h_summable : Summable (fun (_ : α) => (Fintype.card α : ENNReal)⁻¹) := by
       exact ENNReal.summable
     have h_tsum : ∑' (_ : α), (Fintype.card α : ENNReal)⁻¹ = 1 := by
       rw [tsum_fintype]
       simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
       exact ENNReal.mul_inv_cancel h_card_ne_zero h_card_ne_top
     rw [← h_tsum]
     exact h_summable.hasSum
   ⟩

lemma uniform_pmf_apply_fin (n : ℕ) (rand : Fin n.succ) :
    (uniform_pmf : PMF (Fin n.succ)) rand = (n.succ : ENNReal)⁻¹ := by
  classical
  change (Fintype.card (Fin n.succ) : ENNReal)⁻¹ = (n.succ : ENNReal)⁻¹
  simp [Fintype.card_fin]

/-- Uniform distribution excluding one element.

    PMF where each element in α \ {x} has probability 1/(|α| - 1).

    Requires card(α) ≥ 2 to ensure the distribution is well-defined. -/
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

/-- Deterministic snapshot obtained by running the adversary with a fixed
    randomness input.  This helper mirrors the construction performed inside
    `run_adversary` and is useful whenever we want to reason about concrete
    randomness seeds instead of working purely with probability masses. -/
noncomputable def adversarySample
  (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
  (x : PublicInput F cs.nPub) (secParam : ℕ)
  (rand : Fin secParam.succ) :
  AdversaryState F VC × Transcript F VC := by
  classical
  let randNat : ℕ := rand
  let proof := A.run cs x randNat
  let state := A.snapshot cs x randNat
  refine ⟨state, ?_⟩
  exact {
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
    valid := verify VC cs x proof }

/-- Convenience projection extracting the transcript determined by a fixed
    randomness input. -/
noncomputable def transcriptOfRandomness
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (rand : Fin secParam.succ) : Transcript F VC :=
  (adversarySample VC cs A x secParam rand).2

/-- Convenience projection extracting the commitment tuple determined by a
    fixed randomness input. -/
noncomputable def commitTupleOfRandomness
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (rand : Fin secParam.succ) :
    VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment :=
  transcriptCommitTuple VC
    (transcriptOfRandomness VC cs A x secParam rand)

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
noncomputable def run_adversary_state : PMF (AdversaryState F VC) :=
  PMF.bind (run_adversary (VC := VC) (cs := cs) A x secParam) (fun sample =>
    PMF.pure sample.1)

/-- Distribution over transcripts emitted in the first adversary run. -/
noncomputable def run_adversary_transcript : PMF (Transcript F VC) :=
  PMF.bind (run_adversary (VC := VC) (cs := cs) A x secParam) (fun sample =>
    PMF.pure sample.2)

/-- Distribution over commitment tuples emitted by the first adversary run. -/
noncomputable def run_adversary_commit_tuple :
    PMF (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :=
  PMF.map (transcriptCommitTuple VC)
    (run_adversary_transcript (VC := VC) (cs := cs) A x secParam)

/-- Distribution over commitment tuples paired with the Fiat–Shamir challenge. -/
noncomputable def run_adversary_commit_challenge :
    PMF ((VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F) :=
  PMF.map (transcriptCommitChallenge VC)
    (run_adversary_transcript (VC := VC) (cs := cs) A x secParam)

/-- Distribution pairing the first-run commitment tuple with an independent uniformly
    sampled challenge.  This models the random-oracle experiment where the challenge is
    drawn only after the commitment phase has finished. -/
noncomputable def run_adversary_commit_uniform_challenge
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    PMF ((VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F) := by
  classical
  refine PMF.bind (run_adversary_commit_tuple (VC := VC) (cs := cs) A x secParam) ?_
  intro comm
  exact PMF.map (fun α : F => (comm, α)) (uniform_pmf : PMF F)

lemma run_adversary_commit_uniform_challenge_fst
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    [DecidableEq (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)] :
    PMF.map Prod.fst
        (run_adversary_commit_uniform_challenge VC cs A x secParam)
      = run_adversary_commit_tuple VC cs A x secParam := by
  classical
  have h_map :=
    (PMF.bind_pure_map
      (μ := run_adversary_commit_uniform_challenge VC cs A x secParam)
      (f := Prod.fst)).symm
  have h_inner :
      ∀ comm : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment,
        PMF.bind (PMF.map (fun α : F => (comm, α)) (uniform_pmf : PMF F))
            (fun pair : ((VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F) =>
              PMF.pure pair.1)
          = PMF.pure comm := by
    intro comm
    have h_bind :=
      PMF.bind_pure_map
        (μ := PMF.map (fun α : F => (comm, α)) (uniform_pmf : PMF F))
        (f := Prod.fst)
    have h_map_pure :
        PMF.map Prod.fst
            (PMF.map (fun α : F => (comm, α)) (uniform_pmf : PMF F))
          = PMF.pure comm := by
      simpa [Function.comp, PMF.map_const]
        using PMF.map_map_comp
          (μ := (uniform_pmf : PMF F))
          (f := fun α : F => (comm, α))
          (g := Prod.fst)
    exact h_bind.trans h_map_pure
  calc
    PMF.map Prod.fst
        (run_adversary_commit_uniform_challenge VC cs A x secParam)
        = PMF.bind
            (run_adversary_commit_uniform_challenge VC cs A x secParam)
            (fun pair : ((VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F) =>
              PMF.pure pair.1) := h_map
    _ = PMF.bind (run_adversary_commit_tuple VC cs A x secParam)
            (fun comm =>
              PMF.bind (PMF.map (fun α : F => (comm, α)) (uniform_pmf : PMF F))
                (fun pair : ((VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F) =>
                  PMF.pure pair.1)) := by
          simp [run_adversary_commit_uniform_challenge, PMF.bind_bind]
    _ = PMF.bind (run_adversary_commit_tuple VC cs A x secParam)
            (fun comm => PMF.pure comm) := by
          simp [h_inner]
    _ = run_adversary_commit_tuple VC cs A x secParam := by
          exact PMF.bind_pure
            (run_adversary_commit_tuple (VC := VC) (cs := cs) A x secParam)

lemma run_adversary_commit_uniform_challenge_snd
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    PMF.map Prod.snd
        (run_adversary_commit_uniform_challenge VC cs A x secParam)
      = (uniform_pmf : PMF F) := by
  classical
  have h_map :=
    (PMF.bind_pure_map
      (μ := run_adversary_commit_uniform_challenge VC cs A x secParam)
      (f := Prod.snd)).symm
  have h_inner :
      ∀ comm : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment,
        PMF.bind (PMF.map (fun α : F => (comm, α)) (uniform_pmf : PMF F))
            (fun pair : ((VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F) =>
              PMF.pure pair.2)
          = (uniform_pmf : PMF F) := by
    intro comm
    have h_bind :=
      PMF.bind_pure_map
        (μ := PMF.map (fun α : F => (comm, α)) (uniform_pmf : PMF F))
        (f := Prod.snd)
    have h_map_id :
        PMF.map Prod.snd
            (PMF.map (fun α : F => (comm, α)) (uniform_pmf : PMF F))
          = (uniform_pmf : PMF F) := by
      simpa [Function.comp, id, PMF.map_id']
        using PMF.map_map_comp
          (μ := (uniform_pmf : PMF F))
          (f := fun α : F => (comm, α))
          (g := Prod.snd)
    exact h_bind.trans h_map_id
  calc
    PMF.map Prod.snd
        (run_adversary_commit_uniform_challenge VC cs A x secParam)
        = PMF.bind
            (run_adversary_commit_uniform_challenge VC cs A x secParam)
            (fun pair : ((VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F) =>
              PMF.pure pair.2) := h_map
    _ = PMF.bind (run_adversary_commit_tuple VC cs A x secParam)
            (fun comm =>
              PMF.bind (PMF.map (fun α : F => (comm, α)) (uniform_pmf : PMF F))
                (fun pair : ((VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) × F) =>
                  PMF.pure pair.2)) := by
          simp [run_adversary_commit_uniform_challenge, PMF.bind_bind]
    _ = PMF.bind (run_adversary_commit_tuple VC cs A x secParam)
            (fun _ => (uniform_pmf : PMF F)) := by
          simp [h_inner]
    _ = (uniform_pmf : PMF F) := by
          exact PMF.bind_const
            (run_adversary_commit_tuple (VC := VC) (cs := cs) A x secParam)
            (uniform_pmf : PMF F)

lemma run_adversary_commit_uniform_challenge_apply
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    [DecidableEq (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)]
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)
    (α : F) :
    run_adversary_commit_uniform_challenge VC cs A x secParam (comm_tuple, α)
      = run_adversary_commit_tuple VC cs A x secParam comm_tuple *
          (uniform_pmf : PMF F) α := by
  classical
  have h_map :
      ∀ comm : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment,
        (PMF.map (fun β : F => (comm, β)) (uniform_pmf : PMF F))
            (comm_tuple, α)
          = (if comm = comm_tuple then (uniform_pmf : PMF F) α else 0) := by
    intro comm
    by_cases h_comm : comm = comm_tuple
    · subst h_comm
      simp [PMF.map_apply, Prod.mk.injEq, tsum_fintype]
    ·
      have h_comm' : comm_tuple ≠ comm := by simpa [eq_comm] using h_comm
      simp [PMF.map_apply, Prod.mk.injEq, h_comm, h_comm']
  have h_bind :
      run_adversary_commit_uniform_challenge VC cs A x secParam (comm_tuple, α)
        = ∑' comm,
            run_adversary_commit_tuple VC cs A x secParam comm *
              (if comm = comm_tuple then (uniform_pmf : PMF F) α else 0) := by
    simp [run_adversary_commit_uniform_challenge, PMF.bind_apply, h_map]
  have h_sum :
      ∑' comm,
          run_adversary_commit_tuple VC cs A x secParam comm *
            (if comm = comm_tuple then (uniform_pmf : PMF F) α else 0)
        = run_adversary_commit_tuple VC cs A x secParam comm_tuple *
            (uniform_pmf : PMF F) α := by
    refine (tsum_eq_single comm_tuple fun comm h_comm => ?_).trans ?_
    · simp [h_comm]
    · simp
  exact h_bind.trans h_sum


lemma run_adversary_eq_map_adversarySample
    {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    run_adversary VC cs A x secParam =
      PMF.map (adversarySample VC cs A x secParam)
        (uniform_pmf : PMF (Fin (secParam.succ))) := by
  classical
  simpa [run_adversary, adversarySample]
    using (PMF.bind_pure_map
      (μ := (uniform_pmf : PMF (Fin (secParam.succ))))
      (f := adversarySample VC cs A x secParam))

lemma run_adversary_transcript_eq_map_transcriptOfRandomness
    {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    run_adversary_transcript VC cs A x secParam =
      PMF.map (transcriptOfRandomness VC cs A x secParam)
        (uniform_pmf : PMF (Fin (secParam.succ))) := by
  classical
  have h_bind :
      run_adversary_transcript VC cs A x secParam =
        PMF.map (fun sample => sample.2)
          (run_adversary VC cs A x secParam) := by
    simpa [run_adversary_transcript]
      using (PMF.bind_pure_map
        (μ := run_adversary VC cs A x secParam)
        (f := fun sample : AdversaryState F VC × Transcript F VC => sample.2))
  have h_run := run_adversary_eq_map_adversarySample VC cs A x secParam
  have h_comp :
      (fun rand : Fin secParam.succ =>
          (adversarySample VC cs A x secParam rand).2)
        = transcriptOfRandomness VC cs A x secParam := by
    funext rand
    simp [transcriptOfRandomness, adversarySample]
  have h_map_comp :
      PMF.map ((fun sample => sample.2) ∘
          adversarySample VC cs A x secParam)
        (uniform_pmf : PMF (Fin (secParam.succ)))
        = PMF.map (transcriptOfRandomness VC cs A x secParam)
          (uniform_pmf : PMF (Fin (secParam.succ))) := by
    simpa [Function.comp] using congrArg
      (fun f => PMF.map f (uniform_pmf : PMF (Fin (secParam.succ)))) h_comp
  calc
    run_adversary_transcript VC cs A x secParam
        = PMF.map (fun sample => sample.2)
            (run_adversary VC cs A x secParam) := h_bind
    _ = PMF.map (fun sample => sample.2)
          (PMF.map (adversarySample VC cs A x secParam)
            (uniform_pmf : PMF (Fin (secParam.succ)))) := by
          simp [h_run]
    _ = PMF.map ((fun sample => sample.2) ∘
          adversarySample VC cs A x secParam)
            (uniform_pmf : PMF (Fin (secParam.succ))) := by
          exact PMF.map_map_comp
            (μ := (uniform_pmf : PMF (Fin (secParam.succ))))
            (f := adversarySample VC cs A x secParam)
            (g := fun sample => sample.2)
    _ = PMF.map (transcriptOfRandomness VC cs A x secParam)
        (uniform_pmf : PMF (Fin (secParam.succ))) := h_map_comp

lemma run_adversary_commit_tuple_eq_map_commitTupleOfRandomness
    {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    run_adversary_commit_tuple VC cs A x secParam =
      PMF.map (commitTupleOfRandomness VC cs A x secParam)
        (uniform_pmf : PMF (Fin (secParam.succ))) := by
  classical
  have h_trans :=
    run_adversary_transcript_eq_map_transcriptOfRandomness VC cs A x secParam
  have h_comp :
      (fun rand : Fin secParam.succ =>
          transcriptCommitTuple VC
            (transcriptOfRandomness VC cs A x secParam rand))
        = commitTupleOfRandomness VC cs A x secParam := by
    funext rand
    simp [commitTupleOfRandomness, transcriptOfRandomness]
  have h_map_comp :
      PMF.map ((transcriptCommitTuple VC) ∘
          transcriptOfRandomness VC cs A x secParam)
        (uniform_pmf : PMF (Fin (secParam.succ)))
        = PMF.map (commitTupleOfRandomness VC cs A x secParam)
          (uniform_pmf : PMF (Fin (secParam.succ))) := by
    simpa [Function.comp] using congrArg
      (fun f => PMF.map f (uniform_pmf : PMF (Fin (secParam.succ)))) h_comp
  calc
    run_adversary_commit_tuple VC cs A x secParam
        = PMF.map (transcriptCommitTuple VC)
            (run_adversary_transcript VC cs A x secParam) := rfl
    _ = PMF.map (transcriptCommitTuple VC)
          (PMF.map (transcriptOfRandomness VC cs A x secParam)
            (uniform_pmf : PMF (Fin (secParam.succ)))) := by
          simp [h_trans]
    _ = PMF.map ((transcriptCommitTuple VC) ∘
          transcriptOfRandomness VC cs A x secParam)
            (uniform_pmf : PMF (Fin (secParam.succ))) := by
          exact PMF.map_map_comp
            (μ := (uniform_pmf : PMF (Fin (secParam.succ))))
            (f := transcriptOfRandomness VC cs A x secParam)
            (g := transcriptCommitTuple VC)
    _ = PMF.map (commitTupleOfRandomness VC cs A x secParam)
        (uniform_pmf : PMF (Fin (secParam.succ))) := h_map_comp

end SingleRun

lemma uniform_pmf_apply_ne_zero {α : Type*} [Fintype α] [Nonempty α]
    (a : α) : (uniform_pmf : PMF α) a ≠ 0 := by
  classical
  have h_card_ne_zero : (Fintype.card α : ENNReal) ≠ 0 := by
    have h_pos : 0 < Fintype.card α := Fintype.card_pos
    exact_mod_cast (Nat.pos_iff_ne_zero.mp h_pos)
  have h_card_ne_top : (Fintype.card α : ENNReal) ≠ ⊤ :=
    ENNReal.natCast_ne_top (Fintype.card α)
  intro h_zero
  dsimp [uniform_pmf] at h_zero
  have h_const_zero : (Fintype.card α : ENNReal)⁻¹ = 0 := h_zero
  have h_zero' :
      (Fintype.card α : ENNReal) * (Fintype.card α : ENNReal)⁻¹ = 0 := by
    simp [h_const_zero]
  have h_one :
      (Fintype.card α : ENNReal) * (Fintype.card α : ENNReal)⁻¹ = 1 :=
    ENNReal.mul_inv_cancel h_card_ne_zero h_card_ne_top
  have : (0 : ENNReal) = 1 := by
    calc
      (0 : ENNReal)
          = (Fintype.card α : ENNReal) * (Fintype.card α : ENNReal)⁻¹ := by
            simp [h_zero']
      _ = 1 := by simpa using h_one
  exact zero_ne_one this

section SuccessMass

variable {F : Type} [Field F] [Fintype F] [DecidableEq F]
variable (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
variable (x : PublicInput F cs.nPub) (secParam : ℕ)

/-- Convert the success event for the first run into a real-valued probability. -/
noncomputable def successProbability : ℝ :=
      ((run_adversary_transcript (VC := VC) (cs := cs) A x secParam).toOuterMeasure
        {t | success_event VC cs x t}).toReal

/-- ENNReal-valued mass of the success event in the first adversary run. -/
noncomputable def successMass : ENNReal :=
  by
    classical
    let p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam
    exact ∑' t, (if success_event VC cs x t then p t else 0)

/-- Weight contributed by a single randomness seed to the overall success mass. -/
noncomputable def successSeedWeight
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (rand : Fin secParam.succ) : ENNReal :=
  by
    classical
    exact if success_event VC cs x
        (transcriptOfRandomness VC cs A x secParam rand) then
        (uniform_pmf : PMF (Fin (secParam.succ))) rand else 0

/-- Weight contributed by a randomness seed towards a fixed commitment tuple. -/
noncomputable def commitSeedWeight
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)
    (rand : Fin secParam.succ) : ENNReal :=
  by
    classical
    exact if comm_tuple = commitTupleOfRandomness VC cs A x secParam rand then
      (uniform_pmf : PMF (Fin (secParam.succ))) rand else 0

lemma successMass_eq_uniform_randomness_tsum
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ) :
    successMass VC cs A x secParam =
      ∑ rand : Fin secParam.succ,
        successSeedWeight VC cs A x secParam rand := by
  classical
  set S := {t : Transcript F VC | success_event VC cs x t}
  have h_mass :
      successMass VC cs A x secParam =
        (run_adversary_transcript VC cs A x secParam).toOuterMeasure S := by
    simp [successMass, S, PMF.toOuterMeasure_apply, Set.indicator, Set.mem_setOf_eq]
  have h_run :=
    run_adversary_transcript_eq_map_transcriptOfRandomness VC cs A x secParam
  have h_map :=
    PMF.toOuterMeasure_map_apply
      (p := (uniform_pmf : PMF (Fin (secParam.succ))))
      (f := transcriptOfRandomness VC cs A x secParam)
      (s := S)
  have h_preimage :
      (transcriptOfRandomness VC cs A x secParam) ⁻¹' S
        = {rand : Fin secParam.succ |
            success_event VC cs x
              (transcriptOfRandomness VC cs A x secParam rand)} := rfl
  have h_uniform :
      (uniform_pmf : PMF (Fin (secParam.succ))).toOuterMeasure
          ((transcriptOfRandomness VC cs A x secParam) ⁻¹' S)
        = ∑ rand : Fin secParam.succ,
            successSeedWeight VC cs A x secParam rand := by
    simp [Set.indicator, Set.mem_setOf_eq, successSeedWeight, h_preimage]
  calc
    successMass VC cs A x secParam
        = (run_adversary_transcript VC cs A x secParam).toOuterMeasure S := h_mass
    _ = (PMF.map (transcriptOfRandomness VC cs A x secParam)
            (uniform_pmf : PMF (Fin (secParam.succ)))).toOuterMeasure S := by
          simp [S, h_run]
    _ = (uniform_pmf : PMF (Fin (secParam.succ))).toOuterMeasure
          ((transcriptOfRandomness VC cs A x secParam) ⁻¹' S) := h_map
    _ = ∑ rand : Fin secParam.succ,
          successSeedWeight VC cs A x secParam rand := h_uniform

/-- Mass of successful transcripts conditioned on a fixed commitment tuple. -/
noncomputable def successMassGivenCommit
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) : ENNReal :=
  by
    classical
    let p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam
    exact ∑' t,
      (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple then p t else 0)

/-- Mass of successful transcripts conditioned on both a commitment tuple and challenge. -/
noncomputable def successMassGivenCommitAndChallenge
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)
    (α : F) : ENNReal :=
  by
    classical
    let p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam
    exact ∑' t,
      (if success_event VC cs x t ∧ transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α
        then p t else 0)

/-- Marginal mass of transcripts whose commitments match the supplied tuple. -/
noncomputable def commitMass
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
    commitMass VC cs A x secParam comm_tuple =
      run_adversary_commit_tuple VC cs A x secParam comm_tuple := by
  classical
  unfold commitMass
  set p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam with hp
  change (∑' t, (if transcriptCommitTuple VC t = comm_tuple then p t else 0))
      = run_adversary_commit_tuple VC cs A x secParam comm_tuple
  have h_map :
      run_adversary_commit_tuple VC cs A x secParam comm_tuple
        = ∑' t, (if transcriptCommitTuple VC t = comm_tuple then p t else 0) := by
    simpa [run_adversary_commit_tuple, hp.symm, PMF.map_apply, eq_comm]
      using (PMF.map_apply (transcriptCommitTuple VC)
        (run_adversary_transcript (VC := VC) (cs := cs) A x secParam) comm_tuple)
  simpa using h_map.symm

lemma commitMass_eq_uniform_randomness_sum
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    commitMass VC cs A x secParam comm_tuple =
      ∑ rand : Fin secParam.succ,
        commitSeedWeight VC cs A x secParam comm_tuple rand := by
  classical
  have h_commit :=
    commitMass_eq_run_adversary_commit_tuple (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (comm_tuple := comm_tuple)
  have h_map :=
    run_adversary_commit_tuple_eq_map_commitTupleOfRandomness (VC := VC)
      (cs := cs) (A := A) (x := x) (secParam := secParam)
  have h_eval :=
    congrArg (fun (pmf : PMF _ ) => pmf comm_tuple) h_map
  have h_uniform :
      (PMF.map (commitTupleOfRandomness VC cs A x secParam)
          (uniform_pmf : PMF (Fin (secParam.succ)))) comm_tuple
        = ∑ rand : Fin secParam.succ,
            commitSeedWeight VC cs A x secParam comm_tuple rand := by
    classical
    have h_sum :=
      (PMF.map_apply (commitTupleOfRandomness VC cs A x secParam)
        (uniform_pmf : PMF (Fin (secParam.succ))) comm_tuple)
    simpa [commitSeedWeight, PMF.map_apply, tsum_fintype]
      using h_sum
  calc
    commitMass VC cs A x secParam comm_tuple
        = run_adversary_commit_tuple VC cs A x secParam comm_tuple := h_commit
    _ = (PMF.map (commitTupleOfRandomness VC cs A x secParam)
          (uniform_pmf : PMF (Fin (secParam.succ)))) comm_tuple := h_eval
    _ = ∑ rand : Fin secParam.succ,
          commitSeedWeight VC cs A x secParam comm_tuple rand := h_uniform

lemma run_adversary_commit_uniform_challenge_apply_eq_commitMass
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    [DecidableEq (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)]
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)
    (α : F) :
    run_adversary_commit_uniform_challenge VC cs A x secParam (comm_tuple, α)
      = commitMass VC cs A x secParam comm_tuple *
          (uniform_pmf : PMF F) α := by
  simpa [commitMass_eq_run_adversary_commit_tuple (VC := VC) (cs := cs) (A := A)
      (x := x) (secParam := secParam) (comm_tuple := comm_tuple)]
    using run_adversary_commit_uniform_challenge_apply
      (VC := VC) (cs := cs) (A := A) (x := x) (secParam := secParam)
      (comm_tuple := comm_tuple) (α := α)

lemma tsum_run_adversary_commit_uniform_challenge_over_challenges
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    [DecidableEq (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)]
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) :
    ∑' α, run_adversary_commit_uniform_challenge VC cs A x secParam (comm_tuple, α)
      = commitMass VC cs A x secParam comm_tuple := by
  classical
  have h_congr :
      ∑' α, run_adversary_commit_uniform_challenge VC cs A x secParam (comm_tuple, α)
        = ∑' α,
            commitMass VC cs A x secParam comm_tuple *
              (uniform_pmf : PMF F) α := by
    refine tsum_congr ?_
    intro α
    simp [run_adversary_commit_uniform_challenge_apply_eq_commitMass]
  have h_tsum := (uniform_pmf : PMF F).tsum_coe
  have h_factor := ENNReal.tsum_mul_left
      (f := fun α => (uniform_pmf : PMF F) α)
      (a := commitMass VC cs A x secParam comm_tuple)
  calc
    ∑' α, run_adversary_commit_uniform_challenge VC cs A x secParam (comm_tuple, α)
        = ∑' α,
            commitMass VC cs A x secParam comm_tuple *
              (uniform_pmf : PMF F) α := h_congr
    _ = commitMass VC cs A x secParam comm_tuple *
          ∑' α, (uniform_pmf : PMF F) α := h_factor
    _ = commitMass VC cs A x secParam comm_tuple * 1 := by simp [h_tsum]
    _ = commitMass VC cs A x secParam comm_tuple := by simp

lemma tsum_run_adversary_commit_uniform_challenge_over_commitments
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    [DecidableEq (VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)]
    (α : F) :
    ∑' comm,
        run_adversary_commit_uniform_challenge VC cs A x secParam (comm, α)
      = (uniform_pmf : PMF F) α := by
  classical
  have h_congr :
      ∑' comm,
          run_adversary_commit_uniform_challenge VC cs A x secParam (comm, α)
        = ∑' comm,
            run_adversary_commit_tuple VC cs A x secParam comm *
              (uniform_pmf : PMF F) α := by
    refine tsum_congr ?_
    intro comm
    simp [run_adversary_commit_uniform_challenge_apply]
  have h_tsum := (run_adversary_commit_tuple VC cs A x secParam).tsum_coe
  have h_factor := ENNReal.tsum_mul_right
      (f := fun comm => run_adversary_commit_tuple VC cs A x secParam comm)
      (a := (uniform_pmf : PMF F) α)
  calc
    ∑' comm,
        run_adversary_commit_uniform_challenge VC cs A x secParam (comm, α)
        = ∑' comm,
            run_adversary_commit_tuple VC cs A x secParam comm *
              (uniform_pmf : PMF F) α := h_congr
    _ = (∑' comm,
          run_adversary_commit_tuple VC cs A x secParam comm) *
            (uniform_pmf : PMF F) α := h_factor
    _ = 1 * (uniform_pmf : PMF F) α := by simp [h_tsum]
    _ = (uniform_pmf : PMF F) α := by simp

/-- Marginal mass of observing a particular commitment tuple together with a challenge. -/
noncomputable def commitChallengeMass
    (comm_tuple : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment)
    (α : F) : ENNReal :=
  by
    classical
    let p := run_adversary_transcript (VC := VC) (cs := cs) A x secParam
    exact ∑' t,
      (if transcriptCommitTuple VC t = comm_tuple ∧ t.view.alpha = α then p t else 0)

end SuccessMass

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

lemma exists_randomness_of_mem_support_run_adversary_transcript
    {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (A : Adversary F VC)
    (x : PublicInput F cs.nPub) (secParam : ℕ)
    {t : Transcript F VC}
    (h_mem : t ∈ (run_adversary_transcript (VC := VC) (cs := cs) A x secParam).support) :
    ∃ rand : Fin secParam.succ,
      transcriptOfRandomness VC cs A x secParam rand = t := by
  classical
  have h_map :=
    run_adversary_transcript_eq_map_transcriptOfRandomness VC cs A x secParam
  have h_mem' :
      t ∈ (PMF.map (transcriptOfRandomness VC cs A x secParam)
        (uniform_pmf : PMF (Fin (secParam.succ)))).support := by
    simpa [h_map]
      using h_mem
  obtain ⟨rand, -, h_eq⟩ :=
    (PMF.mem_support_map_iff
      (f := transcriptOfRandomness VC cs A x secParam)
      (p := (uniform_pmf : PMF (Fin (secParam.succ))))
      (b := t)).1 h_mem'
  exact ⟨rand, h_eq⟩

lemma adversarySample_mem_support_run_adversary
    {F : Type} [CommRing F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F)
    (A : Adversary F VC) (x : PublicInput F cs.nPub)
    (secParam : ℕ) (rand : Fin secParam.succ) :
    adversarySample VC cs A x secParam rand
      ∈ (run_adversary (VC := VC) (cs := cs) A x secParam).support := by
  classical
  have h_unif_mem :
      rand ∈ (uniform_pmf : PMF (Fin (secParam.succ))).support := by
    have h_ne : (uniform_pmf : PMF (Fin (secParam.succ))) rand ≠ 0 :=
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

end LambdaSNARK
