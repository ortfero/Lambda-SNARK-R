# ΛSNARK-R Formal Verification Roadmap to 100%

**Current Status**: 80% (12/15 theorems), 14 sorry  
**Target**: 100% (15/15 theorems), 0 sorry, 0 axioms beyond crypto assumptions  
**Timeline**: 20-30h total (8 phases)  
**Date**: 2025-11-16

---

## Executive Summary

**Quality Requirements**:
- ✅ **Soundness**: Complete proofs without axioms (except crypto hardness)
- ✅ **Confluence**: All extraction paths deterministic
- ✅ **Completeness**: All sorry closed with verified proofs
- ✅ **Verification**: 100% theorem coverage
- ✅ **Documentation**: Proof strategies documented

**Critical Path**:
1. **Phase 1** (Soundness Foundation): Eliminate extraction_axiom → 5-8h
2. **Phase 2** (Probability Infrastructure): PMF formalization → 3-4h
3. **Phase 3-5** (Proof Completion): Close remaining sorry → 7-9h
4. **Phase 6-8** (Polish & Verify): Final integration → 4-5h

**Total**: 20-26h to 100% verification ✅

---

## Phase 1: Eliminate Extraction Axiom (Priority: P0 Soundness)

**Goal**: Replace `extraction_axiom` with actual proof  
**Duration**: 5-8h  
**Blocking**: All soundness guarantees  
**Status**: 🔴 Critical

### Current State
- File: `ForkingInfrastructure.lean:456`
- Issue: `axiom extraction_axiom` defers verification→extraction proof
- Impact: Core soundness property unproven

### Implementation Strategy

#### Step 1.1: Extend Core.lean Verification (2-3h)
**Goal**: Expose quotient polynomial from verification structure

**Tasks**:
1. Modify `Proof` structure to include quotient polynomial:
   ```lean
   structure Proof (F : Type) [CommRing F] (VC : VectorCommitment F) where
     ...
     quotient : Polynomial F  -- NEW: Add explicit quotient
   ```

2. Extend `verify` function to check quotient opening:
   ```lean
   def verify {F : Type} [CommRing F] [DecidableEq F]
       (VC : VectorCommitment F) (cs : R1CS F)
       (x : PublicInput F cs.nPub) (π : Proof F VC) : Bool :=
     -- Existing checks...
     -- NEW: Verify quotient commitment opens correctly
     VC.verify π.comm_quotient π.challenge_α π.opening_quotient_α ∧
     -- Quotient equation: q(α) * Z_H(α) = constraint_poly(α)
     π.quotient.eval π.challenge_α * vanishing_poly.eval π.challenge_α = 
       constraint_poly cs (extract_from_openings π) π.challenge_α
   ```

3. Prove lemma: `verify_implies_quotient_correct`:
   ```lean
   theorem verify_implies_quotient_correct {F : Type} [Field F]
       (VC : VectorCommitment F) (cs : R1CS F) (π : Proof F VC)
       (h_verify : verify VC cs x π = true) :
       π.quotient.eval π.challenge_α * vanishing_poly.eval π.challenge_α = 
         constraint_poly cs w π.challenge_α
   ```

**Files Modified**: `Core.lean` (+50 lines)

#### Step 1.2: Prove Binding Implies Uniqueness (1-2h)
**Goal**: Connect commitment binding to polynomial uniqueness

**Tasks**:
1. Use existing `VectorCommitment.binding` property:
   ```lean
   -- From Core.lean line ~165
   binding : ∀ pp c v₁ v₂ r₁ r₂ α π₁ π₂,
     commitProof pp v₁ r₁ = c →
     commitProof pp v₂ r₂ = c →
     verify pp c α π₁ = true →
     verify pp c α π₂ = true →
     v₁ = v₂
   ```

2. Extend to polynomial level:
   ```lean
   theorem binding_implies_unique_polynomial {F : Type} [Field F]
       (VC : VectorCommitment F) (cs : R1CS F)
       (t1 t2 : Transcript F VC)
       (h_fork : is_valid_fork VC t1 t2)
       (h_sis : ModuleSIS_Hard 256 2 12289 1024) :
       -- Same commitment → same polynomial
       t1.comm_quotient = t2.comm_quotient →
       extract_quotient_diff VC cs t1 t2 h_fork m ω = 
         (verified_quotient_from_t1 : Polynomial F)
   ```

**Files Modified**: `ForkingInfrastructure.lean` (+30 lines)

#### Step 1.3: Connect to quotient_exists_iff_satisfies (1-2h)
**Goal**: Complete the chain: verified transcript → satisfies

**Tasks**:
1. Implement `extract_quotient_diff` properly (currently returns 0):
   ```lean
   noncomputable def extract_quotient_diff {F : Type} [Field F]
       (VC : VectorCommitment F) (cs : R1CS F)
       (t1 t2 : Transcript F VC)
       (h_fork : is_valid_fork VC t1 t2) (m : ℕ) (ω : F) :
       Polynomial F :=
     -- Extract from verified transcripts using binding property
     t1.proof.quotient  -- Now available from Step 1.1
   ```

2. Prove extraction_soundness without axiom:
   ```lean
   theorem extraction_soundness ... := by
     let q := extract_quotient_diff VC cs t1 t2 h_fork m ω
     let w := extract_witness VC cs q m ω hω h_m
     
     -- Step A: Both transcripts verify
     have h_t1_verify : verify VC cs x t1.proof = true := h_fork.left.right
     have h_t2_verify : verify VC cs x t2.proof = true := h_fork.right.right
     
     -- Step B: Quotient equations hold (from Step 1.1)
     have h_q_eq1 : q.eval t1.challenge_α * Z_H(t1.challenge_α) = 
                     constraint_poly cs w t1.challenge_α :=
       verify_implies_quotient_correct ...
     
     -- Step C: Apply quotient_exists_iff_satisfies (Soundness.lean:95)
     apply (quotient_exists_iff_satisfies cs w m ω h_m_cons hω).mpr
     use q
     constructor
     · -- Prove: ∀ i, q.eval(ωⁱ) = constraint_poly i
       intro i
       -- Use polynomial interpolation from two points (t1.challenge, t2.challenge)
       sorry -- Requires Lagrange uniqueness theorem (1-2h)
     · -- Prove: q %ₘ Z_H = 0
       -- From quotient definition and vanishing property
       sorry -- Direct from division algorithm (30min)
   ```

**Files Modified**: `ForkingInfrastructure.lean` (+80 lines)

#### Step 1.4: Remove Axiom (15min)
**Tasks**:
1. Delete `axiom extraction_axiom` declaration
2. Replace axiom call with actual proof
3. Verify build passes

**Success Criteria**:
- ✅ `extraction_soundness` proven without axiom
- ✅ Build passes: 0 errors
- ✅ Sorry count: 14 → 11 (closes 3: extraction + 2 internal)

---

## Phase 2: Probability Formalization (PMF Infrastructure)

**Goal**: Replace PMF axioms with Mathlib constructions  
**Duration**: 3-4h  
**Blocking**: heavy_row_lemma, forking_lemma  
**Status**: 🟡 Important

### Current State
- Files: `ForkingInfrastructure.lean:132-135, 163, 195`
- Issue: `axiom uniform_pmf`, `uniform_pmf_ne` + 2 sorry in adversary execution
- Impact: Probability bounds unproven

### Implementation Strategy

#### Step 2.1: Replace Uniform PMF Axioms (1h)
**Tasks**:
1. Use Mathlib's `PMF.uniformOfFintype`:
   ```lean
   import Mathlib.Probability.ProbabilityMassFunction.Uniform
   
   -- Replace axiom uniform_pmf
   def uniform_pmf {α : Type*} [Fintype α] [Nonempty α] : PMF α :=
     PMF.uniformOfFintype α
   
   -- Replace axiom uniform_pmf_ne
   def uniform_pmf_ne {α : Type*} [Fintype α] [DecidableEq α]
       (x : α) (h : Fintype.card α ≥ 2) : PMF α :=
     PMF.uniformOfFintype {y : α // y ≠ x}
   ```

2. Prove properties:
   ```lean
   theorem uniform_pmf_ne_prob {α : Type*} [Fintype α] [DecidableEq α]
       (x y : α) (h : Fintype.card α ≥ 2) (h_ne : y ≠ x) :
       (uniform_pmf_ne x h).prob y = 1 / ((Fintype.card α : ℝ) - 1)
   ```

**Files Modified**: `ForkingInfrastructure.lean` (+20 lines, -2 axioms)

#### Step 2.2: Implement run_adversary (1-1.5h)
**Tasks**:
1. Construct adversary execution PMF via bind:
   ```lean
   noncomputable def run_adversary {F : Type} [Field F] [Fintype F]
       (VC : VectorCommitment F) (cs : R1CS F)
       (A : Adversary F VC) (x : PublicInput F cs.nPub)
       (secParam : ℕ) : PMF (Transcript F VC) := do
     -- Sample randomness
     let r ← uniform_pmf (α := ℕ) -- Over bounded range [0, 2^secParam)
     -- Run adversary to get commitments
     let proof := A.run cs x r
     -- Sample challenge
     let α ← uniform_pmf (α := F)
     -- Complete transcript
     pure {
       comm_Az := proof.comm_Az,
       comm_Bz := proof.comm_Bz,
       comm_Cz := proof.comm_Cz,
       comm_quotient := proof.comm_quotient,
       challenge_α := α,
       challenge_β := α, -- Fiat-Shamir
       opening_Az_α := proof.opening_Az_α,
       opening_Bz_β := proof.opening_Bz_β,
       opening_Cz_α := proof.opening_Cz_α,
       opening_quotient_α := proof.opening_quotient_α,
       valid := verify VC cs x proof
     }
   ```

**Files Modified**: `ForkingInfrastructure.lean` (+40 lines, -1 sorry)

#### Step 2.3: Implement rewind_adversary (1-1.5h)
**Tasks**:
1. Construct rewinding PMF:
   ```lean
   noncomputable def rewind_adversary {F : Type} [Field F] [Fintype F]
       (VC : VectorCommitment F) (cs : R1CS F)
       (A : Adversary F VC) (x : PublicInput F cs.nPub)
       (state : AdversaryState F VC)
       (first_challenge : F) (h_card : Fintype.card F ≥ 2) :
       PMF (Transcript F VC) := do
     -- Sample new challenge α' ≠ α
     let α' ← uniform_pmf_ne first_challenge h_card
     -- Resume with new challenge
     let (open_Az, open_Bz, open_Cz, open_quot) := state.respond α' α'
     pure {
       comm_Az := state.comm_Az,
       comm_Bz := state.comm_Bz,
       comm_Cz := state.comm_Cz,
       comm_quotient := state.comm_quotient,
       challenge_α := α',
       challenge_β := α',
       opening_Az_α := open_Az,
       opening_Bz_β := open_Bz,
       opening_Cz_α := open_Cz,
       opening_quotient_α := open_quot,
       valid := true -- Computed from verification
     }
   ```

**Files Modified**: `ForkingInfrastructure.lean` (+35 lines, -1 sorry)

#### Step 2.4: Formalize Success Probability (30min)
**Tasks**:
1. Define success event probability:
   ```lean
   def adversary_success_prob {F : Type} [Field F] [Fintype F]
       (VC : VectorCommitment F) (cs : R1CS F)
       (A : Adversary F VC) (x : PublicInput F cs.nPub)
       (secParam : ℕ) : ℝ :=
     (run_adversary VC cs A x secParam).toMeasure {t | t.valid = true}
   ```

2. Replace `h_success : True` hypotheses:
   ```lean
   -- Old: (h_success : True)  -- TODO: formalize
   -- New: (h_success : adversary_success_prob VC cs A x secParam ≥ ε)
   ```

**Files Modified**: `ForkingInfrastructure.lean, Soundness.lean` (+15 lines)

**Success Criteria**:
- ✅ 0 PMF axioms beyond uniform distribution
- ✅ `run_adversary`, `rewind_adversary` implemented
- ✅ Probability hypothesis formalized
- ✅ Sorry count: 11 → 9 (closes 2 PMF sorry)

---

## Phase 3: Combinatorics (fork_success_bound)

**Goal**: Close 3 sorry in fork_success_bound  
**Duration**: 3-4h  
**Blocking**: None (self-contained)  
**Status**: 🟡 Important

### Current State
- File: `ForkingInfrastructure.lean:362, 378, 394`
- Issue: Complex combinatorial inequalities with C(n,2)
- Impact: Fork probability bound unproven

### Implementation Strategy

#### Step 3.1: Parity Argument (Line 362, 1h)
**Goal**: Prove `2 | v(v-1)` for division by 2

**Tasks**:
1. Use Mathlib's `Nat.even_mul_pred`:
   ```lean
   have h_parity : 2 ∣ valid_challenges.card * (valid_challenges.card - 1) := by
     apply Nat.even_mul_pred
     -- Either v is even or v-1 is even
   ```

2. Apply `Nat.cast_div`:
   ```lean
   have h_vp : (valid_pairs : ℝ) = 
       (valid_challenges.card : ℝ) * ((valid_challenges.card : ℝ) - 1) / 2 := by
     rw [Nat.choose_two_right]
     rw [Nat.cast_div h_parity]
     norm_num
   ```

**Files Modified**: `ForkingInfrastructure.lean:362` (-1 sorry)

#### Step 3.2: Nat.cast for C(n,2) (Line 378, 1h)
**Goal**: Cast `C(n,2) = n(n-1)/2` to ℝ

**Tasks**:
1. Prove divisibility first:
   ```lean
   have h_card_ge_2 : Fintype.card F ≥ 2 := 
     Nat.cast_le.mp (by linarith : (2 : ℝ) ≤ (Fintype.card F : ℝ))
   have h_tp_div : 2 ∣ Fintype.card F * (Fintype.card F - 1) := 
     Nat.even_mul_pred
   ```

2. Apply casting:
   ```lean
   have h_tp : (total_pairs : ℝ) = 
       (Fintype.card F : ℝ) * ((Fintype.card F : ℝ) - 1) / 2 := by
     simp only [total_pairs]
     rw [Nat.choose_two_right]
     rw [Nat.cast_div h_tp_div]
     norm_num
   ```

**Files Modified**: `ForkingInfrastructure.lean:378` (-1 sorry)

#### Step 3.3: Final Calc (Line 394, 1-2h)
**Goal**: Prove `(ε²n² - 2εn) * 2 / (n² - n) ≥ ε² - 2/n`

**Tasks**:
1. Clear denominators with field_simp:
   ```lean
   field_simp [hn_pos, hn_m1_pos]
   -- Goal: (ε²n² - 2εn) * 2 ≥ (ε² - 2/n) * (n² - n)
   ```

2. Expand and simplify:
   ```lean
   ring_nf
   -- Goal: 2ε²n² - 4εn ≥ ε²n² - ε²n - 2n + 2
   ```

3. Rearrange:
   ```lean
   -- ε²n² - 4εn + ε²n + 2n ≥ 2
   -- ε²n(n+1) - 2εn + 2n ≥ 2
   -- Factor: n(ε²(n+1) - 2ε + 2) ≥ 2
   ```

4. Bound for ε close to 1, n ≥ 2:
   ```lean
   have h_main : (ε^2 * (Fintype.card F : ℝ) * ((Fintype.card F : ℝ) + 1) - 
                  2 * ε + 2) * (Fintype.card F : ℝ) ≥ 2 := by
     have h1 : ε^2 * (Fintype.card F : ℝ) * ((Fintype.card F : ℝ) + 1) ≥ 
               ε^2 * 2 * 3 := by nlinarith [sq_nonneg ε, h_ε_pos, h_field_size]
     have h2 : ε^2 * 2 * 3 - 2 * ε + 2 ≥ 1 := by nlinarith [h_ε_bound, h_ε_pos]
     nlinarith [h1, h2, h_field_size]
   linarith [h_main]
   ```

**Files Modified**: `ForkingInfrastructure.lean:394` (-1 sorry)

**Success Criteria**:
- ✅ `fork_success_bound` fully proven
- ✅ Sorry count: 9 → 6 (closes 3)

---

## Phase 4: Probability Counting (heavy_row_lemma)

**Goal**: Prove pigeonhole principle for heavy commitments  
**Duration**: 2-3h  
**Depends**: Phase 2 (PMF formalization)  
**Status**: 🟡 Important

### Implementation Strategy

#### Step 4.1: Define Commitment Distribution (30min)
**Tasks**:
1. Formalize commitment probability:
   ```lean
   def commitment_prob {F : Type} [Field F] [Fintype F]
       (VC : VectorCommitment F) (cs : R1CS F)
       (A : Adversary F VC) (x : PublicInput F cs.nPub)
       (c : VC.Commitment × VC.Commitment × VC.Commitment × VC.Commitment) : ℝ :=
     (run_adversary VC cs A x secParam).toMeasure 
       {t | t.commitments = c}
   ```

**Files Modified**: `ForkingInfrastructure.lean` (+15 lines)

#### Step 4.2: Partition into Heavy/Light (1h)
**Tasks**:
1. Define partition:
   ```lean
   let all_comms := (Finset.univ : Finset CommitmentTuple)
   let heavy_comms := all_comms.filter (is_heavy_commitment VC cs x · ε)
   let light_comms := all_comms.filter (¬is_heavy_commitment VC cs x · ε)
   ```

2. Prove disjoint union:
   ```lean
   have h_partition : all_comms = heavy_comms ∪ light_comms := ...
   have h_disjoint : Disjoint heavy_comms light_comms := ...
   ```

**Files Modified**: `ForkingInfrastructure.lean` (+20 lines)

#### Step 4.3: Weighted Average Argument (1-1.5h)
**Tasks**:
1. Total success as sum:
   ```lean
   have h_total : ε ≤ ∑ c ∈ all_comms, 
       commitment_prob VC cs A x c * (valid_challenges c).card / Fintype.card F :=
     h_success
   ```

2. Bound heavy contribution:
   ```lean
   have h_heavy_contrib : ∑ c ∈ heavy_comms, 
       commitment_prob VC cs A x c * (valid_challenges c).card / Fintype.card F ≥
     ∑ c ∈ heavy_comms, commitment_prob VC cs A x c * ε := by
     -- From is_heavy_commitment definition
     apply Finset.sum_le_sum
     intro c hc
     have : (valid_challenges c).card ≥ ε * Fintype.card F := hc
     nlinarith
   ```

3. Bound light contribution:
   ```lean
   have h_light_contrib : ∑ c ∈ light_comms, 
       commitment_prob VC cs A x c * (valid_challenges c).card / Fintype.card F <
     ∑ c ∈ light_comms, commitment_prob VC cs A x c * ε := by
     -- From ¬is_heavy_commitment
     apply Finset.sum_lt_sum_of_nonempty
     ...
   ```

4. Derive contradiction:
   ```lean
   -- If (heavy_comms.card : ℝ) < (ε - 1/|F|) * secParam, then
   -- total_prob < ε (contradiction with h_success)
   by_contra h_not_enough
   have h_bound : ∑ c ∈ all_comms, ... < ε := by
     -- Weighted average with too few heavy commitments
     calc ...
       _ = (heavy contribution) + (light contribution)
       _ < (ε - 1/|F|) * secParam * ε + (1 - (ε - 1/|F|) * secParam) * ε
       _ < ε  -- Algebraic manipulation
   linarith [h_total, h_bound]
   ```

**Files Modified**: `ForkingInfrastructure.lean:269` (-1 sorry)

**Success Criteria**:
- ✅ `heavy_row_lemma` fully proven
- ✅ Sorry count: 6 → 5

---

## Phase 5: Forking Integration (forking_lemma)

**Goal**: Close 3 sorry in forking_lemma  
**Duration**: 2-3h  
**Depends**: Phases 1, 2  
**Status**: 🟡 Important

### Implementation Strategy

#### Step 5.1: Nonemptiness (Line 184, 30min)
**Tasks**:
1. Prove `(ε - 1/|F|) * secParam > 0`:
   ```lean
   have h_nonempty : heavy_comms.Nonempty := by
     by_contra h_empty
     simp [Finset.not_nonempty_iff_eq_empty] at h_empty
     rw [h_empty] at h_card
     simp at h_card
     -- h_card: 0 ≥ (ε - 1/|F|) * secParam
     -- Need: secParam > 0 (add hypothesis)
     have h_secParam_pos : secParam > 0 := by sorry -- TODO: Add as hypothesis
     have h_rhs_pos : (ε - 1/(Fintype.card F : ℝ)) * secParam > 0 := by
       apply mul_pos
       · linarith [h_ε_pos, h_field_size]
       · exact Nat.cast_pos.mpr h_secParam_pos
     linarith [h_card, h_rhs_pos]
   ```

**Files Modified**: `Soundness.lean:184` (-1 sorry), add secParam > 0 hypothesis

#### Step 5.2: Fork Extraction (Line 213, 1-1.5h)
**Tasks**:
1. Apply heavy_row_lemma + fork_success_bound:
   ```lean
   have h_fork_exists : ∃ (t1 t2 : Transcript F VC),
       is_valid_fork VC t1 t2 := by
     -- Step A: Get heavy commitment from heavy_row_lemma
     have h_heavy_exist := h_heavy
     obtain ⟨heavy_comms, h_card, h_all_heavy⟩ := h_heavy_exist
     have h_nonempty : heavy_comms.Nonempty := ... -- From Step 5.1
     obtain ⟨c, hc⟩ := h_nonempty
     
     -- Step B: c is heavy → many valid challenges
     have h_c_heavy : is_heavy_commitment VC cs x c ε := h_all_heavy c hc
     
     -- Step C: Run adversary with commitment c
     let state : AdversaryState F VC := {
       randomness := ..., -- Extract from adversary
       comm_Az := c.1, comm_Bz := c.2.1, 
       comm_Cz := c.2.2.1, comm_quotient := c.2.2.2,
       respond := ... -- From adversary structure
     }
     
     -- Step D: First run
     let t1_dist := run_adversary VC cs A x secParam
     -- Condition on commitment = c
     let t1_cond := ... -- Conditional probability
     
     -- Step E: Second run (rewind)
     -- Sample t1, if valid, rewind with new challenge
     have h_fork_prob : ... := by
       -- Apply fork_success_bound with heavy commitment
       apply fork_success_bound VC state (valid_challenges_of c) ε
       exact h_c_heavy
       ...
     
     -- Step F: Existence from probability bound
     -- If Pr[fork] ≥ ε²/2 - 1/|F| > 0, then ∃ fork
     sorry -- Requires: probability > 0 → event occurs (measurability)
   ```

**Files Modified**: `Soundness.lean:213` (-1 sorry, may add lemma)

#### Step 5.3: Public Input (Line 242, 30min-1h)
**Tasks**:
1. Connect transcript verification to public input:
   ```lean
   have h_pub : extractPublic cs.h_pub_le w = x := by
     -- Step A: Transcript structure includes public input
     -- verify checks: committed witness opens to consistent values
     -- First nPub values must match x (from verification)
     
     -- Step B: extract_witness deterministic from quotient
     -- w(i) = q.eval(ωⁱ) for i < nVars
     
     -- Step C: extractPublic takes first nPub elements
     unfold extractPublic
     ext i
     -- Goal: w(embed i) = x(i)
     
     -- Step D: Use Phase 1 verification structure
     -- verify checks public input consistency
     have h_t1_verify : verify VC cs x t1.proof = true := h_fork.left.right
     -- From verify definition (Phase 1.1):
     -- ∀ i < nPub, committed_value(i) = x(i)
     
     sorry -- Requires: verify → public input match (from Phase 1.1)
   ```

**Files Modified**: `Soundness.lean:242` (-1 sorry), `Core.lean` (+lemma from Phase 1)

**Success Criteria**:
- ✅ `forking_lemma` fully proven
- ✅ Sorry count: 5 → 2

---

## Phase 6: Polynomial Remainders (Polynomial.lean)

**Goal**: Close 2 sorry (lines 225, 247)  
**Duration**: 1-2h  
**Depends**: None (self-contained)  
**Status**: 🟢 Low Priority (can defer)

### Implementation Strategy

#### Step 6.1: Remainder Bound (Line 225, 30min)
**Tasks**:
1. Use degree_pos_of_ne_zero:
   ```lean
   by_cases h : f % g = 0
   · right; exact h
   · left
     apply Polynomial.natDegree_mod_lt
     exact hg
   ```

**Files Modified**: `Polynomial.lean:225` (-1 sorry)

#### Step 6.2: Quotient Uniqueness (Line 247, 30min-1h)
**Note**: May already be proven (check lines 338-365 for `quotient_uniqueness`)

**Tasks**:
1. If not proven, implement via degree contradiction:
   ```lean
   intro ⟨q', r'⟩ ⟨hq', hdeg'⟩
   -- From: f = q' * g + r' and f = (f/g) * g + (f%g)
   have h_diff : (q' - f/g) * g = (f%g) - r' := by
     linear_combination hq' - (EuclideanDomain.div_add_mod f g).symm
   
   -- Degree contradiction
   by_cases h_q_eq : q' = f/g
   · -- If q' = f/g, then r' = f%g follows
     subst h_q_eq
     have : (0 : Polynomial F) * g = (f%g) - r' := by rwa [sub_self] at h_diff
     simp at this
     linarith [hdeg', Polynomial.degree_mod_lt f hg]
   · -- If q' ≠ f/g, degree of LHS ≥ deg g, but RHS < deg g
     exfalso
     have h_lhs : ((q' - f/g) * g).natDegree ≥ g.natDegree := by
       apply Polynomial.natDegree_mul_ge_of_ne_zero
       · intro h_zero; exact h_q_eq (by simp [h_zero])
       · exact hg
     have h_rhs : ((f%g) - r').natDegree < g.natDegree := by
       apply Nat.lt_of_le_of_lt (Polynomial.natDegree_sub_le _ _)
       apply Nat.max_lt
       · exact Polynomial.natDegree_mod_lt _ hg
       · cases hdeg' with
         | inl h => exact h
         | inr h => rw [h]; exact Polynomial.natDegree_zero
     linarith [h_lhs, h_rhs]
   ```

**Files Modified**: `Polynomial.lean:247` (-1 sorry or verify already complete)

**Success Criteria**:
- ✅ Polynomial.lean sorry closed
- ✅ Sorry count: 2 → 0 or 1 (depending on line 247 status)

---

## Phase 7: Final Composition (knowledge_soundness)

**Goal**: Prove main soundness theorem  
**Duration**: 2-3h  
**Depends**: Phases 1-5 complete  
**Status**: 🟡 Important

### Implementation Strategy

#### Step 7.1: Construct Extractor (1h)
**Tasks**:
1. Implement `forking_extractor` fully:
   ```lean
   noncomputable def forking_extractor {F : Type} [Field F] [Fintype F]
       (VC : VectorCommitment F) (cs : R1CS F) (m : ℕ) (ω : F)
       (hω : IsPrimitiveRoot ω m) (h_m : m = cs.nVars)
       (h_sis : ModuleSIS_Hard 256 2 12289 1024) :
       Extractor F VC := {
     extract := fun A x => do
       -- Run adversary
       let t1_dist := run_adversary VC cs A x secParam
       let t1 ← t1_dist
       if ¬t1.valid then return none
       
       -- Rewind with new challenge
       let state := ... -- Extract state from t1
       let t2_dist := rewind_adversary VC cs A x state t1.challenge_α ...
       let t2 ← t2_dist
       if ¬t2.valid then return none
       
       -- Check fork validity
       if ¬is_valid_fork VC t1 t2 then return none
       
       -- Extract witness
       let q := extract_quotient_diff VC cs t1 t2 ... m ω
       let w := extract_witness VC cs q m ω hω h_m
       return some w
     
     poly_time := by
       -- Runtime = O(adversary_time × 2 + poly(secParam))
       -- Each run: A.poly_time
       -- Extraction: polynomial in secParam
       sorry -- Formal complexity proof (can axiomatize)
   }
   ```

**Files Modified**: `ForkingInfrastructure.lean` (+60 lines)

#### Step 7.2: Prove knowledge_soundness (1-2h)
**Tasks**:
1. Apply forking_lemma:
   ```lean
   theorem knowledge_soundness ... := by
     -- Construct extractor E := forking_extractor
     use forking_extractor VC cs m ω hω h_m h_sis
     constructor
     · -- E.poly_time
       exact (forking_extractor ...).poly_time
     
     · intro x h_verify_exists
       -- If adversary succeeds with ε ≥ 1/poly(λ), then extraction succeeds
       
       -- Step A: Non-negligible success → apply forking_lemma
       have h_fork := forking_lemma VC cs A x ε secParam ... h_success
       obtain ⟨w, h_satisfies, h_pub, _⟩ := h_fork
       
       -- Step B: Return extracted witness
       use w
       exact ⟨h_satisfies, h_pub⟩
   ```

**Files Modified**: `Soundness.lean:310` (-1 sorry)

**Success Criteria**:
- ✅ `knowledge_soundness` fully proven
- ✅ Sorry count: 1 or 0 → 0
- ✅ **100% verification achieved** 🎉

---

## Phase 8: Verification & Documentation

**Goal**: Final audit and certification  
**Duration**: 1-2h  
**Status**: 🟢 Final

### Tasks

#### Step 8.1: Full Build Verification (15min)
```bash
cd formal
lake clean
lake build LambdaSNARK
# Expected: Build completed successfully (N jobs).
# Expected: 0 errors, 0 sorry warnings
```

#### Step 8.2: Axiom Audit (15min)
```bash
grep -r "axiom" LambdaSNARK/*.lean
# Expected results:
# - Core.lean: ModuleLWE_Hard, ModuleSIS_Hard (cryptographic assumptions only)
# - ForkingInfrastructure.lean: 0 axioms
# - Soundness.lean: 0 axioms
# - Polynomial.lean: 0 axioms
```

#### Step 8.3: Theorem Coverage (15min)
```lean
-- Verify all main theorems proven:
#check schwartz_zippel                -- ✅
#check quotient_exists_iff_satisfies  -- ✅
#check heavy_row_lemma                -- ✅
#check fork_success_bound             -- ✅
#check extraction_soundness           -- ✅
#check forking_lemma                  -- ✅
#check knowledge_soundness            -- ✅
-- Total: 15/15 theorems ✅
```

#### Step 8.4: Update Documentation (30min)
1. Update `FORMAL_VERIFICATION_AUDIT.md`:
   - Verification: 79% → 100% ✅
   - Sorry count: 14 → 0 ✅
   - Axioms: Document cryptographic assumptions only
   - Proof strategies: Link to code comments

2. Update `ROADMAP.md`:
   - S3 (Forking Lemma): COMPLETE ✅
   - S4 (Knowledge Soundness): COMPLETE ✅
   - Mark project: **Formally Verified** 🎉

3. Create `VERIFICATION_CERTIFICATE.md`:
   ```markdown
   # ΛSNARK-R Formal Verification Certificate
   
   **Date**: 2025-11-16 (completion)
   **Verification Level**: 100%
   **Theorem Count**: 15/15
   **Proof Lines**: ~2000 lines
   **Cryptographic Assumptions**: Module-SIS, Module-LWE
   
   ## Certified Properties
   - ✅ Soundness: Under Module-SIS, adversary cannot forge proofs
   - ✅ Knowledge Soundness: Extractor exists for any successful adversary
   - ✅ Completeness: Honest prover always convinces verifier
   - ✅ Zero-Knowledge: Simulator exists (deferred to S5)
   
   ## Verification Tools
   - Lean 4: v4.25.0
   - Mathlib: Latest stable
   - Build: 6030+ jobs, 0 errors
   ```

#### Step 8.5: Integration Tests (30min)
1. Test end-to-end proof workflow:
   ```lean
   -- Example: Simple circuit (x² = y)
   example : ∃ (w : Witness F 2), satisfies simple_square_circuit w := by
     use ![x_val, y_val]
     -- Verify constraint: w(0) * w(0) = w(1)
     apply satisfies_iff_constraint_zero.mpr
     intro i
     fin_cases i <;> simp [constraintPoly, simple_square_circuit]
   ```

2. Verify proof generation + verification:
   ```lean
   -- Soundness check: If verify passes, witness exists
   theorem soundness_check {F : Type} [Field F] [Fintype F]
       (VC : VectorCommitment F) (cs : R1CS F)
       (x : PublicInput F cs.nPub) (π : Proof F VC)
       (h_verify : verify VC cs x π = true)
       (h_sis : ModuleSIS_Hard 256 2 12289 1024) :
       ∃ w, satisfies cs w ∧ extractPublic cs.h_pub_le w = x := by
     -- Apply knowledge_soundness with adversary that returns π
     sorry -- Integration test placeholder
   ```

**Success Criteria**:
- ✅ 0 sorry
- ✅ 0 axioms beyond crypto
- ✅ Documentation updated
- ✅ **Project Certified: Formally Verified** 🏆

---

## Execution Strategy

### Recommended Sequence

**Week 1** (12-16h):
- Phase 1: Eliminate extraction axiom (5-8h)
- Phase 2: PMF formalization (3-4h)
- Phase 3: fork_success_bound (3-4h)

**Week 2** (8-12h):
- Phase 4: heavy_row_lemma (2-3h)
- Phase 5: forking_lemma (2-3h)
- Phase 6: Polynomial remainders (1-2h)
- Phase 7: knowledge_soundness (2-3h)

**Week 3** (2h):
- Phase 8: Final verification & documentation

### Parallel Work Opportunities

**Can Parallelize**:
- Phase 3 (combinatorics) ∥ Phase 2 (PMF) — independent
- Phase 6 (polynomial) ∥ Phase 4-5 — low priority, can defer

**Must Serialize**:
- Phase 1 → Phase 5.3 (public input depends on verification structure)
- Phase 2 → Phase 4 (probability counting needs PMF)
- Phase 1-5 → Phase 7 (final composition needs all pieces)

### Risk Mitigation

**High-Risk Items**:
1. **Phase 1.3** (Lagrange uniqueness): Complex interpolation theory
   - **Mitigation**: May axiomatize if Mathlib lemma missing (low risk)
   
2. **Phase 5.2** (Fork extraction): Probability → existence
   - **Mitigation**: Use PMF.exists_of_prob_pos if available

3. **Phase 7** (Complexity bounds): Formal poly-time proof
   - **Mitigation**: Can axiomatize poly_time property (non-critical)

**Contingency**:
- If Phase 1 exceeds 10h: Temporarily axiomatize verification→extraction, continue to Phase 7
- If Phase 4 probability theory missing: Axiomatize pigeonhole principle, document for future
- Target: 80-90% pure proofs, 10-20% documented axioms acceptable

---

## Success Metrics

### Completion Criteria

**Tier 1 (Must Have)** — Core Soundness:
- ✅ `extraction_soundness` proven (no axiom)
- ✅ `forking_lemma` proven
- ✅ `knowledge_soundness` proven
- ✅ Sorry count: 0
- ✅ Axioms: Only Module-SIS, Module-LWE

**Tier 2 (Should Have)** — Mathematical Completeness:
- ✅ All combinatorics proven (fork_success_bound)
- ✅ All probability bounds proven (heavy_row_lemma)
- ✅ All polynomial lemmas proven

**Tier 3 (Nice to Have)** — Polish:
- ✅ PMF constructions (not axioms)
- ✅ Poly-time proofs (not axioms)
- ✅ Integration tests passing

### Quality Gates

**Pre-Phase Checks**:
- [ ] Build passes before starting
- [ ] Sorry count tracked
- [ ] Dependencies verified

**Post-Phase Checks**:
- [ ] Build passes after completion
- [ ] Sorry count decreased
- [ ] No new axioms introduced
- [ ] Code review: proof strategies documented

**Final Certification**:
- [ ] `lake build`: 0 errors
- [ ] `grep sorry`: 0 results (ignoring comments)
- [ ] `grep axiom`: Only crypto assumptions
- [ ] Documentation: 100% coverage
- [ ] Peer review: 2 reviewers approve

---

## Timeline Summary

| Phase | Duration | Status | Blocking |
|-------|----------|--------|----------|
| 1. Extraction Axiom Elimination | 5-8h | 🔴 Critical | All soundness |
| 2. PMF Formalization | 3-4h | 🟡 Important | Phases 4-5 |
| 3. Combinatorics | 3-4h | 🟡 Important | Phase 7 |
| 4. Probability Counting | 2-3h | 🟡 Important | Phase 7 |
| 5. Forking Integration | 2-3h | 🟡 Important | Phase 7 |
| 6. Polynomial Remainders | 1-2h | 🟢 Low Priority | None |
| 7. Final Composition | 2-3h | 🟡 Important | Phases 1-5 |
| 8. Verification & Docs | 1-2h | 🟢 Final | Phase 7 |
| **TOTAL** | **20-30h** | 🎯 **Target** | **→ 100%** |

**Critical Path**: Phase 1 → Phase 2 → Phase 4 → Phase 5 → Phase 7 → Phase 8  
**Estimated Completion**: 2-3 weeks (with parallel work)  
**Final Deliverable**: **100% Formally Verified ΛSNARK-R** 🏆

---

## Contact & Review

**Primary Reviewers**:
- Formal Methods: [TBD]
- Cryptography: [TBD]
- Implementation: [TBD]

**Review Checkpoints**:
- After Phase 1: Extraction soundness architecture
- After Phase 3: Combinatorics correctness
- After Phase 7: Full soundness proof
- After Phase 8: Final certification

**Questions/Issues**: File in GitHub repo issues with `formal-verification` label

---

**Status**: Roadmap approved, ready for execution  
**Next Step**: Begin Phase 1 (Extraction Axiom Elimination)  
**Target**: 100% Verification by Dec 2025 ✅
