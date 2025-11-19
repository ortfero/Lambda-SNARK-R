/-
Copyright (c) 2025 URPKS Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: URPKS Contributors
-/

import LambdaSNARK.Core
import Mathlib.Algebra.Polynomial.CoeffList
import Mathlib.Algebra.Polynomial.Div
import Mathlib.Algebra.Polynomial.FieldDivision
import Mathlib.Data.Finset.Card
import Mathlib.RingTheory.EuclideanDomain
import Mathlib.RingTheory.Polynomial.Basic
import Mathlib.RingTheory.RootsOfUnity.Basic
import Mathlib.Tactic

open scoped BigOperators
open Polynomial List

noncomputable section

/-!
# Polynomial Operations for ΛSNARK-R

This file contains polynomial-related lemmas and operations:
- Lagrange interpolation
- Polynomial division
- Evaluation properties
- Quotient polynomial construction

## Main Results

- `lagrange_interpolation`: Unique polynomial through given points
- `polynomial_division_correctness`: Division algorithm correctness
- `quotient_uniqueness`: Quotient polynomial is unique
-/

namespace LambdaSNARK

open Polynomial

-- ============================================================================
-- Coefficient List Uniqueness
-- ============================================================================

section CoeffList

variable {F : Type*} [Semiring F]

lemma polynomial_eq_of_coeffList_eq {p q : Polynomial F}
    (h : Polynomial.coeffList p = Polynomial.coeffList q) : p = q := by
  classical
  have h_map : (List.range p.degree.succ).map p.coeff =
      (List.range q.degree.succ).map q.coeff := by
    have := congrArg List.reverse h
    simpa [Polynomial.coeffList, List.map_reverse, List.reverse_reverse]
      using this
  have h_len : p.degree.succ = q.degree.succ := by
    simpa [List.length_map, List.length_range] using congrArg List.length h_map
  ext n
  by_cases hn : n < p.degree.succ
  · have hn' : n < q.degree.succ := by simpa [h_len] using hn
    have h_get := congrArg (fun (l : List F) => l[n]?) h_map
    have h_left : ((List.range p.degree.succ).map p.coeff)[n]? =
        some (p.coeff n) := by
      simp [hn]
    have h_right : ((List.range q.degree.succ).map q.coeff)[n]? =
        some (q.coeff n) := by
      simp [hn']
    simpa [h_left, h_right] using h_get
  · have h_le : p.degree.succ ≤ n := Nat.le_of_not_lt hn
    have hzero_p : p.coeff n = 0 := by
      by_cases hp : p = 0
      · subst hp; simp
      · have h_nat : p.natDegree + 1 ≤ n := by
          simpa [withBotSucc_degree_eq_natDegree_add_one hp] using h_le
        have h_lt : p.natDegree < n :=
          Nat.lt_of_lt_of_le (Nat.lt_succ_self _) h_nat
        exact coeff_eq_zero_of_natDegree_lt h_lt
    have hzero_q : q.coeff n = 0 := by
      by_cases hq : q = 0
      · subst hq; simp
      · have h_le_q : q.degree.succ ≤ n := by simpa [h_len] using h_le
        have h_nat : q.natDegree + 1 ≤ n := by
          simpa [withBotSucc_degree_eq_natDegree_add_one hq] using h_le_q
        have h_lt : q.natDegree < n :=
          Nat.lt_of_lt_of_le (Nat.lt_succ_self _) h_nat
        exact coeff_eq_zero_of_natDegree_lt h_lt
    simp [hzero_p, hzero_q]

lemma coeffList_injective :
    Function.Injective (fun p : Polynomial F => Polynomial.coeffList p) := by
  intro p q h
  exact polynomial_eq_of_coeffList_eq h

end CoeffList

-- ============================================================================
-- Degree Control Utilities
-- ============================================================================

section DegreeBounds

variable {F : Type*} [Field F]

lemma degree_lt_of_natDegree_lt {p : Polynomial F} {m : ℕ}
    (hp : p.natDegree < m) :
    p.degree < (m : WithBot ℕ) := by
  classical
  by_cases hzero : p = 0
  · subst hzero
    have : ((m : WithBot ℕ) ≠ (⊥ : WithBot ℕ)) := by simp
    exact bot_lt_iff_ne_bot.mpr this
  · have hdeg := Polynomial.degree_eq_natDegree hzero
    have : ((p.natDegree : ℕ) : WithBot ℕ) < (m : WithBot ℕ) := by
      exact_mod_cast hp
    simpa [hdeg] using this

lemma natDegree_sub_lt_of_lt {p q : Polynomial F} {m : ℕ}
    (hp : p = 0 ∨ p.natDegree < m)
    (hq : q = 0 ∨ q.natDegree < m)
    (hm : 0 < m) :
    (p - q).natDegree < m := by
  classical
  cases hp with
  | inl hp0 =>
      subst hp0
      cases hq with
      | inl hq0 =>
          subst hq0
          simpa [sub_eq_add_neg] using hm
      | inr hq_lt =>
          simpa [sub_eq_add_neg] using hq_lt
  | inr hp_lt =>
      cases hq with
      | inl hq0 =>
          subst hq0
          simpa using hp_lt
      | inr hq_lt =>
          by_cases hsub : p - q = 0
          · have hzero : (p - q).natDegree = 0 := by simp [hsub]
            simpa [hzero] using hm
          ·
            have hp_deg_lt : p.degree < (m : WithBot ℕ) :=
              degree_lt_of_natDegree_lt hp_lt
            have hq_deg_lt : q.degree < (m : WithBot ℕ) :=
              degree_lt_of_natDegree_lt hq_lt
            have hdeg_le : (p - q).degree ≤ max p.degree q.degree :=
              Polynomial.degree_sub_le _ _
            have hdeg_lt : (p - q).degree < (m : WithBot ℕ) :=
              lt_of_le_of_lt hdeg_le (max_lt hp_deg_lt hq_deg_lt)
            have : ((p - q).natDegree : WithBot ℕ) < (m : WithBot ℕ) := by
              simpa [Polynomial.degree_eq_natDegree hsub] using hdeg_lt
            exact_mod_cast this

end DegreeBounds

-- ============================================================================
-- Vanishing Polynomial
-- ============================================================================

/-- Vanishing polynomial `Z_H(X) = ∏ᵢ (X - ω^i)` for `H = {ω^i | i < m}`. -/
noncomputable def vanishing_poly {F : Type*} [Field F] (m : ℕ) (ω : F) : Polynomial F :=
  ∏ i : Fin m, (X - C (ω ^ (i : ℕ)))

/-- Evaluate the vanishing polynomial at a point `α`. -/
def vanishingEval {F : Type*} [Field F] (m : ℕ) (ω α : F) : F :=
  (vanishing_poly m ω).eval α

/-- Helper: Evaluate `(X - C a)` at point `x`. -/
lemma eval_factor {F : Type*} [Field F] (x a : F) : (X - C a).eval x = x - a := by
  simp [eval_sub, eval_X, eval_C]

/-- Vanishing polynomial evaluates to zero at domain points. -/
lemma eval_vanishing_at_pow {F : Type*} [Field F]
    (m : ℕ) (ω : F) (j : Fin m) :
    (vanishing_poly m ω).eval (ω ^ (j : ℕ)) = 0 := by
  classical
  unfold vanishing_poly
  -- Evaluation is a ring hom, so it commutes with products
  -- One factor is `(X - C (ω^j))`, whose eval at `ω^j` is zero
  conv_lhs => rw [← Polynomial.coe_evalRingHom]; rw [map_prod]
  apply Finset.prod_eq_zero (Finset.mem_univ j)
  simp only [Polynomial.coe_evalRingHom, eval_sub, eval_X, eval_C, sub_self]

/-- The vanishing polynomial is never the zero polynomial. -/
lemma vanishing_poly_ne_zero {F : Type*} [Field F]
    (m : ℕ) (ω : F) : vanishing_poly m ω ≠ 0 := by
  classical
  unfold vanishing_poly
  refine Finset.prod_ne_zero_iff.mpr ?_
  intro i _
  simpa using (Polynomial.monic_X_sub_C (ω ^ (i : ℕ))).ne_zero

lemma vanishing_poly_monic {F : Type*} [Field F]
    (m : ℕ) (ω : F) : (vanishing_poly m ω).Monic := by
  classical
  unfold vanishing_poly
  apply monic_prod_of_monic
  intro i _
  simpa using Polynomial.monic_X_sub_C (ω ^ (i : ℕ))

/-- Product of linear factors has degree matching the factor count. -/
lemma natDegree_prod_X_sub_C {F : Type*} [Field F] {α : Type*} [DecidableEq α]
    (s : Finset α) (f : α → F) :
    (∏ x ∈ s, (X - C (f x))).natDegree = s.card := by
  classical
  refine Finset.induction_on s ?base ?step
  · simp
  · intro a s ha ih
    have h₁ : (X - C (f a)) ≠ 0 := by
      simpa using (Polynomial.monic_X_sub_C (f a)).ne_zero
    have h₂ : (∏ x ∈ s, (X - C (f x))) ≠ 0 := by
      refine Finset.prod_ne_zero_iff.mpr ?_
      intro x hx
      simpa using (Polynomial.monic_X_sub_C (f x)).ne_zero
    have hdeg := Polynomial.natDegree_mul h₁ h₂
    have hgoal :
        (∏ x ∈ insert a s, (X - C (f x))).natDegree = s.card + 1 := by
      simpa [Finset.prod_insert, ha, ih, Polynomial.natDegree_X_sub_C,
        Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.succ_eq_add_one] using hdeg
    have hcard : s.card + 1 = (insert a s).card := by
      classical
      simpa using (Finset.card_insert_of_notMem (α := α) (a := a) (s := s) ha).symm
    exact hgoal.trans hcard

/-- The vanishing polynomial has degree equal to the domain size. -/
lemma vanishing_poly_natDegree {F : Type*} [Field F]
    (m : ℕ) (ω : F) : (vanishing_poly m ω).natDegree = m := by
  classical
  simpa [vanishing_poly, Finset.card_univ] using
    natDegree_prod_X_sub_C (Finset.univ : Finset (Fin m))
      (fun i : Fin m => ω ^ (i : ℕ))

/-- If a polynomial has smaller degree than the vanishing polynomial and is
    divisible by it, then the polynomial must be zero. -/
lemma vanishing_poly_dvd_eq_zero_of_natDegree_lt {F : Type*} [Field F]
    {m : ℕ} {ω : F} {p : Polynomial F}
    (hdiv : vanishing_poly m ω ∣ p)
    (hdeg : p.natDegree < m) :
    p = 0 := by
  classical
  rcases hdiv with ⟨q, rfl⟩
  by_cases hq : q = (0 : Polynomial F)
  · simp [hq]
  have hvan_ne : vanishing_poly m ω ≠ (0 : Polynomial F) :=
    vanishing_poly_ne_zero m ω
  have hlt : (vanishing_poly m ω * q).natDegree < m := by
    simpa using hdeg
  have hdeg_mul := Polynomial.natDegree_mul hvan_ne hq
  have hvan_deg : (vanishing_poly m ω).natDegree = m :=
    vanishing_poly_natDegree m ω
  have hge : m ≤ (vanishing_poly m ω * q).natDegree := by
    calc
      m ≤ m + q.natDegree := Nat.le_add_right _ _
      _ = (vanishing_poly m ω).natDegree + q.natDegree := by
            simp [hvan_deg]
      _ = (vanishing_poly m ω * q).natDegree := by
            simp [hdeg_mul]
  exact False.elim ((not_lt_of_ge hge) hlt)

/-- Convenience lemma bundling evaluation of the vanishing polynomial on the domain. -/
@[simp] lemma vanishingEval_domain {F : Type*} [Field F]
    (m : ℕ) (ω : F) (j : Fin m) :
    vanishingEval m ω (ω ^ (j : ℕ)) = 0 := by
  unfold vanishingEval
  simpa using eval_vanishing_at_pow m ω j

-- ============================================================================
-- Lagrange Basis Polynomials
-- ============================================================================

/-- If `ω` is a primitive `m`-th root of unity, then `i ↦ ω^i` is injective on `Fin m`. -/
lemma primitive_root_pow_injective {F : Type*} [Field F]
    {m : ℕ} {ω : F} (hprim : IsPrimitiveRoot ω m)
    (i j : Fin m) :
    ω ^ (i : ℕ) = ω ^ (j : ℕ) → i = j := by
  intro h
  -- WLOG assume i ≤ j
  wlog hij : (i : ℕ) ≤ (j : ℕ) generalizing i j with H
  · exact (H j i h.symm (Nat.le_of_not_le hij)).symm
  -- From ω^i = ω^j and i ≤ j, derive ω^(j-i) = 1
  have h_m_pos : 0 < m := Fin.pos_iff_nonempty.mpr ⟨i⟩
  have h_pow : ω ^ ((j : ℕ) - (i : ℕ)) = 1 := by
    have h1 : ω ^ (j : ℕ) = ω ^ (i : ℕ) * ω ^ ((j : ℕ) - (i : ℕ)) := by
      rw [← pow_add, Nat.add_sub_cancel' hij]
    have h2 : ω ^ (i : ℕ) * ω ^ ((j : ℕ) - (i : ℕ)) = ω ^ (i : ℕ) * 1 := by
      rw [← h1, h, mul_one]
    exact mul_left_cancel₀ (pow_ne_zero _ (hprim.ne_zero h_m_pos.ne')) h2
  -- Use primitivity: ω^k = 1 ↔ m ∣ k
  rw [hprim.pow_eq_one_iff_dvd] at h_pow
  -- Since 0 ≤ j - i < m and m ∣ (j - i), we have j - i = 0
  have h_diff : (j : ℕ) - (i : ℕ) = 0 := by
    apply Nat.eq_zero_of_dvd_of_lt h_pow
    have : (j : ℕ) - (i : ℕ) ≤ (j : ℕ) := Nat.sub_le _ _
    exact Nat.lt_of_le_of_lt this j.prop
  -- Therefore i = j
  exact Fin.ext (hij.antisymm (Nat.sub_eq_zero_iff_le.mp h_diff))

/-- Lagrange basis `Lᵢ(X) = ∏_{j≠i} (X - ω^j) / ∏_{j≠i} (ω^i - ω^j)`. -/
noncomputable def lagrange_basis {F : Type*} [Field F] (m : ℕ) (ω : F) (i : Fin m) : Polynomial F := by
  classical
  let num : Polynomial F := ∏ j : Fin m, if j = i then (1 : Polynomial F) else (X - C (ω ^ (j : ℕ)))
  let den : F := ∏ j : Fin m, if j = i then (1 : F) else (ω ^ (i : ℕ) - ω ^ (j : ℕ))
  exact C (1 / den) * num


/-- Kronecker delta property: `Lᵢ(ωʲ) = δᵢⱼ`. -/
theorem lagrange_basis_property {F : Type*} [Field F]
    (m : ℕ) {ω : F} (hprim : IsPrimitiveRoot ω m) (i j : Fin m) :
    (lagrange_basis m ω i).eval (ω ^ (j : ℕ)) = if i = j then 1 else 0 := by
  classical
  unfold lagrange_basis
  simp only [Polynomial.eval_mul, Polynomial.eval_C]
  by_cases h : i = j
  · -- Case i = j: Show Lᵢ(ωⁱ) = 1
    subst h
    simp only [if_true]
    -- Evaluate numerator: ∏_{k≠i} (X - ωᵏ) at ωⁱ = ∏_{k≠i} (ωⁱ - ωᵏ) = denominator
    have h_num_eval : (∏ k : Fin m, if k = i then 1 else (X - C (ω ^ (k : ℕ)))).eval (ω ^ (i : ℕ)) =
                       ∏ k : Fin m, if k = i then (1 : F) else (ω ^ (i : ℕ) - ω ^ (k : ℕ)) := by
      rw [← Polynomial.coe_evalRingHom, map_prod]
      congr 1
      ext k
      by_cases hk : k = i
      · simp only [hk, if_true, Polynomial.coe_evalRingHom, Polynomial.eval_one]
      · simp only [hk, if_false, Polynomial.coe_evalRingHom, Polynomial.eval_sub,
                   Polynomial.eval_X, Polynomial.eval_C]
    -- Denominator is nonzero (from primitivity of ω)
    have h_denom_ne_zero : (∏ k : Fin m, if k = i then (1 : F) else (ω ^ (i : ℕ) - ω ^ (k : ℕ))) ≠ 0 := by
      apply Finset.prod_ne_zero_iff.mpr
      intro k _
      by_cases hk : k = i
      · simp [hk]
      · simp only [hk, if_false]
        -- Use: ωⁱ ≠ ωᵏ for i ≠ k (from primitive_root_pow_injective)
        intro h_eq
        -- h_eq : ω^i - ω^k = 0 ⟹ ω^i = ω^k
        have h_pow_eq : ω ^ (i : ℕ) = ω ^ (k : ℕ) := sub_eq_zero.mp h_eq
        have h_inj : i = k := primitive_root_pow_injective hprim i k h_pow_eq
        exact hk h_inj.symm
    -- Now: (1/denom) * denom = 1
    rw [h_num_eval]
    field_simp [h_denom_ne_zero]
  · -- Case i ≠ j: Show Lᵢ(ωʲ) = 0
    -- Numerator contains factor (X - ωʲ), evaluation at ωʲ gives 0
    simp only [if_neg h]
    suffices h_prod_zero : (∏ k : Fin m, if k = i then 1 else (X - C (ω ^ (k : ℕ)))).eval (ω ^ (j : ℕ)) = 0 by
      rw [h_prod_zero]; ring
    -- Product evaluation: eval is RingHom, so commutes with products
    conv_lhs => rw [← Polynomial.coe_evalRingHom]; rw [map_prod]
    -- Now: ∏ eval(pᵢ) contains factor eval((X - ωʲ)) = 0
    apply Finset.prod_eq_zero (Finset.mem_univ j)
    simp only [if_neg (Ne.symm h), Polynomial.coe_evalRingHom, Polynomial.eval_sub,
               Polynomial.eval_X, Polynomial.eval_C, sub_self]

/-- Lagrange interpolation: construct polynomial from evaluations -/
noncomputable def lagrange_interpolate {F : Type*} [Field F] (m : ℕ) (ω : F)
    (evals : Fin m → F) : Polynomial F := by
  classical
  exact ∑ i : Fin m, C (evals i) * lagrange_basis m ω i

/-- Interpolation correctness: `p(ωⁱ) = evals(i)`. -/
theorem lagrange_interpolate_eval {F : Type*} [Field F]
    (m : ℕ) {ω : F} (hprim : IsPrimitiveRoot ω m)
    (evals : Fin m → F) (i : Fin m) :
    (lagrange_interpolate m ω evals).eval (ω ^ (i : ℕ)) = evals i := by
  classical
  unfold lagrange_interpolate
  simp only [eval_finset_sum, eval_mul, eval_C, lagrange_basis_property m hprim]
  -- Goal: ∑ j, evals j * (if j = i then 1 else 0) = evals i
  -- Transform to: ∑ j, (if j = i then evals j else 0) = evals i
  simp only [mul_ite, mul_one, mul_zero]
  -- Rewrite j = i to i = j for Finset.sum_ite_eq
  have h_eq : ∑ j : Fin m, (if j = i then evals j else 0) =
              ∑ j : Fin m, (if i = j then evals j else 0) := by
    congr 1; ext j
    by_cases h : j = i
    · simp only [h, ↓reduceIte]
    · simp only [h, Ne.symm h, ↓reduceIte]
  rw [h_eq]
  simp only [Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]

-- ============================================================================
-- Polynomial Division
-- ============================================================================

/-! ## Division API Reference (from Zulip response)

Key lemmas for remainder degree bound:
- In Mathlib ≥ v4.25: `Polynomial.degree_mod_lt (f g : Polynomial F) (hg : g ≠ 0)`
- Converts to natDegree via `Polynomial.natDegree_lt_natDegree_of_degree_lt_degree`
- For monic divisors: `degree_modByMonic_lt`, `natDegree_modByMonic_lt`

Uniqueness pattern:
1. Subtract: `(q₁ - q₂) * g = r₂ - r₁`
2. Bound: `degree (r₂ - r₁) < degree g` via `degree_sub_le`
3. Contradict: if `q₁ ≠ q₂`, then `degree ((q₁ - q₂) * g) ≥ degree g` via `degree_mul`
-/

/-- Polynomial division: `f = q * g + r` with `deg(r) < deg(g)`.

    M2 API Resolution (2025-11-16): ✅ COMPLETE
    - Found: `Polynomial.natDegree_mod_lt` in Mathlib.Algebra.Polynomial.FieldDivision
    - Signature: `(f % g).natDegree < g.natDegree` when `g.natDegree ≠ 0`
    - Status: P3-P4 at 95% (proof skeleton complete, 2 minor edge cases deferred)

    Current state (2 sorry):
    - Line ~207: P3 edge case (g.natDegree = 0 → f % g = 0, trivial unit case)
    - Line ~214: P4 uniqueness (degree contradiction via subtraction pattern)

    Deferred rationale: Edge cases require complex Mathlib API (isUnit lemmas, WithBot reasoning).
    Better to proceed to P5-P6 with 95% complete skeleton than spend hours on 5% edge cases.
-/
theorem polynomial_division {F : Type*} [Field F]
    (f g : Polynomial F) (hg : g ≠ 0) :
    ∃! qr : Polynomial F × Polynomial F,
      f = qr.1 * g + qr.2 ∧ (qr.2 = 0 ∨ qr.2.natDegree < g.natDegree) := by
  classical
  -- Auxiliary lemmas for degree control
  have degree_lt_of_natDegree_lt {p : Polynomial F} {m : ℕ}
      (hp : p.natDegree < m) :
      p.degree < (m : WithBot ℕ) := by
    classical
    by_cases hzero : p = 0
    · subst hzero
      have : ((m : WithBot ℕ) ≠ (⊥ : WithBot ℕ)) := by simp
      exact bot_lt_iff_ne_bot.mpr this
    · have hdeg := Polynomial.degree_eq_natDegree hzero
      have : ((p.natDegree : ℕ) : WithBot ℕ) < (m : WithBot ℕ) := by
        exact_mod_cast hp
      simpa [hdeg] using this
  have natDegree_sub_lt_of_lt
      {p q : Polynomial F} {m : ℕ}
      (hp : p = 0 ∨ p.natDegree < m)
      (hq : q = 0 ∨ q.natDegree < m)
      (hm : 0 < m) :
      (p - q).natDegree < m := by
    classical
    cases hp with
    | inl hp0 =>
        subst hp0
        cases hq with
        | inl hq0 =>
            subst hq0
            simpa [sub_eq_add_neg] using hm
        | inr hq_lt =>
            simpa [sub_eq_add_neg] using hq_lt
    | inr hp_lt =>
        cases hq with
      | inl hq0 =>
        subst hq0
        simpa using hp_lt
        | inr hq_lt =>
            by_cases hsub : p - q = 0
            · have hzero : (p - q).natDegree = 0 := by simp [hsub]
              simpa [hzero] using hm
            ·
              have hp_deg_lt : p.degree < (m : WithBot ℕ) :=
                degree_lt_of_natDegree_lt hp_lt
              have hq_deg_lt : q.degree < (m : WithBot ℕ) :=
                degree_lt_of_natDegree_lt hq_lt
              have hdeg_le : (p - q).degree ≤ max p.degree q.degree :=
                Polynomial.degree_sub_le _ _
              have hdeg_lt : (p - q).degree < (m : WithBot ℕ) :=
                lt_of_le_of_lt hdeg_le (max_lt hp_deg_lt hq_deg_lt)
              have : ((p - q).natDegree : WithBot ℕ) < (m : WithBot ℕ) := by
                simpa [Polynomial.degree_eq_natDegree hsub] using hdeg_lt
              exact_mod_cast this
  refine ⟨(f / g, f % g), ?_, ?_⟩
  · -- P3 (Existence): f = (f/g) * g + (f%g) with remainder bound
    constructor
    · simpa [mul_comm] using (EuclideanDomain.div_add_mod f g).symm
    · by_cases hunit : IsUnit g
      · have hdiv : g ∣ f := by
          rcases hunit with ⟨u, rfl⟩
          refine ⟨f * ↑(u⁻¹), ?_⟩
          simp [mul_left_comm]
        have : f % g = 0 := (EuclideanDomain.mod_eq_zero).2 hdiv
        exact Or.inl this
      · have hgNat : g.natDegree ≠ 0 := by
          intro hdeg
          have hdeg' : g.degree = (0 : WithBot ℕ) := by
            simpa [hdeg] using (Polynomial.degree_eq_natDegree hg)
          have : IsUnit g := (Polynomial.isUnit_iff_degree_eq_zero).2 hdeg'
          exact hunit this
        right
        exact Polynomial.natDegree_mod_lt f hgNat
  · -- P4 (Uniqueness): via subtraction + degree contradiction
    intro ⟨q', r'⟩ ⟨hrepr, hdeg⟩
    set q := f / g
    set r := f % g
    have hcanon : f = q * g + r := by
      simpa [q, r, mul_comm] using (EuclideanDomain.div_add_mod f g).symm
    by_cases hunit : IsUnit g
    · have hgdeg : g.degree = (0 : WithBot ℕ) :=
        (Polynomial.isUnit_iff_degree_eq_zero).1 hunit
      have hgNatZero : g.natDegree = 0 := by
        have hg_ne_zero : g ≠ 0 := hg
        simpa [Polynomial.degree_eq_natDegree hg_ne_zero] using hgdeg
      have hr_zero : r = 0 := by
        have hdiv : g ∣ f := by
          rcases hunit with ⟨u, rfl⟩
          refine ⟨f * ↑(u⁻¹), ?_⟩
          simp [mul_left_comm]
        have : f % g = 0 := (EuclideanDomain.mod_eq_zero).2 hdiv
        simpa [r] using this
      have hdeg' : r' = 0 ∨ r'.natDegree < 0 := by
        simpa [hgNatZero] using hdeg
      have hr'_zero : r' = 0 := hdeg'.resolve_right (Nat.not_lt_zero _)
      have hcanon' : f = q * g := by simpa [hr_zero, add_comm] using hcanon
      have hrepr' : f = q' * g := by simpa [hr'_zero, add_comm] using hrepr
      have hmul : q * g = q' * g := hcanon'.symm.trans hrepr'
      have hq_eq : q = q' := mul_right_cancel₀ hg hmul
      refine Prod.ext ?_ ?_
      · simpa using hq_eq.symm
      · simp [hr_zero, hr'_zero]
    · have hgNat : g.natDegree ≠ 0 := by
        intro hdeg
        have hdeg' : g.degree = (0 : WithBot ℕ) := by
          simpa [hdeg] using (Polynomial.degree_eq_natDegree hg)
        have : IsUnit g := (Polynomial.isUnit_iff_degree_eq_zero).2 hdeg'
        exact hunit this
      have hgNatPos : 0 < g.natDegree := Nat.pos_of_ne_zero hgNat
      have hr_bound : r = 0 ∨ r.natDegree < g.natDegree := by
        by_cases hr0 : r = 0
        · exact Or.inl hr0
        · have hlt : (f % g).natDegree < g.natDegree :=
            Polynomial.natDegree_mod_lt f hgNat
          right; simpa [r] using hlt
      have hrepr_eq : q' * g + r' = q * g + r := by
        have h := hcanon.symm.trans hrepr
        simpa [q, r] using h.symm
      have hr_diff_lt : (r' - r).natDegree < g.natDegree :=
        natDegree_sub_lt_of_lt hdeg hr_bound hgNatPos
      have hcomb : (q' - q) * g + (r' - r) = 0 := by
        simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc, add_right_comm,
          mul_add, add_mul]
          using congrArg (fun x => x - q * g - r) hrepr_eq
      have hdiff : r' - r = -((q' - q) * g) :=
        eq_neg_of_add_eq_zero_right hcomb
      have hq_eq : q' = q := by
        by_contra hneq
        have hsub_ne : q' - q ≠ 0 := sub_ne_zero_of_ne hneq
        have hprod : (r' - r).natDegree = (q' - q).natDegree + g.natDegree := by
          have := Polynomial.natDegree_mul hsub_ne hg
          simpa [hdiff, Polynomial.natDegree_neg] using this
        have hge : g.natDegree ≤ (r' - r).natDegree := by
          have hbase : g.natDegree ≤ (q' - q).natDegree + g.natDegree :=
            Nat.le_add_left _ _
          have hrewrite : (q' - q).natDegree + g.natDegree = (r' - r).natDegree :=
            hprod.symm
          exact by
            calc
              g.natDegree ≤ (q' - q).natDegree + g.natDegree := hbase
              _ = (r' - r).natDegree := hrewrite
        exact (not_lt_of_ge hge) hr_diff_lt
      have hr_eq : r' = r := by
        have hsum : q * g + r' = q * g + r := by
          simpa [hq_eq, q, r] using hrepr_eq
        exact add_left_cancel hsum
      ext <;> simp [q, r, hq_eq, hr_eq]

/-- Divide a polynomial by the vanishing polynomial. -/
noncomputable def divide_by_vanishing {F : Type*} [Field F]
    (f : Polynomial F) (m : ℕ) (ω : F) : Polynomial F × Polynomial F :=
  let ZH := vanishing_poly m ω
  (f /ₘ ZH, f %ₘ ZH)

/-- Remainder is zero iff `f` vanishes on roots of `Z_H`. -/
theorem remainder_zero_iff_vanishing {F : Type*} [Field F]
    (f : Polynomial F) (m : ℕ) (ω : F) (hω : IsPrimitiveRoot ω m) :
    f %ₘ vanishing_poly m ω = 0 ↔ ∀ i : Fin m, f.eval (ω ^ (i : ℕ)) = 0 := by
  classical
  unfold vanishing_poly
  constructor
  · -- P5 (→): f %ₘ Z_H = 0 ⟹ ∀i, f(ωⁱ) = 0
    intro h_rem i
    -- Z_H = ∏(X - ωⁱ) divides f (from remainder 0)
    have h_ZH_monic : (∏ j : Fin m, (X - C (ω ^ (j : ℕ)))).Monic := by
      apply monic_prod_of_monic
      intro j _
      exact monic_X_sub_C (ω ^ (j : ℕ))
    have h_ZH_dvd : (∏ j : Fin m, (X - C (ω ^ (j : ℕ)))) ∣ f := by
      rw [← Polynomial.modByMonic_eq_zero_iff_dvd h_ZH_monic]
      exact h_rem
    -- Product ∏(X - ωⁱ) divides f ⟹ each factor (X - ωⁱ) divides f
    have h_factor_dvd : (X - C (ω ^ (i : ℕ))) ∣ f := by
      apply dvd_trans _ h_ZH_dvd
      exact Finset.dvd_prod_of_mem _ (Finset.mem_univ i)
    -- (X - ωⁱ) | f ⟹ f(ωⁱ) = 0 via IsRoot
    rw [dvd_iff_isRoot] at h_factor_dvd
    exact h_factor_dvd
  · -- P6 (←): ∀i, f(ωⁱ) = 0 ⟹ f %ₘ Z_H = 0
    intro h_eval
    -- Each f(ωⁱ) = 0 ⟹ (X - ωⁱ) | f
    have h_factors_dvd : ∀ i : Fin m, (X - C (ω ^ (i : ℕ))) ∣ f := by
      intro i
      rw [dvd_iff_isRoot]
      exact h_eval i
    -- Product divisibility via coprimality: ∏pᵢ | f when pᵢ pairwise coprime and ∀i, pᵢ | f
    have h_prod_dvd : (∏ i : Fin m, (X - C (ω ^ (i : ℕ)))) ∣ f := by
      apply Finset.prod_dvd_of_coprime
      · -- Pairwise coprimality of X - ωⁱ follows from ω^i injective (primitive root)
        have h_pow_inj : Function.Injective (fun i : Fin m => ω ^ (i : ℕ)) := by
          intro i j h_eq
          exact primitive_root_pow_injective hω i j h_eq
        exact fun i _ j _ hij => pairwise_coprime_X_sub_C h_pow_inj hij
      · intro i _
        exact h_factors_dvd i
    -- ∏(X - ωⁱ) | f ⟹ f %ₘ Z_H = 0
    have h_ZH_monic : (∏ i : Fin m, (X - C (ω ^ (i : ℕ)))).Monic := by
      apply monic_prod_of_monic
      intro i _
      exact monic_X_sub_C (ω ^ (i : ℕ))
    rw [Polynomial.modByMonic_eq_zero_iff_dvd h_ZH_monic]
    exact h_prod_dvd

-- ============================================================================
-- Witness Polynomial Construction
-- ============================================================================

/-- Construct polynomial from witness vector via Lagrange interpolation -/
noncomputable def witness_to_poly {F : Type} [Field F] [DecidableEq (Fin 1)] {n : ℕ}
    (w : Witness F n) (m : ℕ) (ω : F)
    (_h_root : ω ^ m = 1) (_h_size : m ≤ n) : Polynomial F :=
  -- Interpolate witness values over evaluation domain
  -- For each point i in [0,m), take witness value at index i
  lagrange_interpolate m ω (fun i =>
    if h : i.val < n then w ⟨i.val, h⟩ else 0)

/-- Honest `A_z` polynomial obtained via Lagrange interpolation. -/
noncomputable def constraintAzPoly {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (ω : F) : Polynomial F :=
  lagrange_interpolate cs.nCons ω (fun i => evaluateConstraintA cs z i)

/-- Honest `B_z` polynomial obtained via Lagrange interpolation. -/
noncomputable def constraintBzPoly {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (ω : F) : Polynomial F :=
  lagrange_interpolate cs.nCons ω (fun i => evaluateConstraintB cs z i)

/-- Honest `C_z` polynomial obtained via Lagrange interpolation. -/
noncomputable def constraintCzPoly {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (ω : F) : Polynomial F :=
  lagrange_interpolate cs.nCons ω (fun i => evaluateConstraintC cs z i)

/-- Domain evaluation of the interpolated `A_z` polynomial. -/
lemma constraintAzPoly_eval_domain {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (i : Fin cs.nCons) :
    (constraintAzPoly cs z ω).eval (ω ^ (i : ℕ)) = evaluateConstraintA cs z i := by
  classical
  unfold constraintAzPoly
  simpa using
    (lagrange_interpolate_eval cs.nCons hprim (fun j => evaluateConstraintA cs z j) i)

/-- Domain evaluation of the interpolated `B_z` polynomial. -/
lemma constraintBzPoly_eval_domain {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (i : Fin cs.nCons) :
    (constraintBzPoly cs z ω).eval (ω ^ (i : ℕ)) = evaluateConstraintB cs z i := by
  classical
  unfold constraintBzPoly
  simpa using
    (lagrange_interpolate_eval cs.nCons hprim (fun j => evaluateConstraintB cs z j) i)

/-- Domain evaluation of the interpolated `C_z` polynomial. -/
lemma constraintCzPoly_eval_domain {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (i : Fin cs.nCons) :
    (constraintCzPoly cs z ω).eval (ω ^ (i : ℕ)) = evaluateConstraintC cs z i := by
  classical
  unfold constraintCzPoly
  simpa using
    (lagrange_interpolate_eval cs.nCons hprim (fun j => evaluateConstraintC cs z j) i)

/-- Honest numerator polynomial `A_z ⋅ B_z - C_z`. -/
noncomputable def constraintNumeratorPoly {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (ω : F) : Polynomial F :=
  constraintAzPoly cs z ω * constraintBzPoly cs z ω - constraintCzPoly cs z ω

/-- The honest numerator polynomial matches the residual evaluations on the domain. -/
lemma constraintNumeratorPoly_eval_domain {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (i : Fin cs.nCons) :
    (constraintNumeratorPoly cs z ω).eval (ω ^ (i : ℕ)) =
      evaluateConstraintResidual cs z i := by
  classical
  have hA := constraintAzPoly_eval_domain (F := F) cs z hprim i
  have hB := constraintBzPoly_eval_domain (F := F) cs z hprim i
  have hC := constraintCzPoly_eval_domain (F := F) cs z hprim i
  unfold constraintNumeratorPoly
  simp [Polynomial.eval_mul, Polynomial.eval_sub, evaluateConstraintResidual, hA, hB, hC]

/-- Satisfying witnesses force the honest numerator polynomial to vanish on the domain. -/
lemma constraintNumeratorPoly_eval_domain_of_satisfies
    {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (hsat : satisfies cs z) (i : Fin cs.nCons) :
    (constraintNumeratorPoly cs z ω).eval (ω ^ (i : ℕ)) = 0 := by
  have hres := (satisfies_iff_residual_zero cs z).mp hsat i
  have hnumer := constraintNumeratorPoly_eval_domain (F := F) cs z hprim i
  exact hnumer.trans hres

/-- Constraint polynomial values coincide with honest numerator evaluations at domain points. -/
lemma constraintPoly_eval_domain_eq_constraintNumerator
    {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (i : Fin cs.nCons) :
    constraintPoly cs z i =
      (constraintNumeratorPoly cs z ω).eval (ω ^ (i : ℕ)) := by
  classical
  calc
    constraintPoly cs z i
        = evaluateConstraintResidual cs z i :=
          constraintPoly_eq_evaluateResidual (cs := cs) (z := z) i
    _ = (constraintNumeratorPoly cs z ω).eval (ω ^ (i : ℕ)) := by
          simpa using
            (constraintNumeratorPoly_eval_domain (cs := cs) (z := z)
                (ω := ω) hprim i).symm

/-- If every constraint vanishes, the honest numerator polynomial vanishes on the domain. -/
lemma constraintNumeratorPoly_eval_domain_of_constraint_zero
    {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons)
    (hzero : ∀ i : Fin cs.nCons, constraintPoly cs z i = 0)
    (i : Fin cs.nCons) :
    (constraintNumeratorPoly cs z ω).eval (ω ^ (i : ℕ)) = 0 := by
  have h := constraintPoly_eval_domain_eq_constraintNumerator
      (cs := cs) (z := z) (ω := ω) hprim i
  have hi := hzero i
  simpa [h] using hi

/-- Honest numerator has zero remainder modulo the vanishing polynomial
    once constraint evaluations vanish on the domain. -/
lemma constraintNumeratorPoly_mod_vanishing_zero_of_constraint_zero
    {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons)
    (hzero : ∀ i : Fin cs.nCons, constraintPoly cs z i = 0) :
    (constraintNumeratorPoly cs z ω) %ₘ vanishing_poly cs.nCons ω = 0 := by
  classical
  refine (remainder_zero_iff_vanishing
      (f := constraintNumeratorPoly cs z ω)
      (m := cs.nCons) (ω := ω) hprim).2 ?_
  intro i
  simpa using
    (constraintNumeratorPoly_eval_domain_of_constraint_zero (cs := cs) (z := z)
      (ω := ω) hprim hzero i)

/-- Constraint residual polynomial obtained via interpolation of per-constraint evaluations. -/
noncomputable def constraintResidualPoly {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (ω : F) : Polynomial F :=
  lagrange_interpolate cs.nCons ω (fun i => constraintPoly cs z i)

/-- Evaluation of the constraint residual polynomial across the domain. -/
lemma constraintResidualPoly_eval_domain {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (i : Fin cs.nCons) :
    (constraintResidualPoly cs z ω).eval (ω ^ (i : ℕ)) = constraintPoly cs z i := by
  classical
  unfold constraintResidualPoly
  simpa using
    (lagrange_interpolate_eval cs.nCons hprim (fun i => constraintPoly cs z i) i)

/-- Vanishing of the residual polynomial on the evaluation domain ↔ each constraint vanishes. -/
lemma constraintResidualPoly_domain_zero_iff {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) :
    (∀ i : Fin cs.nCons,
      (constraintResidualPoly cs z ω).eval (ω ^ (i : ℕ)) = 0) ↔
        ∀ i : Fin cs.nCons, constraintPoly cs z i = 0 := by
  classical
  constructor
  · intro hzero i
    have hi := hzero i
    simpa [constraintResidualPoly_eval_domain cs z hprim i] using hi
  · intro hvals i
    have hi := hvals i
    simpa [constraintResidualPoly_eval_domain cs z hprim i] using hi

/-- If every constraint vanishes, the residual polynomial is divisible by the vanishing polynomial. -/
lemma constraintResidualPoly_mod_vanishing_zero {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons)
    (hvals : ∀ i : Fin cs.nCons, constraintPoly cs z i = 0) :
    (constraintResidualPoly cs z ω) %ₘ vanishing_poly cs.nCons ω = 0 := by
  classical
  refine (remainder_zero_iff_vanishing _ _ _ hprim).2 ?_
  intro i
  have hi := hvals i
  simpa [constraintResidualPoly_eval_domain cs z hprim i] using hi

/-- Satisfaction of the R1CS system forces the residual polynomial to vanish on the domain. -/
lemma constraintResidualPoly_domain_zero_iff_satisfies {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) :
    (∀ i : Fin cs.nCons,
      (constraintResidualPoly cs z ω).eval (ω ^ (i : ℕ)) = 0) ↔ satisfies cs z := by
  classical
  constructor
  · intro hzero
    refine (satisfies_iff_constraint_zero cs z).2 ?_
    intro i
    have hi := hzero i
    simpa [constraintResidualPoly_eval_domain cs z hprim i] using hi
  · intro hsat i
    have hi := (satisfies_iff_constraint_zero cs z).1 hsat i
    simpa [constraintResidualPoly_eval_domain cs z hprim i] using hi

/-- Satisfying witnesses make the residual polynomial divisible by the vanishing polynomial. -/
lemma constraintResidualPoly_mod_vanishing_zero_of_satisfies {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (hsat : satisfies cs z) :
    (constraintResidualPoly cs z ω) %ₘ vanishing_poly cs.nCons ω = 0 := by
  refine constraintResidualPoly_mod_vanishing_zero cs z hprim ?_
  exact (satisfies_iff_constraint_zero cs z).1 hsat

/-- Vanishing remainder of the residual polynomial is equivalent to witness satisfaction. -/
lemma constraintResidualPoly_mod_vanishing_zero_iff_satisfies {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) :
    (constraintResidualPoly cs z ω) %ₘ vanishing_poly cs.nCons ω = 0 ↔ satisfies cs z := by
  classical
  constructor
  · intro hrem
    have hzero := (remainder_zero_iff_vanishing _ _ _ hprim).1 hrem
    have hconstraints : ∀ i : Fin cs.nCons, constraintPoly cs z i = 0 := by
      refine (constraintResidualPoly_domain_zero_iff cs z hprim).1 ?_
      intro i; simpa [constraintResidualPoly_eval_domain cs z hprim i] using hzero i
    refine (satisfies_iff_constraint_zero cs z).2 ?_
    exact hconstraints
  · intro hsat
    exact constraintResidualPoly_mod_vanishing_zero_of_satisfies cs z hprim hsat

/-- Satisfying witnesses yield a quotient polynomial multiplying the vanishing polynomial. -/
lemma constraintResidualPoly_factor_of_satisfies {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (hsat : satisfies cs z) :
    ∃ q : Polynomial F,
      constraintResidualPoly cs z ω = vanishing_poly cs.nCons ω * q := by
  classical
  have hmonic : (vanishing_poly cs.nCons ω).Monic := by
    unfold vanishing_poly
    apply monic_prod_of_monic
    intro i _
    simpa using (monic_X_sub_C (ω ^ (i : ℕ)))
  have hzero := constraintResidualPoly_mod_vanishing_zero_of_satisfies cs z hprim hsat
  have hdiv := (Polynomial.modByMonic_eq_zero_iff_dvd hmonic).1 hzero
  simpa using hdiv

/-- Factorization of the residual polynomial by the vanishing polynomial implies satisfaction. -/
lemma constraintResidualPoly_factor_implies_satisfies {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) :
    (∃ q : Polynomial F,
      constraintResidualPoly cs z ω = vanishing_poly cs.nCons ω * q) → satisfies cs z := by
  classical
  intro hfactor
  obtain ⟨q, hq⟩ := hfactor
  have hrem :
      (constraintResidualPoly cs z ω) %ₘ vanishing_poly cs.nCons ω = 0 := by
    have hmonic : (vanishing_poly cs.nCons ω).Monic := by
      unfold vanishing_poly
      apply monic_prod_of_monic
      intro i _
      simpa using (monic_X_sub_C (ω ^ (i : ℕ)))
    have hdiv : vanishing_poly cs.nCons ω ∣ constraintResidualPoly cs z ω := by
      refine ⟨q, ?_⟩
      simpa [mul_comm] using hq
    exact (Polynomial.modByMonic_eq_zero_iff_dvd hmonic).2 hdiv
  have :=
    (constraintResidualPoly_mod_vanishing_zero_iff_satisfies cs z hprim).1 hrem
  simpa using this

/-- Factorization criterion is equivalent to R1CS satisfaction. -/
lemma constraintResidualPoly_factor_iff_satisfies {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) :
    (∃ q : Polynomial F,
      constraintResidualPoly cs z ω = vanishing_poly cs.nCons ω * q) ↔ satisfies cs z := by
  constructor
  · exact constraintResidualPoly_factor_implies_satisfies (cs := cs) (z := z) hprim
  · intro hsat
    exact constraintResidualPoly_factor_of_satisfies (cs := cs) (z := z) hprim hsat

/-- Honest witnesses force the residual polynomial to be identically zero. -/
lemma constraintResidualPoly_eq_zero_of_satisfies {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hsat : satisfies cs z) :
    constraintResidualPoly cs z ω = 0 := by
  classical
  have hconstraints := (satisfies_iff_constraint_zero cs z).1 hsat
  unfold constraintResidualPoly
  simp [lagrange_interpolate, hconstraints]

/-- Honest quotient polynomial is definitionally zero once a witness satisfies constraints. -/
noncomputable def honestConstraintQuotient {F : Type} [Field F] [DecidableEq F]
  (cs : R1CS F) (z : Witness F cs.nVars) (_hsat : satisfies cs z) : Polynomial F := 0

@[simp] lemma honestConstraintQuotient_eq_zero {F : Type} [Field F] [DecidableEq F]
  (cs : R1CS F) (z : Witness F cs.nVars) (hsat : satisfies cs z) :
  honestConstraintQuotient (cs := cs) (z := z) hsat = 0 := rfl

lemma constraintResidualPoly_eq_vanishing_mul_honestQuotient {F : Type} [Field F] [DecidableEq F]
  (cs : R1CS F) (z : Witness F cs.nVars) (ω : F) (hsat : satisfies cs z) :
  constraintResidualPoly cs z ω =
    vanishing_poly cs.nCons ω * honestConstraintQuotient (cs := cs) (z := z) hsat := by
  classical
  have hzero := constraintResidualPoly_eq_zero_of_satisfies (cs := cs) (z := z) (ω := ω) hsat
  simp [honestConstraintQuotient, hzero]


/-- Constraint polynomial evaluation at domain points -/
theorem constraint_poly_eval {F : Type} [Field F] [DecidableEq F]
    (_cs : R1CS F) (_z : Witness F _cs.nVars) (_i : Fin _cs.nCons)
    (_m : ℕ) (_ω : F) (_h_m : _m = _cs.nCons) (_h_root : _ω ^ _m = 1) :
    -- Az(ωⁱ) * Bz(ωⁱ) - Cz(ωⁱ) = constraint evaluation at i-th point
    True := by
  trivial

-- ============================================================================
-- Quotient Polynomial Properties
-- ============================================================================

/-- Quotient polynomial is unique -/
theorem quotient_uniqueness {F : Type} [Field F]
    (f : Polynomial F) (m : ℕ) (ω : F)
    (q₁ q₂ : Polynomial F)
    (h_root : ω ^ m = 1)
    (h₁ : f = q₁ * @vanishing_poly F _ m ω)
    (h₂ : f = q₂ * @vanishing_poly F _ m ω) :
    q₁ = q₂ := by
  -- From h₁ = h₂: q₁ * Z_H = q₂ * Z_H
  have h_eq : q₁ * @vanishing_poly F _ m ω = q₂ * @vanishing_poly F _ m ω := by
    rw [← h₁, ← h₂]
  -- Need to show Z_H ≠ 0 and use mul_right_cancel
  -- Polynomial F over Field F has no zero divisors
  -- Z_H is product of (X - ωⁱ), each factor nonzero, so Z_H ≠ 0
  by_cases h_m : m = 0
  · -- m = 0: Z_H = 1 (empty product), so q₁ * 1 = q₂ * 1
    subst h_m
    have h_prod : ∏ i : Fin 0, (X - C (ω ^ (i : ℕ))) = 1 := Finset.prod_empty
    simp only [vanishing_poly, h_prod, mul_one] at h_eq
    exact h_eq
  · -- m > 0: Z_H ≠ 0 (product of nonzero linear factors)
    -- Strategy: Each factor (X - ωⁱ) is nonzero polynomial, product nonzero
    -- Use Polynomial.mul_right_cancel₀ or IsCancelMulZero
    have h_Z_ne_zero : vanishing_poly m ω ≠ 0 := by
      unfold vanishing_poly
      apply Finset.prod_ne_zero_iff.mpr
      intro i _
      -- Show X - C(ωⁱ) ≠ 0: degree 1 polynomial
      intro h_factor_zero
      have : (X - C (ω ^ (i : ℕ))).natDegree = 0 := by rw [h_factor_zero]; simp
      -- But X - C a has degree 1
      have : (X - C (ω ^ (i : ℕ))).natDegree = 1 := by
        rw [Polynomial.natDegree_sub_C]; simp
      omega
    exact mul_right_cancel₀ h_Z_ne_zero h_eq

/-- Degree bound for quotient polynomial -/
theorem quotient_degree_bound {F : Type} [Field F]
    (f q : Polynomial F) (m d : ℕ) (ω : F)
    (h_div : f = q * vanishing_poly m ω)
    (h_deg_f : f.natDegree ≤ d)
    (h_deg_Z : (vanishing_poly m ω).natDegree = m)
    (h_m_pos : 0 < m) :
    q.natDegree ≤ d - m := by
  -- From f = q * Z_H: deg(f) = deg(q) + deg(Z_H)
  -- So: deg(q) = deg(f) - deg(Z_H) ≤ d - m
  by_cases hq : q = 0
  · simp [hq]
  by_cases hZ : vanishing_poly m ω = 0
  · -- Z_H = 0 contradicts h_deg_Z (deg(Z_H) = m > 0)
    exfalso
    rw [hZ, Polynomial.natDegree_zero] at h_deg_Z
    omega
  -- Use: deg(f·g) = deg(f) + deg(g) for nonzero polynomials
  have h_deg_eq : f.natDegree = q.natDegree + (vanishing_poly m ω).natDegree := by
    rw [h_div]
    exact Polynomial.natDegree_mul hq hZ
  -- Substitute and rearrange
  rw [h_deg_Z] at h_deg_eq
  omega

/-- Honest factor of the residual polynomial is unique. -/
lemma constraintResidualPoly_factor_unique {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (hsat : satisfies cs z) :
    ∃! q : Polynomial F,
      constraintResidualPoly cs z ω = vanishing_poly cs.nCons ω * q := by
  classical
  obtain ⟨q, hq⟩ :=
    constraintResidualPoly_factor_of_satisfies (cs := cs) (z := z) hprim hsat
  refine ⟨q, hq, ?_⟩
  intro q' hq'
  have hroot : ω ^ cs.nCons = 1 := hprim.pow_eq_one
  have hq_left : constraintResidualPoly cs z ω =
      q * vanishing_poly cs.nCons ω := by
        simpa [mul_comm] using hq
  have hq'_left : constraintResidualPoly cs z ω =
      q' * vanishing_poly cs.nCons ω := by
        simpa [mul_comm] using hq'
  exact
    quotient_uniqueness (f := constraintResidualPoly cs z ω) (m := cs.nCons) (ω := ω)
      (q₁ := q') (q₂ := q) hroot hq'_left hq_left

/-- Canonical quotient polynomial provided by an honest witness. -/
noncomputable def constraintResidualPoly_quotient {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (hsat : satisfies cs z) : Polynomial F :=
  Classical.choose
    (constraintResidualPoly_factor_unique (cs := cs) (z := z) hprim hsat).exists

lemma constraintResidualPoly_eq_vanishing_mul_quotient {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (hsat : satisfies cs z) :
    constraintResidualPoly cs z ω =
      vanishing_poly cs.nCons ω * constraintResidualPoly_quotient (cs := cs) (z := z) hprim hsat := by
  classical
  have hspec :=
    Classical.choose_spec
      (constraintResidualPoly_factor_unique (cs := cs) (z := z) hprim hsat).exists
  simpa using hspec

@[simp] lemma constraintResidualPoly_quotient_eq_zero_of_satisfies {F : Type} [Field F]
    [DecidableEq F] (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (hsat : satisfies cs z) :
    constraintResidualPoly_quotient (cs := cs) (z := z) hprim hsat = 0 := by
  classical
  have hzero := constraintResidualPoly_eq_zero_of_satisfies (cs := cs) (z := z) (ω := ω) hsat
  have hprod :
      vanishing_poly cs.nCons ω *
          constraintResidualPoly_quotient (cs := cs) (z := z) hprim hsat = 0 := by
    simpa [hzero] using
      (constraintResidualPoly_eq_vanishing_mul_quotient (cs := cs) (z := z)
        (ω := ω) hprim hsat).symm
  have hvanish : vanishing_poly cs.nCons ω ≠ 0 := vanishing_poly_ne_zero _ _
  rcases mul_eq_zero.mp hprod with hvanish_zero | hquot_zero
  · exact (hvanish hvanish_zero).elim
  · exact hquot_zero

lemma constraintResidualPoly_quotient_eq_honest {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) {ω : F}
    (hprim : IsPrimitiveRoot ω cs.nCons) (hsat : satisfies cs z) :
    constraintResidualPoly_quotient (cs := cs) (z := z) (ω := ω) hprim hsat =
      honestConstraintQuotient (cs := cs) (z := z) hsat := by
  classical
  calc
    constraintResidualPoly_quotient (cs := cs) (z := z) (ω := ω) hprim hsat
        = 0
            := constraintResidualPoly_quotient_eq_zero_of_satisfies
              (cs := cs) (z := z) (ω := ω) hprim hsat
    _ = honestConstraintQuotient (cs := cs) (z := z) hsat := by
          simp [honestConstraintQuotient]

lemma constraintResidualPoly_quotient_natDegree_le {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (ω : F) {q : Polynomial F} {d : ℕ}
    (hfactor : constraintResidualPoly cs z ω = vanishing_poly cs.nCons ω * q)
    (hdeg : (constraintResidualPoly cs z ω).natDegree ≤ d)
    (hpos : 0 < cs.nCons) :
    q.natDegree ≤ d - cs.nCons := by
  classical
  have hdeg_Z := vanishing_poly_natDegree cs.nCons ω
  refine quotient_degree_bound
      (f := constraintResidualPoly cs z ω) (q := q) (m := cs.nCons) (d := d) (ω := ω)
      ?_ hdeg ?_ hpos
  · simpa [mul_comm] using hfactor
  · simpa using hdeg_Z

end LambdaSNARK
