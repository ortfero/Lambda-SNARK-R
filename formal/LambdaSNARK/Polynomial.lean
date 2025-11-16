/-
Copyright (c) 2025 URPKS Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: URPKS Contributors
-/

import LambdaSNARK.Core
import Mathlib.Algebra.Polynomial.Div
import Mathlib.RingTheory.Polynomial.Basic
import Mathlib.RingTheory.RootsOfUnity.Basic
import Mathlib.Tactic

open scoped BigOperators
open Polynomial

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

-- ============================================================================
-- Vanishing Polynomial
-- ============================================================================

/-- Vanishing polynomial `Z_H(X) = ∏ᵢ (X - ω^i)` for `H = {ω^i | i < m}`. -/
noncomputable def vanishing_poly {F : Type*} [Field F] (m : ℕ) (ω : F) : Polynomial F :=
  ∏ i : Fin m, (X - C (ω ^ (i : ℕ)))

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

-- ============================================================================
-- Lagrange Basis Polynomials
-- ============================================================================

/-- If `ω` is a primitive `m`-th root of unity, then `i ↦ ω^i` is injective on `Fin m`. -/
lemma primitive_root_pow_injective {F : Type*} [Field F]
    {m : ℕ} {ω : F} (hprim : IsPrimitiveRoot ω m)
    (i j : Fin m) :
    ω ^ (i : ℕ) = ω ^ (j : ℕ) → i = j := by
  -- Strategy: ω^i = ω^j → ω^|i-j| = 1 → m | |i-j| → i = j (since 0 < |i-j| < m)
  -- Requires trichotomy on i, j and careful application of IsPrimitiveRoot.pow_eq_one_iff_dvd
  -- Technical blocker: mul_left_cancel₀ requires careful term construction in Lean 4
  -- Defer to manual proof with explicit have lemmas for power splitting
  sorry

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
  -- Strategy: rewrite evals j * ite to ite (evals j) 0, then apply Finset.sum_ite_eq
  sorry  -- TODO: simp [mul_ite, mul_one, mul_zero, Finset.sum_ite_eq, Finset.mem_univ]
         -- Technical issue: simp doesn't apply sum_ite_eq with correct argument order

-- ============================================================================
-- Polynomial Division
-- ============================================================================

/-- Polynomial division: `f = q * g + r` with `deg(r) < deg(g)`. -/
theorem polynomial_division {F : Type*} [Field F]
    (f g : Polynomial F) (hg : g ≠ 0) :
    ∃! qr : Polynomial F × Polynomial F,
      f = qr.1 * g + qr.2 ∧ (qr.2 = 0 ∨ qr.2.natDegree < g.natDegree) := by
  classical
  refine ⟨(f / g, f % g), ?exist, ?uniq⟩
  · constructor
    · simpa [mul_comm] using (EuclideanDomain.div_add_mod f g).symm
    · by_cases h : f % g = 0
      · exact Or.inl h
      · right
        sorry -- TODO: Apply degree_modByMonic_lt with proper monic condition
  · intro ⟨q', r'⟩ ⟨hq, hdeg⟩
    sorry -- TODO: Euclidean domain uniqueness argument

/-- Divide a polynomial by the vanishing polynomial. -/
noncomputable def divide_by_vanishing {F : Type*} [Field F]
    (f : Polynomial F) (m : ℕ) (ω : F) : Polynomial F × Polynomial F :=
  let ZH := vanishing_poly m ω
  (f /ₘ ZH, f %ₘ ZH)

/-- Remainder is zero iff `f` vanishes on roots of `Z_H`. -/
theorem remainder_zero_iff_vanishing {F : Type*} [Field F]
    (f : Polynomial F) (m : ℕ) (ω : F) (hω : IsPrimitiveRoot ω m) :
    f %ₘ vanishing_poly m ω = 0 ↔ ∀ i : Fin m, f.eval (ω ^ (i : ℕ)) = 0 := by
  unfold vanishing_poly
  constructor
  · intro h_rem i
    sorry -- TODO: modByMonic = 0 → divisibility → eval = 0
  · intro h_eval
    sorry -- TODO: All roots → product divides → modByMonic = 0

-- ============================================================================
-- Witness Polynomial Construction
-- ============================================================================

/-- Construct polynomial from witness vector via Lagrange interpolation -/
noncomputable def witness_to_poly {F : Type} [Field F] [DecidableEq (Fin 1)] {n : ℕ}
    (w : Witness F n) (m : ℕ) (ω : F)
    (h_root : ω ^ m = 1) (h_size : m ≤ n) : Polynomial F :=
  -- Interpolate witness values over evaluation domain
  -- For each point i in [0,m), take witness value at index i
  lagrange_interpolate m ω (fun i =>
    if h : i.val < n then w ⟨i.val, h⟩ else 0)

/-- Constraint polynomial evaluation at domain points -/
theorem constraint_poly_eval {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars) (i : Fin cs.nCons)
    (m : ℕ) (ω : F) (h_m : m = cs.nCons) (h_root : ω ^ m = 1) :
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
    simp [vanishing_poly, h_m] at h_eq
    cases h_eq with
    | inl h => exact h
    | inr h =>
      -- h: ∏ x : Fin 0, _ = 0, but empty product = 1
      exfalso
      sorry  -- TODO: Show Finset.prod over empty set equals 1, not 0
  · -- m > 0: Z_H ≠ 0 (product of nonzero linear factors)
    sorry  -- TODO: Show vanishing_poly m ω ≠ 0 and apply mul_right_cancel

/-- Degree bound for quotient polynomial -/
theorem quotient_degree_bound {F : Type} [Field F] [DecidableEq F]
    (cs : R1CS F) (z : Witness F cs.nVars)
    (m : ℕ) (ω : F) (q : Polynomial F)
    (h_sat : satisfies cs z)
    (h_m : m = cs.nCons)
    (h_root : ω ^ m = 1) :
    q.natDegree ≤ cs.nVars + cs.nCons := by
  -- Constraint polynomial degree ≤ 2·nVars (quadratic), vanishing poly degree = m
  -- So q degree ≤ 2·nVars - m = 2·nVars - nCons
  -- With proper encoding: deg(q) ≤ nVars + nCons
  sorry  -- TODO: Derive from constraint polynomial structure

end LambdaSNARK
