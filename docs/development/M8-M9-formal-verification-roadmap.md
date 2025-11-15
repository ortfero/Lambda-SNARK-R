# M8-M9 Formal Verification Roadmap

**Timeline**: November 2025 - May 2026  
**Goal**: Complete Lean 4 soundness proof for ΛSNARK-R  
**Estimated Effort**: 80-120 hours total

---

## Phase 1: Definitions & Statements ✅ (Complete - November 15, 2025)

**Effort**: 10h actual  
**Deliverables**:
- ✅ Core.lean: R1CS, Witness, Satisfaction, VectorCommitment, Proof
- ✅ Soundness.lean: knowledge_soundness statement
- ✅ Completeness.lean: perfect_completeness statement
- ✅ Polynomial.lean: Lagrange interpolation, polynomial division

**Status**: All definitions formalized, theorem statements complete.

---

## Phase 2: Polynomial Lemmas (M8.1 - December 2025)

**Effort**: 20-30h  
**Dependencies**: Mathlib4 polynomial library

### Tasks

1. **Lagrange Interpolation** (8-10h)
   - [ ] Prove `lagrange_basis_property`: Lᵢ(ωʲ) = δᵢⱼ
   - [ ] Prove `lagrange_interpolate_eval`: p(ωⁱ) = evals(i)
   - [ ] Prove uniqueness: polynomial through m points is unique (degree < m)

2. **Polynomial Division** (6-8h)
   - [ ] Prove `polynomial_division`: f = q*g + r with deg(r) < deg(g)
   - [ ] Prove `remainder_zero_iff_vanishing`: r = 0 ↔ f vanishes on H
   - [ ] Use Mathlib's `Polynomial.div_mod_by_monic_unique`

3. **Vanishing Polynomial** (4-6h)
   - [ ] Prove `vanishing_poly_roots`: Z_H(ωⁱ) = 0 for all i < m
   - [ ] Prove `vanishing_poly_degree`: deg(Z_H) = m
   - [ ] Prove `quotient_uniqueness`: quotient by Z_H is unique

4. **Schwartz-Zippel** (2-4h)
   - [ ] Prove `schwartz_zippel`: |{α : p(α) = 0}| ≤ deg(p) for p ≠ 0
   - [ ] Use Mathlib's `Polynomial.card_roots'` or prove directly

**Deliverables**: Polynomial.lean with all `sorry` replaced by proofs.

---

## Phase 3: Soundness Proof (M8.2 - January-February 2026)

**Effort**: 40-60h  
**Dependencies**: Phase 2 complete, cryptographic assumptions

### Tasks

1. **Forking Lemma** (15-20h)
   - [ ] Formalize adversary model (PPT, oracle access)
   - [ ] Prove rewinding technique: two transcripts (α, π₁), (α', π₂)
   - [ ] Prove success probability: Pr[extract] ≥ ε² - negl(λ)
   - [ ] Handle edge cases: adversary fails, same challenge, etc.

2. **Witness Extraction** (12-18h)
   - [ ] Prove quotient difference formula: q - q' reveals witness
   - [ ] Prove extraction correctness: extracted w satisfies R1CS
   - [ ] Prove public input consistency: extractPublic w = x
   - [ ] Handle field arithmetic (modular operations)

3. **Commitment Binding** (8-10h)
   - [ ] Formalize Module-SIS assumption (lattice hardness)
   - [ ] Prove commitment binding: c₁ = c₂ ∧ v₁ ≠ v₂ → break SIS
   - [ ] Reduction: soundness adversary → SIS solver
   - [ ] Compute concrete security bounds (λ bits)

4. **Random Oracle Model** (5-8h)
   - [ ] Formalize Fiat-Shamir transformation
   - [ ] Prove challenge unpredictability in ROM
   - [ ] Prove rewinding doesn't break ROM security
   - [ ] Document non-programmable ROM limitations

**Deliverables**: Soundness.lean with `knowledge_soundness` proof complete.

---

## Phase 4: Completeness Proof (M8.3 - March 2026)

**Effort**: 10-20h  
**Dependencies**: Phase 2 complete

### Tasks

1. **Honest Prover Construction** (4-6h)
   - [ ] Formalize prove algorithm step-by-step
   - [ ] Prove polynomial construction correctness
   - [ ] Prove quotient polynomial existence (by satisfaction)
   - [ ] Prove commitment generation correctness

2. **Verifier Checks** (4-6h)
   - [ ] Prove opening correctness: verify returns true for honest openings
   - [ ] Prove quotient check: q(α) * Z_H(α) = constraint_poly(α)
   - [ ] Prove public input check: first l elements match
   - [ ] Handle field operations (modular evaluation)

3. **Perfect Completeness** (2-4h)
   - [ ] Prove deterministic verification: no randomness in verifier
   - [ ] Prove completeness error = 0 (not just negligible)
   - [ ] Prove consistency with Rust/C++ implementation (via test vectors)

**Deliverables**: Completeness.lean with `completeness` proof complete.

---

## Phase 5: Zero-Knowledge Proof (M9 - April-May 2026)

**Effort**: 30-40h  
**Dependencies**: Phases 2-4 complete

### Tasks

1. **Simulator Construction** (12-15h)
   - [ ] Formalize simulator that produces fake proofs
   - [ ] Prove simulator doesn't use witness (only public input)
   - [ ] Prove simulator outputs have correct distribution
   - [ ] Handle commitment hiding (Module-LWE assumption)

2. **Indistinguishability** (12-15h)
   - [ ] Formalize distinguisher model (PPT adversary)
   - [ ] Prove hybrid argument: Real ≈ Hybrid₁ ≈ ... ≈ Sim
   - [ ] Reduce to Module-LWE: distinguisher → LWE solver
   - [ ] Compute concrete security bounds

3. **Blinding Factor Analysis** (6-10h)
   - [ ] Prove blinding polynomial hides witness
   - [ ] Prove σ (noise parameter) is sufficient
   - [ ] Prove no information leakage from openings
   - [ ] Document security parameter requirements

**Deliverables**: ZeroKnowledge.lean with `zero_knowledge` proof complete.

---

## Phase 6: Integration & Validation (M9.1 - May 2026)

**Effort**: 10-15h

### Tasks

1. **Test Vector Validation** (4-6h)
   - [ ] Extract R1CS from test-vectors/ as Lean terms
   - [ ] Verify satisfaction predicate matches Rust/C++ implementation
   - [ ] Check commitment values match (deterministic seed)
   - [ ] Automate extraction (JSON → Lean parser)

2. **CI Integration** (3-4h)
   - [ ] Add `lake build` to GitHub Actions
   - [ ] Add proof checking step (fail on `sorry`)
   - [ ] Add Lean version pinning (lean-toolchain)
   - [ ] Cache mathlib build artifacts

3. **Documentation** (3-5h)
   - [ ] Write proof overview (high-level strategy)
   - [ ] Document cryptographic assumptions
   - [ ] Document limitations (ROM, Module-SIS parameters)
   - [ ] Create tutorial (how to extend proofs)

**Deliverables**: 
- CI passing with all proofs verified
- Documentation complete
- Test vector integration working

---

## Success Criteria

**M8 Complete** (Soundness + Completeness):
- ✅ All `sorry` removed from Soundness.lean and Completeness.lean
- ✅ `lake build` succeeds without errors
- ✅ Test vectors validated against Lean formalization
- ✅ Concrete security bounds documented

**M9 Complete** (Zero-Knowledge):
- ✅ All `sorry` removed from ZeroKnowledge.lean
- ✅ Full proof chain: Soundness + Completeness + Zero-Knowledge
- ✅ CI verification passing
- ✅ External review (Lean community feedback)

---

## Risks & Mitigations

### Risk 1: Mathlib API Changes
- **Impact**: HIGH (proof breaks with updates)
- **Mitigation**: Pin mathlib version in lake-manifest.json, document dependencies

### Risk 2: Cryptographic Assumptions Too Strong
- **Impact**: MEDIUM (soundness holds but impractical)
- **Mitigation**: Validate parameters against lattice security estimates (lattice-estimator)

### Risk 3: Proof Complexity Underestimated
- **Impact**: MEDIUM (timeline slips 2-3 months)
- **Mitigation**: Break into smaller milestones, parallelize polynomial lemmas

### Risk 4: ROM Limitations
- **Impact**: LOW (theory vs practice gap)
- **Mitigation**: Document limitations, plan standard model variant for future

---

## Resources

**Lean 4**:
- Theorem Proving in Lean 4: https://leanprover.github.io/theorem_proving_in_lean4/
- Mathlib4 docs: https://leanprover-community.github.io/mathlib4_docs/

**Cryptographic Proofs**:
- EasyCrypt tutorials: https://www.easycrypt.info/
- CryptoVerif examples: https://cryptoverif.inria.fr/

**Reference Implementations**:
- Bulletproofs Lean: https://github.com/Verified-zkEVM/
- Plonk Lean: (to be released)

---

## Estimated Timeline

```
Nov 2025: Phase 1 (Definitions) ✅
Dec 2025: Phase 2 (Polynomial Lemmas)
Jan 2026: Phase 3.1 (Forking Lemma)
Feb 2026: Phase 3.2 (Soundness Proof Complete)
Mar 2026: Phase 4 (Completeness Proof)
Apr 2026: Phase 5 (Zero-Knowledge Proof)
May 2026: Phase 6 (Integration & Validation)
```

**Milestone Dates**:
- M8.1 (Polynomial Lemmas): December 31, 2025
- M8.2 (Soundness): February 28, 2026
- M8.3 (Completeness): March 31, 2026
- M9 (Zero-Knowledge): May 31, 2026

**Alpha Release** (v0.1.0-alpha): November 2025 ✅  
**Production Release** (v1.0.0): August 2026 (with external audit)
