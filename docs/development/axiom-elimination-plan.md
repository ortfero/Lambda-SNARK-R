# Axiom Elimination Roadmap (Lean Core)

_Last updated: 2025-11-21_

The current Lean proof stack for ΛSNARK-R is now axiom-free for the forking
infrastructure. This document remains the live tracker for ensuring the kernel
stays constructive and for recording any future assumptions, aligned with the
“variant 1” plan (remove axioms in Lean first, then sync with Rust test vectors).

## Overview

| Axiom | Location | Role | Status | Planned Replacement |
|-------|----------|------|--------|---------------------|
| `heavy_row_lemma` | `LambdaSNARK/ForkingInfrastructure.lean` | Probabilistic heavy row argument | DONE | Lean proof via PMF counting (`successProbability_eq_successfulRandomness_card_div`, `heavy_randomness_finset_lower_bound`) |
| `fork_success_bound` | `LambdaSNARK/ForkingInfrastructure.lean` | Lower bound on fork probability | DONE | Derived from `heavy_row_lemma` + combinatorial bounds on `Nat.choose` |
| `extraction_soundness` | `LambdaSNARK/ForkingInfrastructure.lean` | Shows extracted witness satisfies R1CS | DONE | Constructive proof using `ForkingEquationsProvider`; see `ForkingInfrastructure.lean` |

`Soundness.lean` and related modules now depend only on proved lemmas; no
probabilistic axioms remain.

## Work Packages

1. **Extraction Soundness (archived)**
   - Inputs: transcripts forming valid fork, `quotient_poly_eq_of_fork`.
   - Required lemmas:
     - `Polynomial.coeffList` ↔ evaluation bridge for quotient polynomial.
     - Normal form of R1CS evaluation (`constraintPoly`) vs commitment openings.
     - Vanishing polynomial properties (already in `Polynomial.lean`).
    - Progress:
       - Added Lean helper lemmas `extract_witness_public`, `extract_witness_private`,
          `extract_witness_public_prefix`, and reduction lemma
          `extraction_soundness_of_constraint_zero` to isolate the remaining goal.
       - `extract_witness` behaviour on public prefix now explicit (needed for
          bridging with transcript commitments).
    - Deliverables:
       - Replace `axiom extraction_soundness` with a proved theorem. ✅ *Now parameterised by `ForkingEquationsProvider`; the global axiom is removed.*
       - Unit test: instantiate simple R1CS (e.g., healthcare circuit) and verify
          extracted witness satisfies constraints. ✅ *See `formal/tests/ForkingCertificateExample.lean`.*

2. **Heavy Row Lemma (archived)**
   - Completed in commit `feat: relate success probability to successful randomness`
     and follow-up cleanups.
   - Key artefacts: helper lemmas on `ENNReal.toReal`, heavy randomness finset
     bounds, and the constructive statement `heavy_row_lemma`.

3. **Fork Success Bound (archived)**
   - Completed alongside (2); relies on the new heavy commitment image finset
     infrastructure and combinatorial bounds for `Nat.choose`.
   - Downstream proofs (`forking_lemma`, soundness) now import this theorem
     directly without axioms.

## Milestones & Checks

- **M1**: `extraction_soundness` proven ⇒ `forking_lemma` uses no extraction
  axioms.
- **M2**: Both probabilistic lemmas proven ⇒ `forking_lemma` completely
   constructive. ✅
- **M3**: `knowledge_soundness` depends only on documented algebraic assumptions
  (e.g., Module-SIS hardness), no Lean axioms.

## Immediate Next Actions

1. Integrate the constructive heavy-row/fork lemmas into the higher-level
   `forking_lemma` pipeline and refresh any derived documentation or diagrams.
2. Resolve outstanding scratch helpers (e.g. `formal/TmpDec.lean`) and ensure CI
   covers `lake build LambdaSNARK` plus the probabilistic regression tests.
3. Extend Lean regression suites with concrete circuits (healthcare, plaquette)
   exercising the new inequalities end-to-end.

Please keep this file up to date after each major step (PR/commit) to maintain
shared visibility on proof obligations.
