# Formal Verification Audit Checklist

## Build & Regression
- [ ] `lake build LambdaSNARK`
- [ ] `lean --version` recorded in report
- [ ] Ensure `formal/README.md` reflects current build instructions

## Proof Coverage
- [ ] Enumerate completed theorems vs roadmap (soundness, ZK)
- [ ] Cross-check Lean statements with Rust documentation
- [ ] Confirm no admitted axioms or `sorry` placeholders

## Traceability
- [ ] Map Lean constants to Rust parameters (`Params`, `Profile`)
- [ ] Verify artifact hashes or seeds recorded in `artifacts/`
- [ ] Update `VERIFICATION_PLAN.md` milestones

## Reporting
- [ ] Summarize outstanding proof obligations
- [ ] File issues for blocked lemmas (assign owner + target milestone)
- [ ] Append findings to `artifacts/audit/B-phase-report.md`
