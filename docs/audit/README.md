# Audit Playbook

The audit is split into two phases:

1. **Phase A (Rust / Constant-Time)** – follow `checklist_rust.md`, capture artifacts in `artifacts/audit/A-phase-report.md` and update security docs.
2. **Phase B (C++ / FFI / Formal)** – follow `checklist_cpp.md` and `checklist_formal.md`, record findings in `artifacts/audit/B-phase-report.md`.

Run `./scripts/setup.sh` before executing tooling in CI or locally. Trigger the `audit-checks` GitHub workflow for recurring smoke validation (also scheduled weekly).

All findings must be tracked as GitHub issues with severity labels (Critical, High, Medium, Low). Critical/High require mitigation plan within 24 hours.
