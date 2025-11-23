# Rust Audit Checklist

## Static Analysis
- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets --all-features -D warnings`
- [ ] `cargo udeps --all-targets` (dependency pruning)

## Unsafe / FFI Review
- [ ] Enumerate all `unsafe` blocks (`RUSTFLAGS="-Zunstable-options" cargo geiger`)
- [ ] Verify `lambda-snark-sys` bindings match C++ signatures
- [ ] Confirm `LweContext` lifetime and pointer invariants

## Constant-Time Verification
- [ ] Re-run `cargo run --manifest-path rust-api/Cargo.toml -p lambda-snark --bin mod_arith_timing`
- [ ] Inspect `artifacts/dudect/mod_arith_report.md` (|t| < 4.5)
- [ ] Capture results in `artifacts/audit/A-phase-report.md`

## Testing & Coverage
- [ ] `cargo test --workspace --all-features`
- [ ] `cargo nextest run` (optional faster harness)
- [ ] `cargo tarpaulin --workspace --out Html` (coverage snapshot)

## Dependency Security
- [ ] `cargo audit`
- [ ] `cargo deny check`
- [ ] Verify git submodules/tags for third-party code

## Documentation Sync
- [ ] Update `SECURITY.md` findings section
- [ ] Refresh `TESTING.md` with any new commands
- [ ] File issues for gaps (Critical/High within 24h)
