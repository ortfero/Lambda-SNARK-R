# C++ / FFI Audit Checklist

## Tooling Pass
- [ ] `cmake -S cpp-core -B cpp-core/build -DCMAKE_BUILD_TYPE=RelWithDebInfo`
- [ ] `cmake --build cpp-core/build --target lambda_snark_core`
- [ ] `ninja -C cpp-core/build clang-tidy` (or `cmake --build ... --target clang-tidy`)
- [ ] Run `include-what-you-use` on public headers

## Sanitizers & Fuzzing
- [ ] Enable `-fsanitize=address,undefined` and rebuild
- [ ] Execute `ctest --output-on-failure`
- [ ] Seed fuzz targets for commitment/opening FFI (record corpus path)

## Constant-Time Checks
- [ ] `./cpp-core/build/dudect_sampler`
- [ ] Verify `artifacts/dudect/gaussian_sampler_report.md` (|t| < 4.5)
- [ ] Inspect SEAL FFI wrappers for early exits/branches on secrets

## Memory & Error Handling
- [ ] Audit all `new`/`delete`, `malloc`/`free` pairs
- [ ] Confirm exception safety (no throws across FFI boundary)
- [ ] Validate input parameter checks in `lambda_snark_r1cs_*` APIs

## Documentation & Artifacts
- [ ] Update `SECURITY.md` → C++ section with new findings
- [ ] Record sanitizer/fuzzer results in `artifacts/audit/B-phase-report.md`
- [ ] Raise GitHub issues for Critical/High findings immediately
