// Baseline benchmarks for Lagrange interpolation (O(m²) naïve implementation)
//
// Purpose: Measure current performance before M5.1 FFT/NTT optimization
// Target: Identify scaling behavior (empirical exponent) and bottlenecks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lambda_snark::{circuit::CircuitBuilder, r1cs::R1CS, LweContext};
use lambda_snark_core::{Params, Profile, SecurityLevel};

const MODULUS: u64 = 17592186044423; // Current prime modulus (NOT NTT-friendly)

/// Create LWE context for benchmarks
fn create_lwe_context() -> LweContext {
    let params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,
            k: 2,
            q: MODULUS,
            sigma: 3.19,
        },
    );

    LweContext::new(params).expect("Failed to create LWE context")
}

/// Build simple circuit with m constraints (x₁ · x₂ = x₃)
fn build_circuit(m: usize) -> R1CS {
    let mut builder = CircuitBuilder::new(MODULUS);

    for _ in 0..m {
        let x1 = builder.alloc_var();
        let x2 = builder.alloc_var();
        let x3 = builder.alloc_var();

        // Constraint: x1 · x2 = x3 (witness: x1=1, x2=1, x3=1)
        builder.add_constraint(
            vec![(x1, 1)], // A: x1
            vec![(x2, 1)], // B: x2
            vec![(x3, 1)], // C: x3
        );
    }

    builder.build()
}

/// Generate valid witness for circuit
///
/// For m constraints, we have:
/// - 1 constant variable (always 1)
/// - 3m variables (x1, x2, x3 per constraint)
/// Total: n = 1 + 3m
fn build_witness(r1cs: &R1CS) -> Vec<u64> {
    // Witness: all variables = 1 (satisfies x1·x2=x3 → 1·1=1)
    vec![1u64; r1cs.n]
}

fn bench_lagrange_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("lagrange_interpolation");

    // Initialize LWE context (reused across benchmarks)
    let ctx = create_lwe_context();

    // Benchmark for m = 10, 20, 30, 50, 100, 200 constraints
    for m in [10, 20, 30, 50, 100, 200].iter() {
        let r1cs = build_circuit(*m);
        let witness = build_witness(&r1cs);
        let seed = 42u64;

        group.bench_with_input(BenchmarkId::from_parameter(m), m, |b, _| {
            b.iter(|| {
                // Prover includes Lagrange interpolation (3 calls for A_z, B_z, C_z)
                lambda_snark::prove_r1cs(
                    black_box(&r1cs),
                    black_box(&witness),
                    black_box(&ctx),
                    black_box(seed),
                )
                .expect("Proof generation failed");
            });
        });
    }

    group.finish();
}

fn bench_circuit_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_building");

    for m in [10, 20, 30, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(m), m, |b, _| {
            b.iter(|| {
                build_circuit(black_box(*m));
            });
        });
    }

    group.finish();
}

fn bench_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("verification");

    let ctx = create_lwe_context();

    for m in [10, 20, 30, 50, 100, 200].iter() {
        let r1cs = build_circuit(*m);
        let witness = build_witness(&r1cs);
        let proof = lambda_snark::prove_r1cs(&r1cs, &witness, &ctx, 42).expect("Proof failed");
        let public_inputs = vec![]; // No public inputs for this simple circuit

        group.bench_with_input(BenchmarkId::from_parameter(m), m, |b, _| {
            b.iter(|| {
                let valid = lambda_snark::verify_r1cs(
                    black_box(&proof),
                    black_box(&public_inputs),
                    black_box(&r1cs),
                );
                assert!(valid, "Verification failed");
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_lagrange_interpolation,
    bench_circuit_building,
    bench_verification,
);
criterion_main!(benches);
