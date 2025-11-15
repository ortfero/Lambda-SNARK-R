//! Benchmark for measuring ZK overhead (prove_r1cs_zk vs prove_r1cs)
//!
//! Target: ZK overhead ≤ 1.10× (ideal) or ≤ 1.30× (acceptable)
//! Current estimate: 1.53× (needs optimization)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lambda_snark::{
    prove_r1cs, prove_r1cs_zk, LweContext, Params, Profile, SecurityLevel, R1CS, SparseMatrix,
};
use rand::thread_rng;
use rand::Rng;

const LEGACY_MODULUS: u64 = 17592186044417; // 2^44 + 1 (prime, NTT-compatible)

/// Create R1CS for m multiplication gates
fn create_multiplication_gates(m: usize, modulus: u64) -> R1CS {
    let n = 1 + 3 * m;
    let l = 1;
    
    let mut a_rows = Vec::new();
    let mut b_rows = Vec::new();
    let mut c_rows = Vec::new();
    
    for i in 0..m {
        let a_idx = 1 + i;
        let b_idx = 1 + m + i;
        let c_idx = 1 + 2 * m + i;
        
        let mut a_row = vec![0u64; n];
        a_row[a_idx] = 1;
        a_rows.push(a_row);
        
        let mut b_row = vec![0u64; n];
        b_row[b_idx] = 1;
        b_rows.push(b_row);
        
        let mut c_row = vec![0u64; n];
        c_row[c_idx] = 1;
        c_rows.push(c_row);
    }
    
    let a = SparseMatrix::from_dense(&a_rows);
    let b = SparseMatrix::from_dense(&b_rows);
    let c = SparseMatrix::from_dense(&c_rows);
    
    R1CS::new(m, n, l, a, b, c, modulus)
}

/// Helper: create witness satisfying multiplication constraints
/// Ensures ALL witness elements (including products) are coprime with modulus
/// Uses rejection sampling for unbiased distribution
/// Structure: [1 (public), a_1..a_m, b_1..b_m, c_1..c_m]
fn create_witness(m: usize, modulus: u64) -> Vec<u64> {
    let mut rng = thread_rng();
    let mut witness = vec![1u64]; // Public input = 1
    
    // Generate a_i, b_i pairs ensuring c_i = a_i * b_i is also coprime
    let mut a_vals = Vec::new();
    let mut b_vals = Vec::new();
    let mut c_vals = Vec::new();
    
    for _ in 0..m {
        // Retry until we find a, b such that both a, b, AND a*b are coprime with modulus
        let (a, b, c) = loop {
            let a = loop {
                let candidate = rng.gen_range(1..modulus);
                if gcd(candidate, modulus) == 1 {
                    break candidate;
                }
            };
            
            let b = loop {
                let candidate = rng.gen_range(1..modulus);
                if gcd(candidate, modulus) == 1 {
                    break candidate;
                }
            };
            
            let c = ((a as u128 * b as u128) % modulus as u128) as u64;
            
            // Check if product is also coprime
            if gcd(c, modulus) == 1 {
                break (a, b, c);
            }
            // Otherwise retry with new a, b
        };
        
        a_vals.push(a);
        b_vals.push(b);
        c_vals.push(c);
    }
    
    // Append all a values
    witness.extend(&a_vals);
    // Append all b values
    witness.extend(&b_vals);
    // Append all c values
    witness.extend(&c_vals);
    
    witness
}

/// Euclidean GCD algorithm
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Benchmark baseline prover (non-ZK)
fn bench_prove_r1cs(c: &mut Criterion) {
    let mut group = c.benchmark_group("prove_r1cs");
    
    // TODO(M7.2): Fix lagrange_basis() to use NTT roots instead of 0,1,2,...
    // Currently m≥32 fails due to non-invertible denominators with sequential points.
    // See r1cs.rs:lagrange_basis() - needs root-of-unity interpolation points.
    for m in [4, 8, 16].iter() {  // Limited to m≤16 until lagrange_basis fixed
        let r1cs = create_multiplication_gates(*m, LEGACY_MODULUS);
        let witness = create_witness(*m, LEGACY_MODULUS);
        
        let params = Params::new(
            SecurityLevel::Bits128,
            Profile::RingB {
                n: 4096,
                k: 2,
                q: LEGACY_MODULUS,
                sigma: 3.19,
            },
        );
        let ctx = LweContext::new(params).expect("Failed to create LWE context");
        
        group.bench_with_input(BenchmarkId::from_parameter(m), m, |b, _| {
            b.iter(|| {
                prove_r1cs(
                    black_box(&r1cs),
                    black_box(&witness),
                    black_box(&ctx),
                    black_box(0x1234),
                )
            });
        });
    }
    
    group.finish();
}

/// Benchmark ZK prover (with blinding)
fn bench_prove_r1cs_zk(c: &mut Criterion) {
    let mut group = c.benchmark_group("prove_r1cs_zk");
    
    // Limited to m≤16 until lagrange_basis supports NTT roots (see bench_prove_r1cs TODO)
    for m in [4, 8, 16].iter() {
        let r1cs = create_multiplication_gates(*m, LEGACY_MODULUS);
        let witness = create_witness(*m, LEGACY_MODULUS);
        
        let params = Params::new(
            SecurityLevel::Bits128,
            Profile::RingB {
                n: 4096,
                k: 2,
                q: LEGACY_MODULUS,
                sigma: 3.19,
            },
        );
        let ctx = LweContext::new(params).expect("Failed to create LWE context");
        
        group.bench_with_input(BenchmarkId::from_parameter(m), m, |b, _| {
            b.iter(|| {
                let mut rng = thread_rng();
                prove_r1cs_zk(
                    black_box(&r1cs),
                    black_box(&witness),
                    black_box(&ctx),
                    black_box(&mut rng),
                    black_box(0x1234),
                )
            });
        });
    }
    
    group.finish();
}

/// Benchmark overhead ratio (ZK / baseline)
fn bench_overhead_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("zk_overhead_ratio");
    
    // Use m=16 for overhead measurement
    let m = 16;
    let r1cs = create_multiplication_gates(m, LEGACY_MODULUS);
    let witness = create_witness(m, LEGACY_MODULUS);
    
    let params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,
            k: 2,
            q: LEGACY_MODULUS,
            sigma: 3.19,
        },
    );
    let ctx = LweContext::new(params).expect("Failed to create LWE context");
    
    // Measure baseline
    group.bench_function("baseline", |b| {
        b.iter(|| {
            prove_r1cs(
                black_box(&r1cs),
                black_box(&witness),
                black_box(&ctx),
                black_box(0x1234),
            )
        });
    });
    
    // Measure ZK
    group.bench_function("zk", |b| {
        b.iter(|| {
            let mut rng = thread_rng();
            prove_r1cs_zk(
                black_box(&r1cs),
                black_box(&witness),
                black_box(&ctx),
                black_box(&mut rng),
                black_box(0x1234),
            )
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_prove_r1cs, bench_prove_r1cs_zk, bench_overhead_ratio);
criterion_main!(benches);
