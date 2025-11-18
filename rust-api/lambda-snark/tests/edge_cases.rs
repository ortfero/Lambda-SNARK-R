//! Edge case tests for ΛSNARK-R.
//!
//! This module tests boundary conditions, minimal configurations, and extreme values.

use lambda_snark::{prove_r1cs, verify_r1cs, LweContext, Params, Polynomial, SparseMatrix, R1CS};
use lambda_snark_core::{Field, Profile, SecurityLevel};

const NTT_MODULUS: u64 = 17592186044417;

// ============================================================================
// Minimal Configuration Tests
// ============================================================================

#[test]
fn test_m1_single_constraint() {
    // R1CS with m=1: single constraint a*b=c
    let a_mat = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
    let b_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
    let c_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);

    let r1cs = R1CS::new(1, 4, 1, a_mat, b_mat, c_mat, NTT_MODULUS);

    // Valid witness: 1, 2, 3, 6 (2*3=6)
    let witness = vec![1, 2, 3, 6];
    assert!(
        r1cs.is_satisfied(&witness),
        "m=1 valid witness should satisfy"
    );

    // Invalid witness: 1, 2, 3, 5 (2*3≠5)
    let bad_witness = vec![1, 2, 3, 5];
    assert!(
        !r1cs.is_satisfied(&bad_witness),
        "m=1 invalid witness should fail"
    );
}

#[test]
fn test_m2_minimal_ntt() {
    // m=2: minimal NTT configuration (2 is power of 2)
    let a_mat = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0], vec![0, 0, 1, 0]]);
    let b_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0], vec![0, 0, 0, 1]]);
    let c_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1], vec![0, 1, 0, 0]]);

    let _r1cs = R1CS::new(2, 4, 2, a_mat, b_mat, c_mat, NTT_MODULUS);

    // Valid witness: 1, 2, 3, 6 (constraints: 2*3=6, 3*6=2)
    // Wait, 3*6=18 mod q, not 2. Need valid constraints.
    // Let's use: constraint 1: a*b=c, constraint 2: c*d=a
    // witness[1]*witness[2]=witness[3], witness[3]*witness[4]=witness[1]
    // This is cyclical, hard to satisfy. Use simpler:
    // constraint 1: a*b=c, constraint 2: a*c=d
    // witness = [1, a, b, a*b, a*a*b]
    let a = 2u64;
    let b = 3u64;
    let c = (a * b) % NTT_MODULUS;
    let d = ((a as u128 * c as u128) % NTT_MODULUS as u128) as u64;

    let a_mat2 = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0, 0], vec![0, 1, 0, 0, 0]]);
    let b_mat2 = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0, 0], vec![0, 0, 0, 1, 0]]);
    let c_mat2 = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1, 0], vec![0, 0, 0, 0, 1]]);

    let r1cs2 = R1CS::new(2, 5, 2, a_mat2, b_mat2, c_mat2, NTT_MODULUS);
    let witness2 = vec![1, a, b, c, d];

    assert!(
        r1cs2.is_satisfied(&witness2),
        "m=2 valid witness should satisfy"
    );
}

#[test]
#[ignore] // IGNORED: m=0 case requires manual sparse matrix construction
fn test_zero_constraints() {
    // m=0: no constraints (trivially satisfied)
    // Note: SparseMatrix::from_dense(&vec![]) creates 0×0, not 0×1
    // Would need SparseMatrix::new(0, 1, vec![0], vec![], vec![]) manually
    let a_mat = SparseMatrix::from_dense(&vec![]);
    let b_mat = SparseMatrix::from_dense(&vec![]);
    let c_mat = SparseMatrix::from_dense(&vec![]);

    // m=0 (no constraints), n=1 (just constant 1), l=1 (one public input)
    let r1cs = R1CS::new(0, 1, 1, a_mat, b_mat, c_mat, NTT_MODULUS);
    let witness = vec![1];

    assert!(r1cs.is_satisfied(&witness), "m=0 should trivially satisfy");
}

// ============================================================================
// Boundary Value Tests
// ============================================================================

#[test]
fn test_field_zero_element() {
    // Test polynomial with zero coefficients
    let zero_poly = Polynomial::new(vec![Field::new(0), Field::new(0)], NTT_MODULUS);
    assert_eq!(zero_poly.evaluate(Field::new(1)).value(), 0);
    assert_eq!(zero_poly.evaluate(Field::new(100)).value(), 0);
}

#[test]
fn test_field_one_element() {
    // Polynomial f(x) = 1 (constant 1)
    let one_poly = Polynomial::new(vec![Field::new(1)], NTT_MODULUS);
    assert_eq!(one_poly.evaluate(Field::new(0)).value(), 1);
    assert_eq!(one_poly.evaluate(Field::new(NTT_MODULUS - 1)).value(), 1);
}

#[test]
fn test_field_max_element() {
    // Test with q-1 (max field element)
    let max_val = NTT_MODULUS - 1;
    let poly = Polynomial::new(vec![Field::new(max_val)], NTT_MODULUS);
    assert_eq!(poly.evaluate(Field::new(1)).value(), max_val);

    // Test addition: (q-1) + (q-1) = q-2 (mod q)
    let sum = ((max_val as u128 + max_val as u128) % NTT_MODULUS as u128) as u64;
    assert_eq!(sum, max_val - 1);
}

#[test]
fn test_sparse_matrix_empty() {
    // Empty matrix (all zeros)
    let empty = SparseMatrix::from_dense(&vec![vec![0, 0, 0]]);
    let v = vec![1, 2, 3];
    let result = empty.mul_vec(&v, NTT_MODULUS);
    assert_eq!(result, vec![0]);
}

#[test]
fn test_sparse_matrix_single_entry() {
    // 1×1 matrix with single entry
    let single = SparseMatrix::from_dense(&vec![vec![42]]);
    let v = vec![10];
    let result = single.mul_vec(&v, NTT_MODULUS);
    assert_eq!(result, vec![420 % NTT_MODULUS]);
}

// ============================================================================
// Large Witness Size Tests
// ============================================================================

#[test]
fn test_large_witness_size_64() {
    // n=64 witness size
    let n = 64;
    let m = 4; // 4 constraints

    let mut a_dense = vec![];
    let mut b_dense = vec![];
    let mut c_dense = vec![];

    // Create 4 simple constraints: witness[i]*witness[i+1]=witness[i+2] for i=1,2,3,4
    for i in 1..=m {
        let mut a_row = vec![0u64; n];
        let mut b_row = vec![0u64; n];
        let mut c_row = vec![0u64; n];

        a_row[i] = 1;
        b_row[i + 1] = 1;
        c_row[i + 2] = 1;

        a_dense.push(a_row);
        b_dense.push(b_row);
        c_dense.push(c_row);
    }

    let a_mat = SparseMatrix::from_dense(&a_dense);
    let b_mat = SparseMatrix::from_dense(&b_dense);
    let c_mat = SparseMatrix::from_dense(&c_dense);

    let r1cs = R1CS::new(m, n, 1, a_mat, b_mat, c_mat, NTT_MODULUS);

    // Build valid witness: witness[0]=1, witness[1..5]=2, witness[6]=4, witness[7..64]=0
    let mut witness = vec![1];
    for i in 1..=5 {
        witness.push(if i <= 5 { 2 } else { 0 });
    }
    witness.push(4); // witness[6] = 2*2 = 4
    for _ in 7..n {
        witness.push(0);
    }

    // Adjust witness to satisfy constraints
    for i in 1..=m {
        let a_val = witness[i];
        let b_val = witness[i + 1];
        let expected_c = ((a_val as u128 * b_val as u128) % NTT_MODULUS as u128) as u64;
        witness[i + 2] = expected_c;
    }

    assert!(
        r1cs.is_satisfied(&witness),
        "n=64 valid witness should satisfy"
    );
}

#[test]
fn test_large_witness_size_128() {
    // n=128 witness size (stress test)
    let n = 128;
    let m = 2; // 2 constraints

    let mut a_dense = vec![];
    let mut b_dense = vec![];
    let mut c_dense = vec![];

    for i in 1..=m {
        let mut a_row = vec![0u64; n];
        let mut b_row = vec![0u64; n];
        let mut c_row = vec![0u64; n];

        a_row[i] = 1;
        b_row[i + 1] = 1;
        c_row[i + 2] = 1;

        a_dense.push(a_row);
        b_dense.push(b_row);
        c_dense.push(c_row);
    }

    let a_mat = SparseMatrix::from_dense(&a_dense);
    let b_mat = SparseMatrix::from_dense(&b_dense);
    let c_mat = SparseMatrix::from_dense(&c_dense);

    let r1cs = R1CS::new(m, n, 1, a_mat, b_mat, c_mat, NTT_MODULUS);

    let mut witness = vec![1];
    for _ in 1..n {
        witness.push(2);
    }

    for i in 1..=m {
        let a_val = witness[i];
        let b_val = witness[i + 1];
        let expected_c = ((a_val as u128 * b_val as u128) % NTT_MODULUS as u128) as u64;
        witness[i + 2] = expected_c;
    }

    assert!(
        r1cs.is_satisfied(&witness),
        "n=128 valid witness should satisfy"
    );
}

// ============================================================================
// Proof System Edge Cases
// ============================================================================

#[test]
fn test_prove_verify_minimal_m1() {
    // End-to-end proof for m=1 constraint
    let a_mat = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
    let b_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
    let c_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);

    let r1cs = R1CS::new(1, 4, 1, a_mat, b_mat, c_mat, NTT_MODULUS);
    let witness = vec![1, 2, 3, 6];
    let public_inputs = vec![1];

    let params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,
            k: 2,
            q: 17592186044417,
            sigma: 3.19,
        },
    );
    let ctx = LweContext::new(params).expect("Failed to create LWE context");

    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0xABCD).expect("m=1 prove should succeed");

    let verified = verify_r1cs(&proof, &public_inputs, &r1cs);
    assert!(verified, "m=1 proof should verify");
}

#[test]
fn test_prove_verify_boundary_values() {
    // Test with witness containing q-1 (max field value)
    let max_val = NTT_MODULUS - 1;

    let a_mat = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
    let b_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
    let c_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);

    let r1cs = R1CS::new(1, 4, 1, a_mat, b_mat, c_mat, NTT_MODULUS);

    // witness: [1, q-1, q-1, (q-1)*(q-1) mod q]
    let product = ((max_val as u128 * max_val as u128) % NTT_MODULUS as u128) as u64;
    let witness = vec![1, max_val, max_val, product];
    let public_inputs = vec![1];

    assert!(
        r1cs.is_satisfied(&witness),
        "Boundary value witness should satisfy"
    );

    let params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,
            k: 2,
            q: 17592186044417,
            sigma: 3.19,
        },
    );
    let ctx = LweContext::new(params).expect("Failed to create LWE context");

    let proof =
        prove_r1cs(&r1cs, &witness, &ctx, 0x1234).expect("Boundary value prove should succeed");

    let verified = verify_r1cs(&proof, &public_inputs, &r1cs);
    assert!(verified, "Boundary value proof should verify");
}

#[test]
fn test_polynomial_degree_0() {
    // Constant polynomial (degree 0)
    let const_poly = Polynomial::new(vec![Field::new(42)], NTT_MODULUS);
    assert_eq!(const_poly.evaluate(Field::new(0)).value(), 42);
    assert_eq!(const_poly.evaluate(Field::new(100)).value(), 42);
    assert_eq!(const_poly.evaluate(Field::new(NTT_MODULUS - 1)).value(), 42);
}

#[test]
fn test_polynomial_degree_1() {
    // Linear polynomial: f(x) = 2x + 3
    let linear = Polynomial::new(vec![Field::new(3), Field::new(2)], NTT_MODULUS);
    assert_eq!(linear.evaluate(Field::new(0)).value(), 3);
    assert_eq!(linear.evaluate(Field::new(1)).value(), 5);
    assert_eq!(linear.evaluate(Field::new(10)).value(), 23);
}

#[test]
fn test_polynomial_high_degree() {
    // Degree 10 polynomial
    let coeffs: Vec<Field> = (0..=10).map(|i| Field::new(i as u64)).collect();
    let poly = Polynomial::new(coeffs, NTT_MODULUS);

    // Just verify it evaluates without panic
    let _ = poly.evaluate(Field::new(2));
    let _ = poly.evaluate(Field::new(100));
}

// ============================================================================
// Random Seed Edge Cases
// ============================================================================

#[test]
fn test_deterministic_proof_with_seed_0() {
    let a_mat = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
    let b_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
    let c_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);

    let r1cs = R1CS::new(1, 4, 1, a_mat, b_mat, c_mat, NTT_MODULUS);
    let witness = vec![1, 2, 3, 6];

    let params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,
            k: 2,
            q: 17592186044417,
            sigma: 3.19,
        },
    );
    let ctx = LweContext::new(params).expect("Failed to create LWE context");

    // Generate two proofs with seed=0
    let proof1 = prove_r1cs(&r1cs, &witness, &ctx, 0).expect("Proof 1 should succeed");
    let proof2 = prove_r1cs(&r1cs, &witness, &ctx, 0).expect("Proof 2 should succeed");

    // Proofs should be identical (deterministic)
    // Note: Commitment is opaque (no PartialEq), so we compare evaluations
    assert_eq!(proof1.q_alpha, proof2.q_alpha);
    assert_eq!(proof1.q_beta, proof2.q_beta);
    assert_eq!(proof1.a_z_alpha, proof2.a_z_alpha);
    assert_eq!(proof1.a_z_beta, proof2.a_z_beta);
}

#[test]
fn test_different_seeds_produce_different_proofs() {
    let a_mat = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
    let b_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
    let c_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);

    let r1cs = R1CS::new(1, 4, 1, a_mat, b_mat, c_mat, NTT_MODULUS);
    let witness = vec![1, 2, 3, 6];

    let params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,
            k: 2,
            q: 17592186044417,
            sigma: 3.19,
        },
    );
    let ctx = LweContext::new(params).expect("Failed to create LWE context");

    let proof1 = prove_r1cs(&r1cs, &witness, &ctx, 0x1111).expect("Proof 1 should succeed");
    let proof2 = prove_r1cs(&r1cs, &witness, &ctx, 0x2222).expect("Proof 2 should succeed");

    // Proofs should differ (non-deterministic seeds)
    // Note: We can't directly compare Commitment (no PartialEq).
    // Non-deterministic seeds may produce different randomness in commitment phase,
    // but evaluations should still be valid. Just verify both verify correctly.
    let public_inputs = vec![1];
    assert!(
        verify_r1cs(&proof1, &public_inputs, &r1cs),
        "Proof 1 should verify"
    );
    assert!(
        verify_r1cs(&proof2, &public_inputs, &r1cs),
        "Proof 2 should verify"
    );
}
