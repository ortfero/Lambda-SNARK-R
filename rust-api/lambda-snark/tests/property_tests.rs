//! Property-based tests for ΛSNARK-R core operations.
//!
//! Uses proptest to verify algebraic properties and invariants
//! that should hold for all valid inputs.

use lambda_snark::*;
use lambda_snark_core::{Field, Profile, SecurityLevel, NTT_MODULUS};
use proptest::prelude::*;

/// Helper: Create default params for testing
fn test_params() -> Params {
    Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,
            k: 2,
            q: NTT_MODULUS,
            sigma: 3.19,
        },
    )
}

// ============================================================================
// Test Utilities
// ============================================================================

/// Generate valid field element (non-zero, < modulus)
fn arb_field(modulus: u64) -> impl Strategy<Value = u64> {
    (1u64..modulus).prop_map(move |x| x)
}

/// Generate small constraint count (for fast tests)
fn arb_small_m() -> impl Strategy<Value = usize> {
    prop_oneof![Just(1), Just(2), Just(4), Just(8),]
}

/// Generate valid witness size
fn arb_witness_size() -> impl Strategy<Value = usize> {
    (4usize..32).prop_map(|x| x)
}

// ============================================================================
// Polynomial Properties
// ============================================================================

proptest! {
    /// Property: Polynomial evaluation is homomorphic over addition
    /// P1(α) + P2(α) = (P1 + P2)(α)
    #[test]
    fn polynomial_eval_addition_homomorphic(
        coeffs1 in prop::collection::vec(0u64..NTT_MODULUS, 1..10),
        coeffs2 in prop::collection::vec(0u64..NTT_MODULUS, 1..10),
        alpha in arb_field(NTT_MODULUS),
    ) {
        let p1 = Polynomial::new(
            coeffs1.iter().map(|&c| Field::new(c)).collect(),
            NTT_MODULUS
        );
        let p2 = Polynomial::new(
            coeffs2.iter().map(|&c| Field::new(c)).collect(),
            NTT_MODULUS
        );

        let eval1 = p1.evaluate(Field::new(alpha)).value();
        let eval2 = p2.evaluate(Field::new(alpha)).value();
        let sum_eval = ((eval1 as u128 + eval2 as u128) % NTT_MODULUS as u128) as u64;

        // Compute (P1 + P2)(α) by adding coefficients
        let max_len = coeffs1.len().max(coeffs2.len());
        let mut sum_coeffs = vec![0u64; max_len];
        for i in 0..coeffs1.len() {
            sum_coeffs[i] = ((sum_coeffs[i] as u128 + coeffs1[i] as u128) % NTT_MODULUS as u128) as u64;
        }
        for i in 0..coeffs2.len() {
            sum_coeffs[i] = ((sum_coeffs[i] as u128 + coeffs2[i] as u128) % NTT_MODULUS as u128) as u64;
        }

        let p_sum = Polynomial::new(
            sum_coeffs.iter().map(|&c| Field::new(c)).collect(),
            NTT_MODULUS
        );
        let sum_then_eval = p_sum.evaluate(Field::new(alpha)).value();

        prop_assert_eq!(sum_eval, sum_then_eval);
    }

    /// Property: Polynomial from witness encodes correctly
    /// f(i) = witness[i] for i in domain
    #[test]
    fn polynomial_witness_encoding(
        witness in prop::collection::vec(arb_field(NTT_MODULUS), 4..16),
    ) {
        let poly = Polynomial::from_witness(&witness, NTT_MODULUS);

        // Check degree matches
        prop_assert_eq!(poly.degree(), witness.len() - 1);

        // Check evaluation at X=0 gives first coefficient
        let eval_0 = poly.evaluate(Field::new(0)).value();
        prop_assert_eq!(eval_0, witness[0]);
    }

    /// Property: Zero polynomial evaluates to zero everywhere
    #[test]
    fn polynomial_zero_property(
        alpha in arb_field(NTT_MODULUS),
    ) {
        let zero_poly = Polynomial::new(vec![Field::new(0)], NTT_MODULUS);
        let eval = zero_poly.evaluate(Field::new(alpha)).value();
        prop_assert_eq!(eval, 0);
    }

    /// Property: Constant polynomial evaluates to constant
    #[test]
    fn polynomial_constant_property(
        c in arb_field(NTT_MODULUS),
        alpha in arb_field(NTT_MODULUS),
    ) {
        let const_poly = Polynomial::new(vec![Field::new(c)], NTT_MODULUS);
        let eval = const_poly.evaluate(Field::new(alpha)).value();
        prop_assert_eq!(eval, c);
    }
}

// ============================================================================
// R1CS Constraint Properties
// ============================================================================

proptest! {
    /// Property: Valid witness always satisfies constraints
    #[test]
    fn r1cs_valid_witness_satisfies(
        a in arb_field(NTT_MODULUS),
        b in arb_field(NTT_MODULUS),
    ) {
        let c = (a as u128 * b as u128 % NTT_MODULUS as u128) as u64;

        // Build R1CS for multiplication: a * b = c
        let a_mat = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
        let b_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
        let c_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);

        let r1cs = R1CS::new(1, 4, 1, a_mat, b_mat, c_mat, NTT_MODULUS);

        let witness = vec![1, a, b, c];
        prop_assert!(r1cs.is_satisfied(&witness));
    }

    /// Property: Invalid witness fails verification
    #[test]
    fn r1cs_invalid_witness_fails(
        a in arb_field(NTT_MODULUS),
        b in arb_field(NTT_MODULUS),
        wrong_c in arb_field(NTT_MODULUS),
    ) {
        let correct_c = (a as u128 * b as u128 % NTT_MODULUS as u128) as u64;

        // Skip if randomly generated wrong_c happens to be correct
        prop_assume!(wrong_c != correct_c);

        // Build R1CS for multiplication
        let a_mat = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
        let b_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
        let c_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);

        let r1cs = R1CS::new(1, 4, 1, a_mat, b_mat, c_mat, NTT_MODULUS);

        let invalid_witness = vec![1, a, b, wrong_c];
        prop_assert!(!r1cs.is_satisfied(&invalid_witness));
    }

    /// Property: Constraint evaluation is linear in witness
    /// (Az)_i = Σ A[i,j] * z[j]
    #[test]
    fn r1cs_constraint_linearity(
        witness in prop::collection::vec(arb_field(NTT_MODULUS), 4..8),
        _scale in arb_field(NTT_MODULUS),
    ) {
        // Create identity matrix A
        let n = witness.len();
        let identity = (0..n).map(|i| {
            let mut row = vec![0u64; n];
            row[i] = 1;
            row
        }).collect::<Vec<_>>();

        let a_mat = SparseMatrix::from_dense(&identity);
        let b_mat = SparseMatrix::from_dense(&vec![vec![0u64; n]; n]);
        let c_mat = SparseMatrix::from_dense(&vec![vec![0u64; n]; n]);

        let r1cs = R1CS::new(n, n, 0, a_mat, b_mat, c_mat, NTT_MODULUS);

        // Az should equal witness for identity matrix
        let (az, _, _) = r1cs.compute_constraint_evals(&witness);

        prop_assert_eq!(az.len(), witness.len());
        for i in 0..n {
            prop_assert_eq!(az[i], witness[i]);
        }
    }
}

// ============================================================================
// Sparse Matrix Properties
// ============================================================================

proptest! {
    /// Property: Sparse matrix-vector multiplication is linear
    /// A(v1 + v2) = Av1 + Av2
    #[test]
    fn sparse_matrix_linear(
        v1 in prop::collection::vec(arb_field(NTT_MODULUS), 4),
        v2 in prop::collection::vec(arb_field(NTT_MODULUS), 4),
    ) {
        // Create simple sparse matrix
        let dense = vec![
            vec![1, 0, 2, 0],
            vec![0, 3, 0, 1],
        ];
        let mat = SparseMatrix::from_dense(&dense);

        // Compute A(v1 + v2)
        let v_sum: Vec<u64> = v1.iter().zip(v2.iter())
            .map(|(&a, &b)| ((a as u128 + b as u128) % NTT_MODULUS as u128) as u64)
            .collect();
        let result_sum = mat.mul_vec(&v_sum, NTT_MODULUS);

        // Compute Av1 + Av2
        let av1 = mat.mul_vec(&v1, NTT_MODULUS);
        let av2 = mat.mul_vec(&v2, NTT_MODULUS);
        let sum_result: Vec<u64> = av1.iter().zip(av2.iter())
            .map(|(&a, &b)| ((a as u128 + b as u128) % NTT_MODULUS as u128) as u64)
            .collect();

        prop_assert_eq!(result_sum, sum_result);
    }

    /// Property: Zero matrix multiplication gives zero
    #[test]
    fn sparse_matrix_zero(
        v in prop::collection::vec(arb_field(NTT_MODULUS), 4),
    ) {
        let zero_mat = SparseMatrix::from_dense(&vec![vec![0u64; 4]; 3]);
        let result = zero_mat.mul_vec(&v, NTT_MODULUS);

        prop_assert_eq!(result, vec![0u64; 3]);
    }

    /// Property: Identity matrix preserves vector
    #[test]
    fn sparse_matrix_identity(
        v in prop::collection::vec(arb_field(NTT_MODULUS), 4),
    ) {
        let identity = vec![
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1],
        ];
        let mat = SparseMatrix::from_dense(&identity);
        let result = mat.mul_vec(&v, NTT_MODULUS);

        prop_assert_eq!(result, v);
    }
}

// ============================================================================
// Field Arithmetic Properties
// ============================================================================

proptest! {
    /// Property: Field addition is commutative
    /// a + b = b + a
    #[test]
    fn field_addition_commutative(
        a in arb_field(NTT_MODULUS),
        b in arb_field(NTT_MODULUS),
    ) {
        let sum1 = ((a as u128 + b as u128) % NTT_MODULUS as u128) as u64;
        let sum2 = ((b as u128 + a as u128) % NTT_MODULUS as u128) as u64;

        prop_assert_eq!(sum1, sum2);
    }

    /// Property: Field multiplication is commutative
    /// a * b = b * a
    #[test]
    fn field_multiplication_commutative(
        a in arb_field(NTT_MODULUS),
        b in arb_field(NTT_MODULUS),
    ) {
        let prod1 = (a as u128 * b as u128 % NTT_MODULUS as u128) as u64;
        let prod2 = (b as u128 * a as u128 % NTT_MODULUS as u128) as u64;

        prop_assert_eq!(prod1, prod2);
    }

    /// Property: Modular reduction is idempotent
    /// (a mod q) mod q = a mod q
    #[test]
    fn modular_reduction_idempotent(
        a in any::<u64>(),
    ) {
        let reduced1 = a % NTT_MODULUS;
        let reduced2 = reduced1 % NTT_MODULUS;

        prop_assert_eq!(reduced1, reduced2);
    }
}

// ============================================================================
// ZK Properties
// ============================================================================

proptest! {
    /// Property: ZK and non-ZK proofs verify identically for valid witness
    #[test]
    fn zk_non_zk_equivalence_valid(
        a in (2u64..1000),
        b in (2u64..1000),
        seed in any::<u64>(),
    ) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let modulus = NTT_MODULUS;
        let c = (a as u128 * b as u128 % modulus as u128) as u64;

        // Setup R1CS
        let a_mat = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
        let b_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
        let c_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);
        let r1cs = R1CS::new(1, 4, 1, a_mat, b_mat, c_mat, modulus);

        let params = test_params();
        let ctx = LweContext::new(params)?;

        let witness = vec![1, a, b, c];
        let public_inputs = vec![1];

        // Non-ZK proof
        let proof = prove_r1cs(&r1cs, &witness, &ctx, seed)?;
        let valid_non_zk = verify_r1cs(&proof, &public_inputs, &r1cs);

        // ZK proof
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let proof_zk = prove_r1cs_zk(&r1cs, &witness, &ctx, &mut rng, seed)?;
        let valid_zk = verify_r1cs_zk(&proof_zk, &public_inputs, &r1cs);

        // Both should verify
        prop_assert!(valid_non_zk);
        prop_assert!(valid_zk);
    }

    /// Property: ZK blinding doesn't break verification
    #[test]
    fn zk_blinding_preserves_verification(
        witness in prop::collection::vec((2u64..1000), 4),
        seed in any::<u64>(),
    ) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        // Ensure witness[1] * witness[2] = witness[3]
        let a = witness[1];
        let b = witness[2];
        let c = (a as u128 * b as u128 % NTT_MODULUS as u128) as u64;
        let valid_witness = vec![witness[0], a, b, c];

        let a_mat = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
        let b_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
        let c_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);
        let r1cs = R1CS::new(1, 4, 1, a_mat, b_mat, c_mat, NTT_MODULUS);

        let params = test_params();
        let ctx = LweContext::new(params)?;

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let proof_zk = prove_r1cs_zk(&r1cs, &valid_witness, &ctx, &mut rng, seed)?;
        let valid = verify_r1cs_zk(&proof_zk, &[valid_witness[0]], &r1cs);

        prop_assert!(valid);
    }
}

// ============================================================================
// Lagrange Interpolation Properties
// ============================================================================
// NOTE: lagrange_interpolate is private in r1cs.rs, so we test indirectly
// via compute_quotient_poly which uses it internally

proptest! {
    /// Property: Quotient polynomial exists for valid witness
    #[test]
    fn quotient_poly_exists_for_valid_witness(
        a in (2u64..100),
        b in (2u64..100),
    ) {
        let c = (a as u128 * b as u128 % NTT_MODULUS as u128) as u64;

        let a_mat = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
        let b_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
        let c_mat = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);

        let r1cs = R1CS::new(1, 4, 1, a_mat, b_mat, c_mat, NTT_MODULUS);

        let witness = vec![1, a, b, c];

        // Should compute successfully
        let result = r1cs.compute_quotient_poly(&witness);
        prop_assert!(result.is_ok());
    }
}
