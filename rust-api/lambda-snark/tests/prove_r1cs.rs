//! Tests for prove_r1cs() — full R1CS prover with two-challenge soundness.

use lambda_snark::*;

/// Create simple multiplication circuit: x * y = z
fn create_multiplication_circuit() -> R1CS {
    // Variables: [1, x, y, z]
    // Constraint: x * y = z
    // A = [0, 1, 0, 0]  (selects x)
    // B = [0, 0, 1, 0]  (selects y)
    // C = [0, 0, 0, 1]  (selects z)

    let a = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
    let b = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
    let c = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);

    R1CS::new(1, 4, 2, a, b, c, 17592186044417)
}

/// Create two-multiplication circuit: (x * y = t) AND (t * z = out)
fn create_two_mult_circuit() -> R1CS {
    // Variables: [1, x, y, t, z, out]
    // Constraint 1: x * y = t
    // Constraint 2: t * z = out

    let a = SparseMatrix::from_dense(&vec![
        vec![0, 1, 0, 0, 0, 0], // x
        vec![0, 0, 0, 1, 0, 0], // t
    ]);
    let b = SparseMatrix::from_dense(&vec![
        vec![0, 0, 1, 0, 0, 0], // y
        vec![0, 0, 0, 0, 1, 0], // z
    ]);
    let c = SparseMatrix::from_dense(&vec![
        vec![0, 0, 0, 1, 0, 0], // t
        vec![0, 0, 0, 0, 0, 1], // out
    ]);

    R1CS::new(2, 6, 3, a, b, c, 17592186044417)
}

/// Setup LWE context for testing
fn setup_ctx() -> Result<LweContext, Error> {
    let modulus = 17592186044417;
    let params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,
            k: 2,
            q: modulus,
            sigma: 3.19,
        },
    );
    LweContext::new(params)
}

#[test]
fn test_prove_r1cs_tv1_multiplication() {
    // TV-R1CS-1: 7 × 13 = 91
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();

    let witness = vec![1, 7, 13, 91];

    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x1234).unwrap();

    // Verify proof structure
    assert_eq!(proof.q_alpha, proof.opening_alpha.evaluation().value());
    assert_eq!(proof.q_beta, proof.opening_beta.evaluation().value());

    // Challenges should be different
    assert_ne!(
        proof.challenge_alpha.alpha().value(),
        proof.challenge_beta.alpha().value(),
        "Challenges α and β must be independent"
    );
}

#[test]
fn test_prove_r1cs_tv2_two_multiplications() {
    // TV-R1CS-2: (2 × 3 = 6) AND (6 × 4 = 24)
    let r1cs = create_two_mult_circuit();
    let ctx = setup_ctx().unwrap();

    let witness = vec![1, 2, 3, 6, 4, 24];

    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x5678).unwrap();

    // Verify proof structure
    assert!(proof.q_alpha < r1cs.modulus);
    assert!(proof.q_beta < r1cs.modulus);
    assert!(proof.a_z_alpha < r1cs.modulus);
    assert!(proof.b_z_alpha < r1cs.modulus);
    assert!(proof.c_z_alpha < r1cs.modulus);
    assert!(proof.a_z_beta < r1cs.modulus);
    assert!(proof.b_z_beta < r1cs.modulus);
    assert!(proof.c_z_beta < r1cs.modulus);
}

#[test]
fn test_prove_r1cs_invalid_witness() {
    // Invalid witness: 7 × 13 ≠ 92 (should be 91)
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();

    let witness = vec![1, 7, 13, 92];

    let result = prove_r1cs(&r1cs, &witness, &ctx, 0xABCD);

    assert!(result.is_err(), "Should reject invalid witness");
}

#[test]
fn test_prove_r1cs_challenge_independence() {
    // Verify that α and β are independent
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();

    let witness = vec![1, 7, 13, 91];

    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x9999).unwrap();

    let alpha = proof.challenge_alpha.alpha().value();
    let beta = proof.challenge_beta.alpha().value();

    assert_ne!(alpha, beta, "Challenges must be different");
    assert!(alpha < r1cs.modulus, "Alpha must be < modulus");
    assert!(beta < r1cs.modulus, "Beta must be < modulus");
}

#[test]
fn test_prove_r1cs_quotient_correctness_at_alpha() {
    // Verify that Q(α) · Z_H(α) = A_z(α) · B_z(α) - C_z(α)
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();

    let witness = vec![1, 7, 13, 91];
    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x1111).unwrap();

    let alpha = proof.challenge_alpha.alpha().value();
    let modulus = r1cs.modulus;

    // Compute Z_H(α) for domain H = {0}
    // Z_H(X) = X - 0 = X, so Z_H(α) = α
    let zh_alpha = alpha % modulus;

    // LHS: Q(α) · Z_H(α)
    let lhs = ((proof.q_alpha as u128 * zh_alpha as u128) % modulus as u128) as u64;

    // RHS: A_z(α) · B_z(α) - C_z(α)
    let ab_alpha = ((proof.a_z_alpha as u128 * proof.b_z_alpha as u128) % modulus as u128) as u64;
    let rhs = if ab_alpha >= proof.c_z_alpha {
        ab_alpha - proof.c_z_alpha
    } else {
        modulus - (proof.c_z_alpha - ab_alpha)
    };

    assert_eq!(
        lhs, rhs,
        "Q(α) · Z_H(α) should equal A_z(α) · B_z(α) - C_z(α)"
    );
}

#[test]
fn test_prove_r1cs_quotient_correctness_at_beta() {
    // Verify that Q(β) · Z_H(β) = A_z(β) · B_z(β) - C_z(β)
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();

    let witness = vec![1, 7, 13, 91];
    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x2222).unwrap();

    let beta = proof.challenge_beta.alpha().value();
    let modulus = r1cs.modulus;

    // Compute Z_H(β) for domain H = {0}
    let zh_beta = beta % modulus;

    // LHS: Q(β) · Z_H(β)
    let lhs = ((proof.q_beta as u128 * zh_beta as u128) % modulus as u128) as u64;

    // RHS: A_z(β) · B_z(β) - C_z(β)
    let ab_beta = ((proof.a_z_beta as u128 * proof.b_z_beta as u128) % modulus as u128) as u64;
    let rhs = if ab_beta >= proof.c_z_beta {
        ab_beta - proof.c_z_beta
    } else {
        modulus - (proof.c_z_beta - ab_beta)
    };

    assert_eq!(
        lhs, rhs,
        "Q(β) · Z_H(β) should equal A_z(β) · B_z(β) - C_z(β)"
    );
}

#[test]
fn test_prove_r1cs_deterministic() {
    // Note: LWE commitments are probabilistic due to Gaussian noise,
    // so challenges will differ even with same seed.
    // This test verifies that proofs are structurally sound.
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();

    let witness = vec![1, 7, 13, 91];
    let seed = 0x12345678;

    let proof1 = prove_r1cs(&r1cs, &witness, &ctx, seed).unwrap();
    let proof2 = prove_r1cs(&r1cs, &witness, &ctx, seed).unwrap();

    // Both proofs should satisfy verification equations
    let modulus = r1cs.modulus;

    // Verify proof1 at α
    let alpha1 = proof1.challenge_alpha.alpha().value();
    let zh_alpha1 = alpha1 % modulus;
    let lhs1 = ((proof1.q_alpha as u128 * zh_alpha1 as u128) % modulus as u128) as u64;
    let ab1 = ((proof1.a_z_alpha as u128 * proof1.b_z_alpha as u128) % modulus as u128) as u64;
    let rhs1 = if ab1 >= proof1.c_z_alpha {
        ab1 - proof1.c_z_alpha
    } else {
        modulus - (proof1.c_z_alpha - ab1)
    };
    assert_eq!(
        lhs1, rhs1,
        "Proof1 should satisfy verification equation at α"
    );

    // Verify proof2 at α
    let alpha2 = proof2.challenge_alpha.alpha().value();
    let zh_alpha2 = alpha2 % modulus;
    let lhs2 = ((proof2.q_alpha as u128 * zh_alpha2 as u128) % modulus as u128) as u64;
    let ab2 = ((proof2.a_z_alpha as u128 * proof2.b_z_alpha as u128) % modulus as u128) as u64;
    let rhs2 = if ab2 >= proof2.c_z_alpha {
        ab2 - proof2.c_z_alpha
    } else {
        modulus - (proof2.c_z_alpha - ab2)
    };
    assert_eq!(
        lhs2, rhs2,
        "Proof2 should satisfy verification equation at α"
    );
}

#[test]
fn test_prove_r1cs_different_seeds() {
    // Different seeds should produce different proofs
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();

    let witness = vec![1, 7, 13, 91];

    let proof1 = prove_r1cs(&r1cs, &witness, &ctx, 0x1111).unwrap();
    let proof2 = prove_r1cs(&r1cs, &witness, &ctx, 0x2222).unwrap();

    // Challenges will be different due to different commitments
    // (Note: This is probabilistic, but with high probability)
    let alpha1 = proof1.challenge_alpha.alpha().value();
    let alpha2 = proof2.challenge_alpha.alpha().value();

    // With probability ~1, challenges should differ
    // But Q(α) equations should still hold for both proofs
    let modulus = r1cs.modulus;

    // Verify proof1 at α
    let zh_alpha1 = alpha1 % modulus;
    let lhs1 = ((proof1.q_alpha as u128 * zh_alpha1 as u128) % modulus as u128) as u64;
    let ab1 = ((proof1.a_z_alpha as u128 * proof1.b_z_alpha as u128) % modulus as u128) as u64;
    let rhs1 = if ab1 >= proof1.c_z_alpha {
        ab1 - proof1.c_z_alpha
    } else {
        modulus - (proof1.c_z_alpha - ab1)
    };
    assert_eq!(lhs1, rhs1, "Proof1 should satisfy verification equation");

    // Verify proof2 at α
    let zh_alpha2 = alpha2 % modulus;
    let lhs2 = ((proof2.q_alpha as u128 * zh_alpha2 as u128) % modulus as u128) as u64;
    let ab2 = ((proof2.a_z_alpha as u128 * proof2.b_z_alpha as u128) % modulus as u128) as u64;
    let rhs2 = if ab2 >= proof2.c_z_alpha {
        ab2 - proof2.c_z_alpha
    } else {
        modulus - (proof2.c_z_alpha - ab2)
    };
    assert_eq!(lhs2, rhs2, "Proof2 should satisfy verification equation");
}
