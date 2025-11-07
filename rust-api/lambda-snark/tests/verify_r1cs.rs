//! Tests for verify_r1cs() — R1CS verifier with two-challenge soundness.

use lambda_snark::*;

/// Create simple multiplication circuit: x * y = z
fn create_multiplication_circuit() -> R1CS {
    let a = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
    let b = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
    let c = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);
    
    R1CS::new(1, 4, 2, a, b, c, 17592186044417)
}

/// Create two-multiplication circuit
fn create_two_mult_circuit() -> R1CS {
    let a = SparseMatrix::from_dense(&vec![
        vec![0, 1, 0, 0, 0, 0],
        vec![0, 0, 0, 1, 0, 0],
    ]);
    let b = SparseMatrix::from_dense(&vec![
        vec![0, 0, 1, 0, 0, 0],
        vec![0, 0, 0, 0, 1, 0],
    ]);
    let c = SparseMatrix::from_dense(&vec![
        vec![0, 0, 0, 1, 0, 0],
        vec![0, 0, 0, 0, 0, 1],
    ]);
    
    R1CS::new(2, 6, 3, a, b, c, 17592186044417)
}

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
fn test_verify_r1cs_valid_proof() {
    // Honest prover should always produce verifying proof
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x1234).unwrap();
    
    let public_inputs = r1cs.public_inputs(&witness);
    let valid = verify_r1cs(&proof, public_inputs, &r1cs);
    
    assert!(valid, "Valid proof should verify");
}

#[test]
fn test_verify_r1cs_tv1() {
    // TV-R1CS-1: 7 × 13 = 91
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0xABCD).unwrap();
    
    let public_inputs = r1cs.public_inputs(&witness);
    assert!(verify_r1cs(&proof, public_inputs, &r1cs));
}

#[test]
fn test_verify_r1cs_tv2() {
    // TV-R1CS-2: (2×3=6) AND (6×4=24)
    let r1cs = create_two_mult_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 2, 3, 6, 4, 24];
    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x5678).unwrap();
    
    let public_inputs = r1cs.public_inputs(&witness);
    assert!(verify_r1cs(&proof, public_inputs, &r1cs));
}

#[test]
fn test_verify_r1cs_wrong_public_inputs() {
    // Verifier uses wrong public inputs
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x1111).unwrap();
    
    // Use wrong public inputs (should cause challenge mismatch)
    let wrong_public = vec![1, 8]; // Changed from [1, 7]
    let valid = verify_r1cs(&proof, &wrong_public, &r1cs);
    
    assert!(!valid, "Proof should fail with wrong public inputs");
}

#[test]
fn test_verify_r1cs_forged_q_alpha() {
    // Attacker tries to forge Q(α) evaluation
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let mut proof = prove_r1cs(&r1cs, &witness, &ctx, 0x2222).unwrap();
    
    // Forge Q(α) (should fail verification equation at α)
    proof.q_alpha = (proof.q_alpha + 1) % r1cs.modulus;
    
    let public_inputs = r1cs.public_inputs(&witness);
    let valid = verify_r1cs(&proof, &public_inputs, &r1cs);
    
    assert!(!valid, "Forged Q(α) should fail verification");
}

#[test]
fn test_verify_r1cs_forged_q_beta() {
    // Attacker tries to forge Q(β) evaluation
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let mut proof = prove_r1cs(&r1cs, &witness, &ctx, 0x3333).unwrap();
    
    // Forge Q(β) (should fail verification equation at β)
    proof.q_beta = (proof.q_beta + 1) % r1cs.modulus;
    
    let public_inputs = r1cs.public_inputs(&witness);
    let valid = verify_r1cs(&proof, &public_inputs, &r1cs);
    
    assert!(!valid, "Forged Q(β) should fail verification");
}

#[test]
fn test_verify_r1cs_forged_a_z_alpha() {
    // Attacker tries to forge A_z(α)
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let mut proof = prove_r1cs(&r1cs, &witness, &ctx, 0x4444).unwrap();
    
    // Forge A_z(α)
    proof.a_z_alpha = (proof.a_z_alpha + 1) % r1cs.modulus;
    
    let public_inputs = r1cs.public_inputs(&witness);
    let valid = verify_r1cs(&proof, &public_inputs, &r1cs);
    
    assert!(!valid, "Forged A_z(α) should fail verification");
}

#[test]
fn test_verify_r1cs_forged_b_z_beta() {
    // Attacker tries to forge B_z(β)
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let mut proof = prove_r1cs(&r1cs, &witness, &ctx, 0x5555).unwrap();
    
    // Forge B_z(β)
    proof.b_z_beta = (proof.b_z_beta + 1) % r1cs.modulus;
    
    let public_inputs = r1cs.public_inputs(&witness);
    let valid = verify_r1cs(&proof, &public_inputs, &r1cs);
    
    assert!(!valid, "Forged B_z(β) should fail verification");
}

#[test]
fn test_verify_r1cs_opening_mismatch_alpha() {
    // Opening evaluation doesn't match claimed Q(α)
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let mut proof = prove_r1cs(&r1cs, &witness, &ctx, 0x6666).unwrap();
    
    // Change Q(α) but keep opening the same (mismatch)
    proof.q_alpha = (proof.q_alpha + 1) % r1cs.modulus;
    
    let public_inputs = r1cs.public_inputs(&witness);
    let valid = verify_r1cs(&proof, &public_inputs, &r1cs);
    
    assert!(!valid, "Opening mismatch should fail verification");
}

#[test]
fn test_verify_r1cs_multiple_witnesses() {
    // Verify multiple different valid proofs
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let test_cases = vec![
        vec![1, 2, 3, 6],      // 2 × 3 = 6
        vec![1, 5, 7, 35],     // 5 × 7 = 35
        vec![1, 11, 13, 143],  // 11 × 13 = 143
    ];
    
    for (i, witness) in test_cases.iter().enumerate() {
        let proof = prove_r1cs(&r1cs, witness, &ctx, (i + 1) as u64 * 0x1000).unwrap();
        let public_inputs = r1cs.public_inputs(witness);
        
        assert!(
            verify_r1cs(&proof, &public_inputs, &r1cs),
            "Proof {} should verify", i
        );
    }
}

#[test]
fn test_verify_r1cs_soundness_two_challenges() {
    // Verify that both challenges are actually checked
    // Forging one equation should fail even if the other is correct
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let mut proof = prove_r1cs(&r1cs, &witness, &ctx, 0x7777).unwrap();
    
    // Save original values
    let orig_q_alpha = proof.q_alpha;
    let orig_q_beta = proof.q_beta;
    
    // Test 1: Forge only α equation
    proof.q_alpha = (orig_q_alpha + 1) % r1cs.modulus;
    proof.q_beta = orig_q_beta; // Keep β correct
    
    let public_inputs = r1cs.public_inputs(&witness);
    assert!(
        !verify_r1cs(&proof, &public_inputs, &r1cs),
        "Should fail with forged α equation"
    );
    
    // Test 2: Forge only β equation
    proof.q_alpha = orig_q_alpha; // Restore α
    proof.q_beta = (orig_q_beta + 1) % r1cs.modulus;
    
    assert!(
        !verify_r1cs(&proof, &public_inputs, &r1cs),
        "Should fail with forged β equation"
    );
}

#[test]
fn test_verify_r1cs_challenge_independence() {
    // Verify that α and β are actually different
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x8888).unwrap();
    
    let alpha = proof.challenge_alpha.alpha().value();
    let beta = proof.challenge_beta.alpha().value();
    
    assert_ne!(alpha, beta, "Challenges must be independent");
}

#[test]
fn test_verify_r1cs_completeness() {
    // Completeness: All valid witnesses should produce verifying proofs
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let valid_witnesses = vec![
        vec![1, 1, 1, 1],        // 1 × 1 = 1
        vec![1, 0, 5, 0],        // 0 × 5 = 0
        vec![1, 100, 200, 20000], // 100 × 200 = 20000
    ];
    
    for witness in valid_witnesses {
        if !r1cs.is_satisfied(&witness) {
            continue; // Skip if witness doesn't satisfy (e.g., overflow)
        }
        
        let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x9999).unwrap();
        let public_inputs = r1cs.public_inputs(&witness);
        
        assert!(
            verify_r1cs(&proof, &public_inputs, &r1cs),
            "Valid witness {:?} should verify", witness
        );
    }
}

#[test]
fn test_verify_r1cs_deterministic_verification() {
    // Same proof should always verify or reject deterministically
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let proof = prove_r1cs(&r1cs, &witness, &ctx, 0xAAAA).unwrap();
    let public_inputs = r1cs.public_inputs(&witness);
    
    // Verify multiple times
    for _ in 0..5 {
        assert!(verify_r1cs(&proof, &public_inputs, &r1cs));
    }
}

#[test]
fn test_verify_r1cs_all_evaluations_checked() {
    // Verify that all 8 evaluations (Q, A_z, B_z, C_z at α and β) matter
    let r1cs = create_multiplication_circuit();
    let ctx = setup_ctx().unwrap();
    
    let witness = vec![1, 7, 13, 91];
    let public_inputs = r1cs.public_inputs(&witness);
    
    let fields_to_forge = vec![
        "q_alpha", "q_beta",
        "a_z_alpha", "a_z_beta",
        "b_z_alpha", "b_z_beta",
        "c_z_alpha", "c_z_beta",
    ];
    
    for field in fields_to_forge {
        let mut proof = prove_r1cs(&r1cs, &witness, &ctx, 0xBBBB).unwrap();
        
        // Forge one field
        match field {
            "q_alpha" => proof.q_alpha = (proof.q_alpha + 1) % r1cs.modulus,
            "q_beta" => proof.q_beta = (proof.q_beta + 1) % r1cs.modulus,
            "a_z_alpha" => proof.a_z_alpha = (proof.a_z_alpha + 1) % r1cs.modulus,
            "a_z_beta" => proof.a_z_beta = (proof.a_z_beta + 1) % r1cs.modulus,
            "b_z_alpha" => proof.b_z_alpha = (proof.b_z_alpha + 1) % r1cs.modulus,
            "b_z_beta" => proof.b_z_beta = (proof.b_z_beta + 1) % r1cs.modulus,
            "c_z_alpha" => proof.c_z_alpha = (proof.c_z_alpha + 1) % r1cs.modulus,
            "c_z_beta" => proof.c_z_beta = (proof.c_z_beta + 1) % r1cs.modulus,
            _ => unreachable!(),
        }
        
        assert!(
            !verify_r1cs(&proof, &public_inputs, &r1cs),
            "Forging {} should fail verification", field
        );
    }
}
