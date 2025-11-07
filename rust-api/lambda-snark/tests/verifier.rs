//! Integration tests for SNARK verifier.

use lambda_snark::{
    prove_simple, verify_simple, Proof, Challenge,
    LweContext, Polynomial, Commitment, Opening,
    generate_opening,
};
use lambda_snark_core::{Field, Params, Profile, SecurityLevel};

const TEST_MODULUS: u64 = 17592186044417; // 2^44 + 1

fn test_context() -> LweContext {
    let params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,
            k: 2,
            q: TEST_MODULUS,
            sigma: 3.19,
        },
    );
    LweContext::new(params).expect("Failed to create LWE context")
}

#[test]
fn test_verify_valid_proof() {
    // Valid proof should verify
    let ctx = test_context();
    
    let witness = vec![1, 7, 13, 91];
    let public_inputs = vec![1, 91];
    
    let proof = prove_simple(&witness, &public_inputs, &ctx, TEST_MODULUS, 0x1234)
        .expect("Proof generation failed");
    
    let valid = verify_simple(&proof, &public_inputs, TEST_MODULUS);
    assert!(valid, "Valid proof should verify");
}

#[test]
fn test_verify_tv1() {
    // TV-1: 7 × 13 = 91
    let ctx = test_context();
    
    let witness = vec![1, 7, 13, 91];
    let public_inputs = vec![1, 91];
    
    let proof = prove_simple(&witness, &public_inputs, &ctx, TEST_MODULUS, 0x5678)
        .expect("TV-1 proof failed");
    
    let valid = verify_simple(&proof, &public_inputs, TEST_MODULUS);
    assert!(valid, "TV-1 proof should verify");
}

#[test]
fn test_verify_tv2() {
    // TV-2: Plaquette gauge configuration
    let ctx = test_context();
    
    let witness = vec![1, 314, 628, 471, 471];
    let public_inputs = vec![1, 471];
    
    let proof = prove_simple(&witness, &public_inputs, &ctx, TEST_MODULUS, 0xABCD)
        .expect("TV-2 proof failed");
    
    let valid = verify_simple(&proof, &public_inputs, TEST_MODULUS);
    assert!(valid, "TV-2 proof should verify");
}

#[test]
fn test_verify_rejects_wrong_challenge() {
    // Proof with manipulated challenge should fail
    let ctx = test_context();
    
    let witness = vec![1, 2, 3, 4];
    let public_inputs = vec![1, 4];
    
    let proof = prove_simple(&witness, &public_inputs, &ctx, TEST_MODULUS, 0x1111)
        .expect("Proof generation failed");
    
    // Tamper with challenge by using different public inputs
    let wrong_public_inputs = vec![1, 5]; // Different
    
    let valid = verify_simple(&proof, &wrong_public_inputs, TEST_MODULUS);
    assert!(!valid, "Proof with wrong challenge should be rejected");
}

#[test]
fn test_verify_rejects_forged_opening() {
    // Proof with forged opening should fail
    let ctx = test_context();
    
    let witness = vec![1, 7, 13, 91];
    let public_inputs = vec![1, 91];
    
    let polynomial = Polynomial::from_witness(&witness, TEST_MODULUS);
    let commitment = Commitment::new(&ctx, polynomial.coefficients(), 0x1234)
        .expect("Commitment failed");
    
    let challenge = Challenge::derive(&public_inputs, &commitment, TEST_MODULUS);
    
    // Create opening with wrong evaluation
    let correct_opening = generate_opening(&polynomial, challenge.alpha(), 0x1234);
    let wrong_eval = Field::new(correct_opening.evaluation().value() + 1);
    let forged_opening = Opening::new(wrong_eval, correct_opening.witness().to_vec());
    
    let forged_proof = Proof::new(commitment, challenge, forged_opening);
    
    let valid = verify_simple(&forged_proof, &public_inputs, TEST_MODULUS);
    assert!(!valid, "Forged opening should be rejected");
}

#[test]
fn test_verify_deterministic() {
    // Same proof should verify consistently
    let ctx = test_context();
    
    let witness = vec![1, 2, 3, 4];
    let public_inputs = vec![1, 4];
    
    let proof = prove_simple(&witness, &public_inputs, &ctx, TEST_MODULUS, 0x7777)
        .expect("Proof generation failed");
    
    // Verify multiple times
    for i in 0..10 {
        let valid = verify_simple(&proof, &public_inputs, TEST_MODULUS);
        assert!(valid, "Verification should be deterministic (iteration {})", i);
    }
}

#[test]
fn test_verify_multiple_proofs() {
    // Different proofs should verify independently
    let ctx = test_context();
    
    let witness1 = vec![1, 2, 3, 4];
    let witness2 = vec![5, 6, 7, 8];
    let public_inputs1 = vec![1, 4];
    let public_inputs2 = vec![5, 8];
    
    let proof1 = prove_simple(&witness1, &public_inputs1, &ctx, TEST_MODULUS, 0x1111)
        .expect("Proof 1 failed");
    let proof2 = prove_simple(&witness2, &public_inputs2, &ctx, TEST_MODULUS, 0x2222)
        .expect("Proof 2 failed");
    
    let valid1 = verify_simple(&proof1, &public_inputs1, TEST_MODULUS);
    let valid2 = verify_simple(&proof2, &public_inputs2, TEST_MODULUS);
    
    assert!(valid1, "Proof 1 should verify");
    assert!(valid2, "Proof 2 should verify");
    
    // Cross-verification should fail
    let cross_valid1 = verify_simple(&proof1, &public_inputs2, TEST_MODULUS);
    let cross_valid2 = verify_simple(&proof2, &public_inputs1, TEST_MODULUS);
    
    assert!(!cross_valid1, "Proof 1 with wrong public inputs should fail");
    assert!(!cross_valid2, "Proof 2 with wrong public inputs should fail");
}

#[test]
fn test_verify_completeness() {
    // Completeness: all valid proofs should verify
    let ctx = test_context();
    
    let test_cases = vec![
        (vec![1], vec![1]),
        (vec![1, 2], vec![1, 2]),
        (vec![1, 7, 13, 91], vec![1, 91]),
        (vec![1, 314, 628, 471, 471], vec![1, 471]),
        (vec![1, 2, 3, 4, 5, 6, 7, 8], vec![1, 8]),
    ];
    
    for (witness, public_inputs) in test_cases {
        let proof = prove_simple(&witness, &public_inputs, &ctx, TEST_MODULUS, 0xBEEF)
            .expect(&format!("Proof failed for witness {:?}", witness));
        
        let valid = verify_simple(&proof, &public_inputs, TEST_MODULUS);
        assert!(valid, "Valid proof for witness {:?} should verify", witness);
    }
}

#[test]
fn test_verify_soundness_challenge_binding() {
    // Soundness: proof for different public inputs should fail
    let ctx = test_context();
    
    let witness = vec![1, 7, 13, 91];
    let public_inputs1 = vec![1, 91];
    let public_inputs2 = vec![1, 92]; // Different
    
    let proof = prove_simple(&witness, &public_inputs1, &ctx, TEST_MODULUS, 0xCAFE)
        .expect("Proof generation failed");
    
    let valid1 = verify_simple(&proof, &public_inputs1, TEST_MODULUS);
    let valid2 = verify_simple(&proof, &public_inputs2, TEST_MODULUS);
    
    assert!(valid1, "Proof should verify with correct public inputs");
    assert!(!valid2, "Proof should NOT verify with different public inputs");
}

#[test]
fn test_verify_performance() {
    // Benchmark verification time
    let ctx = test_context();
    
    let witness = vec![1, 7, 13, 91];
    let public_inputs = vec![1, 91];
    
    let proof = prove_simple(&witness, &public_inputs, &ctx, TEST_MODULUS, 0xDEAD)
        .expect("Proof generation failed");
    
    let runs = 10;
    let mut times = Vec::new();
    
    for _ in 0..runs {
        let start = std::time::Instant::now();
        let valid = verify_simple(&proof, &public_inputs, TEST_MODULUS);
        let elapsed = start.elapsed();
        
        assert!(valid, "Proof should verify");
        times.push(elapsed);
    }
    
    let avg = times.iter().sum::<std::time::Duration>() / runs as u32;
    let min = times.iter().min().unwrap();
    let max = times.iter().max().unwrap();
    
    println!("\nVerification Performance ({} runs):", runs);
    println!("  Average: {:?}", avg);
    println!("  Min:     {:?}", min);
    println!("  Max:     {:?}", max);
    
    // Verification should be fast (<10ms)
    assert!(avg.as_millis() < 10, "Verification should be <10ms, got {:?}", avg);
}
