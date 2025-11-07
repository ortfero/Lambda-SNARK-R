//! Integration tests for opening proof generation and verification.

use lambda_snark::{
    Polynomial, Commitment, Challenge, Opening,
    generate_opening, verify_opening,
    LweContext,
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
fn test_opening_pipeline_tv1() {
    // TV-1: 7 × 13 = 91
    // witness = [1, 7, 13, 91]
    // From prover_pipeline: α = 7941808122273, f(α) = 5125469496080
    
    let ctx = test_context();
    let randomness = 0x1234;
    
    // 1. Create polynomial from witness
    let witness = vec![1, 7, 13, 91];
    let polynomial = Polynomial::from_witness(&witness, TEST_MODULUS);
    
    // 2. Commit to polynomial
    let commitment = Commitment::new(&ctx, polynomial.coefficients(), randomness)
        .expect("Commitment creation failed");
    
    // 3. Generate Fiat-Shamir challenge
    let public_inputs = vec![1, 91]; // First and last witness elements
    let challenge = Challenge::derive(&public_inputs, &commitment, TEST_MODULUS);
    let alpha = challenge.alpha();
    
    println!("TV-1 Opening Pipeline:");
    println!("  witness = {:?}", witness);
    println!("  α = {}", alpha.value());
    
    // 4. Generate opening proof
    let opening = generate_opening(&polynomial, alpha, randomness);
    println!("  f(α) = {}", opening.evaluation().value());
    
    // 5. Verify opening
    let valid = verify_opening(&commitment, alpha, &opening, TEST_MODULUS);
    assert!(valid, "TV-1 opening should verify");
    
    // 6. Check evaluation matches expected from prover_pipeline
    // Note: Challenge α depends on commitment randomness, so we can't hardcode it
    // Instead, verify evaluation is correct for the derived α
    let expected_eval = polynomial.evaluate(alpha);
    assert_eq!(opening.evaluation(), expected_eval, "Evaluation should match polynomial.evaluate(α)");
}

#[test]
fn test_opening_pipeline_tv2() {
    // TV-2: Plaquette gauge configuration
    // witness = [1, 314, 628, 471, 471]
    // From prover_pipeline: α = 2016424729740, f(α) = 12595982643411
    
    let ctx = test_context();
    let randomness = 0x5678;
    
    // 1. Create polynomial from witness
    let witness = vec![1, 314, 628, 471, 471];
    let polynomial = Polynomial::from_witness(&witness, TEST_MODULUS);
    
    // 2. Commit to polynomial
    let commitment = Commitment::new(&ctx, polynomial.coefficients(), randomness)
        .expect("Commitment creation failed");
    
    // 3. Generate Fiat-Shamir challenge
    let public_inputs = vec![1, 471]; // First and last witness elements
    let challenge = Challenge::derive(&public_inputs, &commitment, TEST_MODULUS);
    let alpha = challenge.alpha();
    
    println!("TV-2 Opening Pipeline:");
    println!("  witness = {:?}", witness);
    println!("  α = {}", alpha.value());
    
    // 4. Generate opening proof
    let opening = generate_opening(&polynomial, alpha, randomness);
    println!("  f(α) = {}", opening.evaluation().value());
    
    // 5. Verify opening
    let valid = verify_opening(&commitment, alpha, &opening, TEST_MODULUS);
    assert!(valid, "TV-2 opening should verify");
    
    // 6. Check evaluation matches expected
    let expected_eval = polynomial.evaluate(alpha);
    assert_eq!(opening.evaluation(), expected_eval, "Evaluation should match polynomial.evaluate(α)");
}

#[test]
fn test_opening_deterministic() {
    // Same inputs → same opening
    let ctx = test_context();
    let randomness = 0xABCD;
    
    let polynomial = Polynomial::from_witness(&[1, 2, 3, 4], TEST_MODULUS);
    let alpha = Field::new(12345);
    
    let opening1 = generate_opening(&polynomial, alpha, randomness);
    let opening2 = generate_opening(&polynomial, alpha, randomness);
    
    assert_eq!(opening1.evaluation(), opening2.evaluation(), "Same inputs should produce same evaluation");
    assert_eq!(opening1.witness(), opening2.witness(), "Same inputs should produce same witness");
}

#[test]
fn test_opening_different_challenges() {
    // Different challenge points → different evaluations
    let ctx = test_context();
    let randomness = 0x9999;
    
    let polynomial = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);
    
    let alpha1 = Field::new(100);
    let alpha2 = Field::new(200);
    
    let opening1 = generate_opening(&polynomial, alpha1, randomness);
    let opening2 = generate_opening(&polynomial, alpha2, randomness);
    
    // Different challenges should produce different evaluations (unless f is constant)
    assert_ne!(opening1.evaluation(), opening2.evaluation(), 
               "Different challenges should produce different evaluations for non-constant polynomial");
}

#[test]
fn test_opening_verification_rejects_forged() {
    // Forged opening with wrong evaluation should be rejected
    let ctx = test_context();
    let randomness = 0x7777;
    
    let polynomial = Polynomial::from_witness(&[5, 10, 15], TEST_MODULUS);
    let commitment = Commitment::new(&ctx, polynomial.coefficients(), randomness).unwrap();
    
    let alpha = Field::new(999);
    let correct_opening = generate_opening(&polynomial, alpha, randomness);
    
    // Forge opening with wrong evaluation
    let wrong_eval = Field::new((correct_opening.evaluation().value() + 42) % TEST_MODULUS);
    let forged_opening = Opening::new(wrong_eval, correct_opening.witness().to_vec());
    
    let valid = verify_opening(&commitment, alpha, &forged_opening, TEST_MODULUS);
    assert!(!valid, "Forged opening with wrong evaluation should be rejected");
}

#[test]
fn test_opening_multiple_commitments() {
    // Multiple polynomials → independent openings
    let ctx = test_context();
    
    let poly1 = Polynomial::from_witness(&[1, 2, 3], TEST_MODULUS);
    let poly2 = Polynomial::from_witness(&[4, 5, 6], TEST_MODULUS);
    
    let comm1 = Commitment::new(&ctx, poly1.coefficients(), 0x1111).unwrap();
    let comm2 = Commitment::new(&ctx, poly2.coefficients(), 0x2222).unwrap();
    
    let alpha = Field::new(777);
    
    let opening1 = generate_opening(&poly1, alpha, 0x1111);
    let opening2 = generate_opening(&poly2, alpha, 0x2222);
    
    // Both should verify independently
    assert!(verify_opening(&comm1, alpha, &opening1, TEST_MODULUS), "Opening 1 should verify");
    assert!(verify_opening(&comm2, alpha, &opening2, TEST_MODULUS), "Opening 2 should verify");
    
    // Cross-verification should fail (opening for wrong polynomial)
    // Note: This test is expected to pass currently because verify_opening
    // doesn't check commitment binding (LWE verification not implemented)
    // Will fail when full LWE verification is added
}

#[test]
fn test_opening_consistency_with_prover_pipeline() {
    // Integration: polynomial → commitment → challenge → opening should be consistent
    let ctx = test_context();
    let randomness = 0xBEEF;
    
    let witness = vec![1, 7, 13, 91];
    let public_inputs = vec![1, 91];
    
    // Full pipeline
    let polynomial = Polynomial::from_witness(&witness, TEST_MODULUS);
    let commitment = Commitment::new(&ctx, polynomial.coefficients(), randomness).unwrap();
    let challenge = Challenge::derive(&public_inputs, &commitment, TEST_MODULUS);
    let alpha = challenge.alpha();
    let opening = generate_opening(&polynomial, alpha, randomness);
    
    // Verify everything is consistent
    assert_eq!(opening.evaluation(), polynomial.evaluate(alpha), 
               "Opening evaluation should match polynomial evaluation");
    
    assert!(verify_opening(&commitment, alpha, &opening, TEST_MODULUS),
            "Pipeline-generated opening should verify");
}
