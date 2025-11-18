//! Integration tests for polynomial encoding and commitment.

use lambda_snark::{Commitment, Field, LweContext, Params, Polynomial, Profile, SecurityLevel};

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
fn test_witness_to_polynomial() {
    // TV-1: witness = [1, 7, 13, 91]
    let witness = vec![1, 7, 13, 91];
    let p = Polynomial::from_witness(&witness, TEST_MODULUS);

    // Verify degree
    assert_eq!(p.degree(), 3, "Expected degree 3 for 4-element witness");

    // Verify coefficients
    assert_eq!(p.coeff(0).unwrap().value(), 1);
    assert_eq!(p.coeff(1).unwrap().value(), 7);
    assert_eq!(p.coeff(2).unwrap().value(), 13);
    assert_eq!(p.coeff(3).unwrap().value(), 91);
}

#[test]
fn test_polynomial_evaluation() {
    // TV-1: f(X) = 1 + 7X + 13X² + 91X³
    let p = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);

    // f(0) = 1
    assert_eq!(p.evaluate(Field::new(0)).value(), 1);

    // f(1) = 1 + 7 + 13 + 91 = 112
    assert_eq!(p.evaluate(Field::new(1)).value(), 112);

    // f(2) = 1 + 14 + 52 + 728 = 795
    assert_eq!(p.evaluate(Field::new(2)).value(), 795);
}

#[test]
fn test_commitment_to_polynomial() {
    let ctx = test_context();

    // TV-1: witness → polynomial
    let witness = vec![1, 7, 13, 91];
    let p = Polynomial::from_witness(&witness, TEST_MODULUS);

    // Commit to polynomial coefficients
    let seed = 0x5EED_2024;
    let comm = Commitment::new(&ctx, p.coefficients(), seed);

    assert!(comm.is_ok(), "Commitment should succeed");
    let comm = comm.unwrap();

    // Commitment data should be non-empty
    let data = comm.as_bytes();
    assert!(!data.is_empty(), "Commitment data should not be empty");
}

#[test]
fn test_commitment_reproducible_structure() {
    // Commitment structure should be consistent (size, format)
    let ctx = test_context();

    let p = Polynomial::from_witness(&[1, 2, 3, 4], TEST_MODULUS);

    let seed = 42;
    let comm1 = Commitment::new(&ctx, p.coefficients(), seed).unwrap();
    let comm2 = Commitment::new(&ctx, p.coefficients(), seed).unwrap();

    // Note: LWE commitments use randomness for security (IND-CPA)
    // So exact bytes will differ even with same seed.
    // But structure (size) should be consistent.
    assert_eq!(
        comm1.as_bytes().len(),
        comm2.as_bytes().len(),
        "Same input should produce same structure size"
    );
}

#[test]
fn test_commitment_distinct() {
    let ctx = test_context();

    // Different polynomials → different commitments
    let p1 = Polynomial::from_witness(&[1, 2, 3], TEST_MODULUS);
    let p2 = Polynomial::from_witness(&[1, 2, 4], TEST_MODULUS);

    let seed = 999;
    let comm1 = Commitment::new(&ctx, p1.coefficients(), seed).unwrap();
    let comm2 = Commitment::new(&ctx, p2.coefficients(), seed).unwrap();

    // Commitments should differ (binding property)
    assert_ne!(
        comm1.as_bytes(),
        comm2.as_bytes(),
        "Different polynomials should produce different commitments"
    );
}

#[test]
fn test_tv1_polynomial_commitment() {
    // Full pipeline: TV-1 witness → polynomial → commitment
    let ctx = test_context();

    // TV-1: 7 × 13 = 91
    let witness = vec![1, 7, 13, 91];

    // Encode
    let p = Polynomial::from_witness(&witness, TEST_MODULUS);

    // Verify polynomial encodes witness correctly
    for (i, &expected) in witness.iter().enumerate() {
        assert_eq!(
            p.coeff(i).unwrap().value(),
            expected,
            "Coefficient mismatch at index {}",
            i
        );
    }

    // Commit
    let comm = Commitment::new(&ctx, p.coefficients(), 0xDEADBEEF).unwrap();

    // Basic sanity checks
    assert!(!comm.as_bytes().is_empty());
    println!(
        "TV-1 commitment size: {} u64 elements",
        comm.as_bytes().len()
    );
}

#[test]
fn test_tv2_polynomial_commitment() {
    // Full pipeline: TV-2 witness → polynomial → commitment
    let ctx = test_context();

    // TV-2: plaquette closure (θ₁ + θ₂ - θ₃ - θ₄ = 0)
    let witness = vec![1, 314, 628, 471, 471];

    // Encode
    let p = Polynomial::from_witness(&witness, TEST_MODULUS);

    // Verify
    assert_eq!(p.degree(), 4);
    assert_eq!(p.coeff(0).unwrap().value(), 1);
    assert_eq!(p.coeff(1).unwrap().value(), 314);

    // Commit
    let comm = Commitment::new(&ctx, p.coefficients(), 0xCAFEBABE).unwrap();

    assert!(!comm.as_bytes().is_empty());
    println!(
        "TV-2 commitment size: {} u64 elements",
        comm.as_bytes().len()
    );
}
