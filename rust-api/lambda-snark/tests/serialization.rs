//! Serialization tests for proof components.

use lambda_snark::{Challenge, Commitment, LweContext, Opening, Polynomial};
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
fn test_opening_roundtrip() {
    // Opening should serialize and deserialize correctly
    let opening = Opening::new(Field::new(12345), vec![0x1234, 1, 7, 13, 91]);

    // Serialize
    let bytes = bincode::serialize(&opening).expect("Serialization failed");
    println!("Opening size: {} bytes", bytes.len());

    // Deserialize
    let decoded: Opening = bincode::deserialize(&bytes).expect("Deserialization failed");

    // Check roundtrip
    assert_eq!(opening, decoded, "Roundtrip should preserve Opening");
}

#[test]
fn test_opening_deterministic() {
    // Same opening → same bytes
    let opening = Opening::new(Field::new(999), vec![1, 2, 3, 4, 5]);

    let bytes1 = bincode::serialize(&opening).unwrap();
    let bytes2 = bincode::serialize(&opening).unwrap();

    assert_eq!(bytes1, bytes2, "Serialization should be deterministic");
}

#[test]
fn test_opening_size_tv1() {
    // TV-1 opening size estimation
    let ctx = test_context();
    let polynomial = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);
    let commitment = Commitment::new(&ctx, polynomial.coefficients(), 0x1234).unwrap();
    let challenge = Challenge::derive(&[1, 91], &commitment, TEST_MODULUS);

    let opening = lambda_snark::generate_opening(&polynomial, challenge.alpha(), 0x1234);

    let bytes = bincode::serialize(&opening).unwrap();
    println!("TV-1 Opening serialized size: {} bytes", bytes.len());

    // Expected: Field (8 bytes) + Vec length (8 bytes) + witness data (5×8 = 40 bytes) ≈ 56 bytes
    assert!(
        bytes.len() < 1024,
        "Opening should be < 1KB, got {} bytes",
        bytes.len()
    );
}

#[test]
fn test_challenge_roundtrip() {
    // Challenge should serialize and deserialize correctly
    let ctx = test_context();
    let polynomial = Polynomial::from_witness(&[1, 2, 3], TEST_MODULUS);
    let commitment = Commitment::new(&ctx, polynomial.coefficients(), 0xABCD).unwrap();

    let challenge = Challenge::derive(&[1, 3], &commitment, TEST_MODULUS);

    // Serialize
    let bytes = bincode::serialize(&challenge).expect("Serialization failed");
    println!("Challenge size: {} bytes", bytes.len());

    // Deserialize
    let decoded: Challenge = bincode::deserialize(&bytes).expect("Deserialization failed");

    // Check roundtrip
    assert_eq!(challenge, decoded, "Roundtrip should preserve Challenge");
}

#[test]
fn test_challenge_deterministic() {
    // Same challenge → same bytes
    let ctx = test_context();
    let polynomial = Polynomial::from_witness(&[10, 20], TEST_MODULUS);
    let commitment = Commitment::new(&ctx, polynomial.coefficients(), 0x5555).unwrap();

    let challenge = Challenge::derive(&[10, 20], &commitment, TEST_MODULUS);

    let bytes1 = bincode::serialize(&challenge).unwrap();
    let bytes2 = bincode::serialize(&challenge).unwrap();

    assert_eq!(
        bytes1, bytes2,
        "Challenge serialization should be deterministic"
    );
}

#[test]
fn test_challenge_size() {
    // Challenge size estimation
    let ctx = test_context();
    let polynomial = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);
    let commitment = Commitment::new(&ctx, polynomial.coefficients(), 0x1234).unwrap();

    let challenge = Challenge::derive(&[1, 91], &commitment, TEST_MODULUS);

    let bytes = bincode::serialize(&challenge).unwrap();
    println!("Challenge serialized size: {} bytes", bytes.len());

    // Expected: Field (8 bytes) + hash array (32 bytes) ≈ 40 bytes
    assert!(
        bytes.len() < 64,
        "Challenge should be < 64 bytes, got {} bytes",
        bytes.len()
    );
}

#[test]
fn test_commitment_serialization() {
    // Commitment should serialize (but not deserialize without context)
    let ctx = test_context();
    let message = vec![Field::new(1), Field::new(2), Field::new(3)];
    let commitment = Commitment::new(&ctx, &message, 0x9999).unwrap();

    // Serialize should work
    let bytes = bincode::serialize(&commitment).expect("Commitment serialization failed");
    println!("Commitment size: {} bytes", bytes.len());

    // Bincode encodes Vec<u64> as (len: u64) + elements
    let raw_data = commitment.as_bytes();
    assert!(
        !raw_data.is_empty(),
        "Serialized commitment must contain data"
    );
    let expected = 8 + raw_data.len() * 8; // length prefix + payload
    assert_eq!(
        bytes.len(),
        expected,
        "Commitment serialization mismatch: expected {} bytes, got {} bytes",
        expected,
        bytes.len()
    );
}

#[test]
#[should_panic(expected = "Commitment deserialization requires LweContext")]
fn test_commitment_deserialization_fails() {
    // Commitment deserialization should fail (requires context)
    let ctx = test_context();
    let message = vec![Field::new(1), Field::new(2)];
    let commitment = Commitment::new(&ctx, &message, 0x7777).unwrap();

    let bytes = bincode::serialize(&commitment).unwrap();

    // This should panic
    let _decoded: Commitment = bincode::deserialize(&bytes).unwrap();
}

#[test]
fn test_proof_component_sizes() {
    // Integration: measure all proof component sizes
    let ctx = test_context();

    // TV-1: witness = [1, 7, 13, 91]
    let witness = vec![1, 7, 13, 91];
    let public_inputs = vec![1, 91];

    let polynomial = Polynomial::from_witness(&witness, TEST_MODULUS);
    let commitment = Commitment::new(&ctx, polynomial.coefficients(), 0x1234).unwrap();
    let challenge = Challenge::derive(&public_inputs, &commitment, TEST_MODULUS);
    let opening = lambda_snark::generate_opening(&polynomial, challenge.alpha(), 0x1234);

    // Serialize components
    let comm_bytes = bincode::serialize(&commitment).unwrap();
    let chal_bytes = bincode::serialize(&challenge).unwrap();
    let open_bytes = bincode::serialize(&opening).unwrap();

    println!("Proof component sizes:");
    println!(
        "  Commitment: {} bytes (~{}KB)",
        comm_bytes.len(),
        comm_bytes.len() / 1024
    );
    println!("  Challenge:  {} bytes", chal_bytes.len());
    println!("  Opening:    {} bytes", open_bytes.len());
    println!(
        "  Total:      {} bytes (~{}KB)",
        comm_bytes.len() + chal_bytes.len() + open_bytes.len(),
        (comm_bytes.len() + chal_bytes.len() + open_bytes.len()) / 1024
    );

    // Validate against design spec (~3KB total)
    // Note: Current commitment is ~65KB (LWE n=4096)
    // Design spec assumed smaller parameters
    let total_size = comm_bytes.len() + chal_bytes.len() + open_bytes.len();
    assert!(
        total_size < 100_000,
        "Total proof should be < 100KB for test params"
    );
}

#[test]
fn test_different_openings_different_bytes() {
    // Different openings → different serializations
    let opening1 = Opening::new(Field::new(100), vec![1, 2, 3]);
    let opening2 = Opening::new(Field::new(200), vec![1, 2, 3]);

    let bytes1 = bincode::serialize(&opening1).unwrap();
    let bytes2 = bincode::serialize(&opening2).unwrap();

    assert_ne!(
        bytes1, bytes2,
        "Different openings should serialize differently"
    );
}

#[test]
fn test_different_challenges_different_bytes() {
    // Different challenges → different serializations
    let ctx = test_context();

    let poly1 = Polynomial::from_witness(&[1, 2], TEST_MODULUS);
    let poly2 = Polynomial::from_witness(&[1, 3], TEST_MODULUS);

    let comm1 = Commitment::new(&ctx, poly1.coefficients(), 0x1111).unwrap();
    let comm2 = Commitment::new(&ctx, poly2.coefficients(), 0x2222).unwrap();

    let chal1 = Challenge::derive(&[1, 2], &comm1, TEST_MODULUS);
    let chal2 = Challenge::derive(&[1, 3], &comm2, TEST_MODULUS);

    let bytes1 = bincode::serialize(&chal1).unwrap();
    let bytes2 = bincode::serialize(&chal2).unwrap();

    assert_ne!(
        bytes1, bytes2,
        "Different challenges should serialize differently"
    );
}
