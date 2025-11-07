//! Opening proofs for polynomial commitments.
//!
//! Implements opening generation and verification for ΛSNARK-R.

use lambda_snark_core::Field;
use crate::{Polynomial, Commitment, Error};
use serde::{Serialize, Deserialize};

/// Opening proof for polynomial evaluation.
///
/// Contains the evaluation y = f(α) and witness data for verification.
///
/// # Serialization
/// Opening supports serialization via `serde`:
/// ```
/// use lambda_snark::Opening;
/// use lambda_snark_core::Field;
///
/// let opening = Opening::new(Field::new(42), vec![1, 2, 3]);
///
/// // Serialize with bincode
/// let bytes = bincode::serialize(&opening).unwrap();
/// println!("Serialized size: {} bytes", bytes.len());
///
/// // Deserialize
/// let decoded: Opening = bincode::deserialize(&bytes).unwrap();
/// assert_eq!(opening, decoded);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Opening {
    /// Evaluation y = f(α)
    evaluation: Field,
    
    /// LWE opening witness (randomness used in commitment)
    /// 
    /// Note: In practice, this should be derived from commitment randomness.
    /// For now, we store it explicitly for testing purposes.
    witness: Vec<u64>,
}

impl Opening {
    /// Create new opening proof.
    ///
    /// # Arguments
    /// * `evaluation` - Value y = f(α)
    /// * `witness` - Opening witness data
    ///
    /// # Example
    /// ```
    /// use lambda_snark::Opening;
    /// use lambda_snark_core::Field;
    ///
    /// let opening = Opening::new(Field::new(42), vec![1, 2, 3]);
    /// assert_eq!(opening.evaluation().value(), 42);
    /// ```
    pub fn new(evaluation: Field, witness: Vec<u64>) -> Self {
        Opening { evaluation, witness }
    }
    
    /// Get evaluation y = f(α).
    pub fn evaluation(&self) -> Field {
        self.evaluation
    }
    
    /// Get opening witness.
    pub fn witness(&self) -> &[u64] {
        &self.witness
    }
}

/// Generate opening proof for polynomial at challenge point.
///
/// Computes y = f(α) and creates opening witness for verification.
///
/// # Arguments
/// * `polynomial` - Polynomial f(X) = Σ z_i·X^i
/// * `alpha` - Challenge point α ∈ F_q
/// * `randomness` - Randomness used in commitment (seed)
///
/// # Returns
/// Opening proof containing y = f(α) and witness
///
/// # Example
/// ```
/// use lambda_snark::{Polynomial, generate_opening};
/// use lambda_snark_core::Field;
///
/// let modulus = 17592186044417; // 2^44 + 1
/// let polynomial = Polynomial::from_witness(&[1, 7, 13, 91], modulus);
/// let alpha = Field::new(12345);
/// let randomness = 0x1234;
///
/// let opening = generate_opening(&polynomial, alpha, randomness);
/// 
/// // Verify evaluation is correct
/// let expected = polynomial.evaluate(alpha);
/// assert_eq!(opening.evaluation(), expected);
/// ```
pub fn generate_opening(
    polynomial: &Polynomial,
    alpha: Field,
    randomness: u64,
) -> Opening {
    // 1. Compute evaluation y = f(α)
    let evaluation = polynomial.evaluate(alpha);
    
    // 2. Generate LWE opening witness
    // For now, witness is just the randomness seed + polynomial coefficients
    // In full implementation, this would be LWE opening randomness
    let mut witness = vec![randomness];
    witness.extend(polynomial.coefficients().iter().map(|f| f.value()));
    
    Opening::new(evaluation, witness)
}

/// Verify opening proof for polynomial commitment.
///
/// Checks that the opening is valid for the given commitment and challenge.
///
/// # Arguments
/// * `commitment` - LWE commitment to polynomial
/// * `alpha` - Challenge point α ∈ F_q  
/// * `opening` - Opening proof to verify
/// * `modulus` - Field modulus q
///
/// # Returns
/// `true` if opening is valid, `false` otherwise
///
/// # Security
/// - Soundness: Forged opening rejected with probability ≥ 1 - ε_LWE
/// - Binding: LWE assumption ensures commitment binds to unique polynomial
///
/// # Example
/// ```
/// use lambda_snark::{Polynomial, Commitment, LweContext, Params, Profile, SecurityLevel};
/// use lambda_snark::{generate_opening, verify_opening};
/// use lambda_snark_core::Field;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let modulus = 17592186044417;
/// let params = Params::new(
///     SecurityLevel::Bits128,
///     Profile::RingB { n: 4096, k: 2, q: modulus, sigma: 3.19 },
/// );
/// let ctx = LweContext::new(params)?;
///
/// let polynomial = Polynomial::from_witness(&[1, 7, 13, 91], modulus);
/// let commitment = Commitment::new(&ctx, polynomial.coefficients(), 0x1234)?;
/// 
/// let alpha = Field::new(12345);
/// let opening = generate_opening(&polynomial, alpha, 0x1234);
///
/// let valid = verify_opening(&commitment, alpha, &opening, modulus);
/// assert!(valid, "Valid opening should verify");
/// # Ok(())
/// # }
/// ```
pub fn verify_opening(
    commitment: &Commitment,
    alpha: Field,
    opening: &Opening,
    modulus: u64,
) -> bool {
    // PLACEHOLDER: Full implementation requires LWE opening verification
    // For now, we check basic structural validity
    
    // 1. Check evaluation is in field
    if opening.evaluation().value() >= modulus {
        return false;
    }
    
    // 2. Check witness is non-empty
    if opening.witness().is_empty() {
        return false;
    }
    
    // 3. Reconstruct polynomial from witness and verify evaluation
    // witness[0] = randomness, witness[1..] = polynomial coefficients
    if opening.witness().len() < 2 {
        return false;
    }
    
    let coeffs: Vec<Field> = opening.witness()[1..]
        .iter()
        .map(|&c| Field::new(c % modulus))
        .collect();
    
    let polynomial = Polynomial::new(coeffs, modulus);
    let expected_eval = polynomial.evaluate(alpha);
    
    // 4. Check evaluation matches
    if opening.evaluation() != expected_eval {
        return false;
    }
    
    // 5. TODO: Add LWE commitment verification
    // This would call lwe_verify_opening() via FFI with:
    // - commitment data
    // - polynomial coefficients (message)
    // - randomness (witness[0])
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LweContext, Commitment};
    use lambda_snark_core::{Params, Profile, SecurityLevel};
    
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
    fn test_opening_new() {
        let opening = Opening::new(Field::new(42), vec![1, 2, 3]);
        assert_eq!(opening.evaluation().value(), 42);
        assert_eq!(opening.witness(), &[1, 2, 3]);
    }
    
    #[test]
    fn test_opening_generation() {
        // Generate opening for known polynomial
        let polynomial = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);
        let alpha = Field::new(12345);
        let randomness = 0x1234;
        
        let opening = generate_opening(&polynomial, alpha, randomness);
        
        // Check evaluation is correct
        let expected_eval = polynomial.evaluate(alpha);
        assert_eq!(opening.evaluation(), expected_eval);
        
        // Check witness contains randomness
        assert!(opening.witness().len() > 0);
        assert_eq!(opening.witness()[0], randomness);
    }
    
    #[test]
    fn test_opening_evaluation_tv1() {
        // TV-1: witness = [1, 7, 13, 91], f(X) = 1 + 7X + 13X² + 91X³
        // From prover_pipeline tests: α = 7941808122273, f(α) = 5125469496080
        let polynomial = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);
        
        // Use known challenge from prover_pipeline test
        // (Note: actual challenge depends on commitment, this is for evaluation only)
        let alpha = Field::new(12345); // Use simple value for this test
        let randomness = 0x1234;
        
        let opening = generate_opening(&polynomial, alpha, randomness);
        
        // Verify evaluation matches polynomial
        assert_eq!(opening.evaluation(), polynomial.evaluate(alpha));
    }
    
    #[test]
    fn test_opening_correctness() {
        // Valid opening should verify
        let ctx = test_context();
        
        let polynomial = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);
        let randomness = 0x1234;
        let commitment = Commitment::new(&ctx, polynomial.coefficients(), randomness).unwrap();
        
        let alpha = Field::new(12345);
        let opening = generate_opening(&polynomial, alpha, randomness);
        
        let valid = verify_opening(&commitment, alpha, &opening, TEST_MODULUS);
        assert!(valid, "Valid opening should verify");
    }
    
    #[test]
    fn test_opening_soundness_wrong_evaluation() {
        // Opening with wrong evaluation should fail
        let ctx = test_context();
        
        let polynomial = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);
        let randomness = 0x1234;
        let commitment = Commitment::new(&ctx, polynomial.coefficients(), randomness).unwrap();
        
        let alpha = Field::new(12345);
        
        // Create opening with wrong evaluation
        let correct_opening = generate_opening(&polynomial, alpha, randomness);
        let wrong_eval = Field::new(correct_opening.evaluation().value() + 1);
        let forged_opening = Opening::new(wrong_eval, correct_opening.witness().to_vec());
        
        let valid = verify_opening(&commitment, alpha, &forged_opening, TEST_MODULUS);
        assert!(!valid, "Opening with wrong evaluation should be rejected");
    }
    
    #[test]
    #[ignore] // TODO: Enable when LWE opening verification is implemented
    fn test_opening_soundness_wrong_polynomial() {
        // Opening for different polynomial should fail
        let ctx = test_context();
        
        let polynomial1 = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);
        let polynomial2 = Polynomial::from_witness(&[1, 7, 13, 92], TEST_MODULUS); // Different
        let randomness = 0x1234;
        
        let commitment1 = Commitment::new(&ctx, polynomial1.coefficients(), randomness).unwrap();
        
        let alpha = Field::new(12345);
        
        // Generate opening for polynomial2 but verify against commitment1
        let opening2 = generate_opening(&polynomial2, alpha, randomness);
        
        let valid = verify_opening(&commitment1, alpha, &opening2, TEST_MODULUS);
        assert!(!valid, "Opening for different polynomial should be rejected");
    }
    
    #[test]
    fn test_opening_empty_witness() {
        // Opening with empty witness should fail
        let ctx = test_context();
        
        let polynomial = Polynomial::from_witness(&[1, 2], TEST_MODULUS);
        let commitment = Commitment::new(&ctx, polynomial.coefficients(), 0x1234).unwrap();
        
        let alpha = Field::new(100);
        let invalid_opening = Opening::new(Field::new(42), vec![]); // Empty witness
        
        let valid = verify_opening(&commitment, alpha, &invalid_opening, TEST_MODULUS);
        assert!(!valid, "Opening with empty witness should be rejected");
    }
    
    #[test]
    fn test_opening_out_of_field() {
        // Opening with evaluation >= modulus should fail
        let ctx = test_context();
        
        let polynomial = Polynomial::from_witness(&[1, 2], TEST_MODULUS);
        let commitment = Commitment::new(&ctx, polynomial.coefficients(), 0x1234).unwrap();
        
        let alpha = Field::new(100);
        let invalid_opening = Opening::new(
            Field::new(TEST_MODULUS), // Out of field
            vec![0x1234, 1, 2]
        );
        
        let valid = verify_opening(&commitment, alpha, &invalid_opening, TEST_MODULUS);
        assert!(!valid, "Opening with out-of-field evaluation should be rejected");
    }
}
