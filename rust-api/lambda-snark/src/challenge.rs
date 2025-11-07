//! Fiat-Shamir challenge generation for ΛSNARK-R.
//!
//! Implements non-interactive challenge derivation using SHA3-256 hash function.

use lambda_snark_core::Field;
use crate::Commitment;
use sha3::{Sha3_256, Digest};
use serde::{Serialize, Deserialize};

/// Fiat-Shamir challenge point.
///
/// # Serialization
/// Challenge supports serialization via `serde`:
/// ```
/// use lambda_snark::{Challenge, Polynomial, Commitment, LweContext, Params, Profile, SecurityLevel};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let params = Params::new(
///     SecurityLevel::Bits128,
///     Profile::RingB { n: 4096, k: 2, q: 17592186044417, sigma: 3.19 },
/// );
/// let ctx = LweContext::new(params)?;
/// let polynomial = Polynomial::from_witness(&[1, 2, 3], 17592186044417);
/// let commitment = Commitment::new(&ctx, polynomial.coefficients(), 0x1234)?;
///
/// let challenge = Challenge::derive(&[1, 3], &commitment, 17592186044417);
///
/// // Serialize with bincode
/// let bytes = bincode::serialize(&challenge).unwrap();
/// println!("Challenge size: {} bytes", bytes.len());
///
/// // Deserialize
/// let decoded: Challenge = bincode::deserialize(&bytes).unwrap();
/// assert_eq!(challenge, decoded);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Challenge {
    /// Challenge point α ∈ F_q
    alpha: Field,
    
    /// SHA3-256 hash used for derivation
    hash: [u8; 32],
}

impl Challenge {
    /// Get challenge point α.
    pub fn alpha(&self) -> Field {
        self.alpha
    }
    
    /// Get SHA3-256 hash.
    pub fn hash(&self) -> &[u8; 32] {
        &self.hash
    }
    
    /// Derive Fiat-Shamir challenge from public inputs and commitment.
    ///
    /// Implements non-interactive challenge generation:
    /// ```text
    /// τ = "LAMBDA-SNARK-R-FS-v1" || len(public_inputs) || public_inputs || commitment
    /// h = SHA3-256(τ)
    /// α = h[0..8] mod q
    /// ```
    ///
    /// # Security
    /// - SHA3-256 collision resistance → challenge unpredictable
    /// - Domain separation prevents cross-protocol attacks
    /// - Uniform distribution over F_q (negligible bias for 64-bit → 44-bit)
    ///
    /// # Arguments
    /// * `public_inputs` - Public witness elements
    /// * `commitment` - LWE commitment to witness polynomial
    /// * `modulus` - Field modulus q
    ///
    /// # Returns
    /// Challenge with α ∈ F_q and derivation hash
    ///
    /// # Example
    /// ```
    /// use lambda_snark::{Challenge, Commitment, LweContext, Params, Profile, SecurityLevel};
    /// use lambda_snark_core::Field;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let params = Params::new(
    ///     SecurityLevel::Bits128,
    ///     Profile::RingB { n: 4096, k: 2, q: 17592186044417, sigma: 3.19 },
    /// );
    /// let ctx = LweContext::new(params)?;
    /// 
    /// let message = vec![Field::new(1), Field::new(2)];
    /// let comm = Commitment::new(&ctx, &message, 42)?;
    /// 
    /// let public_inputs = vec![1, 91];
    /// let challenge = Challenge::derive(&public_inputs, &comm, 17592186044417);
    /// 
    /// println!("Challenge α = {}", challenge.alpha().value());
    /// # Ok(())
    /// # }
    /// ```
    pub fn derive(public_inputs: &[u64], commitment: &Commitment, modulus: u64) -> Self {
        // 1. Construct transcript
        let mut hasher = Sha3_256::new();
        
        // Domain separation
        hasher.update(b"LAMBDA-SNARK-R-FS-v1");
        
        // Public inputs length
        hasher.update(&(public_inputs.len() as u64).to_le_bytes());
        
        // Public inputs
        for &inp in public_inputs {
            hasher.update(&inp.to_le_bytes());
        }
        
        // Commitment data
        let comm_bytes = commitment.as_bytes();
        hasher.update(&(comm_bytes.len() as u64).to_le_bytes());
        for &word in comm_bytes {
            hasher.update(&word.to_le_bytes());
        }
        
        // 2. Hash
        let hash: [u8; 32] = hasher.finalize().into();
        
        // 3. Reduce to field element
        // Take first 8 bytes as little-endian u64, then reduce modulo q
        let alpha_bytes = &hash[0..8];
        let alpha_raw = u64::from_le_bytes(alpha_bytes.try_into().unwrap());
        let alpha = Field::new(alpha_raw % modulus);
        
        Challenge { alpha, hash }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LweContext, Polynomial};
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
    fn test_challenge_deterministic() {
        // Same inputs → same challenge
        let ctx = test_context();
        
        let p = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);
        let comm = Commitment::new(&ctx, p.coefficients(), 0x1234).unwrap();
        
        let public_inputs = vec![1, 91];
        
        let ch1 = Challenge::derive(&public_inputs, &comm, TEST_MODULUS);
        let ch2 = Challenge::derive(&public_inputs, &comm, TEST_MODULUS);
        
        assert_eq!(ch1.alpha(), ch2.alpha(), "Same inputs should produce same challenge");
        assert_eq!(ch1.hash(), ch2.hash(), "Same inputs should produce same hash");
    }
    
    #[test]
    fn test_challenge_collision_resistance() {
        // Different inputs → different challenges
        let ctx = test_context();
        
        let p1 = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);
        let p2 = Polynomial::from_witness(&[1, 7, 13, 92], TEST_MODULUS);
        
        let comm1 = Commitment::new(&ctx, p1.coefficients(), 0x1234).unwrap();
        let comm2 = Commitment::new(&ctx, p2.coefficients(), 0x1234).unwrap();
        
        let public_inputs = vec![1, 91];
        
        let ch1 = Challenge::derive(&public_inputs, &comm1, TEST_MODULUS);
        let ch2 = Challenge::derive(&public_inputs, &comm2, TEST_MODULUS);
        
        // Different commitments should produce different challenges
        // Note: Very small probability of collision (2^-256)
        assert_ne!(ch1.hash(), ch2.hash(), "Different commitments should produce different hashes");
    }
    
    #[test]
    fn test_challenge_public_input_sensitivity() {
        // Different public inputs → different challenges
        let ctx = test_context();
        
        let p = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);
        let comm = Commitment::new(&ctx, p.coefficients(), 0x1234).unwrap();
        
        let public1 = vec![1, 91];
        let public2 = vec![1, 92];
        
        let ch1 = Challenge::derive(&public1, &comm, TEST_MODULUS);
        let ch2 = Challenge::derive(&public2, &comm, TEST_MODULUS);
        
        assert_ne!(ch1.alpha(), ch2.alpha(), "Different public inputs should produce different challenges");
        assert_ne!(ch1.hash(), ch2.hash(), "Different public inputs should produce different hashes");
    }
    
    #[test]
    fn test_challenge_in_field() {
        // Challenge α should be in F_q
        let ctx = test_context();
        
        let p = Polynomial::from_witness(&[1, 2, 3, 4, 5], TEST_MODULUS);
        let comm = Commitment::new(&ctx, p.coefficients(), 999).unwrap();
        
        let public_inputs = vec![1, 5];
        let ch = Challenge::derive(&public_inputs, &comm, TEST_MODULUS);
        
        assert!(ch.alpha().value() < TEST_MODULUS, "Challenge should be in field F_q");
    }
    
    #[test]
    fn test_challenge_domain_separation() {
        // Domain separation: different contexts should produce different challenges
        let ctx = test_context();
        
        let p = Polynomial::from_witness(&[1, 2], TEST_MODULUS);
        let comm = Commitment::new(&ctx, p.coefficients(), 42).unwrap();
        
        let public_inputs = vec![1];
        
        // Normal derivation
        let ch1 = Challenge::derive(&public_inputs, &comm, TEST_MODULUS);
        
        // Simulate different domain by modifying hasher (test internal)
        let mut hasher = Sha3_256::new();
        hasher.update(b"DIFFERENT-DOMAIN");  // Different prefix
        hasher.update(&(public_inputs.len() as u64).to_le_bytes());
        for &inp in &public_inputs {
            hasher.update(&inp.to_le_bytes());
        }
        let comm_bytes = comm.as_bytes();
        hasher.update(&(comm_bytes.len() as u64).to_le_bytes());
        for &word in comm_bytes {
            hasher.update(&word.to_le_bytes());
        }
        let hash2: [u8; 32] = hasher.finalize().into();
        
        // Should be different due to domain separation
        assert_ne!(ch1.hash(), &hash2, "Different domains should produce different hashes");
    }
    
    #[test]
    fn test_challenge_uniform_distribution() {
        // Statistical test: check challenge distribution over F_q
        // Generate 100 challenges and verify spread
        let ctx = test_context();
        
        let mut challenges = Vec::new();
        
        for i in 0..100 {
            let witness = vec![1, i, i*2, i*3];
            let p = Polynomial::from_witness(&witness, TEST_MODULUS);
            let comm = Commitment::new(&ctx, p.coefficients(), i).unwrap();
            
            let public_inputs = vec![1, i];
            let ch = Challenge::derive(&public_inputs, &comm, TEST_MODULUS);
            
            challenges.push(ch.alpha().value());
        }
        
        // Check: all challenges are distinct (collision probability ~100²/2^44 ≈ 10^-9)
        let mut sorted = challenges.clone();
        sorted.sort_unstable();
        sorted.dedup();
        
        assert_eq!(sorted.len(), 100, "All 100 challenges should be distinct");
        
        // Check: challenges span wide range (at least 50% of field)
        let min = *sorted.first().unwrap();
        let max = *sorted.last().unwrap();
        let range = max - min;
        
        // Expect at least 30% of field covered (conservative)
        let expected_range = TEST_MODULUS * 3 / 10;
        assert!(range > expected_range, 
                "Challenges should span at least 30% of field, got {}% (range={}, expected={})",
                (range * 100) / TEST_MODULUS, range, expected_range);
    }
}
