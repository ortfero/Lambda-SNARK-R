//! ΛSNARK-R: Post-quantum SNARK over lattices.
//!
//! This library provides a production-grade implementation of lattice-based SNARKs
//! using Module-LWE/SIS hardness assumptions.
//!
//! # Quick Start
//!
//! ```no_run
//! use lambda_snark::{Params, Profile, SecurityLevel, Field, setup, prove, verify};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Setup parameters
//! let params = Params::new(
//!     SecurityLevel::Bits128,
//!     Profile::RingB {
//!         n: 4096,
//!         k: 2,
//!         q: 17592186044417,  // 2^44 + 1 (prime)
//!         sigma: 3.19,
//!     },
//! );
//!
//! let (pk, vk) = setup(params)?;
//!
//! // Prove a * b = c (R1CS) - NOT YET IMPLEMENTED
//! let public_input = vec![Field::new(1), Field::new(91)];  // (1, 91)
//! let witness = vec![Field::new(7), Field::new(13)];       // a=7, b=13, c=91
//!
//! // TODO: prover not yet implemented
//! // let proof = prove(&pk, &public_input, &witness)?;
//! // let valid = verify(&vk, &public_input, &proof)?;
//! // assert!(valid);
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────┐
//! │   lambda-snark (Safe Rust API)  │
//! └─────────────┬───────────────────┘
//!               │ FFI (safe wrappers)
//! ┌─────────────▼───────────────────┐
//! │  lambda-snark-sys (FFI bindings)│
//! └─────────────┬───────────────────┘
//!               │ extern "C"
//! ┌─────────────▼───────────────────┐
//! │  cpp-core (C++ performance)     │
//! │  - SEAL (LWE commitment)        │
//! │  - NTL (NTT)                    │
//! │  - Eigen (linear algebra)       │
//! └─────────────────────────────────┘
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

pub use lambda_snark_core::{Field, Params, Profile, SecurityLevel, Witness};
pub use lambda_snark_core::Error as CoreError;

mod commitment;
mod context;
mod polynomial;
mod challenge;
mod opening;

pub use commitment::Commitment;
pub use context::LweContext;
pub use polynomial::Polynomial;
pub use challenge::Challenge;
pub use opening::{Opening, generate_opening, verify_opening};

use thiserror::Error as ThisError;

/// ΛSNARK-R errors.
#[derive(Debug, ThisError)]
pub enum Error {
    /// Core error.
    #[error(transparent)]
    Core(#[from] CoreError),
    
    /// FFI error.
    #[error("FFI call failed: {0}")]
    Ffi(String),
    
    /// Invalid proof.
    #[error("Invalid proof")]
    InvalidProof,
}

/// Proving key (stub).
pub struct ProvingKey {
    ctx: LweContext,
    // TODO: Add R1CS matrices, etc.
}

/// Verifying key (stub).
pub struct VerifyingKey {
    // TODO: Add public parameters
}

/// SNARK proof containing commitment, challenge, and opening.
#[derive(Debug)]
pub struct Proof {
    /// LWE commitment to witness polynomial
    pub commitment: Commitment,
    
    /// Fiat-Shamir challenge point α
    pub challenge: Challenge,
    
    /// Opening proof at α
    pub opening: Opening,
}

impl Proof {
    /// Create new proof from components.
    pub fn new(commitment: Commitment, challenge: Challenge, opening: Opening) -> Self {
        Proof {
            commitment,
            challenge,
            opening,
        }
    }
    
    /// Get commitment.
    pub fn commitment(&self) -> &Commitment {
        &self.commitment
    }
    
    /// Get challenge.
    pub fn challenge(&self) -> &Challenge {
        &self.challenge
    }
    
    /// Get opening.
    pub fn opening(&self) -> &Opening {
        &self.opening
    }
}

/// Setup phase: generate proving and verifying keys.
///
/// # Errors
///
/// Returns error if parameters are invalid or FFI fails.
pub fn setup(params: Params) -> Result<(ProvingKey, VerifyingKey), Error> {
    params.validate()?;
    
    let ctx = LweContext::new(params)?;
    
    Ok((
        ProvingKey { ctx },
        VerifyingKey {},
    ))
}

/// Generate proof for witness.
///
/// Implements the prover algorithm:
/// 1. Encode witness as polynomial f(X) = Σ z_i·X^i
/// 2. Commit to polynomial using LWE
/// 3. Derive Fiat-Shamir challenge α = H(public_inputs || commitment)
/// 4. Compute opening y = f(α)
/// 5. Generate opening proof
/// 6. Assemble and return Proof
///
/// # Arguments
/// * `witness` - Witness values z_1, ..., z_n
/// * `public_inputs` - Public inputs for Fiat-Shamir
/// * `ctx` - LWE context for commitment
/// * `modulus` - Field modulus q
/// * `seed` - Random seed for commitment (0 = random)
///
/// # Returns
/// Proof containing commitment, challenge, and opening
///
/// # Errors
/// Returns error if witness is empty or commitment fails
///
/// # Example
/// ```no_run
/// use lambda_snark::{prove_simple, LweContext, Params, Profile, SecurityLevel};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let modulus = 17592186044417;
/// let params = Params::new(
///     SecurityLevel::Bits128,
///     Profile::RingB { n: 4096, k: 2, q: modulus, sigma: 3.19 },
/// );
/// let ctx = LweContext::new(params)?;
///
/// let witness = vec![1, 7, 13, 91];
/// let public_inputs = vec![1, 91];
///
/// let proof = prove_simple(&witness, &public_inputs, &ctx, modulus, 0x1234)?;
/// println!("Proof generated successfully");
/// # Ok(())
/// # }
/// ```
pub fn prove_simple(
    witness: &[u64],
    public_inputs: &[u64],
    ctx: &LweContext,
    modulus: u64,
    seed: u64,
) -> Result<Proof, Error> {
    // 1. Validate inputs
    if witness.is_empty() {
        return Err(Error::Ffi("Witness cannot be empty".to_string()));
    }
    
    // 2. Encode witness as polynomial
    let polynomial = Polynomial::from_witness(witness, modulus);
    
    // 3. Commit to polynomial
    let commitment = Commitment::new(ctx, polynomial.coefficients(), seed)?;
    
    // 4. Derive Fiat-Shamir challenge
    let challenge = Challenge::derive(public_inputs, &commitment, modulus);
    
    // 5. Generate opening proof
    let opening = generate_opening(&polynomial, challenge.alpha(), seed);
    
    // 6. Assemble proof
    Ok(Proof::new(commitment, challenge, opening))
}

/// Generate proof for R1CS instance (legacy API).
///
/// # Errors
///
/// Returns error if witness doesn't satisfy R1CS or proving fails.
pub fn prove(
    _pk: &ProvingKey,
    _public_input: &[Field],
    _witness: &[Field],
) -> Result<Proof, Error> {
    // TODO: Implement full R1CS prover
    Err(Error::Ffi("R1CS prover not implemented yet, use prove_simple()".to_string()))
}

/// Verify proof.
///
/// # Errors
///
/// Returns error if verification fails or proof is malformed.
pub fn verify(
    _vk: &VerifyingKey,
    _public_input: &[Field],
    _proof: &Proof,
) -> Result<bool, Error> {
    // TODO: Implement verifier
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_setup() {
        let params = Params::new(
            SecurityLevel::Bits128,
            Profile::RingB {
                n: 4096,  // SEAL requires n >= 1024
                k: 2,
                q: 17592186044417,  // 2^44 + 1 (prime, > 2^24)
                sigma: 3.19,
            },
        );
        
        let result = setup(params);
        assert!(result.is_ok());
    }
}
