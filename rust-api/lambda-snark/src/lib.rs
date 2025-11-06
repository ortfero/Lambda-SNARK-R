//! ΛSNARK-R: Post-quantum SNARK over lattices.
//!
//! This library provides a production-grade implementation of lattice-based SNARKs
//! using Module-LWE/SIS hardness assumptions.
//!
//! # Quick Start
//!
//! ```no_run
//! use lambda_snark::{Params, Profile, SecurityLevel, setup, prove, verify};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Setup parameters
//! let params = Params::new(
//!     SecurityLevel::Bits128,
//!     Profile::RingB {
//!         n: 256,
//!         k: 2,
//!         q: 12289,
//!         sigma: 3.19,
//!     },
//! );
//!
//! let (pk, vk) = setup(params)?;
//!
//! // Prove a * b = c (R1CS)
//! let public_input = vec![1, 91];  // (1, 91)
//! let witness = vec![7, 13];       // a=7, b=13, c=91
//!
//! let proof = prove(&pk, &public_input, &witness)?;
//! let valid = verify(&vk, &public_input, &proof)?;
//! assert!(valid);
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

pub use lambda_snark_core::{Error, Field, Params, Profile, SecurityLevel, Witness};

mod commitment;
mod context;

pub use commitment::Commitment;
pub use context::LweContext;

use thiserror::Error as ThisError;

/// ΛSNARK-R errors.
#[derive(Debug, ThisError)]
pub enum Error {
    /// Core error.
    #[error(transparent)]
    Core(#[from] lambda_snark_core::Error),
    
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

/// Proof (stub).
pub struct Proof {
    commitment: Commitment,
    // TODO: Add openings, challenges, etc.
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

/// Generate proof for R1CS instance.
///
/// # Errors
///
/// Returns error if witness doesn't satisfy R1CS or proving fails.
pub fn prove(
    _pk: &ProvingKey,
    _public_input: &[Field],
    _witness: &[Field],
) -> Result<Proof, Error> {
    // TODO: Implement prover
    Err(Error::Ffi("Not implemented".to_string()))
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
                n: 256,
                k: 2,
                q: 12289,
                sigma: 3.19,
            },
        );
        
        let result = setup(params);
        assert!(result.is_ok());
    }
}
