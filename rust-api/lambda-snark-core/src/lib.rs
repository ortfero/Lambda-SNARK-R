//! Core types for ΛSNARK-R (no_std compatible).
//!
//! This crate defines fundamental types used across the ΛSNARK-R stack:
//! - Field elements
//! - Parameter profiles
//! - Error types
//!
//! # no_std Support
//!
//! This crate is `no_std` compatible by default. Enable the `std` feature
//! for standard library support.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Field element in Z_q.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Zeroize)]
#[repr(transparent)]
pub struct Field(pub u64);

impl Field {
    /// Create a new field element.
    #[inline]
    pub const fn new(value: u64) -> Self {
        Field(value)
    }
    
    /// Get the raw value.
    #[inline]
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Security level in bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum SecurityLevel {
    /// 128-bit security (post-quantum).
    Bits128 = 128,
    /// 192-bit security.
    Bits192 = 192,
    /// 256-bit security.
    Bits256 = 256,
}

/// Parameter profile.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Profile {
    /// Scalar profile: R = Z_q.
    ScalarA {
        /// Prime modulus q.
        q: u64,
        /// Gaussian parameter σ.
        sigma: f64,
    },
    /// Ring profile: R_q = Z_q[X]/(X^n + 1).
    RingB {
        /// Ring degree (power of 2).
        n: usize,
        /// Module rank.
        k: usize,
        /// Prime modulus q.
        q: u64,
        /// Gaussian parameter σ.
        sigma: f64,
    },
}

/// Public parameters for ΛSNARK-R.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Params {
    /// Security level.
    pub security_level: SecurityLevel,
    /// Parameter profile.
    pub profile: Profile,
}

impl Params {
    /// Create new parameters.
    pub fn new(security_level: SecurityLevel, profile: Profile) -> Self {
        Self {
            security_level,
            profile,
        }
    }
    
    /// Validate parameters for correctness.
    pub fn validate(&self) -> Result<(), Error> {
        match &self.profile {
            Profile::ScalarA { q, sigma } => {
                if *q < (1u64 << 24) {
                    return Err(Error::InvalidModulus);
                }
                if *sigma < 3.0 {
                    return Err(Error::InvalidSigma);
                }
            }
            Profile::RingB { n, k, q, sigma } => {
                if !n.is_power_of_two() {
                    return Err(Error::InvalidRingDegree);
                }
                if *k == 0 {
                    return Err(Error::InvalidModuleRank);
                }
                if *q < (1u64 << 24) {
                    return Err(Error::InvalidModulus);
                }
                if *sigma < 3.0 {
                    return Err(Error::InvalidSigma);
                }
            }
        }
        Ok(())
    }
}

/// Secret witness (zeroized on drop).
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop, Serialize, Deserialize)]
pub struct Witness {
    data: Vec<Field>,
}

impl Witness {
    /// Create witness from vector.
    pub fn new(data: Vec<Field>) -> Self {
        Self { data }
    }
    
    /// Get witness data.
    pub fn as_slice(&self) -> &[Field] {
        &self.data
    }
}

/// Error type.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Invalid modulus (too small).
    InvalidModulus,
    /// Invalid sigma (too small).
    InvalidSigma,
    /// Invalid ring degree (not power of 2).
    InvalidRingDegree,
    /// Invalid module rank (zero).
    InvalidModuleRank,
    /// Invalid dimensions.
    InvalidDimensions,
    /// Commitment failed.
    CommitmentFailed,
    /// Verification failed.
    VerificationFailed,
    /// FFI error.
    FfiError,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::InvalidModulus => write!(f, "Invalid modulus (must be prime > 2^24)"),
            Error::InvalidSigma => write!(f, "Invalid sigma (must be >= 3.0)"),
            Error::InvalidRingDegree => write!(f, "Invalid ring degree (must be power of 2)"),
            Error::InvalidModuleRank => write!(f, "Invalid module rank (must be > 0)"),
            Error::InvalidDimensions => write!(f, "Invalid dimensions"),
            Error::CommitmentFailed => write!(f, "Commitment generation failed"),
            Error::VerificationFailed => write!(f, "Verification failed"),
            Error::FfiError => write!(f, "FFI call failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_field_element() {
        let f = Field::new(42);
        assert_eq!(f.value(), 42);
    }
    
    #[test]
    fn test_params_validation() {
        let params = Params::new(
            SecurityLevel::Bits128,
            Profile::RingB {
                n: 256,
                k: 2,
                q: 12289,
                sigma: 3.19,
            },
        );
        assert!(params.validate().is_ok());
        
        // Invalid: sigma too small
        let bad_params = Params::new(
            SecurityLevel::Bits128,
            Profile::RingB {
                n: 256,
                k: 2,
                q: 12289,
                sigma: 1.0,
            },
        );
        assert_eq!(bad_params.validate(), Err(Error::InvalidSigma));
    }
}
