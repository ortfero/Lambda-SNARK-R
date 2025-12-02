//! Core types for ΛSNARK-R (no_std compatible).
//!
//! This crate defines fundamental types used across the ΛSNARK-R stack:
//! - Field elements
//! - Parameter profiles
//! - Error types
//! - R1CS constraint system
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

// Re-export lambda-snark-sys to propagate link flags
#[doc(hidden)]
pub use lambda_snark_sys;

pub mod r1cs;

// ============================================================================
// NTT-Friendly Modulus Constants (M5.1.2)
// ============================================================================

/// NTT-friendly modulus for FFT/NTT polynomial operations.
///
/// q = 2^64 - 2^32 + 1 = 18,446,744,069,414,584,321
///
/// # Properties
///
/// - **Prime**: Verified by primality testing
/// - **NTT Support**: q-1 = 2^32 · (2^32 - 1) supports 2^32-point NTT
/// - **Primitive Root**: ω = 1,753,635,133,440,165,772 (from generator g=7)
/// - **Security**: 128-bit quantum security (Module-LWE with n=4096, k=2, σ=3.19)
/// - **Bit Length**: 64 bits (larger than legacy 44-bit modulus → stronger security)
///
/// # NTT Capacity
///
/// Supports circuits with up to **2^32 = 4,294,967,296 constraints** (far beyond practical limits).
/// For m ≤ 2^32, use Cooley-Tukey NTT with O(m log m) complexity.
///
/// # See Also
///
/// - [`LEGACY_MODULUS`]: Original 44-bit modulus (NOT NTT-friendly)
/// - [`NTT_PRIMITIVE_ROOT`]: Primitive 2^32-th root of unity for this modulus
pub const NTT_MODULUS: u64 = 18_446_744_069_414_584_321; // 2^64 - 2^32 + 1

/// Primitive 2^32-th root of unity modulo [`NTT_MODULUS`].
///
/// ω = 1,753,635,133,440,165,772 (computed as 7^((q-1)/2^32) mod q)
///
/// # Properties
///
/// - ω^(2^32) ≡ 1 (mod q)
/// - ω^(2^31) ≡ q-1 ≡ -1 (mod q) (primitivity check)
/// - ω^(2^k) for k < 32 are primitive 2^k-th roots (root hierarchy)
///
/// # Usage
///
/// ```rust,ignore
/// use lambda_snark_core::{NTT_MODULUS, NTT_PRIMITIVE_ROOT};
///
/// // Cooley-Tukey NTT uses powers of ω as twiddle factors
/// let omega_k = mod_pow(NTT_PRIMITIVE_ROOT, k, NTT_MODULUS);
/// ```
pub const NTT_PRIMITIVE_ROOT: u64 = 1_753_635_133_440_165_772; // ω for 2^32-point NTT

/// Legacy modulus (44-bit, NOT NTT-friendly).
///
/// q = 17,592,186,044,423
///
/// # Limitations
///
/// - q-1 = 2 · 8,796,093,022,211 (only 1 factor of 2)
/// - **Supports 2-point NTT only** (insufficient for FFT/NTT optimization)
/// - Lower security margin (44-bit vs. 64-bit)
///
/// # Migration
///
/// Code using this modulus should migrate to [`NTT_MODULUS`] for M5.1 FFT/NTT optimization.
/// See `docs/ntt-modulus.md` for migration guide.
pub const LEGACY_MODULUS: u64 = 17_592_186_044_423;

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
    /// Ring profile: R_q = Z_q\[X\]/(X^n + 1).
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

    /// Helper: Modular exponentiation (a^b mod m)
    fn mod_pow(base: u64, mut exp: u64, modulus: u64) -> u64 {
        if modulus == 1 {
            return 0;
        }
        let mut result = 1u128;
        let mut b = base as u128;
        let m = modulus as u128;

        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * b) % m;
            }
            exp >>= 1;
            b = (b * b) % m;
        }
        result as u64
    }

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
                q: 17592186044417, // Must be > 2^24
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
                q: 17592186044417,
                sigma: 1.0,
            },
        );
        assert_eq!(bad_params.validate(), Err(Error::InvalidSigma));
    }

    // ========================================================================
    // M5.1.2: NTT Modulus Tests
    // ========================================================================

    #[test]
    fn test_ntt_modulus_properties() {
        const Q: u64 = NTT_MODULUS;

        // Test 1: q = 2^64 - 2^32 + 1
        let expected = (1u128 << 64) - (1u128 << 32) + 1;
        assert_eq!(Q as u128, expected, "NTT_MODULUS != 2^64 - 2^32 + 1");

        // Test 2: q-1 = 2^32 · (2^32 - 1)
        let q_minus_1 = Q - 1;
        assert_eq!(q_minus_1 % (1u64 << 32), 0, "q-1 not divisible by 2^32");

        let k = q_minus_1 / (1u64 << 32);
        assert_eq!(k, (1u64 << 32) - 1, "q-1 / 2^32 != 2^32 - 1");
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_primitive_root_of_unity() {
        const Q: u64 = NTT_MODULUS;
        const OMEGA: u64 = NTT_PRIMITIVE_ROOT;
        const N: u64 = 1u64 << 32; // 2^32

        // Test 1: ω^(2^31) ≡ -1 ≡ q-1 (mod q) (primitivity)
        let omega_half = mod_pow(OMEGA, N / 2, Q);
        assert_eq!(
            omega_half,
            Q - 1,
            "ω^(2^31) must equal -1 (mod q) for primitivity"
        );

        // Test 2: ω ≠ 1 (sanity check)
        assert_ne!(OMEGA, 1, "ω cannot be 1 (trivial root)");
        assert!(OMEGA < Q, "ω must be < q");
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_ntt_root_hierarchy() {
        const Q: u64 = NTT_MODULUS;
        const OMEGA: u64 = NTT_PRIMITIVE_ROOT;

        // ω is primitive 2^32-th root
        // ω^2 is primitive 2^31-th root
        // ω^4 is primitive 2^30-th root
        // ... and so on

        for k in 1..10 {
            let n_k = 1u64 << k; // 2^k
            let omega_k = mod_pow(OMEGA, (1u64 << 32) / n_k, Q);

            // Check: omega_k^(2^k) ≡ 1 (mod q)
            let test = mod_pow(omega_k, n_k, Q);
            assert_eq!(
                test, 1,
                "ω^(2^32 / 2^{}) must be primitive 2^{}-th root",
                k, k
            );

            // Check primitivity: omega_k^(2^(k-1)) ≡ -1 (mod q)
            if k > 1 {
                let test_half = mod_pow(omega_k, n_k / 2, Q);
                assert_eq!(test_half, Q - 1, "ω_{} not primitive", k);
            }
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_legacy_vs_ntt_modulus() {
        // NTT modulus should be larger (better security)
        assert!(
            NTT_MODULUS > LEGACY_MODULUS,
            "NTT modulus should be larger than legacy"
        );

        // NTT modulus is 64-bit, legacy is 45-bit
        assert_eq!(
            NTT_MODULUS.leading_zeros(),
            0,
            "NTT modulus should use full 64 bits"
        );
        assert!(
            LEGACY_MODULUS.leading_zeros() >= 19,
            "Legacy modulus is ~45 bits (19 leading zeros)"
        );
    }
}
