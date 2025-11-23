//! LWE commitment wrapper.

use crate::arith::add_mod;
#[cfg(test)]
use crate::arith::mul_mod;
use crate::{CoreError, Error, LweContext};
use lambda_snark_core::Field;
use lambda_snark_sys as ffi;
use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::slice;

/// LWE commitment (safe wrapper).
#[derive(Debug)]
pub struct Commitment {
    inner: *mut ffi::LweCommitment,
}

impl Clone for Commitment {
    fn clone(&self) -> Self {
        let inner = unsafe { ffi::lwe_commitment_clone(self.as_ffi_ptr()) };
        if inner.is_null() {
            panic!("lwe_commitment_clone returned null");
        }
        Commitment { inner }
    }
}

impl Commitment {
    /// Commit to a message vector.
    pub fn new(ctx: &LweContext, message: &[Field], seed: u64) -> Result<Self, Error> {
        let modulus = ctx.modulus();
        let msg_u64: Vec<u64> = message
            .iter()
            .map(|f| add_mod(f.value() % modulus, 0, modulus))
            .collect();

        let inner = unsafe { ffi::lwe_commit(ctx.as_ptr(), msg_u64.as_ptr(), msg_u64.len(), seed) };

        if inner.is_null() {
            return Err(Error::Core(CoreError::CommitmentFailed));
        }

        Ok(Commitment { inner })
    }

    /// Compute linear combination of commitments with scalar coefficients.
    pub fn linear_combine(
        ctx: &LweContext,
        commitments: &[&Commitment],
        coeffs: &[Field],
    ) -> Result<Self, Error> {
        if commitments.is_empty() {
            return Err(Error::InvalidInput("no commitments provided".into()));
        }

        if commitments.len() != coeffs.len() {
            return Err(Error::InvalidInput(
                "commitments/coeffs length mismatch".into(),
            ));
        }

        let mut ptrs: Vec<*const ffi::LweCommitment> =
            commitments.iter().map(|c| c.as_ffi_ptr()).collect();
        let modulus = ctx.modulus();
        let coeff_words: Vec<u64> = coeffs
            .iter()
            .map(|f| add_mod(f.value() % modulus, 0, modulus))
            .collect();

        let combined = unsafe {
            ffi::lwe_linear_combine(
                ctx.as_ptr(),
                ptrs.as_mut_ptr(),
                coeff_words.as_ptr(),
                ptrs.len(),
            )
        };

        if combined.is_null() {
            return Err(Error::Core(CoreError::CommitmentFailed));
        }

        Ok(Commitment { inner: combined })
    }

    /// Get commitment data as bytes.
    pub fn as_bytes(&self) -> &[u64] {
        unsafe {
            let raw = &*self.inner;
            slice::from_raw_parts(raw.data, raw.len)
        }
    }

    /// Get raw FFI pointer (for internal use).
    pub(crate) fn as_ffi_ptr(&self) -> *const ffi::LweCommitment {
        self.inner as *const ffi::LweCommitment
    }
}

impl Drop for Commitment {
    fn drop(&mut self) {
        unsafe {
            ffi::lwe_commitment_free(self.inner);
        }
    }
}

// Safety: Commitment owns its data
unsafe impl Send for Commitment {}

impl Serialize for Commitment {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize commitment as vector of u64
        let data = self.as_bytes();
        data.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Commitment {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // PLACEHOLDER: Cannot reconstruct Commitment without LweContext
        // This requires storing context info or accepting context on deserialization
        // For now, return error
        Err(de::Error::custom(
            "Commitment deserialization requires LweContext (not implemented)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambda_snark_core::{Params, Profile, SecurityLevel};
    use std::ptr;

    #[test]
    fn test_commitment_create() {
        let params = Params::new(
            SecurityLevel::Bits128,
            Profile::RingB {
                n: 4096, // SEAL requires n >= 1024
                k: 2,
                q: 17592186044417, // 2^44 + 1 (prime, > 2^24)
                sigma: 3.19,
            },
        );

        let ctx = LweContext::new(params).unwrap();
        let message = vec![Field::new(1), Field::new(2), Field::new(3)];

        let comm = Commitment::new(&ctx, &message, 0x1234);
        assert!(comm.is_ok());
    }

    #[test]
    fn test_linear_combination_roundtrip() {
        let params = Params::new(
            SecurityLevel::Bits128,
            Profile::RingB {
                n: 4096,
                k: 2,
                q: 17592186044417,
                sigma: 3.19,
            },
        );

        let ctx = LweContext::new(params).unwrap();
        let msg_len = 4;
        let msg1: Vec<Field> = (0..msg_len).map(|i| Field::new(i as u64 + 1)).collect();
        let msg2: Vec<Field> = (0..msg_len)
            .map(|i| Field::new((i as u64 + 1) * 2))
            .collect();

        let comm1 = Commitment::new(&ctx, &msg1, 0).unwrap();
        let comm2 = Commitment::new(&ctx, &msg2, 1).unwrap();

        let coeffs = vec![Field::new(2), Field::new(3)];
        let inputs = [&comm1, &comm2];

        let combined = Commitment::linear_combine(&ctx, &inputs, &coeffs).unwrap();

        let modulus = ctx.modulus();
        let expected: Vec<u64> = msg1
            .iter()
            .zip(msg2.iter())
            .map(|(a, b)| {
                let term1 = mul_mod(coeffs[0].value(), a.value(), modulus);
                let term2 = mul_mod(coeffs[1].value(), b.value(), modulus);
                add_mod(term1, term2, modulus)
            })
            .collect();

        let opening = ffi::LweOpening {
            randomness: ptr::null_mut(),
            rand_len: 0,
        };

        let valid = unsafe {
            ffi::lwe_verify_opening(
                ctx.as_ptr(),
                combined.as_ffi_ptr(),
                expected.as_ptr(),
                expected.len(),
                &opening,
            )
        };

        assert_eq!(
            valid, 1,
            "homomorphic combination should match expected message"
        );
    }
}
