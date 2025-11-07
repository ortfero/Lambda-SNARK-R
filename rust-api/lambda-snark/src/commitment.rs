//! LWE commitment wrapper.

use crate::{Error, LweContext, CoreError};
use lambda_snark_core::Field;
use lambda_snark_sys as ffi;
use std::slice;
use zeroize::{Zeroize, ZeroizeOnDrop};
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::de::{self, Visitor, SeqAccess};

/// LWE commitment (safe wrapper).
#[derive(Debug)]
pub struct Commitment {
    inner: *mut ffi::LweCommitment,
}

impl Commitment {
    /// Commit to a message vector.
    pub fn new(
        ctx: &LweContext,
        message: &[Field],
        seed: u64,
    ) -> Result<Self, Error> {
        let msg_u64: Vec<u64> = message.iter().map(|f| f.value()).collect();
        
        let inner = unsafe {
            ffi::lwe_commit(
                ctx.as_ptr(),
                msg_u64.as_ptr(),
                msg_u64.len(),
                seed,
            )
        };
        
        if inner.is_null() {
            return Err(Error::Core(CoreError::CommitmentFailed));
        }
        
        Ok(Commitment { inner })
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
            "Commitment deserialization requires LweContext (not implemented)"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambda_snark_core::{Params, Profile, SecurityLevel};
    
    #[test]
    fn test_commitment_create() {
        let params = Params::new(
            SecurityLevel::Bits128,
            Profile::RingB {
                n: 4096,  // SEAL requires n >= 1024
                k: 2,
                q: 17592186044417,  // 2^44 + 1 (prime, > 2^24)
                sigma: 3.19,
            },
        );
        
        let ctx = LweContext::new(params).unwrap();
        let message = vec![Field::new(1), Field::new(2), Field::new(3)];
        
        let comm = Commitment::new(&ctx, &message, 0x1234);
        assert!(comm.is_ok());
    }
}
