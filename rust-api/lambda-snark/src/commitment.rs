//! LWE commitment wrapper.

use crate::{Error, LweContext};
use lambda_snark_core::Field;
use lambda_snark_sys as ffi;
use std::slice;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// LWE commitment (safe wrapper).
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
            return Err(Error::Core(lambda_snark_core::Error::CommitmentFailed));
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

#[cfg(test)]
mod tests {
    use super::*;
    use lambda_snark_core::{Params, Profile, SecurityLevel};
    
    #[test]
    fn test_commitment_create() {
        let params = Params::new(
            SecurityLevel::Bits128,
            Profile::RingB {
                n: 256,
                k: 2,
                q: 12289,
                sigma: 3.19,
            },
        );
        
        let ctx = LweContext::new(params).unwrap();
        let message = vec![Field::new(1), Field::new(2), Field::new(3)];
        
        let comm = Commitment::new(&ctx, &message, 0x1234);
        assert!(comm.is_ok());
    }
}
