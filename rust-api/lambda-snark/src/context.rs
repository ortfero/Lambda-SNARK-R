//! LWE context wrapper.

use crate::{CoreError, Error};
use lambda_snark_core::{Params, Profile};
use lambda_snark_sys as ffi;
/// Safe wrapper for LWE context.
pub struct LweContext {
    inner: *mut ffi::LweContext,
    modulus: u64,
}

impl LweContext {
    /// Create new LWE context from parameters.
    pub fn new(params: Params) -> Result<Self, Error> {
        params.validate().map_err(Error::Core)?;

        // Convert Rust params to C params
        let c_params = ffi::PublicParams {
            profile: match params.profile {
                Profile::ScalarA { .. } => ffi::ProfileType_PROFILE_SCALAR_A,
                Profile::RingB { .. } => ffi::ProfileType_PROFILE_RING_B,
            },
            security_level: match params.security_level {
                lambda_snark_core::SecurityLevel::Bits128 => 128,
                lambda_snark_core::SecurityLevel::Bits192 => 192,
                lambda_snark_core::SecurityLevel::Bits256 => 256,
            },
            modulus: match params.profile {
                Profile::ScalarA { q, .. } | Profile::RingB { q, .. } => q,
            },
            ring_degree: match params.profile {
                Profile::ScalarA { .. } => 1,
                Profile::RingB { n, .. } => n as u32,
            },
            module_rank: match params.profile {
                Profile::ScalarA { .. } => 1,
                Profile::RingB { k, .. } => k as u32,
            },
            sigma: match params.profile {
                Profile::ScalarA { sigma, .. } | Profile::RingB { sigma, .. } => sigma,
            },
        };

        let inner = unsafe { ffi::lwe_context_create(&c_params) };

        if inner.is_null() {
            return Err(Error::Core(CoreError::FfiError));
        }

        Ok(LweContext {
            inner,
            modulus: c_params.modulus,
        })
    }

    /// Get raw pointer (for FFI calls).
    pub(crate) fn as_ptr(&self) -> *mut ffi::LweContext {
        self.inner
    }

    /// Retrieve the prime modulus associated with this context.
    pub fn modulus(&self) -> u64 {
        self.modulus
    }
}

impl Drop for LweContext {
    fn drop(&mut self) {
        unsafe {
            ffi::lwe_context_free(self.inner);
        }
    }
}

// Safety: LweContext is Send as long as C++ implementation is thread-safe
unsafe impl Send for LweContext {}

#[cfg(test)]
mod tests {
    use super::*;
    use lambda_snark_core::SecurityLevel;

    #[test]
    fn test_context_create_and_drop() {
        let params = Params::new(
            SecurityLevel::Bits128,
            Profile::RingB {
                n: 4096, // SEAL requires n >= 1024
                k: 2,
                q: 17592186044417, // 2^44 + 1 (prime, > 2^24)
                sigma: 3.19,
            },
        );

        let ctx = LweContext::new(params);
        assert!(ctx.is_ok());

        let ctx = ctx.unwrap();
        assert_eq!(ctx.modulus(), 17592186044417);
        // Drop happens here when ctx goes out of scope
    }
}
