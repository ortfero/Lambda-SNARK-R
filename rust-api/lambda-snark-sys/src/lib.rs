//! FFI bindings to ΛSNARK-R C++ core.
//!
//! This crate provides low-level, unsafe bindings to the C++ performance kernel.
//! You probably want to use the `lambda-snark` crate instead, which provides a
//! safe Rust API.
//!
//! # Safety
//!
//! All functions in this crate are `unsafe` and require careful handling of:
//! - Pointer validity and alignment
//! - Lifetime management
//! - Thread safety
//! - Memory allocation/deallocation across FFI boundary

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    
    #[test]
    fn test_lwe_context_create_null() {
        unsafe {
            let ctx = lwe_context_create(ptr::null());
            assert!(ctx.is_null());
        }
    }
    
    #[test]
    fn test_ntt_context_create_free() {
        unsafe {
            let ctx = ntt_context_create(12289, 256);
            if !ctx.is_null() {
                ntt_context_free(ctx);
            }
        }
    }
}
