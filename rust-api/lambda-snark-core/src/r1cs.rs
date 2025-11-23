//! R1CS (Rank-1 Constraint System) implementation.
//!
//! This module provides safe Rust wrappers around the C++ R1CS core.
//! R1CS represents arithmetic constraints as:
//!     (A·z) ∘ (B·z) = (C·z)
//! where:
//! - A, B, C are sparse matrices (m constraints × n variables)
//! - z is the witness vector [1, x₁, ..., xₙ₋₁]
//! - ∘ is element-wise (Hadamard) product
//!
//! # Example
//!
//! ```no_run
//! use lambda_snark_core::r1cs::{SparseMatrix, R1CS};
//!
//! // Create constraint: a * b = c
//! let a = SparseMatrix::from_entries(1, 4, vec![(0, 1, 1)]);
//! let b = SparseMatrix::from_entries(1, 4, vec![(0, 2, 1)]);
//! let c = SparseMatrix::from_entries(1, 4, vec![(0, 3, 1)]);
//!
//! let r1cs = R1CS::new(a, b, c, 17592186044417)?;
//!
//! // Validate witness: [1, 7, 13, 91] (7 * 13 = 91)
//! assert!(r1cs.validate_witness(&[1, 7, 13, 91])?);
//! # Ok::<(), lambda_snark_core::Error>(())
//! ```

use crate::Error;
use std::ptr;

/// Sparse matrix entry (row, col, value).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseEntry {
    /// Row index
    pub row: u32,
    /// Column index
    pub col: u32,
    /// Field element value
    pub value: u64,
}

/// Sparse matrix in COO (Coordinate) format.
#[repr(C)]
pub struct SparseMatrix {
    /// Pointer to entries array
    entries: *mut SparseEntry,
    /// Number of entries
    n_entries: usize,
    /// Number of rows
    n_rows: u32,
    /// Number of columns
    n_cols: u32,
}

impl SparseMatrix {
    /// Create sparse matrix from list of (row, col, value) entries.
    ///
    /// # Arguments
    /// * `n_rows` - Number of rows
    /// * `n_cols` - Number of columns
    /// * `entries` - List of non-zero entries
    pub fn from_entries(n_rows: u32, n_cols: u32, entries: Vec<(u32, u32, u64)>) -> Self {
        let mut sparse_entries: Vec<SparseEntry> = entries
            .into_iter()
            .map(|(r, c, v)| SparseEntry {
                row: r,
                col: c,
                value: v,
            })
            .collect();

        let n_entries = sparse_entries.len();
        let ptr = sparse_entries.as_mut_ptr();
        std::mem::forget(sparse_entries); // Transfer ownership

        Self {
            entries: ptr,
            n_entries,
            n_rows,
            n_cols,
        }
    }

    /// Number of non-zero entries.
    pub fn len(&self) -> usize {
        self.n_entries
    }

    /// Check if matrix is empty (no entries).
    pub fn is_empty(&self) -> bool {
        self.n_entries == 0
    }

    /// Number of rows.
    pub fn rows(&self) -> u32 {
        self.n_rows
    }

    /// Number of columns.
    pub fn cols(&self) -> u32 {
        self.n_cols
    }
}

impl Drop for SparseMatrix {
    fn drop(&mut self) {
        if !self.entries.is_null() {
            unsafe {
                drop(Vec::from_raw_parts(
                    self.entries,
                    self.n_entries,
                    self.n_entries,
                ));
            }
        }
    }
}

// FFI declarations
extern "C" {
    fn lambda_snark_r1cs_create(
        a: *const SparseMatrix,
        b: *const SparseMatrix,
        c: *const SparseMatrix,
        modulus: u64,
        out_r1cs: *mut *mut std::ffi::c_void,
    ) -> u32;

    fn lambda_snark_r1cs_validate_witness(
        r1cs: *mut std::ffi::c_void,
        witness: *const R1CSWitness,
        out_valid: *mut bool,
    ) -> u32;

    fn lambda_snark_r1cs_free(r1cs: *mut std::ffi::c_void);

    fn lambda_snark_r1cs_num_constraints(r1cs: *mut std::ffi::c_void) -> u32;

    fn lambda_snark_r1cs_num_variables(r1cs: *mut std::ffi::c_void) -> u32;
}

#[repr(C)]
struct R1CSWitness {
    values: *const u64,
    len: usize,
}

/// R1CS constraint system.
///
/// Safe RAII wrapper around C++ R1CS.
pub struct R1CS {
    handle: *mut std::ffi::c_void,
}

impl R1CS {
    /// Create R1CS from sparse matrices A, B, C.
    ///
    /// # Arguments
    /// * `a` - Left matrix
    /// * `b` - Right matrix
    /// * `c` - Output matrix
    /// * `modulus` - Field modulus q (must be prime)
    ///
    /// # Errors
    /// Returns `Error::InvalidParams` if dimensions mismatch.
    pub fn new(
        a: SparseMatrix,
        b: SparseMatrix,
        c: SparseMatrix,
        modulus: u64,
    ) -> Result<Self, Error> {
        let mut handle: *mut std::ffi::c_void = ptr::null_mut();

        let err = unsafe {
            lambda_snark_r1cs_create(
                &a as *const SparseMatrix,
                &b as *const SparseMatrix,
                &c as *const SparseMatrix,
                modulus,
                &mut handle,
            )
        };

        // Matrices ownership transferred to C++
        std::mem::forget(a);
        std::mem::forget(b);
        std::mem::forget(c);

        match err {
            0 => Ok(Self { handle }),
            2 => Err(Error::InvalidDimensions),
            3 => Err(Error::FfiError),
            _ => Err(Error::FfiError),
        }
    }

    /// Validate witness satisfies all constraints.
    ///
    /// Checks: (A·z) ∘ (B·z) = (C·z) for all rows.
    ///
    /// # Arguments
    /// * `witness` - Witness vector z (first element must be 1)
    ///
    /// # Errors
    /// Returns `Error::InvalidParams` if witness length mismatch.
    pub fn validate_witness(&self, witness: &[u64]) -> Result<bool, Error> {
        let w = R1CSWitness {
            values: witness.as_ptr(),
            len: witness.len(),
        };

        let mut valid = false;

        let err = unsafe { lambda_snark_r1cs_validate_witness(self.handle, &w, &mut valid) };

        match err {
            0 => Ok(valid),
            2 => Err(Error::InvalidDimensions),
            _ => Err(Error::VerificationFailed),
        }
    }

    /// Get number of constraints (m).
    pub fn num_constraints(&self) -> u32 {
        unsafe { lambda_snark_r1cs_num_constraints(self.handle) }
    }

    /// Get number of variables (n).
    pub fn num_variables(&self) -> u32 {
        unsafe { lambda_snark_r1cs_num_variables(self.handle) }
    }
}

impl Drop for R1CS {
    fn drop(&mut self) {
        unsafe {
            lambda_snark_r1cs_free(self.handle);
        }
    }
}

// R1CS is thread-safe (C++ uses immutable operations after construction)
unsafe impl Send for R1CS {}
