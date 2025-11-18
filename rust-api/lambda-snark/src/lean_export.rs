//! Lean 4 term export for verification key and parameters.
//!
//! This module provides functionality to export R1CS verification keys and security
//! parameters as Lean 4 terms, enabling bidirectional verification between Rust
//! implementation and formal proofs.
//!
//! # Example
//!
//! ```no_run
//! use lambda_snark::{LeanExportable, VerificationKey, R1CS, SparseMatrix};
//!
//! fn main() {
//!     // Simple 1×3 R1CS: witness [1, x, y], constraint x * y = 1
//!     let a = SparseMatrix::from_dense(&vec![vec![0, 1, 0]]);
//!     let b = SparseMatrix::from_dense(&vec![vec![0, 0, 1]]);
//!     let c = SparseMatrix::from_dense(&vec![vec![1, 0, 0]]);
//!
//!     let r1cs = R1CS::new(1, 3, 1, a, b, c, 12289);
//!     let vk = VerificationKey::from_r1cs(&r1cs);
//!     let lean_term = vk.to_lean_term();
//!     println!("{}", lean_term);
//! }
//! ```
//!
//! # Milestone
//!
//! Prototype for M10 (Completeness & Integration)
//! - M10.2: Export VK → Lean term format
//! - Target: April 2026

use crate::r1cs::R1CS;
use crate::sparse_matrix::SparseMatrix;

/// Trait for types that can be exported as Lean 4 terms.
pub trait LeanExportable {
    /// Convert to Lean 4 term representation.
    fn to_lean_term(&self) -> String;
}

/// Verification Key (simplified for prototype).
///
/// Contains minimal information needed for verification:
/// - R1CS constraint system structure
/// - Security parameters
#[derive(Debug, Clone)]
pub struct VerificationKey {
    /// Number of constraints (m)
    pub num_constraints: usize,

    /// Total witness size (n)
    pub num_vars: usize,

    /// Number of public inputs (l)
    pub num_public_inputs: usize,

    /// Field modulus q
    pub modulus: u64,

    /// Constraint matrix A (sparse)
    pub a_matrix: SparseMatrix,

    /// Constraint matrix B (sparse)
    pub b_matrix: SparseMatrix,

    /// Constraint matrix C (sparse)
    pub c_matrix: SparseMatrix,
}

impl VerificationKey {
    /// Create verification key from R1CS instance.
    pub fn from_r1cs(r1cs: &R1CS) -> Self {
        Self {
            num_constraints: r1cs.m,
            num_vars: r1cs.n,
            num_public_inputs: r1cs.l,
            modulus: r1cs.modulus,
            a_matrix: r1cs.a.clone(),
            b_matrix: r1cs.b.clone(),
            c_matrix: r1cs.c.clone(),
        }
    }
}

impl LeanExportable for SparseMatrix {
    /// Export SparseMatrix as Lean term.
    ///
    /// Format: `SparseMatrix.mk rows cols [(row, col, val), ...]`
    fn to_lean_term(&self) -> String {
        let mut entries = String::new();
        let mut first = true;

        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let val = self.get(row, col);
                if val != 0 {
                    if !first {
                        entries.push_str(", ");
                    }
                    entries.push_str(&format!("({}, {}, {})", row, col, val));
                    first = false;
                }
            }
        }

        format!(
            "SparseMatrix.mk {} {} [{}]",
            self.rows(),
            self.cols(),
            entries
        )
    }
}

impl LeanExportable for VerificationKey {
    /// Export VerificationKey as Lean term.
    ///
    /// Format:
    /// ```lean
    /// ⟨nCons, nVars, nPublic, modulus,
    ///   SparseMatrix.mk ...,
    ///   SparseMatrix.mk ...,
    ///   SparseMatrix.mk ...⟩
    /// ```
    fn to_lean_term(&self) -> String {
        let a_term = self.a_matrix.to_lean_term();
        let b_term = self.b_matrix.to_lean_term();
        let c_term = self.c_matrix.to_lean_term();

        format!(
            "⟨{}, {}, {}, {},\n  {},\n  {},\n  {}⟩",
            self.num_constraints,
            self.num_vars,
            self.num_public_inputs,
            self.modulus,
            a_term,
            b_term,
            c_term
        )
    }
}

impl LeanExportable for R1CS {
    /// Export the entire R1CS instance as a Lean verification-key term.
    ///
    /// This is a convenience wrapper around `VerificationKey::from_r1cs`
    /// allowing direct serialization of constraint systems produced on the
    /// Rust side.
    fn to_lean_term(&self) -> String {
        VerificationKey::from_r1cs(self).to_lean_term()
    }
}

/// Security parameters for export to Lean.
#[derive(Debug, Clone)]
pub struct SecurityParams {
    /// LWE dimension (n)
    pub n: usize,

    /// Module rank (k)
    pub k: usize,

    /// Field modulus (q)
    pub q: u64,

    /// Gaussian width (σ)
    pub sigma: f64,

    /// Security level (λ in bits)
    pub lambda: usize,
}

impl SecurityParams {
    /// Create default parameters (128-bit security).
    pub fn default_128bit() -> Self {
        Self {
            n: 4096,
            k: 2,
            q: 12289, // NTT-friendly prime
            sigma: 3.2,
            lambda: 128,
        }
    }
}

impl LeanExportable for SecurityParams {
    /// Export SecurityParams as Lean term.
    ///
    /// Format:
    /// ```lean
    /// { n := 4096, k := 2, q := 12289, σ := 3.2, λ := 128 }
    /// ```
    fn to_lean_term(&self) -> String {
        format!(
            "{{ n := {}, k := {}, q := {}, σ := {}, λ := {} }}",
            self.n, self.k, self.q, self.sigma, self.lambda
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matrix_export() {
        // Matrix:
        // [0, 42, 0]
        // [0, 0, 99]
        let rows = vec![vec![0, 42, 0], vec![0, 0, 99]];
        let matrix = SparseMatrix::from_dense(&rows);

        let lean_term = matrix.to_lean_term();
        assert!(lean_term.contains("SparseMatrix.mk"));
        assert!(lean_term.contains("2 3"));
        assert!(lean_term.contains("(0, 1, 42)"));
        assert!(lean_term.contains("(1, 2, 99)"));
    }

    #[test]
    fn test_security_params_export() {
        let params = SecurityParams::default_128bit();
        let lean_term = params.to_lean_term();

        assert!(lean_term.contains("n := 4096"));
        assert!(lean_term.contains("q := 12289"));
        assert!(lean_term.contains("λ := 128"));
    }

    #[test]
    fn test_vk_export_format() {
        // Simple 1×2 R1CS: witness [1, x], constraint x * 1 = 1
        let a = SparseMatrix::from_dense(&vec![vec![0, 1]]); // A[0,1] = 1
        let b = SparseMatrix::from_dense(&vec![vec![1, 0]]); // B[0,0] = 1
        let c = SparseMatrix::from_dense(&vec![vec![1, 0]]); // C[0,0] = 1

        let r1cs = R1CS::new(1, 2, 1, a, b, c, 12289);
        let vk = VerificationKey::from_r1cs(&r1cs);

        let lean_term = vk.to_lean_term();

        // Validate structure
        assert!(lean_term.starts_with("⟨"));
        assert!(lean_term.ends_with("⟩"));
        assert!(lean_term.contains("12289")); // modulus
        assert!(lean_term.contains("SparseMatrix.mk"));
    }

    #[test]
    fn test_r1cs_export_via_trait() {
        let a = SparseMatrix::from_dense(&vec![vec![0, 1]]);
        let b = SparseMatrix::from_dense(&vec![vec![1, 0]]);
        let c = SparseMatrix::from_dense(&vec![vec![1, 0]]);

        let r1cs = R1CS::new(1, 2, 1, a, b, c, 12289);
        let lean_term = r1cs.to_lean_term();

        assert!(lean_term.starts_with("⟨"));
        assert!(lean_term.contains("2")); // number of vars
        assert!(lean_term.contains("SparseMatrix.mk"));
    }
}
