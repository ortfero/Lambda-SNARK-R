//! R1CS (Rank-1 Constraint System) implementation.
//!
//! R1CS defines a system of constraints over a finite field F_q:
//!   (Az) ⊙ (Bz) = Cz
//!
//! Where:
//! - z ∈ F_q^n is the witness vector
//! - A, B, C ∈ F_q^{m×n} are constraint matrices (sparse)
//! - ⊙ denotes element-wise (Hadamard) product
//! - m = number of constraints
//! - n = witness size (including public inputs)
//!
//! # Example
//!
//! ```no_run
//! use lambda_snark::r1cs::R1CS;
//! use lambda_snark::sparse_matrix::SparseMatrix;
//!
//! // Multiplication gate: a * b = c
//! // Witness: z = [1, 7, 13, 91] (constant, a, b, c)
//! let modulus = 17592186044417; // 2^44 + 1
//!
//! let a_matrix = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
//! let b_matrix = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
//! let c_matrix = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);
//!
//! let r1cs = R1CS::new(1, 4, 2, a_matrix, b_matrix, c_matrix, modulus);
//!
//! let witness = vec![1, 7, 13, 91];
//! assert!(r1cs.is_satisfied(&witness));
//! ```

use crate::sparse_matrix::SparseMatrix;
use crate::Error;

/// R1CS (Rank-1 Constraint System) instance.
#[derive(Debug, Clone)]
pub struct R1CS {
    /// Number of constraints (m)
    pub m: usize,
    
    /// Total witness size (n), including public inputs
    pub n: usize,
    
    /// Number of public inputs (l)
    /// Public inputs are z[0..l]
    pub l: usize,
    
    /// Constraint matrix A ∈ F_q^{m×n}
    pub a: SparseMatrix,
    
    /// Constraint matrix B ∈ F_q^{m×n}
    pub b: SparseMatrix,
    
    /// Constraint matrix C ∈ F_q^{m×n}
    pub c: SparseMatrix,
    
    /// Field modulus q
    pub modulus: u64,
}

impl R1CS {
    /// Create new R1CS instance.
    ///
    /// # Arguments
    ///
    /// * `m` - Number of constraints
    /// * `n` - Total witness size (including public inputs)
    /// * `l` - Number of public inputs (z[0..l])
    /// * `a` - Constraint matrix A
    /// * `b` - Constraint matrix B
    /// * `c` - Constraint matrix C
    /// * `modulus` - Field modulus q
    ///
    /// # Panics
    ///
    /// Panics if matrix dimensions don't match:
    /// - A, B, C must all be m×n
    /// - l must be ≤ n
    pub fn new(
        m: usize,
        n: usize,
        l: usize,
        a: SparseMatrix,
        b: SparseMatrix,
        c: SparseMatrix,
        modulus: u64,
    ) -> Self {
        assert_eq!(a.rows(), m, "Matrix A must have m rows");
        assert_eq!(a.cols(), n, "Matrix A must have n columns");
        assert_eq!(b.rows(), m, "Matrix B must have m rows");
        assert_eq!(b.cols(), n, "Matrix B must have n columns");
        assert_eq!(c.rows(), m, "Matrix C must have m rows");
        assert_eq!(c.cols(), n, "Matrix C must have n columns");
        assert!(l <= n, "Number of public inputs must be <= n");
        
        R1CS {
            m,
            n,
            l,
            a,
            b,
            c,
            modulus,
        }
    }
    
    /// Check if witness satisfies R1CS constraints: (Az) ⊙ (Bz) = Cz
    ///
    /// Returns true if all m constraints are satisfied modulo q.
    ///
    /// # Arguments
    ///
    /// * `witness` - Witness vector z (length = n)
    ///
    /// # Returns
    ///
    /// `true` if witness satisfies all constraints, `false` otherwise
    ///
    /// # Panics
    ///
    /// Panics if witness.len() != n
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lambda_snark::r1cs::R1CS;
    /// # use lambda_snark::sparse_matrix::SparseMatrix;
    /// # let modulus = 17592186044417;
    /// # let a = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
    /// # let b = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
    /// # let c = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);
    /// # let r1cs = R1CS::new(1, 4, 2, a, b, c, modulus);
    /// let valid_witness = vec![1, 7, 13, 91];
    /// assert!(r1cs.is_satisfied(&valid_witness));
    ///
    /// let invalid_witness = vec![1, 7, 13, 90]; // 7*13 = 91 ≠ 90
    /// assert!(!r1cs.is_satisfied(&invalid_witness));
    /// ```
    pub fn is_satisfied(&self, witness: &[u64]) -> bool {
        assert_eq!(
            witness.len(),
            self.n,
            "Witness length {} must equal n={}",
            witness.len(),
            self.n
        );
        
        // Compute Az, Bz, Cz
        let az = self.a.mul_vec(witness, self.modulus);
        let bz = self.b.mul_vec(witness, self.modulus);
        let cz = self.c.mul_vec(witness, self.modulus);
        
        // Check (Az)_i · (Bz)_i = (Cz)_i for all i ∈ [m]
        for i in 0..self.m {
            let lhs = ((az[i] as u128) * (bz[i] as u128)) % (self.modulus as u128);
            let rhs = cz[i] as u128;
            
            if lhs != rhs {
                return false;
            }
        }
        
        true
    }
    
    /// Get public inputs from witness.
    ///
    /// Returns z[0..l] (first l elements of witness).
    pub fn public_inputs<'a>(&self, witness: &'a [u64]) -> &'a [u64] {
        assert_eq!(
            witness.len(),
            self.n,
            "Witness length must equal n"
        );
        &witness[0..self.l]
    }
    
    /// Validate R1CS structure (consistency checks).
    ///
    /// Checks:
    /// - Matrix dimensions match (m, n)
    /// - l ≤ n
    /// - modulus is valid (should be prime, > 2^24)
    ///
    /// Returns Ok(()) if valid, Err otherwise.
    pub fn validate(&self) -> Result<(), Error> {
        if self.m == 0 {
            return Err(Error::Ffi("R1CS must have at least one constraint".to_string()));
        }
        
        if self.n == 0 {
            return Err(Error::Ffi("R1CS witness size must be > 0".to_string()));
        }
        
        if self.l > self.n {
            return Err(Error::Ffi(format!(
                "Public inputs count {} cannot exceed witness size {}",
                self.l, self.n
            )));
        }
        
        if self.modulus < (1 << 24) {
            return Err(Error::Ffi(format!(
                "Modulus {} too small, should be > 2^24 for security",
                self.modulus
            )));
        }
        
        // Matrix dimension checks (already done in constructor, but double-check)
        if self.a.rows() != self.m || self.a.cols() != self.n {
            return Err(Error::Ffi("Matrix A dimensions mismatch".to_string()));
        }
        if self.b.rows() != self.m || self.b.cols() != self.n {
            return Err(Error::Ffi("Matrix B dimensions mismatch".to_string()));
        }
        if self.c.rows() != self.m || self.c.cols() != self.n {
            return Err(Error::Ffi("Matrix C dimensions mismatch".to_string()));
        }
        
        Ok(())
    }
    
    /// Get number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.m
    }
    
    /// Get witness size.
    pub fn witness_size(&self) -> usize {
        self.n
    }
    
    /// Get number of public inputs.
    pub fn num_public_inputs(&self) -> usize {
        self.l
    }
    
    /// Get total number of non-zero entries across all matrices.
    pub fn total_nnz(&self) -> usize {
        self.a.nnz() + self.b.nnz() + self.c.nnz()
    }
    
    /// Get density (percentage of non-zero entries).
    ///
    /// Returns value in [0.0, 1.0].
    pub fn density(&self) -> f64 {
        let total_entries = 3 * self.m * self.n;
        if total_entries == 0 {
            return 0.0;
        }
        self.total_nnz() as f64 / total_entries as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    fn create_multiplication_gate() -> R1CS {
        // TV-R1CS-1: Simple multiplication gate a * b = c
        // Witness: z = [1, 7, 13, 91]
        // Constraint: a * b = c
        
        let modulus = 17592186044417; // 2^44 + 1
        
        // A = [0, 1, 0, 0] (select a=z[1])
        let a_matrix = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
        
        // B = [0, 0, 1, 0] (select b=z[2])
        let b_matrix = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
        
        // C = [0, 0, 0, 1] (select c=z[3])
        let c_matrix = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);
        
        R1CS::new(1, 4, 2, a_matrix, b_matrix, c_matrix, modulus)
    }
    
    fn create_two_multiplications() -> R1CS {
        // TV-R1CS-2: Two multiplication gates
        // Circuit: a*b=c, c*d=e
        // Witness: z = [1, 2, 3, 6, 4, 24]
        
        let modulus = 17592186044417;
        
        // A = [[0, 1, 0, 0, 0, 0],   (select a=z[1])
        //      [0, 0, 0, 1, 0, 0]]   (select c=z[3])
        let a_rows = vec![
            vec![0, 1, 0, 0, 0, 0],
            vec![0, 0, 0, 1, 0, 0],
        ];
        let a_matrix = SparseMatrix::from_dense(&a_rows);
        
        // B = [[0, 0, 1, 0, 0, 0],   (select b=z[2])
        //      [0, 0, 0, 0, 1, 0]]   (select d=z[4])
        let b_rows = vec![
            vec![0, 0, 1, 0, 0, 0],
            vec![0, 0, 0, 0, 1, 0],
        ];
        let b_matrix = SparseMatrix::from_dense(&b_rows);
        
        // C = [[0, 0, 0, 1, 0, 0],   (select c=z[3])
        //      [0, 0, 0, 0, 0, 1]]   (select e=z[5])
        let c_rows = vec![
            vec![0, 0, 0, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 1],
        ];
        let c_matrix = SparseMatrix::from_dense(&c_rows);
        
        R1CS::new(2, 6, 2, a_matrix, b_matrix, c_matrix, modulus)
    }
    
    #[test]
    fn test_multiplication_gate_satisfied() {
        let r1cs = create_multiplication_gate();
        
        // Valid witness: 7 * 13 = 91
        let witness = vec![1, 7, 13, 91];
        assert!(r1cs.is_satisfied(&witness));
    }
    
    #[test]
    fn test_multiplication_gate_not_satisfied() {
        let r1cs = create_multiplication_gate();
        
        // Invalid witness: 7 * 13 = 91 ≠ 90
        let witness = vec![1, 7, 13, 90];
        assert!(!r1cs.is_satisfied(&witness));
    }
    
    #[test]
    fn test_two_multiplications_satisfied() {
        let r1cs = create_two_multiplications();
        
        // Valid witness: 2*3=6, 6*4=24
        let witness = vec![1, 2, 3, 6, 4, 24];
        assert!(r1cs.is_satisfied(&witness));
    }
    
    #[test]
    fn test_two_multiplications_first_fails() {
        let r1cs = create_two_multiplications();
        
        // First constraint fails: 2*3=6 ≠ 7
        let witness = vec![1, 2, 3, 7, 4, 24];
        assert!(!r1cs.is_satisfied(&witness));
    }
    
    #[test]
    fn test_two_multiplications_second_fails() {
        let r1cs = create_two_multiplications();
        
        // Second constraint fails: 6*4=24 ≠ 25
        let witness = vec![1, 2, 3, 6, 4, 25];
        assert!(!r1cs.is_satisfied(&witness));
    }
    
    #[test]
    fn test_public_inputs() {
        let r1cs = create_multiplication_gate();
        let witness = vec![1, 7, 13, 91];
        
        let public = r1cs.public_inputs(&witness);
        assert_eq!(public, &[1, 7]); // First 2 elements
    }
    
    #[test]
    fn test_validate_valid() {
        let r1cs = create_multiplication_gate();
        assert!(r1cs.validate().is_ok());
    }
    
    #[test]
    fn test_validate_small_modulus() {
        let a = SparseMatrix::from_dense(&vec![vec![1]]);
        let b = SparseMatrix::from_dense(&vec![vec![1]]);
        let c = SparseMatrix::from_dense(&vec![vec![1]]);
        
        let r1cs = R1CS::new(1, 1, 0, a, b, c, 100); // modulus too small
        
        let result = r1cs.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }
    
    #[test]
    fn test_density() {
        let r1cs = create_multiplication_gate();
        
        // 3 matrices × 1 row × 4 cols = 12 total entries
        // 3 non-zero entries total (one per matrix)
        // Density = 3/12 = 0.25 = 25%
        let density = r1cs.density();
        assert!((density - 0.25).abs() < 1e-6);
    }
    
    #[test]
    fn test_total_nnz() {
        let r1cs = create_multiplication_gate();
        assert_eq!(r1cs.total_nnz(), 3); // One non-zero per matrix
        
        let r1cs2 = create_two_multiplications();
        assert_eq!(r1cs2.total_nnz(), 6); // Two non-zeros per matrix
    }
    
    #[test]
    fn test_getters() {
        let r1cs = create_multiplication_gate();
        
        assert_eq!(r1cs.num_constraints(), 1);
        assert_eq!(r1cs.witness_size(), 4);
        assert_eq!(r1cs.num_public_inputs(), 2);
    }
    
    #[test]
    fn test_modular_arithmetic() {
        // Test constraint with modular reduction
        let modulus = 101; // Small prime
        
        // Constraint: a * b = c mod 101
        // Let a=50, b=3, c=150 mod 101 = 49
        
        let a_matrix = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);
        let b_matrix = SparseMatrix::from_dense(&vec![vec![0, 0, 1, 0]]);
        let c_matrix = SparseMatrix::from_dense(&vec![vec![0, 0, 0, 1]]);
        
        let r1cs = R1CS::new(1, 4, 1, a_matrix, b_matrix, c_matrix, modulus);
        
        let witness = vec![1, 50, 3, 49]; // 50*3 = 150 ≡ 49 (mod 101)
        assert!(r1cs.is_satisfied(&witness));
        
        let witness_wrong = vec![1, 50, 3, 50]; // 50*3 ≠ 50 (mod 101)
        assert!(!r1cs.is_satisfied(&witness_wrong));
    }
    
    #[test]
    #[should_panic(expected = "Witness length")]
    fn test_wrong_witness_length() {
        let r1cs = create_multiplication_gate();
        let witness = vec![1, 7, 13]; // Too short
        r1cs.is_satisfied(&witness);
    }
    
    #[test]
    #[should_panic(expected = "Matrix A must have m rows")]
    fn test_matrix_dimension_mismatch_rows() {
        let a = SparseMatrix::from_dense(&vec![vec![1, 0]]);
        let b = SparseMatrix::from_dense(&vec![vec![1, 0], vec![0, 1]]);
        let c = SparseMatrix::from_dense(&vec![vec![1, 0], vec![0, 1]]);
        
        R1CS::new(2, 2, 0, a, b, c, 1000); // A has wrong number of rows
    }
    
    #[test]
    #[should_panic(expected = "Matrix B must have n columns")]
    fn test_matrix_dimension_mismatch_cols() {
        let a = SparseMatrix::from_dense(&vec![vec![1, 0]]);
        let b = SparseMatrix::from_dense(&vec![vec![1]]);
        let c = SparseMatrix::from_dense(&vec![vec![1, 0]]);
        
        R1CS::new(1, 2, 0, a, b, c, 1000); // B has wrong number of cols
    }
}
