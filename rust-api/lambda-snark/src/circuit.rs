//! Circuit Builder API for ergonomic R1CS construction
//!
//! This module provides a high-level DSL for constructing R1CS instances
//! without manually managing sparse matrices. Users can allocate variables,
//! add constraints via linear combinations, and build the final R1CS.
//!
//! # Example
//!
//! ```
//! use lambda_snark::{CircuitBuilder, R1CS};
//!
//! // Build circuit for multiplication gate: a * b = c
//! let modulus = 17592186044417;  // 2^44 + 1
//! let mut builder = CircuitBuilder::new(modulus);
//!
//! // Allocate variables (indices: 0, 1, 2, 3)
//! let one = builder.alloc_var();   // z_0 = 1 (constant)
//! let a = builder.alloc_var();     // z_1 = a
//! let b = builder.alloc_var();     // z_2 = b
//! let c = builder.alloc_var();     // z_3 = c
//!
//! // Set public inputs (first 2 variables: constant and result)
//! builder.set_public_inputs(2);
//!
//! // Add constraint: a * b = c
//! // (1*a) * (1*b) = (1*c)
//! builder.add_constraint(
//!     vec![(a, 1)],  // A: select variable a with coefficient 1
//!     vec![(b, 1)],  // B: select variable b with coefficient 1
//!     vec![(c, 1)],  // C: select variable c with coefficient 1
//! );
//!
//! // Build final R1CS
//! let r1cs: R1CS = builder.build();
//!
//! // Validate with witness
//! let witness = vec![1, 7, 13, 91];  // 7 * 13 = 91
//! assert!(r1cs.is_satisfied(&witness));
//! ```
//!
//! # Linear Combinations
//!
//! Constraints support arbitrary linear combinations:
//! ```text
//! (Σ_j a_j * z_j) * (Σ_k b_k * z_k) = (Σ_l c_l * z_l)
//! ```
//!
//! Example: `(2*a + 3*b) * (c) = (d + 5*e)`
//! ```no_run
//! # use lambda_snark::CircuitBuilder;
//! # let modulus = 17592186044417;
//! # let mut builder = CircuitBuilder::new(modulus);
//! # let a = 0; let b = 1; let c = 2; let d = 3; let e = 4;
//! builder.add_constraint(
//!     vec![(a, 2), (b, 3)],  // A: 2*a + 3*b
//!     vec![(c, 1)],          // B: c
//!     vec![(d, 1), (e, 5)],  // C: d + 5*e
//! );
//! ```

use crate::{SparseMatrix, R1CS};
use std::collections::HashMap;

/// Circuit builder for constructing R1CS instances
///
/// Provides ergonomic API for:
/// - Variable allocation (`alloc_var()`)
/// - Constraint addition (`add_constraint()`)
/// - Public input specification (`set_public_inputs()`)
/// - R1CS generation (`build()`)
///
/// # Workflow
///
/// 1. Create builder with modulus
/// 2. Allocate variables (returns indices 0, 1, 2, ...)
/// 3. Mark first `l` variables as public inputs
/// 4. Add constraints as (A, B, C) linear combination triples
/// 5. Build final R1CS (converts to sparse matrices)
///
/// # Constraints
///
/// Each constraint is a rank-1 equation:
/// ```text
/// (A * witness) * (B * witness) = (C * witness)
/// ```
///
/// Where A, B, C are sparse row vectors represented as `Vec<(var_index, coefficient)>`.
#[derive(Debug)]
pub struct CircuitBuilder {
    /// Constraints as triples of linear combinations
    /// Format: (A_terms, B_terms, C_terms)
    /// Each term: Vec<(variable_index, coefficient)>
    constraints: Vec<(Vec<(usize, u64)>, Vec<(usize, u64)>, Vec<(usize, u64)>)>,

    /// Total number of variables allocated (including public inputs)
    num_vars: usize,

    /// Number of public inputs (first l variables)
    num_public: usize,

    /// Field modulus q
    modulus: u64,
}

impl CircuitBuilder {
    /// Create new circuit builder with field modulus
    ///
    /// # Arguments
    ///
    /// * `modulus` - Prime field modulus q (should be > 2^24 for security)
    ///
    /// # Example
    ///
    /// ```
    /// use lambda_snark::CircuitBuilder;
    ///
    /// let modulus = 17592186044417;  // 2^44 + 1
    /// let builder = CircuitBuilder::new(modulus);
    /// ```
    pub fn new(modulus: u64) -> Self {
        Self {
            constraints: Vec::new(),
            num_vars: 0,
            num_public: 0,
            modulus,
        }
    }

    /// Allocate new variable, returns its index
    ///
    /// Variables are indexed sequentially: 0, 1, 2, ...
    /// Convention: z_0 = 1 (constant), then z_1, z_2, ... are circuit-specific
    ///
    /// # Returns
    ///
    /// Index of newly allocated variable
    ///
    /// # Example
    ///
    /// ```
    /// # use lambda_snark::CircuitBuilder;
    /// # let modulus = 17592186044417;
    /// # let mut builder = CircuitBuilder::new(modulus);
    /// let one = builder.alloc_var();  // index 0
    /// let a = builder.alloc_var();    // index 1
    /// let b = builder.alloc_var();    // index 2
    /// assert_eq!(one, 0);
    /// assert_eq!(a, 1);
    /// assert_eq!(b, 2);
    /// ```
    pub fn alloc_var(&mut self) -> usize {
        let index = self.num_vars;
        self.num_vars += 1;
        index
    }

    /// Mark first `l` variables as public inputs
    ///
    /// By convention, public inputs should be allocated first (indices 0..l).
    /// Verifier will receive these values to check proof validity.
    ///
    /// # Arguments
    ///
    /// * `l` - Number of public inputs (must be ≤ num_vars)
    ///
    /// # Panics
    ///
    /// Panics if `l > num_vars` (cannot mark more public inputs than variables)
    ///
    /// # Example
    ///
    /// ```
    /// # use lambda_snark::CircuitBuilder;
    /// # let modulus = 17592186044417;
    /// # let mut builder = CircuitBuilder::new(modulus);
    /// let one = builder.alloc_var();
    /// let result = builder.alloc_var();
    /// builder.set_public_inputs(2);  // First 2 variables are public
    /// ```
    pub fn set_public_inputs(&mut self, l: usize) {
        if l > self.num_vars {
            panic!(
                "Cannot set {} public inputs with only {} variables allocated",
                l, self.num_vars
            );
        }
        self.num_public = l;
    }

    /// Add constraint to circuit
    ///
    /// Adds rank-1 constraint: (A * witness) * (B * witness) = (C * witness)
    ///
    /// # Arguments
    ///
    /// * `a` - Linear combination for left operand (variable_index, coefficient pairs)
    /// * `b` - Linear combination for right operand
    /// * `c` - Linear combination for result
    ///
    /// # Example
    ///
    /// ```
    /// # use lambda_snark::CircuitBuilder;
    /// # let modulus = 17592186044417;
    /// # let mut builder = CircuitBuilder::new(modulus);
    /// # let a = builder.alloc_var();
    /// # let b = builder.alloc_var();
    /// # let c = builder.alloc_var();
    ///
    /// // Simple multiplication: a * b = c
    /// builder.add_constraint(
    ///     vec![(a, 1)],
    ///     vec![(b, 1)],
    ///     vec![(c, 1)],
    /// );
    ///
    /// // Linear combination: (2*a + 3*b) * c = d
    /// # let d = builder.alloc_var();
    /// builder.add_constraint(
    ///     vec![(a, 2), (b, 3)],  // 2*a + 3*b
    ///     vec![(c, 1)],          // c
    ///     vec![(d, 1)],          // d
    /// );
    /// ```
    pub fn add_constraint(
        &mut self,
        a: Vec<(usize, u64)>,
        b: Vec<(usize, u64)>,
        c: Vec<(usize, u64)>,
    ) {
        self.constraints.push((a, b, c));
    }

    /// Build final R1CS instance
    ///
    /// Converts accumulated constraints into sparse matrices (CSR format).
    /// Validates that all variable indices are within bounds.
    ///
    /// # Returns
    ///
    /// R1CS instance ready for proving/verification
    ///
    /// # Panics
    ///
    /// Panics if any constraint references variable index ≥ num_vars
    ///
    /// # Example
    ///
    /// ```
    /// # use lambda_snark::CircuitBuilder;
    /// # let modulus = 17592186044417;
    /// # let mut builder = CircuitBuilder::new(modulus);
    /// # let a = builder.alloc_var();
    /// # let b = builder.alloc_var();
    /// # let c = builder.alloc_var();
    /// # builder.add_constraint(vec![(a, 1)], vec![(b, 1)], vec![(c, 1)]);
    ///
    /// let r1cs = builder.build();
    /// assert_eq!(r1cs.num_constraints(), 1);
    /// assert_eq!(r1cs.witness_size(), 3);
    /// ```
    pub fn build(self) -> R1CS {
        let m = self.constraints.len();
        let n = self.num_vars;
        let l = self.num_public;

        // Build sparse matrices via HashMap (flexible insertion)
        let mut a_map: HashMap<(usize, usize), u64> = HashMap::new();
        let mut b_map: HashMap<(usize, usize), u64> = HashMap::new();
        let mut c_map: HashMap<(usize, usize), u64> = HashMap::new();

        for (constraint_idx, (a_terms, b_terms, c_terms)) in self.constraints.iter().enumerate() {
            // Populate A matrix (row = constraint_idx)
            for &(var_idx, coeff) in a_terms {
                if var_idx >= n {
                    panic!(
                        "Constraint {} references invalid variable index {} (num_vars={})",
                        constraint_idx, var_idx, n
                    );
                }
                if coeff != 0 {
                    *a_map.entry((constraint_idx, var_idx)).or_insert(0) =
                        (a_map.get(&(constraint_idx, var_idx)).unwrap_or(&0) + coeff)
                            % self.modulus;
                }
            }

            // Populate B matrix
            for &(var_idx, coeff) in b_terms {
                if var_idx >= n {
                    panic!(
                        "Constraint {} references invalid variable index {} (num_vars={})",
                        constraint_idx, var_idx, n
                    );
                }
                if coeff != 0 {
                    *b_map.entry((constraint_idx, var_idx)).or_insert(0) =
                        (b_map.get(&(constraint_idx, var_idx)).unwrap_or(&0) + coeff)
                            % self.modulus;
                }
            }

            // Populate C matrix
            for &(var_idx, coeff) in c_terms {
                if var_idx >= n {
                    panic!(
                        "Constraint {} references invalid variable index {} (num_vars={})",
                        constraint_idx, var_idx, n
                    );
                }
                if coeff != 0 {
                    *c_map.entry((constraint_idx, var_idx)).or_insert(0) =
                        (c_map.get(&(constraint_idx, var_idx)).unwrap_or(&0) + coeff)
                            % self.modulus;
                }
            }
        }

        // Convert to sparse matrices (CSR format)
        let a_matrix = SparseMatrix::from_map(m, n, &a_map);
        let b_matrix = SparseMatrix::from_map(m, n, &b_map);
        let c_matrix = SparseMatrix::from_map(m, n, &c_map);

        R1CS::new(m, n, l, a_matrix, b_matrix, c_matrix, self.modulus)
    }

    /// Get current number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Get current number of variables
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Get number of public inputs
    pub fn num_public_inputs(&self) -> usize {
        self.num_public
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODULUS: u64 = 17592186044417; // 2^44 + 1

    #[test]
    fn test_empty_circuit() {
        let builder = CircuitBuilder::new(MODULUS);
        assert_eq!(builder.num_constraints(), 0);
        assert_eq!(builder.num_vars(), 0);
        assert_eq!(builder.num_public_inputs(), 0);
    }

    #[test]
    fn test_alloc_var_sequential() {
        let mut builder = CircuitBuilder::new(MODULUS);
        assert_eq!(builder.alloc_var(), 0);
        assert_eq!(builder.alloc_var(), 1);
        assert_eq!(builder.alloc_var(), 2);
        assert_eq!(builder.num_vars(), 3);
    }

    #[test]
    fn test_set_public_inputs() {
        let mut builder = CircuitBuilder::new(MODULUS);
        builder.alloc_var();
        builder.alloc_var();
        builder.set_public_inputs(2);
        assert_eq!(builder.num_public_inputs(), 2);
    }

    #[test]
    #[should_panic(expected = "Cannot set 3 public inputs with only 2 variables")]
    fn test_set_public_inputs_too_many() {
        let mut builder = CircuitBuilder::new(MODULUS);
        builder.alloc_var();
        builder.alloc_var();
        builder.set_public_inputs(3); // More than num_vars
    }

    #[test]
    fn test_add_constraint() {
        let mut builder = CircuitBuilder::new(MODULUS);
        let a = builder.alloc_var();
        let b = builder.alloc_var();
        let c = builder.alloc_var();

        builder.add_constraint(vec![(a, 1)], vec![(b, 1)], vec![(c, 1)]);

        assert_eq!(builder.num_constraints(), 1);
    }

    #[test]
    fn test_multiplication_gate() {
        // TV-R1CS-1: a * b = c
        let mut builder = CircuitBuilder::new(MODULUS);

        let one = builder.alloc_var(); // z_0 = 1
        assert_eq!(one, 0, "First allocation must bind the constant 1");
        let a = builder.alloc_var(); // z_1 = 7
        let b = builder.alloc_var(); // z_2 = 13
        let c = builder.alloc_var(); // z_3 = 91

        builder.set_public_inputs(2); // Public: one, result

        builder.add_constraint(vec![(a, 1)], vec![(b, 1)], vec![(c, 1)]);

        let r1cs = builder.build();

        assert_eq!(r1cs.num_constraints(), 1);
        assert_eq!(r1cs.witness_size(), 4);
        assert_eq!(r1cs.num_public_inputs(), 2);

        // Validate with witness
        let witness = vec![1, 7, 13, 91];
        assert!(r1cs.is_satisfied(&witness));

        // Invalid witness
        let witness_bad = vec![1, 7, 13, 90];
        assert!(!r1cs.is_satisfied(&witness_bad));
    }

    #[test]
    fn test_two_multiplications() {
        // TV-R1CS-2: a*b=c, c*d=e
        let mut builder = CircuitBuilder::new(MODULUS);

        let one = builder.alloc_var(); // z_0 = 1
        assert_eq!(one, 0, "First allocation must bind the constant 1");
        let a = builder.alloc_var(); // z_1 = 2
        let b = builder.alloc_var(); // z_2 = 3
        let c = builder.alloc_var(); // z_3 = 6
        let d = builder.alloc_var(); // z_4 = 4
        let e = builder.alloc_var(); // z_5 = 24

        builder.set_public_inputs(2);

        // Constraint 1: a * b = c
        builder.add_constraint(vec![(a, 1)], vec![(b, 1)], vec![(c, 1)]);

        // Constraint 2: c * d = e
        builder.add_constraint(vec![(c, 1)], vec![(d, 1)], vec![(e, 1)]);

        let r1cs = builder.build();

        assert_eq!(r1cs.num_constraints(), 2);
        assert_eq!(r1cs.witness_size(), 6);

        let witness = vec![1, 2, 3, 6, 4, 24];
        assert!(r1cs.is_satisfied(&witness));
    }

    #[test]
    fn test_linear_combination() {
        // (2*a + 3*b) * c = d
        let mut builder = CircuitBuilder::new(MODULUS);

        let a = builder.alloc_var(); // 5
        let b = builder.alloc_var(); // 7
        let c = builder.alloc_var(); // 11
        let d = builder.alloc_var(); // (2*5 + 3*7) * 11 = 31 * 11 = 341

        builder.add_constraint(
            vec![(a, 2), (b, 3)], // 2*a + 3*b = 10 + 21 = 31
            vec![(c, 1)],         // c = 11
            vec![(d, 1)],         // d = 341
        );

        let r1cs = builder.build();

        let witness = vec![5, 7, 11, 341];
        assert!(r1cs.is_satisfied(&witness));
    }

    #[test]
    fn test_addition_via_multiplication_by_one() {
        // Addition: a + b = c (via (a + b) * 1 = c)
        let mut builder = CircuitBuilder::new(MODULUS);

        let one = builder.alloc_var(); // 1
        let a = builder.alloc_var(); // 10
        let b = builder.alloc_var(); // 23
        let c = builder.alloc_var(); // 33

        builder.add_constraint(
            vec![(a, 1), (b, 1)], // a + b
            vec![(one, 1)],       // 1
            vec![(c, 1)],         // c
        );

        let r1cs = builder.build();

        let witness = vec![1, 10, 23, 33];
        assert!(r1cs.is_satisfied(&witness));
    }

    #[test]
    #[should_panic(expected = "references invalid variable index")]
    fn test_invalid_variable_index() {
        let mut builder = CircuitBuilder::new(MODULUS);
        let a = builder.alloc_var();

        // Reference non-existent variable 5
        builder.add_constraint(
            vec![(a, 1)],
            vec![(5, 1)], // Out of bounds
            vec![(a, 1)],
        );

        builder.build(); // Should panic
    }

    #[test]
    fn test_zero_coefficient_ignored() {
        let mut builder = CircuitBuilder::new(MODULUS);
        let a = builder.alloc_var(); // 0: 7
        let b = builder.alloc_var(); // 1: 13
        let c = builder.alloc_var(); // 2: 91

        // Add zero coefficient (should be ignored in sparse matrix)
        builder.add_constraint(
            vec![(a, 1), (b, 0)], // b has zero coeff, so A*witness = 7
            vec![(b, 1)],         // B*witness = 13
            vec![(c, 1)],         // C*witness = 91
        );

        let r1cs = builder.build();

        // Witness: a=7, b=13, c=91
        // Check: 7 * 13 = 91 ✓
        let witness = vec![7, 13, 91];
        assert!(r1cs.is_satisfied(&witness));
    }

    #[test]
    fn test_modular_reduction_in_coefficients() {
        // Test that coefficients are reduced modulo q
        let modulus = 101;
        let mut builder = CircuitBuilder::new(modulus);

        let a = builder.alloc_var(); // 50
        let b = builder.alloc_var(); // 3
        let c = builder.alloc_var(); // 49 (since 50*3 = 150 ≡ 49 mod 101)

        builder.add_constraint(vec![(a, 1)], vec![(b, 1)], vec![(c, 1)]);

        let r1cs = builder.build();

        let witness = vec![50, 3, 49];
        assert!(r1cs.is_satisfied(&witness));
    }

    #[test]
    fn test_empty_linear_combination() {
        // Edge case: empty linear combination (evaluates to 0)
        let mut builder = CircuitBuilder::new(MODULUS);
        let a = builder.alloc_var();

        // 0 * a = 0
        builder.add_constraint(
            vec![], // Empty = 0
            vec![(a, 1)],
            vec![], // Empty = 0
        );

        let r1cs = builder.build();

        let witness = vec![42]; // Any value for a
        assert!(r1cs.is_satisfied(&witness));
    }
}
