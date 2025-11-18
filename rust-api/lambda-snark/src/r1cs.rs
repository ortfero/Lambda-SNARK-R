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

#[cfg(feature = "fft-ntt")]
use crate::ntt::{compute_root_of_unity, ntt_inverse};
#[cfg(feature = "fft-ntt")]
use lambda_snark_core::{NTT_MODULUS, NTT_PRIMITIVE_ROOT};

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
        assert_eq!(witness.len(), self.n, "Witness length must equal n");
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
            return Err(Error::Ffi(
                "R1CS must have at least one constraint".to_string(),
            ));
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

    /// Compute constraint evaluations: (Az), (Bz), (Cz)
    ///
    /// Returns vectors a, b, c where:
    /// - a[i] = (Az)_i = Σ_j A[i,j] · z[j]
    /// - b[i] = (Bz)_i = Σ_j B[i,j] · z[j]
    /// - c[i] = (Cz)_i = Σ_j C[i,j] · z[j]
    ///
    /// These are the evaluations of constraint polynomials A_z, B_z, C_z
    /// at domain points H = {ω^0, ω^1, ..., ω^{m-1}}.
    ///
    /// # Arguments
    ///
    /// * `witness` - Witness vector z ∈ F_q^n
    ///
    /// # Returns
    ///
    /// (a_evals, b_evals, c_evals) where each is `Vec<u64>` of length m
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lambda_snark::{R1CS, SparseMatrix};
    /// # let modulus = 17592186044417;
    /// # let a = SparseMatrix::from_dense(&vec![vec![0,1,0,0]]);
    /// # let b = SparseMatrix::from_dense(&vec![vec![0,0,1,0]]);
    /// # let c = SparseMatrix::from_dense(&vec![vec![0,0,0,1]]);
    /// # let r1cs = R1CS::new(1, 4, 2, a, b, c, modulus);
    /// let witness = vec![1, 7, 13, 91];
    /// let (a_evals, b_evals, c_evals) = r1cs.compute_constraint_evals(&witness);
    ///
    /// // For multiplication gate: a_evals[0]=7, b_evals[0]=13, c_evals[0]=91
    /// assert_eq!(a_evals[0], 7);
    /// assert_eq!(b_evals[0], 13);
    /// assert_eq!(c_evals[0], 91);
    /// ```
    pub fn compute_constraint_evals(&self, witness: &[u64]) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        assert_eq!(witness.len(), self.n, "Witness length must equal n");

        let a_evals = self.a.mul_vec(witness, self.modulus);
        let b_evals = self.b.mul_vec(witness, self.modulus);
        let c_evals = self.c.mul_vec(witness, self.modulus);

        (a_evals, b_evals, c_evals)
    }

    /// Compute quotient polynomial Q(X) coefficients
    ///
    /// Computes Q(X) = (A_z(X) · B_z(X) - C_z(X)) / Z_H(X)
    ///
    /// Where:
    /// - A_z, B_z, C_z are Lagrange interpolations of constraint evaluations
    /// - Z_H(X) = X^m - 1 (vanishing polynomial for domain H)
    ///
    /// **Note**: This is naïve O(m²) implementation using direct Lagrange basis.
    /// For production (m > 10^4), use NTT-based O(m log m) version (M5.2).
    ///
    /// # Arguments
    ///
    /// * `witness` - Witness vector z ∈ F_q^n
    ///
    /// # Returns
    ///
    /// Ok(q_coeffs) where q_coeffs is `Vec<u64>` of quotient polynomial coefficients
    /// Err if witness doesn't satisfy R1CS (Q would not be polynomial)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lambda_snark::{R1CS, SparseMatrix};
    /// # let modulus = 17592186044417;
    /// # let a = SparseMatrix::from_dense(&vec![vec![0,1,0,0]]);
    /// # let b = SparseMatrix::from_dense(&vec![vec![0,0,1,0]]);
    /// # let c = SparseMatrix::from_dense(&vec![vec![0,0,0,1]]);
    /// # let r1cs = R1CS::new(1, 4, 2, a, b, c, modulus);
    /// let witness = vec![1, 7, 13, 91];
    /// let q_coeffs = r1cs.compute_quotient_poly(&witness).unwrap();
    ///
    /// // Q(X) should have degree < m (since division by Z_H is exact)
    /// assert!(q_coeffs.len() <= r1cs.num_constraints());
    /// ```

    /// Evaluate polynomial at point x using Horner's method.
    ///
    /// # Arguments
    ///
    /// * `poly` - Polynomial coefficients in ascending degree order [a₀, a₁, ..., aₙ]
    /// * `x` - Evaluation point in F_q
    ///
    /// # Returns
    ///
    /// f(x) mod modulus where f(X) = a₀ + a₁X + ... + aₙXⁿ
    ///
    /// # Example
    ///
    /// ```ignore
    /// let poly = vec![2, 3, 1]; // f(X) = 2 + 3X + X²
    /// let r1cs = R1CS::new(..., modulus: 97);
    /// assert_eq!(r1cs.eval_poly(&poly, 0), 2);   // f(0) = 2
    /// assert_eq!(r1cs.eval_poly(&poly, 1), 6);   // f(1) = 2+3+1 = 6
    /// assert_eq!(r1cs.eval_poly(&poly, 2), 12);  // f(2) = 2+6+4 = 12
    /// ```
    pub fn eval_poly(&self, poly: &[u64], x: u64) -> u64 {
        let mut result = 0u64;
        let mut power = 1u64;

        for &coeff in poly {
            let term = ((coeff as u128 * power as u128) % self.modulus as u128) as u64;
            result = ((result as u128 + term as u128) % self.modulus as u128) as u64;
            power = ((power as u128 * x as u128) % self.modulus as u128) as u64;
        }

        result
    }

    /// Check if NTT interpolation should be used (vs baseline Lagrange).
    ///
    /// NTT requires:
    /// 1. m is power of 2
    /// 2. modulus is NTT_MODULUS
    /// 3. feature "fft-ntt" is enabled
    ///
    /// # Returns
    ///
    /// `true` if NTT path will be taken, `false` for baseline
    #[cfg(feature = "fft-ntt")]
    pub fn should_use_ntt(&self) -> bool {
        use lambda_snark_core::NTT_MODULUS;
        self.m.is_power_of_two() && self.modulus == NTT_MODULUS
    }

    #[cfg(not(feature = "fft-ntt"))]
    pub fn should_use_ntt(&self) -> bool {
        false
    }

    /// Evaluate vanishing polynomial Z_H(x) at point x.
    ///
    /// Domain H depends on interpolation method:
    /// - **NTT path**: H = {1, ω, ω², ..., ω^{m-1}} (roots of unity)
    ///   → Z_H(X) = X^m - 1
    /// - **Baseline path**: H = {0, 1, 2, ..., m-1} (integers)
    ///   → Z_H(X) = X(X-1)(X-2)...(X-(m-1))
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate Z_H
    ///
    /// # Returns
    ///
    /// Z_H(x) mod modulus
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lambda_snark::{R1CS, SparseMatrix};
    /// # let r1cs = R1CS::new(4, 10, 1,
    /// #     SparseMatrix::from_dense(&vec![vec![0u64; 10]; 4]),
    /// #     SparseMatrix::from_dense(&vec![vec![0u64; 10]; 4]),
    /// #     SparseMatrix::from_dense(&vec![vec![0u64; 10]; 4]),
    /// #     18446744069414584321);
    /// // For NTT: Z_H(ω) = ω^4 - 1 = 0 (ω is 4th root of unity)
    /// // For baseline: Z_H(2) = 2·1·0·(-1) = 0
    /// ```
    pub fn eval_vanishing(&self, x: u64) -> u64 {
        if self.should_use_ntt() {
            // NTT path: Z_H(X) = X^m - 1
            // Compute x^m mod modulus
            let x_pow_m = mod_pow(x, self.m as u64, self.modulus);
            if x_pow_m >= 1 {
                x_pow_m - 1
            } else {
                self.modulus - (1 - x_pow_m)
            }
        } else {
            // Baseline path: Z_H(X) = ∏(X - i) for i=0..m-1
            let mut result = 1u64;
            for i in 0..self.m {
                let diff = if x >= (i as u64) {
                    x - (i as u64)
                } else {
                    self.modulus - ((i as u64) - x)
                };
                result = ((result as u128 * diff as u128) % self.modulus as u128) as u64;
            }
            result
        }
    }

    /// Compute quotient polynomial Q(X) = (A_z(X) · B_z(X) - C_z(X)) / Z_H(X).
    ///
    /// This is the core operation in R1CS proving. The quotient polynomial exists
    /// if and only if the witness satisfies all R1CS constraints.
    ///
    /// # Arguments
    ///
    /// * `witness` - Full witness vector z ∈ F_q^n (including public inputs)
    ///
    /// # Returns
    ///
    /// * `Ok(coeffs)` - Coefficients of Q(X) in ascending degree order
    /// * `Err` - If witness doesn't satisfy R1CS (exact division fails)
    ///
    /// # Complexity
    ///
    /// O(m²) time using naïve Lagrange interpolation.
    /// For m > 10^4, use NTT-based version (M5.1).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lambda_snark::{R1CS, SparseMatrix};
    /// # let modulus = 17592186044417;
    /// # let a = SparseMatrix::from_dense(&vec![vec![0,1,0,0]]);
    /// # let b = SparseMatrix::from_dense(&vec![vec![0,0,1,0]]);
    /// # let c = SparseMatrix::from_dense(&vec![vec![0,0,0,1]]);
    /// # let r1cs = R1CS::new(1, 4, 2, a, b, c, modulus);
    /// let witness = vec![1, 7, 13, 91]; // 7 * 13 = 91
    /// let q = r1cs.compute_quotient_poly(&witness).unwrap();
    /// assert!(q.len() <= r1cs.num_constraints());
    /// ```
    pub fn compute_quotient_poly(&self, witness: &[u64]) -> Result<Vec<u64>, Error> {
        // 1. Verify witness satisfies constraints
        if !self.is_satisfied(witness) {
            return Err(Error::Ffi(
                "Witness does not satisfy R1CS constraints".to_string(),
            ));
        }

        // 2. Compute constraint evaluations at domain points
        let (a_evals, b_evals, c_evals) = self.compute_constraint_evals(witness);

        // 3. Interpolate polynomials from evaluations
        //    Domain H = {0, 1, 2, ..., m-1}
        let a_poly = lagrange_interpolate(&a_evals, self.modulus);
        let b_poly = lagrange_interpolate(&b_evals, self.modulus);
        let c_poly = lagrange_interpolate(&c_evals, self.modulus);

        // 4. Compute A_z(X) · B_z(X)
        let ab_poly = poly_mul(&a_poly, &b_poly, self.modulus);

        // 5. Compute numerator: A_z(X) · B_z(X) - C_z(X)
        let numerator = poly_sub(&ab_poly, &c_poly, self.modulus);

        // 6. Divide by vanishing polynomial Z_H(X)
        //    Use domain-aware Z_H: X^m - 1 (NTT) or ∏(X-i) (baseline)
        let use_ntt = self.should_use_ntt();
        let quotient = poly_div_vanishing(&numerator, self.m, self.modulus, use_ntt)?;

        Ok(quotient)
    }
}

// ============================================================================
// Polynomial Helper Functions
// ============================================================================

/// Modular exponentiation: base^exp mod modulus
///
/// Uses binary exponentiation for O(log exp) performance.
fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }

    let mut result = 1u64;
    let mut base = base % modulus;
    let mut exp = exp;

    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp >>= 1;
    }

    result
}

/// Modular multiplicative inverse using Extended Euclidean Algorithm
///
/// Returns x such that (a * x) ≡ 1 (mod m)
/// Panics if a and m are not coprime.
fn mod_inverse(a: u64, m: u64) -> u64 {
    if a == 0 {
        panic!("Cannot compute inverse of 0");
    }

    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (m as i128, a as i128);

    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }

    if r > 1 {
        panic!("a={} is not invertible mod m={}", a, m);
    }

    if t < 0 {
        t += m as i128;
    }

    t as u64
}

/// NTT-friendly prime modulus (supports up to 2^13 = 8192 NTT)
/// q = 17592169062401 (prime), φ(q) = 2147481575 × 2^13
const NTT_FRIENDLY_MODULUS: u64 = 17592169062401;

/// Precomputed primitive roots of unity for NTT_FRIENDLY_MODULUS
/// Each entry: (order m, primitive m-th root ω) where ω^m ≡ 1 (mod q)
/// Generator g = 3
const ROOTS_OF_UNITY: &[(usize, u64)] = &[
    (4, 981206394875),      // 2^2
    (8, 4268641988953),     // 2^3
    (16, 9400386778549),    // 2^4
    (32, 15690227524213),   // 2^5
    (64, 8332322609789),    // 2^6
    (128, 9249819209096),   // 2^7
    (256, 5221410271124),   // 2^8
    (512, 9594533594163),   // 2^9
    (1024, 11016271016603), // 2^10
    (2048, 14373677444369), // 2^11
    (4096, 11176258803537), // 2^12
    (8192, 9037003627149),  // 2^13
];

/// Get primitive root of unity for domain size m
///
/// Returns ω such that ω^m ≡ 1 (mod modulus) and ω^(m/2) ≡ -1 (mod modulus)
///
/// # Strategy (Variant C - Hybrid)
/// 1. If modulus == NTT_FRIENDLY_MODULUS and m in precomputed table → O(1) lookup
/// 2. Otherwise → Fallback to sequential points 0,1,2,... (legacy behavior)
///
/// # Arguments
/// * `m` - Domain size (must be power of 2 for NTT)
/// * `modulus` - Field modulus
///
/// # Returns
/// Primitive m-th root of unity, or None if not available
fn get_ntt_root(m: usize, modulus: u64) -> Option<u64> {
    if modulus == NTT_FRIENDLY_MODULUS {
        // Fast path: precomputed roots
        ROOTS_OF_UNITY
            .iter()
            .find(|&&(order, _)| order == m)
            .map(|&(_, root)| root)
    } else {
        // Fallback: no NTT roots for non-standard modulus
        None
    }
}

/// Lagrange basis polynomial L_i(X) evaluated at domain points
///
/// For NTT-friendly modulus with precomputed roots, uses domain H = {1, ω, ω², ..., ω^(m-1)}
/// Otherwise falls back to H = {0, 1, 2, ..., m-1} (legacy, may fail for large m)
///
/// Computes: L_i(X) = Π_{j≠i} (X - ω^j) / (ω^i - ω^j)
///
/// Returns coefficients of L_i(X) as a polynomial.
///
/// **Note**: Naïve O(m²) implementation. For production, use NTT (M5.2).
///
/// # Arguments
///
/// * `i` - Index of Lagrange basis polynomial (0 ≤ i < m)
/// * `m` - Domain size |H| = m
/// * `modulus` - Field modulus q
///
/// # Returns
///
/// Coefficients of L_i(X), length m
fn lagrange_basis(i: usize, m: usize, modulus: u64) -> Vec<u64> {
    assert!(i < m, "Lagrange index i={} must be < m={}", i, m);

    // Try NTT roots first (Variant C - Hybrid)
    if let Some(omega) = get_ntt_root(m, modulus) {
        // Fast path: Use NTT domain {1, ω, ω², ..., ω^(m-1)}
        lagrange_basis_ntt(i, m, omega, modulus)
    } else {
        // Fallback: Sequential domain {0, 1, 2, ..., m-1}
        lagrange_basis_sequential(i, m, modulus)
    }
}

/// Lagrange basis using NTT roots domain
fn lagrange_basis_ntt(i: usize, m: usize, omega: u64, modulus: u64) -> Vec<u64> {
    // Domain points: x_j = ω^j for j = 0, 1, ..., m-1
    // L_i(X) = Π_{j≠i} (X - ω^j) / (ω^i - ω^j)

    let mut poly = vec![1u64];

    // Compute domain points
    let mut omega_powers = vec![1u64];
    for _ in 1..m {
        let prev = omega_powers.last().unwrap();
        omega_powers.push(((*prev as u128 * omega as u128) % modulus as u128) as u64);
    }

    // Multiply by (X - ω^j) for all j ≠ i
    for j in 0..m {
        if j == i {
            continue;
        }
        poly = poly_mul_linear(&poly, omega_powers[j], modulus);
    }

    // Compute denominator: Π_{j≠i} (ω^i - ω^j)
    let mut denom = 1u64;
    let omega_i = omega_powers[i];

    for j in 0..m {
        if j == i {
            continue;
        }

        let omega_j = omega_powers[j];
        let diff = if omega_i >= omega_j {
            (omega_i - omega_j) % modulus
        } else {
            modulus - ((omega_j - omega_i) % modulus)
        };

        denom = ((denom as u128 * diff as u128) % modulus as u128) as u64;
    }

    // Divide all coefficients by denominator
    let denom_inv = mod_inverse(denom, modulus);
    for coeff in poly.iter_mut() {
        *coeff = ((*coeff as u128 * denom_inv as u128) % modulus as u128) as u64;
    }

    poly.resize(m, 0);
    poly
}

/// Lagrange basis using sequential domain (legacy, may fail for large m)
fn lagrange_basis_sequential(i: usize, m: usize, modulus: u64) -> Vec<u64> {
    // Domain points: x_j = j for j = 0, 1, ..., m-1
    // L_i(X) = Π_{j≠i} (X - j) / (i - j)

    let mut poly = vec![1u64];

    // Multiply by (X - j) for all j ≠ i
    for j in 0..m {
        if j == i {
            continue;
        }

        // poly *= (X - j)
        poly = poly_mul_linear(&poly, j as u64, modulus);
    }

    // Compute denominator: Π_{j≠i} (i - j)
    let mut denom = 1u64;
    for j in 0..m {
        if j == i {
            continue;
        }

        let diff = if i >= j {
            ((i - j) as u64) % modulus
        } else {
            // i < j, so (i - j) is negative
            // In modular arithmetic: -x ≡ q - x (mod q)
            let abs_diff = ((j - i) as u64) % modulus;
            if abs_diff == 0 {
                0
            } else {
                modulus - abs_diff
            }
        };

        denom = ((denom as u128 * diff as u128) % modulus as u128) as u64;
    }

    // Divide all coefficients by denominator
    let denom_inv = mod_inverse(denom, modulus);
    for coeff in poly.iter_mut() {
        *coeff = ((*coeff as u128 * denom_inv as u128) % modulus as u128) as u64;
    }

    // Pad to degree m-1
    poly.resize(m, 0);

    poly
}

/// Multiply polynomial by linear factor (X - a)
///
/// Given poly p(X) of degree d, computes p(X) * (X - a)
///
/// Result has degree d+1
fn poly_mul_linear(poly: &[u64], a: u64, modulus: u64) -> Vec<u64> {
    if poly.is_empty() {
        return vec![0];
    }

    let n = poly.len();
    let mut result = vec![0u64; n + 1];

    // (p₀ + p₁X + ... + pₙXⁿ) * (X - a)
    // = -a·p₀ + (p₀ - a·p₁)X + (p₁ - a·p₂)X² + ... + pₙX^{n+1}

    for i in 0..n {
        // Coefficient of X^i: poly[i-1] - a * poly[i]
        // But for i=0: just -a * poly[0]

        // Add poly[i] to result[i+1] (shift left by X)
        result[i + 1] = ((result[i + 1] as u128 + poly[i] as u128) % modulus as u128) as u64;

        // Subtract a * poly[i] from result[i]
        let term = ((a as u128 * poly[i] as u128) % modulus as u128) as u64;
        if result[i] >= term {
            result[i] = result[i] - term;
        } else {
            result[i] = modulus - (term - result[i]);
        }
    }

    result
}

/// Lagrange interpolation from evaluations
///
/// Given evaluations [f(0), f(1), ..., f(m-1)], computes polynomial f(X)
/// of degree < m such that f(i) = evals[i] for all i.
///
/// Uses formula: f(X) = Σᵢ f(i) · L_i(X)
///
/// **Note**: Naïve O(m²) implementation. For production, use NTT (M5.2).
///
/// # Arguments
///
/// * `evals` - Evaluations at domain points [f(0), ..., f(m-1)]
/// * `modulus` - Field modulus q
///
/// # Returns
///
/// Coefficients of interpolated polynomial f(X), length m
pub(crate) fn lagrange_interpolate(evals: &[u64], modulus: u64) -> Vec<u64> {
    #[cfg(feature = "fft-ntt")]
    {
        lagrange_interpolate_ntt(evals, modulus)
    }

    #[cfg(not(feature = "fft-ntt"))]
    {
        lagrange_interpolate_baseline(evals, modulus)
    }
}

/// NTT-based polynomial interpolation (O(m log m)).
///
/// Uses inverse NTT to compute coefficients from evaluations.
/// Requires m to be power of 2 and modulus to be NTT_MODULUS.
///
/// # Arguments
///
/// * `evals` - Evaluations at roots of unity {1, ω, ω², ..., ω^(m-1)}
/// * `modulus` - Field modulus (must be NTT_MODULUS for NTT)
///
/// # Returns
///
/// Coefficients of interpolated polynomial f(X)
#[cfg(feature = "fft-ntt")]
fn lagrange_interpolate_ntt(evals: &[u64], modulus: u64) -> Vec<u64> {
    let m = evals.len();

    // Check if m is power of 2
    if !m.is_power_of_two() {
        // Fallback to baseline for non-power-of-2
        return lagrange_interpolate_baseline(evals, modulus);
    }

    // Check if modulus is NTT-friendly
    if modulus != NTT_MODULUS {
        // Fallback to baseline for incompatible modulus
        return lagrange_interpolate_baseline(evals, modulus);
    }

    // Compute primitive m-th root of unity
    let omega = compute_root_of_unity(m, NTT_MODULUS, NTT_PRIMITIVE_ROOT);

    // Inverse NTT: evaluations → coefficients
    ntt_inverse(evals, NTT_MODULUS, omega)
        .expect("NTT inverse failed (should not happen for valid inputs)")
}

/// Baseline Lagrange interpolation (O(m²)).
///
/// Naïve algorithm using Lagrange basis polynomials.
/// Used as fallback when NTT is disabled or m is not power of 2.
///
/// # Arguments
///
/// * `evals` - Evaluations at domain points {0, 1, ..., m-1}
/// * `modulus` - Field modulus q
///
/// # Returns
///
/// Coefficients of interpolated polynomial f(X)
fn lagrange_interpolate_baseline(evals: &[u64], modulus: u64) -> Vec<u64> {
    let m = evals.len();
    if m == 0 {
        return vec![];
    }

    let mut result = vec![0u64; m];

    for i in 0..m {
        // Compute L_i(X)
        let basis = lagrange_basis(i, m, modulus);

        // Add evals[i] * L_i(X) to result
        for j in 0..m {
            let term = ((evals[i] as u128 * basis[j] as u128) % modulus as u128) as u64;
            result[j] = ((result[j] as u128 + term as u128) % modulus as u128) as u64;
        }
    }

    result
}

/// Polynomial multiplication (naïve O(m²) convolution)
///
/// Computes c(X) = a(X) * b(X) where:
/// - a(X) has degree deg_a
/// - b(X) has degree deg_b
/// - c(X) has degree deg_a + deg_b
///
/// # Arguments
///
/// * `a` - Coefficients of polynomial a(X)
/// * `b` - Coefficients of polynomial b(X)
/// * `modulus` - Field modulus q
///
/// # Returns
///
/// Coefficients of product polynomial c(X)
fn poly_mul(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }

    let deg_a = a.len() - 1;
    let deg_b = b.len() - 1;
    let mut result = vec![0u64; deg_a + deg_b + 1];

    for i in 0..a.len() {
        for j in 0..b.len() {
            let term = ((a[i] as u128 * b[j] as u128) % modulus as u128) as u64;
            result[i + j] = ((result[i + j] as u128 + term as u128) % modulus as u128) as u64;
        }
    }

    result
}

/// Polynomial subtraction: a(X) - b(X)
///
/// # Arguments
///
/// * `a` - Coefficients of polynomial a(X)
/// * `b` - Coefficients of polynomial b(X)
/// * `modulus` - Field modulus q
///
/// # Returns
///
/// Coefficients of difference polynomial a(X) - b(X)
fn poly_sub(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
    let max_len = a.len().max(b.len());
    let mut result = vec![0u64; max_len];

    for i in 0..max_len {
        let a_val = if i < a.len() { a[i] } else { 0 };
        let b_val = if i < b.len() { b[i] } else { 0 };

        if a_val >= b_val {
            result[i] = a_val - b_val;
        } else {
            result[i] = modulus - (b_val - a_val);
        }
    }

    // Remove leading zeros
    while result.len() > 1 && result[result.len() - 1] == 0 {
        result.pop();
    }

    result
}

/// Polynomial addition: a(X) + b(X)
///
/// # Arguments
///
/// * `a` - Coefficients of polynomial a(X)
/// * `b` - Coefficients of polynomial b(X)
/// * `modulus` - Field modulus q
///
/// # Returns
///
/// Coefficients of sum polynomial a(X) + b(X)
pub fn poly_add(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
    let max_len = a.len().max(b.len());
    let mut result = vec![0u64; max_len];

    for i in 0..max_len {
        let a_val = if i < a.len() { a[i] } else { 0 };
        let b_val = if i < b.len() { b[i] } else { 0 };

        result[i] = ((a_val as u128 + b_val as u128) % modulus as u128) as u64;
    }

    // Remove leading zeros
    while result.len() > 1 && result[result.len() - 1] == 0 {
        result.pop();
    }

    result
}

/// Scalar multiplication: scalar · poly(X)
///
/// # Arguments
///
/// * `poly` - Coefficients of polynomial p(X)
/// * `scalar` - Scalar value r ∈ F_q
/// * `modulus` - Field modulus q
///
/// # Returns
///
/// Coefficients of product r · p(X)
pub fn poly_mul_scalar(poly: &[u64], scalar: u64, modulus: u64) -> Vec<u64> {
    poly.iter()
        .map(|&coeff| ((coeff as u128 * scalar as u128) % modulus as u128) as u64)
        .collect()
}

/// Compute vanishing polynomial Z_H(X) coefficients.
///
/// Domain H depends on `use_ntt`:
/// - **use_ntt = true**: H = roots of unity {1, ω, ..., ω^{m-1}}
///   → Z_H(X) = X^m - 1
/// - **use_ntt = false**: H = integers {0, 1, ..., m-1}  
///   → Z_H(X) = X(X-1)(X-2)...(X-(m-1))
///
/// # Arguments
///
/// * `m` - Domain size |H| = m
/// * `modulus` - Field modulus q
/// * `use_ntt` - If true, use roots-of-unity domain (X^m - 1)
///
/// # Returns
///
/// Coefficients of Z_H(X)
pub fn vanishing_poly(m: usize, modulus: u64, use_ntt: bool) -> Vec<u64> {
    if use_ntt {
        // NTT path: Z_H(X) = X^m - 1
        // Coefficients: [-1, 0, 0, ..., 0, 1]
        //                ^0  1  2      m-1 ^m
        let mut poly = vec![0u64; m + 1];
        poly[0] = modulus - 1; // -1 mod modulus
        poly[m] = 1; // X^m coefficient
        poly
    } else {
        // Baseline path: Z_H(X) = ∏(X - i) for i=0..m-1
        let mut poly = vec![1u64];
        for i in 0..m {
            poly = poly_mul_linear(&poly, i as u64, modulus);
        }
        poly
    }
}

/// Polynomial division by vanishing polynomial Z_H(X)
///
/// Computes quotient q(X) such that numerator(X) = q(X) * Z_H(X)
///
/// Returns Err if division is not exact (remainder ≠ 0), which indicates
/// the witness doesn't satisfy R1CS constraints.
///
/// # Arguments
///
/// * `numerator` - Coefficients of dividend polynomial
/// * `m` - Domain size (Z_H has roots at 0, 1, ..., m-1 or roots of unity)
/// * `modulus` - Field modulus q
/// * `use_ntt` - If true, Z_H(X) = X^m - 1; else Z_H(X) = ∏(X-i)
///
/// # Returns
///
/// Ok(q_coeffs) if division is exact, Err otherwise
fn poly_div_vanishing(
    numerator: &[u64],
    m: usize,
    modulus: u64,
    use_ntt: bool,
) -> Result<Vec<u64>, crate::Error> {
    if numerator.is_empty() {
        return Ok(vec![0]);
    }

    // Compute Z_H(X) with correct domain
    let divisor = vanishing_poly(m, modulus, use_ntt);

    // Perform polynomial long division
    let mut remainder = numerator.to_vec();
    let deg_num = remainder.len().saturating_sub(1);
    let deg_div = divisor.len().saturating_sub(1); // Should be m

    if deg_num < deg_div {
        // numerator has degree < deg(divisor), so quotient is 0
        // Check if numerator is zero
        if remainder.iter().all(|&x| x == 0) {
            return Ok(vec![0]);
        } else {
            return Err(crate::Error::Ffi(
                "Polynomial division by Z_H: remainder non-zero (witness invalid)".to_string(),
            ));
        }
    }

    let deg_quot = deg_num - deg_div;
    let mut quotient = vec![0u64; deg_quot + 1];

    // Long division: divide high-degree terms first
    for i in (0..=deg_quot).rev() {
        let idx = i + deg_div;
        if idx < remainder.len() && idx > 0 {
            // Leading coefficient of divisor
            let lead_div = divisor[deg_div];
            let lead_div_inv = mod_inverse(lead_div, modulus);

            // Quotient coefficient
            let q_coeff =
                ((remainder[idx] as u128 * lead_div_inv as u128) % modulus as u128) as u64;
            quotient[i] = q_coeff;

            // Subtract q_coeff * divisor from remainder
            for j in 0..divisor.len() {
                let term = ((q_coeff as u128 * divisor[j] as u128) % modulus as u128) as u64;
                let pos = i + j;
                if pos < remainder.len() {
                    if remainder[pos] >= term {
                        remainder[pos] -= term;
                    } else {
                        remainder[pos] = modulus - (term - remainder[pos]);
                    }
                }
            }
        }
    }

    // Check remainder: should be zero
    let has_nonzero = remainder.iter().any(|&x| x != 0);
    if has_nonzero {
        return Err(crate::Error::Ffi(
            "Polynomial division by Z_H: remainder non-zero (witness invalid)".to_string(),
        ));
    }

    // Remove leading zeros from quotient
    while quotient.len() > 1 && quotient[quotient.len() - 1] == 0 {
        quotient.pop();
    }

    Ok(quotient)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let a_rows = vec![vec![0, 1, 0, 0, 0, 0], vec![0, 0, 0, 1, 0, 0]];
        let a_matrix = SparseMatrix::from_dense(&a_rows);

        // B = [[0, 0, 1, 0, 0, 0],   (select b=z[2])
        //      [0, 0, 0, 0, 1, 0]]   (select d=z[4])
        let b_rows = vec![vec![0, 0, 1, 0, 0, 0], vec![0, 0, 0, 0, 1, 0]];
        let b_matrix = SparseMatrix::from_dense(&b_rows);

        // C = [[0, 0, 0, 1, 0, 0],   (select c=z[3])
        //      [0, 0, 0, 0, 0, 1]]   (select e=z[5])
        let c_rows = vec![vec![0, 0, 0, 1, 0, 0], vec![0, 0, 0, 0, 0, 1]];
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

    #[test]
    fn test_compute_constraint_evals_multiplication_gate() {
        let r1cs = create_multiplication_gate();
        let witness = vec![1, 7, 13, 91];

        let (a_evals, b_evals, c_evals) = r1cs.compute_constraint_evals(&witness);

        assert_eq!(a_evals.len(), 1);
        assert_eq!(b_evals.len(), 1);
        assert_eq!(c_evals.len(), 1);

        // For a*b=c: (Az)_0 = 7, (Bz)_0 = 13, (Cz)_0 = 91
        assert_eq!(a_evals[0], 7);
        assert_eq!(b_evals[0], 13);
        assert_eq!(c_evals[0], 91);
    }

    #[test]
    fn test_compute_constraint_evals_two_multiplications() {
        let r1cs = create_two_multiplications();
        let witness = vec![1, 2, 3, 6, 4, 24];

        let (a_evals, b_evals, c_evals) = r1cs.compute_constraint_evals(&witness);

        assert_eq!(a_evals.len(), 2);

        // Constraint 1: a*b=c → 2*3=6
        assert_eq!(a_evals[0], 2);
        assert_eq!(b_evals[0], 3);
        assert_eq!(c_evals[0], 6);

        // Constraint 2: c*d=e → 6*4=24
        assert_eq!(a_evals[1], 6);
        assert_eq!(b_evals[1], 4);
        assert_eq!(c_evals[1], 24);
    }

    #[test]
    fn test_compute_quotient_poly_valid_witness() {
        let r1cs = create_multiplication_gate();
        let witness = vec![1, 7, 13, 91];

        let result = r1cs.compute_quotient_poly(&witness);
        assert!(result.is_ok());

        let q_coeffs = result.unwrap();
        // Should return polynomial of degree < m
        assert_eq!(q_coeffs.len(), r1cs.num_constraints());
    }

    #[test]
    fn test_compute_quotient_poly_invalid_witness() {
        let r1cs = create_multiplication_gate();
        let witness = vec![1, 7, 13, 90]; // 7*13 = 91 ≠ 90

        let result = r1cs.compute_quotient_poly(&witness);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("does not satisfy"));
    }

    #[test]
    fn test_compute_quotient_poly_correctness() {
        // Test mathematical correctness: Q(α) * Z_H(α) = A_z(α) * B_z(α) - C_z(α)
        let r1cs = create_multiplication_gate();
        let witness = vec![1, 7, 13, 91];
        let modulus = r1cs.modulus;

        let q_coeffs = r1cs.compute_quotient_poly(&witness).unwrap();

        // Compute A_z, B_z, C_z polynomials
        let (a_evals, b_evals, c_evals) = r1cs.compute_constraint_evals(&witness);
        let a_poly = super::lagrange_interpolate(&a_evals, modulus);
        let b_poly = super::lagrange_interpolate(&b_evals, modulus);
        let c_poly = super::lagrange_interpolate(&c_evals, modulus);

        // Test at random point α = 42
        let alpha = 42u64;

        // Evaluate Q(α)
        let mut q_alpha = 0u64;
        for (i, &coeff) in q_coeffs.iter().enumerate() {
            let power = super::mod_pow(alpha, i as u64, modulus);
            let term = ((coeff as u128 * power as u128) % modulus as u128) as u64;
            q_alpha = ((q_alpha as u128 + term as u128) % modulus as u128) as u64;
        }

        // Evaluate A_z(α), B_z(α), C_z(α)
        let mut a_alpha = 0u64;
        let mut b_alpha = 0u64;
        let mut c_alpha = 0u64;

        for (i, &coeff) in a_poly.iter().enumerate() {
            let power = super::mod_pow(alpha, i as u64, modulus);
            let term = ((coeff as u128 * power as u128) % modulus as u128) as u64;
            a_alpha = ((a_alpha as u128 + term as u128) % modulus as u128) as u64;
        }

        for (i, &coeff) in b_poly.iter().enumerate() {
            let power = super::mod_pow(alpha, i as u64, modulus);
            let term = ((coeff as u128 * power as u128) % modulus as u128) as u64;
            b_alpha = ((b_alpha as u128 + term as u128) % modulus as u128) as u64;
        }

        for (i, &coeff) in c_poly.iter().enumerate() {
            let power = super::mod_pow(alpha, i as u64, modulus);
            let term = ((coeff as u128 * power as u128) % modulus as u128) as u64;
            c_alpha = ((c_alpha as u128 + term as u128) % modulus as u128) as u64;
        }

        // Evaluate Z_H(α) = α(α-1)(α-2)...(α-(m-1)) [integer domain]
        let m = r1cs.num_constraints();
        let z_h_poly = super::vanishing_poly(m, modulus, false); // baseline domain
        let mut z_h_alpha = 0u64;
        for (i, &coeff) in z_h_poly.iter().enumerate() {
            let power = super::mod_pow(alpha, i as u64, modulus);
            let term = ((coeff as u128 * power as u128) % modulus as u128) as u64;
            z_h_alpha = ((z_h_alpha as u128 + term as u128) % modulus as u128) as u64;
        }

        // Check: Q(α) * Z_H(α) = A_z(α) * B_z(α) - C_z(α)
        let lhs = ((q_alpha as u128 * z_h_alpha as u128) % modulus as u128) as u64;

        let ab_product = ((a_alpha as u128 * b_alpha as u128) % modulus as u128) as u64;
        let rhs = if ab_product >= c_alpha {
            ab_product - c_alpha
        } else {
            modulus - (c_alpha - ab_product)
        };

        assert_eq!(
            lhs, rhs,
            "Q(α) * Z_H(α) must equal A_z(α) * B_z(α) - C_z(α)"
        );
    }

    #[test]
    fn test_compute_quotient_poly_two_constraints() {
        // Test with two multiplication constraints
        let r1cs = create_two_multiplications();
        let witness = vec![1, 2, 3, 6, 4, 24];

        let q_coeffs = r1cs.compute_quotient_poly(&witness).unwrap();

        // Q(X) should have degree < m = 2
        assert!(q_coeffs.len() <= 2);

        // Verify at point α = 17
        let alpha = 17u64;
        let modulus = r1cs.modulus;

        let (a_evals, b_evals, c_evals) = r1cs.compute_constraint_evals(&witness);
        let a_poly = super::lagrange_interpolate(&a_evals, modulus);
        let b_poly = super::lagrange_interpolate(&b_evals, modulus);
        let c_poly = super::lagrange_interpolate(&c_evals, modulus);

        // Evaluate polynomials at α
        let eval_poly = |poly: &[u64]| -> u64 {
            let mut result = 0u64;
            for (i, &coeff) in poly.iter().enumerate() {
                let power = super::mod_pow(alpha, i as u64, modulus);
                let term = ((coeff as u128 * power as u128) % modulus as u128) as u64;
                result = ((result as u128 + term as u128) % modulus as u128) as u64;
            }
            result
        };

        let q_alpha = eval_poly(&q_coeffs);
        let a_alpha = eval_poly(&a_poly);
        let b_alpha = eval_poly(&b_poly);
        let c_alpha = eval_poly(&c_poly);

        // Z_H(α) = α(α-1) for domain {0, 1} [baseline]
        let m = r1cs.num_constraints();
        let z_h_poly = super::vanishing_poly(m, modulus, false); // baseline domain
        let z_h_alpha = eval_poly(&z_h_poly);

        // Verify Q(α) * Z_H(α) = A_z(α) * B_z(α) - C_z(α)
        let lhs = ((q_alpha as u128 * z_h_alpha as u128) % modulus as u128) as u64;
        let ab = ((a_alpha as u128 * b_alpha as u128) % modulus as u128) as u64;
        let rhs = if ab >= c_alpha {
            ab - c_alpha
        } else {
            modulus - (c_alpha - ab)
        };

        assert_eq!(lhs, rhs);
    }

    // ========================================================================
    // Polynomial Helper Tests
    // ========================================================================

    #[test]
    fn test_mod_pow() {
        let modulus = 17592186044417;

        // 2^10 mod q
        assert_eq!(super::mod_pow(2, 10, modulus), 1024);

        // 5^0 = 1
        assert_eq!(super::mod_pow(5, 0, modulus), 1);

        // 7^1 = 7
        assert_eq!(super::mod_pow(7, 1, modulus), 7);

        // Large exponent
        let result = super::mod_pow(123, 456, modulus);
        assert!(result < modulus);
    }

    #[test]
    fn test_mod_inverse() {
        let modulus = 17592186044417;

        // 2 * inv(2) ≡ 1 (mod q)
        let inv2 = super::mod_inverse(2, modulus);
        assert_eq!((2u128 * inv2 as u128) % modulus as u128, 1);

        // 7 * inv(7) ≡ 1 (mod q)
        let inv7 = super::mod_inverse(7, modulus);
        assert_eq!((7u128 * inv7 as u128) % modulus as u128, 1);

        // 13 * inv(13) ≡ 1 (mod q)
        let inv13 = super::mod_inverse(13, modulus);
        assert_eq!((13u128 * inv13 as u128) % modulus as u128, 1);
    }

    #[test]
    fn test_poly_mul_linear() {
        let modulus = 17592186044417;

        // (2) * (X - 3) = -6 + 2X = (q-6) + 2X
        let poly = vec![2];
        let result = super::poly_mul_linear(&poly, 3, modulus);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], modulus - 6); // -6 ≡ q-6
        assert_eq!(result[1], 2);

        // (1 + X) * (X - 0) = X + X² = 0 + 1·X + 1·X²
        let poly = vec![1, 1];
        let result = super::poly_mul_linear(&poly, 0, modulus);
        assert_eq!(result, vec![0, 1, 1]);
    }

    #[test]
    fn test_lagrange_basis_simple() {
        let modulus = 17592186044417;
        let m = 2; // Domain H = {0, 1}

        // L_0(X) for domain {0, 1}
        // L_0(X) = (X - 1) / (0 - 1) = -(X - 1) = 1 - X
        let l0 = super::lagrange_basis(0, m, modulus);
        assert_eq!(l0.len(), m);
        assert_eq!(l0[0], 1); // Constant term
        assert_eq!(l0[1], modulus - 1); // Coefficient of X (which is -1)

        // L_1(X) for domain {0, 1}
        // L_1(X) = (X - 0) / (1 - 0) = X
        let l1 = super::lagrange_basis(1, m, modulus);
        assert_eq!(l1.len(), m);
        assert_eq!(l1[0], 0); // No constant term
        assert_eq!(l1[1], 1); // Coefficient of X is 1
    }

    #[test]
    fn test_lagrange_basis_properties() {
        let modulus = 17592186044417;
        let m = 3; // Domain H = {0, 1, 2}

        // Kronecker delta: L_i(j) = δ_{ij}
        for i in 0..m {
            let li = super::lagrange_basis(i, m, modulus);

            for j in 0..m {
                // Evaluate L_i at point j
                let mut val = 0u64;
                for (k, &coeff) in li.iter().enumerate() {
                    let power = super::mod_pow(j as u64, k as u64, modulus);
                    let term = ((coeff as u128 * power as u128) % modulus as u128) as u64;
                    val = ((val as u128 + term as u128) % modulus as u128) as u64;
                }

                if i == j {
                    assert_eq!(val, 1, "L_{}({}) should be 1", i, j);
                } else {
                    assert_eq!(val, 0, "L_{}({}) should be 0", i, j);
                }
            }
        }
    }

    #[test]
    fn test_lagrange_interpolate_simple() {
        let modulus = 17592186044417;

        // Interpolate constant function f(X) = 5
        let evals = vec![5, 5, 5];
        let poly = super::lagrange_interpolate(&evals, modulus);

        // Should get constant polynomial [5, 0, 0]
        assert_eq!(poly[0], 5);
        assert_eq!(poly[1], 0);
        assert_eq!(poly[2], 0);
    }

    #[test]
    fn test_lagrange_interpolate_linear() {
        let modulus = 17592186044417;

        // Interpolate f(X) = X from evaluations [0, 1, 2]
        let evals = vec![0, 1, 2];
        let poly = super::lagrange_interpolate(&evals, modulus);

        // Should get f(X) = X, i.e., [0, 1, 0]
        assert_eq!(poly.len(), 3);
        assert_eq!(poly[0], 0); // Constant term
        assert_eq!(poly[1], 1); // X coefficient
        assert_eq!(poly[2], 0); // X² coefficient
    }

    #[test]
    fn test_poly_mul() {
        let modulus = 17592186044417;

        // (2 + 3X) * (1 + X) = 2 + 2X + 3X + 3X² = 2 + 5X + 3X²
        let a = vec![2, 3];
        let b = vec![1, 1];
        let c = super::poly_mul(&a, &b, modulus);

        assert_eq!(c.len(), 3);
        assert_eq!(c[0], 2);
        assert_eq!(c[1], 5);
        assert_eq!(c[2], 3);
    }

    #[test]
    fn test_poly_sub() {
        let modulus = 17592186044417;

        // (5 + 7X) - (2 + 3X) = 3 + 4X
        let a = vec![5, 7];
        let b = vec![2, 3];
        let c = super::poly_sub(&a, &b, modulus);

        assert_eq!(c.len(), 2);
        assert_eq!(c[0], 3);
        assert_eq!(c[1], 4);

        // Test underflow: (1) - (5) = -4 ≡ q-4 (mod q)
        let a = vec![1];
        let b = vec![5];
        let c = super::poly_sub(&a, &b, modulus);
        assert_eq!(c[0], modulus - 4);
    }

    #[test]
    fn test_poly_div_vanishing_exact() {
        let modulus = 17592186044417;
        let m = 2; // Domain H = {0, 1}

        // Z_H(X) = X(X-1) = X² - X [baseline domain]
        // Test: (X² - X) / (X² - X) should give quotient = 1

        let z_h = super::vanishing_poly(m, modulus, false); // baseline
                                                            // Z_H(X) = X² - X, so z_h = [0, modulus-1, 1] (constant=-0, X=-1, X²=1)
                                                            // Actually: X(X-1) = X² - X so coeffs are [0, -1, 1] in standard form

        let q = super::poly_div_vanishing(&z_h, m, modulus, false).unwrap();

        // Quotient should be constant 1
        assert_eq!(q, vec![1]);
    }

    #[test]
    fn test_vanishing_poly() {
        let modulus = 17592186044417;

        // For m=1, Z_H(X) = X - 0 = X [baseline domain]
        let z1 = super::vanishing_poly(1, modulus, false);
        assert_eq!(z1, vec![0, 1]); // 0 + 1·X

        // For m=2, Z_H(X) = X(X-1) = X² - X [baseline domain]
        let z2 = super::vanishing_poly(2, modulus, false);
        assert_eq!(z2.len(), 3);
        assert_eq!(z2[0], 0); // constant term
        assert_eq!(z2[1], modulus - 1); // X coefficient (-1)
        assert_eq!(z2[2], 1); // X² coefficient

        // For m=3, Z_H(X) = X(X-1)(X-2) = X³ - 3X² + 2X [baseline domain]
        let z3 = super::vanishing_poly(3, modulus, false);
        assert_eq!(z3.len(), 4);
        assert_eq!(z3[0], 0); // constant
        assert_eq!(z3[1], 2); // X coefficient (2)
        assert_eq!(z3[2], modulus - 3); // X² coefficient (-3)
        assert_eq!(z3[3], 1); // X³ coefficient
    }

    #[test]
    fn test_poly_div_vanishing_non_exact() {
        let modulus = 17592186044417;
        let m = 2;

        // Numerator: X² (not divisible by X² - X) [baseline domain]
        let numerator = vec![0, 0, 1];

        let result = super::poly_div_vanishing(&numerator, m, modulus, false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("remainder non-zero"));
    }

    // ========================================================================
    // Integration Tests for compute_quotient_poly
    // ========================================================================

    #[test]
    fn test_compute_quotient_poly_multiplication_gate() {
        let r1cs = create_multiplication_gate();
        let witness = vec![1, 7, 13, 91]; // 7 * 13 = 91

        let q = r1cs.compute_quotient_poly(&witness).unwrap();

        // Quotient should have degree < m (since division is exact)
        assert!(q.len() <= r1cs.num_constraints());

        // Verify: Q(X) * Z_H(X) = A_z(X) * B_z(X) - C_z(X)
        // For single constraint (m=1), Z_H(X) = X - 0 = X
        // We'll just check Q exists and has reasonable coefficients
        assert!(!q.is_empty());

        // All coefficients should be < modulus
        for &coeff in &q {
            assert!(coeff < r1cs.modulus);
        }
    }

    #[test]
    fn test_compute_quotient_poly_two_multiplications() {
        let r1cs = create_two_multiplications();
        let witness = vec![1, 2, 3, 6, 4, 24]; // 2*3=6, 6*4=24

        let q = r1cs.compute_quotient_poly(&witness).unwrap();

        // Quotient should have degree < m
        assert!(q.len() <= r1cs.num_constraints());
        assert!(!q.is_empty());

        // All coefficients should be < modulus
        for &coeff in &q {
            assert!(coeff < r1cs.modulus);
        }
    }

    #[test]
    fn test_compute_quotient_poly_evaluates_correctly() {
        let r1cs = create_multiplication_gate();
        let witness = vec![1, 7, 13, 91];

        let q = r1cs.compute_quotient_poly(&witness).unwrap();
        let (a_evals, b_evals, c_evals) = r1cs.compute_constraint_evals(&witness);

        // Interpolate polynomials
        let a_poly = super::lagrange_interpolate(&a_evals, r1cs.modulus);
        let b_poly = super::lagrange_interpolate(&b_evals, r1cs.modulus);
        let c_poly = super::lagrange_interpolate(&c_evals, r1cs.modulus);

        // Compute A_z(X) * B_z(X)
        let ab_poly = super::poly_mul(&a_poly, &b_poly, r1cs.modulus);

        // Verify at random point α
        // Q(α) * Z_H(α) should equal A_z(α) * B_z(α) - C_z(α)
        let alpha = 12345u64;

        let q_alpha = eval_poly(&q, alpha, r1cs.modulus);
        let a_alpha = eval_poly(&a_poly, alpha, r1cs.modulus);
        let b_alpha = eval_poly(&b_poly, alpha, r1cs.modulus);
        let c_alpha = eval_poly(&c_poly, alpha, r1cs.modulus);

        // Z_H(α) = α^m - 1 (for m=1, this is α - 1 = α^1 - 1)
        let zh_alpha = if alpha >= 1 {
            ((super::mod_pow(alpha, r1cs.m as u64, r1cs.modulus) as u128 + r1cs.modulus as u128
                - 1)
                % r1cs.modulus as u128) as u64
        } else {
            r1cs.modulus - 1
        };

        // Q(α) * Z_H(α)
        let lhs = ((q_alpha as u128 * zh_alpha as u128) % r1cs.modulus as u128) as u64;

        // A_z(α) * B_z(α) - C_z(α)
        let ab_alpha = ((a_alpha as u128 * b_alpha as u128) % r1cs.modulus as u128) as u64;
        let rhs = if ab_alpha >= c_alpha {
            ab_alpha - c_alpha
        } else {
            r1cs.modulus - (c_alpha - ab_alpha)
        };

        assert_eq!(
            lhs, rhs,
            "Q(α) * Z_H(α) should equal A_z(α) * B_z(α) - C_z(α)"
        );
    }

    /// Helper: Evaluate polynomial at point x
    fn eval_poly(poly: &[u64], x: u64, modulus: u64) -> u64 {
        let mut result = 0u64;
        let mut power = 1u64;

        for &coeff in poly {
            let term = ((coeff as u128 * power as u128) % modulus as u128) as u64;
            result = ((result as u128 + term as u128) % modulus as u128) as u64;
            power = ((power as u128 * x as u128) % modulus as u128) as u64;
        }

        result
    }
}
