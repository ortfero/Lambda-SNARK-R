//! Polynomial encoding and operations for ΛSNARK-R.
//!
//! This module provides polynomial representation and evaluation for witness encoding.

use crate::arith::{add_mod, mul_mod};
use lambda_snark_core::Field;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Polynomial over F_q.
#[derive(Clone, Debug, PartialEq)]
pub struct Polynomial {
    /// Coefficients: f(X) = coeffs[0] + coeffs[1]*X + coeffs[2]*X² + ...
    coeffs: Vec<Field>,

    /// Field modulus
    modulus: u64,
}

impl Polynomial {
    /// Create polynomial from coefficients.
    ///
    /// # Arguments
    /// * `coeffs` - Polynomial coefficients (constant term first)
    /// * `modulus` - Field modulus q
    ///
    /// # Example
    /// ```
    /// use lambda_snark::Polynomial;
    /// use lambda_snark_core::Field;
    ///
    /// // f(X) = 1 + 7X + 13X²
    /// let f = Polynomial::new(vec![
    ///     Field::new(1),
    ///     Field::new(7),
    ///     Field::new(13),
    /// ], 17592186044417);
    /// ```
    pub fn new(coeffs: Vec<Field>, modulus: u64) -> Self {
        Self { coeffs, modulus }
    }

    /// Encode witness as polynomial.
    ///
    /// Maps witness vector `z = [z_0, z_1, ..., z_{n-1}]` to polynomial
    /// `f(X) = Σ z_i · X^i`.
    ///
    /// # Arguments
    /// * `witness` - Witness vector
    /// * `modulus` - Field modulus q
    ///
    /// # Returns
    /// Polynomial f with deg(f) = len(witness) - 1
    ///
    /// # Example
    /// ```
    /// use lambda_snark::Polynomial;
    ///
    /// // TV-1: witness = [1, 7, 13, 91]
    /// let witness = vec![1, 7, 13, 91];
    /// let f = Polynomial::from_witness(&witness, 17592186044417);
    ///
    /// // f(X) = 1 + 7X + 13X² + 91X³
    /// assert_eq!(f.degree(), 3);
    /// ```
    pub fn from_witness(witness: &[u64], modulus: u64) -> Self {
        let coeffs = witness
            .iter()
            .map(|&val| Field::new(val % modulus))
            .collect();

        Self { coeffs, modulus }
    }

    /// Evaluate polynomial at point α.
    ///
    /// Computes f(α) = Σ coeffs\[i\] · α^i mod q using Horner's method.
    ///
    /// # Arguments
    /// * `alpha` - Evaluation point
    ///
    /// # Returns
    /// f(α) ∈ F_q
    ///
    /// # Example
    /// ```
    /// use lambda_snark::Polynomial;
    /// use lambda_snark_core::Field;
    ///
    /// let f = Polynomial::from_witness(&[1, 7, 13, 91], 17592186044417);
    /// let alpha = Field::new(2);
    ///
    /// // f(2) = 1 + 7·2 + 13·4 + 91·8 = 1 + 14 + 52 + 728 = 795
    /// let result = f.evaluate(alpha);
    /// assert_eq!(result.value(), 795);
    /// ```
    pub fn evaluate(&self, alpha: Field) -> Field {
        if self.coeffs.is_empty() {
            return Field::new(0);
        }

        // Horner's method: f(x) = a_0 + x(a_1 + x(a_2 + x(...)))
        let mut result = self.coeffs.last().unwrap().value();
        let alpha_val = alpha.value();

        for coeff in self.coeffs.iter().rev().skip(1) {
            // Horner step with constant-time modular helpers
            let product = mul_mod(result, alpha_val, self.modulus);
            result = add_mod(product, coeff.value(), self.modulus);
        }

        Field::new(result)
    }

    /// Get polynomial degree.
    ///
    /// Degree is highest non-zero coefficient index, or 0 for zero polynomial.
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Get number of coefficients.
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    /// Check if polynomial is zero.
    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Get coefficient at index i.
    pub fn coeff(&self, i: usize) -> Option<Field> {
        self.coeffs.get(i).copied()
    }

    /// Convert polynomial coefficients to Field vector for commitment.
    ///
    /// Returns coefficients as `Vec<Field>` suitable for passing to `Commitment::new()`.
    pub fn coefficients(&self) -> &[Field] {
        &self.coeffs
    }

    /// Generate random blinding polynomial for zero-knowledge.
    ///
    /// Creates polynomial r(X) = r₀ + r₁·X + ... + r_d·X^d where each r_i is
    /// uniformly random in F_q. Used to hide witness polynomial f(X) by computing
    /// f'(X) = f(X) + r(X).
    ///
    /// # Arguments
    /// * `degree` - Polynomial degree (must match witness polynomial degree)
    /// * `modulus` - Field modulus q
    /// * `seed` - Optional RNG seed (None = cryptographically secure random)
    ///
    /// # Returns
    /// Random polynomial r with deg(r) = degree
    ///
    /// # Security
    /// - Uses ChaCha20 CSPRNG (cryptographically secure)
    /// - If seed=None, uses OS entropy via `from_entropy()`
    /// - Coefficients uniformly distributed over F_q
    ///
    /// # Example
    /// ```
    /// use lambda_snark::Polynomial;
    ///
    /// // Generate random degree-3 blinding polynomial
    /// let r = Polynomial::random_blinding(3, 17592186044417, None);
    /// assert_eq!(r.degree(), 3);
    ///
    /// // Deterministic blinding (for testing)
    /// let r1 = Polynomial::random_blinding(3, 17592186044417, Some(42));
    /// let r2 = Polynomial::random_blinding(3, 17592186044417, Some(42));
    /// assert_eq!(r1, r2); // Same seed → same polynomial
    /// ```
    pub fn random_blinding(degree: usize, modulus: u64, seed: Option<u64>) -> Self {
        let mut rng = if let Some(s) = seed {
            ChaCha20Rng::seed_from_u64(s)
        } else {
            ChaCha20Rng::from_entropy()
        };

        let coeffs: Vec<Field> = (0..=degree)
            .map(|_| Field::new(rng.gen::<u64>() % modulus))
            .collect();

        Self { coeffs, modulus }
    }

    /// Add two polynomials: (f + g)(X) = f(X) + g(X) mod q.
    ///
    /// Performs coefficient-wise addition in F_q. If polynomials have different
    /// degrees, the result has max degree. Used for blinding: f'(X) = f(X) + r(X).
    ///
    /// # Arguments
    /// * `other` - Polynomial to add
    ///
    /// # Panics
    /// Panics if polynomials have different moduli
    ///
    /// # Example
    /// ```
    /// use lambda_snark::Polynomial;
    ///
    /// let q = 17592186044417;
    /// let f = Polynomial::from_witness(&[1, 2, 3], q);
    /// let r = Polynomial::from_witness(&[10, 20, 30], q);
    ///
    /// // (f + r)(X) = (1+10) + (2+20)X + (3+30)X² = 11 + 22X + 33X²
    /// let blinded = f.add(&r);
    /// assert_eq!(blinded.coeff(0).unwrap().value(), 11);
    /// assert_eq!(blinded.coeff(1).unwrap().value(), 22);
    /// assert_eq!(blinded.coeff(2).unwrap().value(), 33);
    /// ```
    pub fn add(&self, other: &Polynomial) -> Self {
        assert_eq!(
            self.modulus, other.modulus,
            "Cannot add polynomials with different moduli: {} vs {}",
            self.modulus, other.modulus
        );

        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = self.coeffs.get(i).map(|f| f.value()).unwrap_or(0);
            let b = other.coeffs.get(i).map(|f| f.value()).unwrap_or(0);
            result.push(Field::new(add_mod(a, b, self.modulus)));
        }

        Self {
            coeffs: result,
            modulus: self.modulus,
        }
    }

    /// Get field modulus.
    pub fn modulus(&self) -> u64 {
        self.modulus
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODULUS: u64 = 17592186044417; // 2^44 + 1

    #[test]
    fn test_polynomial_new() {
        let coeffs = vec![Field::new(1), Field::new(2), Field::new(3)];
        let p = Polynomial::new(coeffs.clone(), TEST_MODULUS);

        assert_eq!(p.degree(), 2);
        assert_eq!(p.len(), 3);
        assert_eq!(p.coeff(0), Some(Field::new(1)));
        assert_eq!(p.coeff(1), Some(Field::new(2)));
        assert_eq!(p.coeff(2), Some(Field::new(3)));
    }

    #[test]
    fn test_from_witness() {
        // TV-1: witness = [1, 7, 13, 91]
        let witness = vec![1, 7, 13, 91];
        let p = Polynomial::from_witness(&witness, TEST_MODULUS);

        assert_eq!(p.degree(), 3);
        assert_eq!(p.coeff(0), Some(Field::new(1)));
        assert_eq!(p.coeff(1), Some(Field::new(7)));
        assert_eq!(p.coeff(2), Some(Field::new(13)));
        assert_eq!(p.coeff(3), Some(Field::new(91)));
    }

    #[test]
    fn test_evaluate_simple() {
        // f(X) = 1 + 2X + 3X²
        let p = Polynomial::from_witness(&[1, 2, 3], TEST_MODULUS);

        // f(0) = 1
        assert_eq!(p.evaluate(Field::new(0)).value(), 1);

        // f(1) = 1 + 2 + 3 = 6
        assert_eq!(p.evaluate(Field::new(1)).value(), 6);

        // f(2) = 1 + 4 + 12 = 17
        assert_eq!(p.evaluate(Field::new(2)).value(), 17);
    }

    #[test]
    fn test_evaluate_tv1() {
        // TV-1: f(X) = 1 + 7X + 13X² + 91X³
        let p = Polynomial::from_witness(&[1, 7, 13, 91], TEST_MODULUS);

        // f(2) = 1 + 14 + 52 + 728 = 795
        assert_eq!(p.evaluate(Field::new(2)).value(), 795);

        // f(10) = 1 + 70 + 1300 + 91000 = 92371
        assert_eq!(p.evaluate(Field::new(10)).value(), 92371);
    }

    #[test]
    fn test_evaluate_modular() {
        // Large coefficients that wrap around modulus
        let large_val = TEST_MODULUS - 1;
        let p = Polynomial::from_witness(&[large_val, large_val], TEST_MODULUS);

        // f(X) = (q-1) + (q-1)X
        // f(2) = (q-1) + 2(q-1) = (q-1)(1 + 2) = 3(q-1) mod q
        // 3(q-1) = 3q - 3 ≡ -3 ≡ q-3 (mod q)
        let result = p.evaluate(Field::new(2));
        let expected = TEST_MODULUS - 3;
        assert_eq!(result.value(), expected);
    }

    #[test]
    fn test_empty_polynomial() {
        let p = Polynomial::new(vec![], TEST_MODULUS);

        assert!(p.is_empty());
        assert_eq!(p.degree(), 0);
        assert_eq!(p.evaluate(Field::new(5)).value(), 0);
    }

    #[test]
    fn test_horner_correctness() {
        // Verify Horner's method matches direct evaluation
        let witness = vec![1, 2, 3, 4, 5];
        let p = Polynomial::from_witness(&witness, TEST_MODULUS);
        let alpha = 123u64;

        // Direct evaluation via modular helpers
        let mut direct = 0u64;
        let mut power = 1u64;
        for &coeff in &witness {
            let term = mul_mod(coeff % TEST_MODULUS, power, TEST_MODULUS);
            direct = add_mod(direct, term, TEST_MODULUS);
            power = mul_mod(power, alpha % TEST_MODULUS, TEST_MODULUS);
        }

        // Horner
        let horner = p.evaluate(Field::new(alpha)).value();

        assert_eq!(horner, direct);
    }

    // --- Zero-Knowledge: Blinding Polynomial Tests ---

    #[test]
    fn test_random_blinding_degree() {
        for degree in [0, 1, 3, 10] {
            let r = Polynomial::random_blinding(degree, TEST_MODULUS, None);
            assert_eq!(r.degree(), degree);
            assert_eq!(r.len(), degree + 1);
        }
    }

    #[test]
    fn test_random_blinding_deterministic() {
        let seed = 42u64;

        let r1 = Polynomial::random_blinding(5, TEST_MODULUS, Some(seed));
        let r2 = Polynomial::random_blinding(5, TEST_MODULUS, Some(seed));

        // Same seed → identical polynomials
        assert_eq!(r1, r2);
        for i in 0..=5 {
            assert_eq!(r1.coeff(i), r2.coeff(i));
        }
    }

    #[test]
    fn test_random_blinding_different_seeds() {
        let r1 = Polynomial::random_blinding(3, TEST_MODULUS, Some(1));
        let r2 = Polynomial::random_blinding(3, TEST_MODULUS, Some(2));

        // Different seeds → different polynomials (with high probability)
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_random_blinding_range() {
        // Coefficients should be in [0, modulus)
        let r = Polynomial::random_blinding(10, TEST_MODULUS, Some(123));

        for i in 0..=10 {
            let coeff = r.coeff(i).unwrap().value();
            assert!(
                coeff < TEST_MODULUS,
                "Coefficient {} = {} exceeds modulus",
                i,
                coeff
            );
        }
    }

    #[test]
    fn test_random_blinding_non_deterministic() {
        // Without seed, should produce different results (high probability)
        let r1 = Polynomial::random_blinding(3, TEST_MODULUS, None);
        let r2 = Polynomial::random_blinding(3, TEST_MODULUS, None);

        // Probability of collision: (1/q)^4 ≈ 2^-176 (negligible)
        assert_ne!(
            r1, r2,
            "Random polynomials should differ (collision is negligible)"
        );
    }

    #[test]
    fn test_polynomial_add_simple() {
        let f = Polynomial::from_witness(&[1, 2, 3], TEST_MODULUS);
        let g = Polynomial::from_witness(&[10, 20, 30], TEST_MODULUS);

        let sum = f.add(&g);

        // (f + g)(X) = 11 + 22X + 33X²
        assert_eq!(sum.coeff(0).unwrap().value(), 11);
        assert_eq!(sum.coeff(1).unwrap().value(), 22);
        assert_eq!(sum.coeff(2).unwrap().value(), 33);
        assert_eq!(sum.degree(), 2);
    }

    #[test]
    fn test_polynomial_add_different_degrees() {
        let f = Polynomial::from_witness(&[1, 2], TEST_MODULUS); // deg = 1
        let g = Polynomial::from_witness(&[10, 20, 30, 40], TEST_MODULUS); // deg = 3

        let sum = f.add(&g);

        // (f + g)(X) = 11 + 22X + 30X² + 40X³
        assert_eq!(sum.coeff(0).unwrap().value(), 11);
        assert_eq!(sum.coeff(1).unwrap().value(), 22);
        assert_eq!(sum.coeff(2).unwrap().value(), 30);
        assert_eq!(sum.coeff(3).unwrap().value(), 40);
        assert_eq!(sum.degree(), 3);
    }

    #[test]
    fn test_polynomial_add_modular() {
        let f = Polynomial::from_witness(&[TEST_MODULUS - 1], TEST_MODULUS);
        let g = Polynomial::from_witness(&[5], TEST_MODULUS);

        let sum = f.add(&g);

        // (q-1) + 5 ≡ 4 (mod q)
        assert_eq!(sum.coeff(0).unwrap().value(), 4);
    }

    #[test]
    fn test_polynomial_add_commutativity() {
        let f = Polynomial::from_witness(&[1, 2, 3], TEST_MODULUS);
        let g = Polynomial::from_witness(&[10, 20, 30], TEST_MODULUS);

        let sum1 = f.add(&g);
        let sum2 = g.add(&f);

        assert_eq!(sum1, sum2);
    }

    #[test]
    fn test_polynomial_add_evaluation() {
        let f = Polynomial::from_witness(&[1, 2, 3], TEST_MODULUS);
        let g = Polynomial::from_witness(&[10, 20, 30], TEST_MODULUS);
        let alpha = Field::new(5);

        let sum = f.add(&g);

        // (f + g)(α) = f(α) + g(α)
        let expected = add_mod(
            f.evaluate(alpha).value(),
            g.evaluate(alpha).value(),
            TEST_MODULUS,
        );
        assert_eq!(sum.evaluate(alpha).value(), expected);
    }

    #[test]
    #[should_panic(expected = "Cannot add polynomials with different moduli")]
    fn test_polynomial_add_different_moduli() {
        let f = Polynomial::from_witness(&[1, 2], 101);
        let g = Polynomial::from_witness(&[3, 4], 103);

        let _ = f.add(&g); // Should panic
    }

    #[test]
    fn test_blinding_hides_witness() {
        let witness = vec![1, 7, 13, 91];
        let f = Polynomial::from_witness(&witness, TEST_MODULUS);

        let r = Polynomial::random_blinding(f.degree(), TEST_MODULUS, Some(999));
        let blinded = f.add(&r);

        // Blinded polynomial should differ from original
        assert_ne!(blinded, f);

        // But evaluation relationship holds: blinded(α) = f(α) + r(α)
        let alpha = Field::new(123);
        let expected = add_mod(
            f.evaluate(alpha).value(),
            r.evaluate(alpha).value(),
            TEST_MODULUS,
        );
        assert_eq!(blinded.evaluate(alpha).value(), expected);
    }
}
