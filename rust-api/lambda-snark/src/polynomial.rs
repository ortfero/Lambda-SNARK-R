//! Polynomial encoding and operations for ΛSNARK-R.
//!
//! This module provides polynomial representation and evaluation for witness encoding.

use lambda_snark_core::Field;
use crate::Error;

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
        let coeffs = witness.iter()
            .map(|&val| Field::new(val % modulus))
            .collect();
        
        Self { coeffs, modulus }
    }
    
    /// Evaluate polynomial at point α.
    ///
    /// Computes f(α) = Σ coeffs[i] · α^i mod q using Horner's method.
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
        
        for coeff in self.coeffs.iter().rev().skip(1) {
            // result = result * alpha + coeff (mod q)
            result = (mul_mod(result, alpha.value(), self.modulus) + coeff.value()) % self.modulus;
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
}

/// Modular multiplication: (a * b) mod m
///
/// Uses 128-bit intermediate to avoid overflow.
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
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
        
        // Direct: 1 + 2·123 + 3·123² + 4·123³ + 5·123⁴
        let mut direct = 1u128;
        let mut power = 1u128;
        for &coeff in &witness[1..] {
            power = (power * alpha as u128) % TEST_MODULUS as u128;
            direct = (direct + coeff as u128 * power) % TEST_MODULUS as u128;
        }
        
        // Horner
        let horner = p.evaluate(Field::new(alpha)).value();
        
        assert_eq!(horner, direct as u64);
    }
}
