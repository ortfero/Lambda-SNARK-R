//! Number Theoretic Transform (NTT) for polynomial operations.
//!
//! Implements Cooley-Tukey FFT/NTT algorithm for O(m log m) polynomial interpolation
//! and evaluation, replacing the O(m²) naïve Lagrange interpolation.
//!
//! # Algorithm
//!
//! Uses radix-2 Decimation-In-Time (DIT) Cooley-Tukey algorithm:
//! - Forward NTT: coefficients → evaluations at roots of unity
//! - Inverse NTT: evaluations → coefficients
//!
//! # Complexity
//!
//! - Time: O(m log m) where m is transform size (must be power of 2)
//! - Space: O(m) for in-place computation with bit-reversal
//!
//! # Requirements
//!
//! - Modulus q must be NTT-friendly: q ≡ 1 (mod 2^k) for k-bit transform
//! - Primitive root of unity ω exists: ω^(2^k) ≡ 1 (mod q), ω^(2^(k-1)) ≠ 1
//!
//! # Example
//!
//! ```ignore
//! use lambda_snark::ntt::{ntt_forward, ntt_inverse};
//! use lambda_snark_core::{NTT_MODULUS, NTT_PRIMITIVE_ROOT};
//!
//! // Polynomial coefficients (degree 3)
//! let coeffs = vec![1, 2, 3, 4];
//!
//! // Forward NTT: coefficients → evaluations
//! let evals = ntt_forward(&coeffs, NTT_MODULUS, NTT_PRIMITIVE_ROOT);
//!
//! // Inverse NTT: evaluations → coefficients
//! let recovered = ntt_inverse(&evals, NTT_MODULUS, NTT_PRIMITIVE_ROOT);
//!
//! assert_eq!(coeffs, recovered);
//! ```

use crate::arith::{add_mod, mod_inverse as arith_mod_inverse, mod_pow, mul_mod, sub_mod};
use crate::Error;

#[inline]
fn mod_inverse(a: u64, modulus: u64) -> Result<u64, Error> {
    arith_mod_inverse(a, modulus).ok_or(Error::InvalidDimensions)
}

/// Bit-reversal permutation for in-place NTT.
///
/// Rearranges array elements so that element at index i is swapped with
/// element at bit-reversed index.
///
/// # Arguments
///
/// * `data` - Array to permute (length must be power of 2)
///
/// # Example
///
/// ```ignore
/// let mut data = vec![0, 1, 2, 3];
/// bit_reverse_permutation(&mut data);
/// // data = [0, 2, 1, 3] (bit-reversed indices)
/// ```
fn bit_reverse_permutation(data: &mut [u64]) {
    let n = data.len();
    let log_n = n.trailing_zeros() as usize;

    for i in 0..n {
        let j = reverse_bits(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Reverse the lower `bits` bits of `x`.
///
/// # Example
///
/// ```ignore
/// reverse_bits(0b101, 3) = 0b101 (palindrome)
/// reverse_bits(0b110, 3) = 0b011
/// reverse_bits(5, 4) = 10  // 0b0101 → 0b1010
/// ```
#[inline]
fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Forward NTT: coefficients → evaluations at roots of unity.
///
/// Transforms polynomial f(X) = Σ f_i X^i into evaluations at ω^0, ω^1, ..., ω^(n-1)
/// where ω is a primitive n-th root of unity.
///
/// # Arguments
///
/// * `coeffs` - Polynomial coefficients [f_0, f_1, ..., f_(n-1)]
/// * `modulus` - Prime modulus q (must be NTT-friendly)
/// * `omega` - Primitive n-th root of unity (ω^n ≡ 1 mod q)
///
/// # Returns
///
/// Evaluations [f(ω^0), f(ω^1), ..., f(ω^(n-1))]
///
/// # Panics
///
/// Panics if `coeffs.len()` is not a power of 2.
///
/// # Complexity
///
/// O(n log n) where n = coeffs.len()
pub fn ntt_forward(coeffs: &[u64], modulus: u64, omega: u64) -> Vec<u64> {
    let n = coeffs.len();
    assert!(
        n.is_power_of_two(),
        "NTT size must be power of 2, got {}",
        n
    );

    if n == 1 {
        return coeffs.to_vec();
    }

    let mut data = coeffs.to_vec();

    // Step 1: Bit-reversal permutation
    bit_reverse_permutation(&mut data);

    // Step 2: Cooley-Tukey butterfly operations
    let log_n = n.trailing_zeros() as usize;

    for s in 1..=log_n {
        let m = 1 << s; // 2^s
        let m_half = m >> 1;

        // Twiddle factor: ω^(n/m) is primitive m-th root
        let omega_m = mod_pow(omega, (n / m) as u64, modulus);

        for k in (0..n).step_by(m) {
            let mut omega_power = 1u64;

            for j in 0..m_half {
                let t = mul_mod(data[k + j + m_half], omega_power, modulus);
                let u = data[k + j];

                // Butterfly: (u, t) → (u + t, u - t)
                data[k + j] = add_mod(u, t, modulus);
                data[k + j + m_half] = sub_mod(u, t, modulus);

                // Update twiddle factor: ω^j → ω^(j+1)
                omega_power = mul_mod(omega_power, omega_m, modulus);
            }
        }
    }

    data
}

/// Inverse NTT: evaluations → coefficients.
///
/// Inverse transform: given evaluations [f(ω^0), ..., f(ω^(n-1))],
/// recovers coefficients [f_0, ..., f_(n-1)].
///
/// # Arguments
///
/// * `evals` - Evaluations at roots of unity
/// * `modulus` - Prime modulus q
/// * `omega` - Primitive n-th root of unity
///
/// # Returns
///
/// Polynomial coefficients
///
/// # Complexity
///
/// O(n log n) where n = evals.len()
pub fn ntt_inverse(evals: &[u64], modulus: u64, omega: u64) -> Result<Vec<u64>, Error> {
    let n = evals.len();
    assert!(n.is_power_of_two(), "NTT size must be power of 2");

    if n == 1 {
        return Ok(evals.to_vec());
    }

    // Inverse NTT = Forward NTT with ω^(-1) and division by n
    let omega_inv = mod_inverse(omega, modulus)?;
    let mut coeffs = ntt_forward(evals, modulus, omega_inv);

    // Divide by n (normalization)
    let n_inv = mod_inverse(n as u64, modulus)?;
    for c in coeffs.iter_mut() {
        *c = mul_mod(*c, n_inv, modulus);
    }

    Ok(coeffs)
}

/// Compute primitive n-th root of unity for NTT-friendly modulus.
///
/// For q = 2^64 - 2^32 + 1, uses generator g=7 to compute ω = g^((q-1)/n).
///
/// # Arguments
///
/// * `n` - Transform size (must be power of 2, n ≤ 2^32)
/// * `modulus` - NTT-friendly prime modulus
/// * `primitive_root` - Primitive 2^32-th root of unity (for n ≤ 2^32)
///
/// # Returns
///
/// Primitive n-th root of unity ω
///
/// # Example
///
/// ```ignore
/// use lambda_snark_core::{NTT_MODULUS, NTT_PRIMITIVE_ROOT};
///
/// // For 1024-point NTT (2^10)
/// let omega = compute_root_of_unity(1024, NTT_MODULUS, NTT_PRIMITIVE_ROOT);
/// assert_eq!(mod_pow(omega, 1024, NTT_MODULUS), 1);
/// ```
pub fn compute_root_of_unity(n: usize, modulus: u64, primitive_root: u64) -> u64 {
    assert!(n.is_power_of_two(), "n must be power of 2");
    assert!(n <= (1 << 32), "n must be ≤ 2^32 for NTT_MODULUS");

    // For primitive_root being 2^32-th root, compute ω = primitive_root^(2^32 / n)
    let exponent = (1u64 << 32) / n as u64;
    mod_pow(primitive_root, exponent, modulus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambda_snark_core::{NTT_MODULUS, NTT_PRIMITIVE_ROOT};

    const Q: u64 = NTT_MODULUS;
    const OMEGA_32: u64 = NTT_PRIMITIVE_ROOT; // Primitive 2^32-th root

    #[test]
    fn test_bit_reverse() {
        assert_eq!(reverse_bits(0b000, 3), 0b000);
        assert_eq!(reverse_bits(0b001, 3), 0b100);
        assert_eq!(reverse_bits(0b010, 3), 0b010);
        assert_eq!(reverse_bits(0b011, 3), 0b110);
        assert_eq!(reverse_bits(0b100, 3), 0b001);
        assert_eq!(reverse_bits(0b101, 3), 0b101);
        assert_eq!(reverse_bits(0b110, 3), 0b011);
        assert_eq!(reverse_bits(0b111, 3), 0b111);
    }

    #[test]
    fn test_bit_reverse_permutation() {
        let mut data = vec![0, 1, 2, 3];
        bit_reverse_permutation(&mut data);
        assert_eq!(data, vec![0, 2, 1, 3]);

        let mut data8 = vec![0, 1, 2, 3, 4, 5, 6, 7];
        bit_reverse_permutation(&mut data8);
        assert_eq!(data8, vec![0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    fn test_compute_root_of_unity() {
        // For 2-point NTT
        let omega_2 = compute_root_of_unity(2, Q, OMEGA_32);
        assert_eq!(mod_pow(omega_2, 2, Q), 1, "ω_2^2 ≡ 1");
        assert_eq!(omega_2, Q - 1, "ω_2 ≡ -1 (primitive 2nd root)");

        // For 4-point NTT
        let omega_4 = compute_root_of_unity(4, Q, OMEGA_32);
        assert_eq!(mod_pow(omega_4, 4, Q), 1, "ω_4^4 ≡ 1");
        assert_eq!(mod_pow(omega_4, 2, Q), Q - 1, "ω_4^2 ≡ -1 (primitivity)");

        // For 8-point NTT
        let omega_8 = compute_root_of_unity(8, Q, OMEGA_32);
        assert_eq!(mod_pow(omega_8, 8, Q), 1, "ω_8^8 ≡ 1");
        assert_eq!(mod_pow(omega_8, 4, Q), Q - 1, "ω_8^4 ≡ -1 (primitivity)");
    }

    #[test]
    fn test_ntt_2_point() {
        // f(X) = 1 + 2X, evaluate at {1, -1}
        let coeffs = vec![1, 2];
        let omega = compute_root_of_unity(2, Q, OMEGA_32); // ω = -1

        let evals = ntt_forward(&coeffs, Q, omega);

        // f(1) = 1 + 2 = 3
        // f(-1) = 1 - 2 = q-1 (≡ -1)
        assert_eq!(evals[0], 3);
        assert_eq!(evals[1], Q - 1);

        // Inverse NTT should recover coefficients
        let recovered = ntt_inverse(&evals, Q, omega).unwrap();
        assert_eq!(recovered, coeffs);
    }

    #[test]
    fn test_ntt_4_point() {
        // f(X) = 1 + 2X + 3X^2 + 4X^3
        let coeffs = vec![1, 2, 3, 4];
        let omega = compute_root_of_unity(4, Q, OMEGA_32);

        let evals = ntt_forward(&coeffs, Q, omega);

        // Verify: f(ω^k) for k=0,1,2,3
        // f(1) = 1 + 2 + 3 + 4 = 10
        assert_eq!(evals[0], 10);

        // Inverse NTT
        let recovered = ntt_inverse(&evals, Q, omega).unwrap();
        assert_eq!(recovered, coeffs);
    }

    #[test]
    fn test_ntt_8_point() {
        let coeffs = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let omega = compute_root_of_unity(8, Q, OMEGA_32);

        let evals = ntt_forward(&coeffs, Q, omega);

        // f(1) = 1+2+3+4+5+6+7+8 = 36
        assert_eq!(evals[0], 36);

        let recovered = ntt_inverse(&evals, Q, omega).unwrap();
        assert_eq!(recovered, coeffs);
    }

    #[test]
    fn test_ntt_inverse_correctness() {
        for log_n in 1..=10 {
            let n = 1 << log_n;
            let omega = compute_root_of_unity(n, Q, OMEGA_32);

            // Random coefficients
            let coeffs: Vec<u64> = (0..n).map(|i| mul_mod(i as u64, 123456789, Q)).collect();

            let evals = ntt_forward(&coeffs, Q, omega);
            let recovered = ntt_inverse(&evals, Q, omega).unwrap();

            assert_eq!(recovered, coeffs, "NTT roundtrip failed for n={}", n);
        }
    }

    #[test]
    fn test_ntt_linearity() {
        // NTT is linear: NTT(a·f + b·g) = a·NTT(f) + b·NTT(g)
        let f = vec![1, 2, 3, 4];
        let g = vec![5, 6, 7, 8];
        let a = 3u64;
        let b = 7u64;
        let omega = compute_root_of_unity(4, Q, OMEGA_32);

        // Compute a·f + b·g
        let mut linear_combo = vec![0u64; 4];
        for i in 0..4 {
            let af = mul_mod(a, f[i], Q);
            let bg = mul_mod(b, g[i], Q);
            linear_combo[i] = add_mod(af, bg, Q);
        }

        // NTT(a·f + b·g)
        let ntt_combo = ntt_forward(&linear_combo, Q, omega);

        // a·NTT(f) + b·NTT(g)
        let ntt_f = ntt_forward(&f, Q, omega);
        let ntt_g = ntt_forward(&g, Q, omega);
        let mut expected = vec![0u64; 4];
        for i in 0..4 {
            let af = mul_mod(a, ntt_f[i], Q);
            let bg = mul_mod(b, ntt_g[i], Q);
            expected[i] = add_mod(af, bg, Q);
        }

        assert_eq!(ntt_combo, expected, "NTT linearity violated");
    }
}
