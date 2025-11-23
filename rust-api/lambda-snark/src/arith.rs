//! Modular arithmetic utilities shared across the ΛSNARK-R stack.
//!
//! These helpers provide a single implementation of modular exponentiation
//! and inversion so that timing characterisation and constant-time rewrites
//! have a single place to update.

#[inline]
pub fn mul_mod(a: u64, b: u64, modulus: u64) -> u64 {
    if modulus <= 1 {
        return 0;
    }

    ((a as u128 * b as u128) % modulus as u128) as u64
}

#[inline]
pub fn add_mod(a: u64, b: u64, modulus: u64) -> u64 {
    if modulus <= 1 {
        return 0;
    }

    let modulus_128 = modulus as u128;
    let sum = a as u128 + b as u128;
    let needs_sub = (sum >= modulus_128) as u128;
    (sum - modulus_128 * needs_sub) as u64
}

#[inline]
pub fn sub_mod(a: u64, b: u64, modulus: u64) -> u64 {
    if modulus <= 1 {
        return 0;
    }

    let modulus_128 = modulus as u128;
    let diff = a as u128 + modulus_128 - b as u128;
    let needs_sub = (diff >= modulus_128) as u128;
    (diff - modulus_128 * needs_sub) as u64
}

/// Compute (base^exp) mod modulus using a constant-time square-and-multiply.
#[inline]
pub fn mod_pow(mut base: u64, exponent: u64, modulus: u64) -> u64 {
    if modulus <= 1 {
        return 0;
    }

    let mut result = 1u64;
    base %= modulus;
    let mut exp = exponent;

    for _ in 0..64 {
        let multiplied = mul_mod(result, base, modulus);
        let mask = 0u64.wrapping_sub((exp & 1) as u64);
        result = (multiplied & mask) | (result & !mask);

        base = mul_mod(base, base, modulus);
        exp >>= 1;
    }

    result
}

/// Compute the modular inverse using Fermat's little theorem for odd moduli,
/// falling back to the extended Euclidean algorithm when necessary.
#[inline]
pub fn mod_inverse(value: u64, modulus: u64) -> Option<u64> {
    if value == 0 || modulus <= 1 {
        return None;
    }

    let reduced = value % modulus;
    if reduced == 0 {
        return None;
    }

    if modulus & 1 == 1 {
        let exponent = modulus.wrapping_sub(2);
        let candidate = mod_pow(reduced, exponent, modulus);
        if mul_mod(candidate, reduced, modulus) == 1 % modulus {
            return Some(candidate);
        }
    }

    mod_inverse_euclid(reduced, modulus)
}

fn mod_inverse_euclid(value: u64, modulus: u64) -> Option<u64> {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (modulus as i128, value as i128);

    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }

    if r != 1 {
        return None;
    }

    if t < 0 {
        t += modulus as i128;
    }

    Some(t as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: u64 = 17592186044417; // 2^44 + 1

    #[test]
    fn test_add_mod_wraps_correctly() {
        let a = M - 5;
        let b = 10;
        assert_eq!(add_mod(a, b, M), 5);

        let c = 12345;
        let d = 67890;
        assert_eq!(add_mod(c, d, M), (c + d) % M);
    }

    #[test]
    fn test_sub_mod_wraps_correctly() {
        let a = 3;
        let b = 5;
        assert_eq!(sub_mod(a, b, M), M - 2);

        let c = 987654321;
        let d = 123456789;
        assert_eq!(sub_mod(c, d, M), (c + M - d) % M);
    }

    #[test]
    fn test_mul_mod_matches_reference() {
        let a = M - 12345;
        let b = 67890;
        assert_eq!(
            mul_mod(a, b, M),
            ((a as u128 * b as u128) % M as u128) as u64
        );
    }
}
