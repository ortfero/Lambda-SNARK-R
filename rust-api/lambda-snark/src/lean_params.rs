//! Lean 4 parameter validation and import.
//!
//! This module provides functionality to import security parameters from Lean 4 definitions
//! and validate them against implementation requirements. This prevents parameter mismatch
//! bugs like VULN-001 (non-prime modulus).
//!
//! # Example
//!
//! ```no_run
//! use lambda_snark::lean_params::{SecurityParams, validate_params};
//!
//! let lean_def = r#"
//!     { n := 4096, k := 2, q := 12289, σ := 3.2, λ := 128 }
//! "#;
//!
//! let params = SecurityParams::from_lean(lean_def).unwrap();
//! assert!(validate_params(&params).is_ok());
//! ```
//!
//! # Milestone
//!
//! Prototype for M10 (Completeness & Integration)
//! - M10.3: Import Lean params → Rust validation
//! - Target: April 2026

use crate::Error;

/// Security parameters (matches Lean definition).
#[derive(Debug, Clone, PartialEq)]
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
    /// Parse security parameters from Lean 4 record syntax.
    ///
    /// Expected format:
    /// ```lean
    /// { n := 4096, k := 2, q := 12289, σ := 3.2, λ := 128 }
    /// ```
    pub fn from_lean(lean_str: &str) -> Result<Self, Error> {
        // Simple parser for Lean record syntax
        let trimmed = lean_str.trim();

        if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
            return Err(Error::InvalidInput(
                "Lean params must be in record syntax { ... }".to_string(),
            ));
        }

        let content = &trimmed[1..trimmed.len() - 1];
        let mut n = None;
        let mut k = None;
        let mut q = None;
        let mut sigma = None;
        let mut lambda = None;

        for field in content.split(',') {
            let parts: Vec<&str> = field.split(":=").collect();
            if parts.len() != 2 {
                continue;
            }

            let key = parts[0].trim();
            let value = parts[1].trim();

            match key {
                "n" => {
                    n = Some(
                        value
                            .parse()
                            .map_err(|_| Error::InvalidInput(format!("Invalid n: {}", value)))?,
                    )
                }
                "k" => {
                    k = Some(
                        value
                            .parse()
                            .map_err(|_| Error::InvalidInput(format!("Invalid k: {}", value)))?,
                    )
                }
                "q" => {
                    q = Some(
                        value
                            .parse()
                            .map_err(|_| Error::InvalidInput(format!("Invalid q: {}", value)))?,
                    )
                }
                "σ" | "sigma" => {
                    sigma = Some(
                        value
                            .parse()
                            .map_err(|_| Error::InvalidInput(format!("Invalid σ: {}", value)))?,
                    )
                }
                "λ" | "lambda" => {
                    lambda = Some(
                        value
                            .parse()
                            .map_err(|_| Error::InvalidInput(format!("Invalid λ: {}", value)))?,
                    )
                }
                _ => {} // Ignore unknown fields
            }
        }

        Ok(Self {
            n: n.ok_or_else(|| Error::InvalidInput("Missing field: n".to_string()))?,
            k: k.ok_or_else(|| Error::InvalidInput("Missing field: k".to_string()))?,
            q: q.ok_or_else(|| Error::InvalidInput("Missing field: q".to_string()))?,
            sigma: sigma.ok_or_else(|| Error::InvalidInput("Missing field: σ".to_string()))?,
            lambda: lambda.ok_or_else(|| Error::InvalidInput("Missing field: λ".to_string()))?,
        })
    }
}

/// Validate security parameters against implementation requirements.
///
/// Checks:
/// 1. Modulus q is prime (prevents VULN-001)
/// 2. LWE dimension n is power of 2 (for NTT)
/// 3. Gaussian width σ ≥ 3.0 (security requirement)
/// 4. Security level λ ∈ {128, 192, 256}
pub fn validate_params(params: &SecurityParams) -> Result<(), Error> {
    // Check 1: Modulus primality (critical for field operations)
    if !is_prime(params.q) {
        return Err(Error::InvalidInput(format!(
            "Modulus q={} is not prime! This breaks field assumption (see VULN-001).",
            params.q
        )));
    }

    // Check 2: n is power of 2 (required for NTT)
    if !params.n.is_power_of_two() {
        return Err(Error::InvalidInput(format!(
            "LWE dimension n={} must be power of 2 for NTT optimization",
            params.n
        )));
    }

    // Check 3: Gaussian width (security requirement)
    if params.sigma < 3.0 {
        return Err(Error::InvalidInput(format!(
            "Gaussian width σ={} is too small (minimum 3.0 for security)",
            params.sigma
        )));
    }

    // Check 4: Security level
    if ![128, 192, 256].contains(&params.lambda) {
        return Err(Error::InvalidInput(format!(
            "Security level λ={} not supported (must be 128, 192, or 256)",
            params.lambda
        )));
    }

    Ok(())
}

/// Simple primality test (Miller-Rabin, deterministic for q < 2^64).
///
/// Uses first 12 primes as witnesses (sufficient for u64).
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    // Miller-Rabin with deterministic witnesses for u64
    let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    // Write n-1 as 2^r * d
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    'witness: for &a in &witnesses {
        if a >= n {
            continue;
        }

        let mut x = mod_pow(a, d, n);

        if x == 1 || x == n - 1 {
            continue 'witness;
        }

        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }

        return false; // Composite
    }

    true // Probably prime
}

/// Modular exponentiation: (base^exp) mod m
fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;

    while exp > 0 {
        if exp % 2 == 1 {
            result = mod_mul(result, base, m);
        }
        base = mod_mul(base, base, m);
        exp /= 2;
    }

    result
}

/// Modular multiplication: (a * b) mod m (avoiding overflow)
fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_lean_params() {
        let lean_str = r#"{ n := 4096, k := 2, q := 12289, σ := 3.2, λ := 128 }"#;
        let params = SecurityParams::from_lean(lean_str).unwrap();

        assert_eq!(params.n, 4096);
        assert_eq!(params.k, 2);
        assert_eq!(params.q, 12289);
        assert_eq!(params.sigma, 3.2);
        assert_eq!(params.lambda, 128);
    }

    #[test]
    fn test_parse_with_whitespace() {
        let lean_str = r#"
            {
                n := 4096,
                k := 2,
                q := 12289,
                σ := 3.2,
                λ := 128
            }
        "#;
        let params = SecurityParams::from_lean(lean_str).unwrap();
        assert_eq!(params.q, 12289);
    }

    #[test]
    fn test_validate_valid_params() {
        let params = SecurityParams {
            n: 4096,
            k: 2,
            q: 12289, // Prime
            sigma: 3.2,
            lambda: 128,
        };

        assert!(validate_params(&params).is_ok());
    }

    #[test]
    fn test_validate_non_prime_modulus() {
        let params = SecurityParams {
            n: 4096,
            k: 2,
            q: 17592186044417, // Composite (VULN-001 example)
            sigma: 3.2,
            lambda: 128,
        };

        let result = validate_params(&params);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not prime"));
    }

    #[test]
    fn test_validate_non_power_of_two_n() {
        let params = SecurityParams {
            n: 4095, // Not power of 2
            k: 2,
            q: 12289,
            sigma: 3.2,
            lambda: 128,
        };

        let result = validate_params(&params);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("power of 2"));
    }

    #[test]
    fn test_validate_small_sigma() {
        let params = SecurityParams {
            n: 4096,
            k: 2,
            q: 12289,
            sigma: 2.5, // Too small
            lambda: 128,
        };

        let result = validate_params(&params);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_primality() {
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(12289));
        assert!(is_prime(17592186044423)); // Next prime after VULN-001

        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(!is_prime(4));
        assert!(!is_prime(17592186044417)); // VULN-001 composite
    }
}
